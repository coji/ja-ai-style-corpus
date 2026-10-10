# /// script
# requires-python = ">=3.11"
# ///
"""generate.py — AI 以前の人間の文書と同じ話題で、生成AIに文章を書かせる(data/generated の作り方)。

プロンプト(どちらも「書く」条件。人間の文章を推敲させると、AI 時代の言い回しがあまり入らない):
    qiita  AI 以前(2019〜2021年8月)の Qiita 記事のタイトル
           「「{title}」というタイトルで技術ブログの記事を書いて。本文は{chars}字程度で。」
    biz    LLM-jp コーパス由来の AI 以前の文章(組織の発信・報道・行政・コラム・大学)の、冒頭の地の文3文
           「次の内容を伝える{kind}を書いて。本文は{chars}字程度で。\\n\\n内容: {lead}」
    {chars} は元の文書の地の文の長さを500字単位に丸め、1,500〜5,000字に収めたもの

書き手(PROVENANCE。各文書の出どころとして、ベンダーだけでなくモデル・effort・ハーネス(または API)・設定を残す):
    claude  Claude Code の claude -p
    codex   Codex CLI の codex exec
    gemini  Gemini API の Batch モード
    CLI は実行者の個人設定(ユーザー設定・ルール・AGENTS.md・プラグイン)を読まないよう、ホームディレクトリと
    このリポジトリの外に置いた空の作業ディレクトリと HOME で動かす。システムプロンプトは各 CLI の標準のまま

段階:
    build   プロンプトを作る(取り出した人間の文書から。既に使った文書と検査用の文書は除く)
    run     書かせる(<out>/<engine>/<id>.md と <out>/meta.jsonl)。済んだものは飛ばす

使い方:
    uv run scripts/generate.py build --qiita work/qiita-pre2022.jsonl.gz --llmjp work/llmjp.jsonl \
        --n-qiita 750 --n-biz 750 --out work/gen/prompts.json
    uv run scripts/generate.py run --prompts work/gen/prompts.json --out work/gen [--engines claude codex gemini]
    # 公開したプロンプトで書かせ直す(同じモデルでも出力は毎回変わる)
    uv run scripts/generate.py run --prompts-from data/generated/generated-v1.jsonl.gz --out work/regen --limit 20
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QIITA_PROMPT = "「{title}」というタイトルで技術ブログの記事を書いて。本文は{chars}字程度で。"
BIZ_PROMPT = "次の内容を伝える{kind}を書いて。本文は{chars}字程度で。\n\n内容: {lead}"
KIND = {"組織の発信": "会社や団体のお知らせ", "報道・ビジネス誌": "記事", "行政": "自治体や官公庁のお知らせ",
        "コラム・解説": "解説記事", "大学・研究機関": "大学や研究機関のお知らせ"}
# CLI に本文だけを書かせるための、運用上の指示。文体や自然さには触れない。
# ないと、短いプロンプトに対して確認の質問だけを返し、本文が得られないことがある
NONINTERACTIVE = ("これは1往復のみの非対話バッチ呼び出しです。フォローアップの質問はできません。"
                  "確認や許可を求めず、指示に対して適切と思う内容を自分で判断し、記事本文だけをそのまま出力してください。")
PROVENANCE = {
    "claude": {"vendor": "Anthropic", "model": "claude-opus-5-5", "model_arg": "opus", "effort": "low",
               "harness": "Claude Code (claude -p)",
               "settings": "--setting-sources project(ユーザー設定・ルール・プラグイン・hook を読まない)、"
                           "--disallowed-tools *、作業ディレクトリは空、"
                           "--append-system-prompt で「1往復のみの非対話バッチ。確認を求めず本文だけを出力」と追記。"
                           "それ以外は Claude Code 標準のシステムプロンプトのまま"},
    "codex": {"vendor": "OpenAI", "model": "gpt-6.1-sol", "model_arg": "gpt-6.1-sol", "effort": "low",
              "harness": "Codex CLI (codex exec)",
              "settings": "-c model_reasoning_effort=low、-c web_search=\"disabled\"、--sandbox read-only、"
                          "空の CODEX_HOME と HOME(個人設定・AGENTS.md を読まない)、"
                          "プロンプトの先頭に「1往復のみの非対話バッチ。確認を求めず本文だけを出力」を付ける。"
                          "それ以外は Codex 標準のシステムプロンプトのまま"},
    "gemini": {"vendor": "Google", "model": "gemini-3.8-flash", "model_arg": "gemini-3.8-flash", "effort": "low",
               "harness": "Gemini API v1beta Batch (models/gemini-3.8-flash:batchGenerateContent)",
               "settings": "generation_config.thinkingConfig.thinkingLevel=low、システムプロンプトなし、"
                           "非対話の指示も付けない(そのため冒頭に「承知いたしました」などの返事が付くことがある)"},
}
API = "https://generativelanguage.googleapis.com/v1beta"
# 隔離ディレクトリ。各 CLI は作業ディレクトリから親をたどって CLAUDE.md・AGENTS.md などを読むので、
# ホームディレクトリとリポジトリの外に置く
ISO = Path(os.environ.get("GEN_ISOLATION_DIR", "/tmp/ja-ai-style-gen"))
CWD, HOME, CODEX_HOME = ISO / "cwd", ISO / "home", ISO / "codex-home"
_lock = threading.Lock()


def clamp_chars(n: int) -> int:
    return max(1500, min(5000, round(n / 500) * 500))


def read_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


# ---------- build ----------

def build(a) -> None:
    rng = random.Random(a.seed)
    root = Path(__file__).resolve().parents[1]
    used = {r["prompt_id"] for r in read_jsonl(str(root / "data/generated/sources-v1.jsonl"))}
    tests = {l.strip() for f in (root / "data/eval").glob("test-*.txt") for l in open(f) if l.strip()}
    q = [d for d in read_jsonl(a.qiita) if d["year"] <= 2021 and (d.get("prose_chars") or 0) >= 1000 and d.get("title")]
    q = [d for d in q if f"q-{d['id']}" not in used and d["id"] not in tests]
    biz = [d for d in read_jsonl(a.llmjp) if (d.get("score") or 0) >= 0.5 and d.get("type") in KIND and len(d["text"]) >= 1000]
    biz = [d for d in biz if f"b-{d['docId']}" not in used and d["docId"] not in tests]
    prompts = []
    for d in rng.sample(q, min(a.n_qiita, len(q))):
        prompts.append({"id": f"q-{d['id']}", "src": "qiita",
                        "prompt": QIITA_PROMPT.format(title=d["title"], chars=clamp_chars(d["prose_chars"]))})
    for d in rng.sample(biz, min(a.n_biz, len(biz))):
        # メニューや見出しの残骸を避け、句点で終わる20字以上の行(地の文)から冒頭の3文を取る
        prose = [l.strip() for l in d["text"].split("\n") if re.fullmatch(r"[^#|/>＞]{20,}[。！？]", l.strip())]
        sents = [s for s in re.split(r"(?<=[。！？])", "".join(prose)) if len(s) >= 15]
        lead = "".join(sents[:3])[:300]
        if len(lead) < 60:
            continue
        prompts.append({"id": f"b-{d['docId']}", "src": "biz", "type": d["type"],
                        "prompt": BIZ_PROMPT.format(kind=KIND[d["type"]], chars=clamp_chars(len(d["text"])), lead=lead)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(prompts)} 件のプロンプト")


# ---------- run ----------

def prepare_isolation() -> None:
    for d in (CWD, HOME, CODEX_HOME):
        d.mkdir(parents=True, exist_ok=True)
    auth = Path.home() / ".codex" / "auth.json"  # 認証情報だけは実環境から写す
    if auth.exists():
        shutil.copy2(auth, CODEX_HOME / "auth.json")


def run_claude(prompt: str) -> str:
    pv = PROVENANCE["claude"]
    r = subprocess.run(["claude", "-p", prompt, "--model", pv["model_arg"], "--effort", pv["effort"],
                        "--setting-sources", "project", "--disallowed-tools", "*", "--append-system-prompt", NONINTERACTIVE],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600, check=True, cwd=CWD)
    return r.stdout.strip()


def run_codex(prompt: str, codex_home: Path) -> str:
    pv = PROVENANCE["codex"]
    cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-c", f"model_reasoning_effort={pv['effort']}",
           "-c", 'web_search="disabled"', "-m", pv["model_arg"], f"{NONINTERACTIVE}\n\n{prompt}"]
    env = {**os.environ, "HOME": str(HOME), "CODEX_HOME": str(codex_home)}
    # 標準入力は閉じる(開いていると codex exec が追加の入力を待つ)
    r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600, check=True, cwd=CWD, env=env)
    return r.stdout.strip()


def harness_version(engine: str) -> str:
    cmd = {"claude": ["claude", "--version"], "codex": ["codex", "--version"]}.get(engine)
    if not cmd:
        return "v1beta"
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_meta(out: Path, engine: str, p: dict, version: str, **extra) -> None:
    pv = PROVENANCE[engine]
    row = {"id": p["id"], "engine": engine, "src": p.get("src"), "type": p.get("type"), "vendor": pv["vendor"],
           "model": pv["model"], "effort": pv["effort"], "harness": pv["harness"], "harness_version": version,
           "settings": pv["settings"], "prompt": p["prompt"], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    with _lock, open(out / "meta.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cli(jobs: list[tuple[str, dict]], out: Path, parallel: int) -> None:
    prepare_isolation()
    versions = {e: harness_version(e) for e in ("claude", "codex")}

    def work(job):
        i, (engine, p) = job
        path = out / engine / f"{p['id']}.md"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if engine == "claude":
                text = run_claude(p["prompt"])
            else:
                # codex はプロセス間で認証情報を共有するとトークンの更新がぶつかるので、並列ごとに分ける
                home = ISO / f"codex-home-{i % parallel}"
                home.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CODEX_HOME / "auth.json", home / "auth.json")
                text = run_codex(p["prompt"], home)
            if len(text) > 300:
                path.write_text(text, encoding="utf-8")
                write_meta(out, engine, p, versions[engine], chars=len(text))
            print(f"  {engine} {p['id']} {len(text)} 字", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {engine} {p['id']}: {e}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(work, enumerate(jobs)))


def api(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("GEMINI_API_KEY") or sys.exit("環境変数 GEMINI_API_KEY を設定する")
    req = urllib.request.Request(f"{API}/{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run_gemini_batch(prompts: list[dict], out: Path) -> None:
    pv = PROVENANCE["gemini"]
    (out / "gemini").mkdir(parents=True, exist_ok=True)
    state_path = out / "gemini-batch.json"
    todo = [p for p in prompts if not (out / "gemini" / f"{p['id']}.md").exists()]
    if state_path.exists():  # 作成済みのバッチの結果を待つ
        state = json.loads(state_path.read_text())
    elif todo:
        reqs = [{"request": {"contents": [{"role": "user", "parts": [{"text": p["prompt"]}]}],
                             "generation_config": {"thinkingConfig": {"thinkingLevel": pv["effort"]}}},
                 "metadata": {"key": p["id"]}} for p in todo]
        res = api("POST", f"models/{pv['model_arg']}:batchGenerateContent",
                  {"batch": {"display_name": f"ja-ai-style-{int(time.time())}", "input_config": {"requests": {"requests": reqs}}}})
        state = {"name": res["name"], "n": len(reqs)}
        state_path.write_text(json.dumps(state))
        print(f"Gemini のバッチを作った: {state['name']}({len(reqs)} 件)", flush=True)
    else:
        return
    while True:
        st = api("GET", state["name"])
        s = st.get("metadata", {}).get("state", "")
        if st.get("done"):
            break
        print(f"  Gemini: {s}", flush=True)
        time.sleep(60)
    if s not in ("JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"):
        print(f"Gemini のバッチが失敗した: {s} {st.get('error')}", file=sys.stderr)
        return
    inl = st.get("response", {}).get("inlinedResponses", [])
    if isinstance(inl, dict):  # {"inlinedResponses": [...]} の形で返ることがある
        inl = inl.get("inlinedResponses", [])
    by_id = {p["id"]: p for p in prompts}
    for r in inl:
        key, resp = r.get("metadata", {}).get("key"), r.get("response")
        if not key or not resp:
            continue
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(pt.get("text", "") for pt in parts if not pt.get("thought"))
        if len(text) > 300:
            (out / "gemini" / f"{key}.md").write_text(text, encoding="utf-8")
            write_meta(out, "gemini", by_id.get(key, {"id": key, "prompt": ""}), "v1beta", chars=len(text),
                       model_version=resp.get("modelVersion"), response_id=resp.get("responseId"), batch=state["name"],
                       usage=resp.get("usageMetadata"))
    state_path.unlink()
    print("Gemini: 完了", flush=True)


def run(a) -> None:
    if a.prompts:
        prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    else:  # 公開データのプロンプトを使う(同じプロンプトは1回だけ)
        seen, prompts = set(), []
        for r in read_jsonl(a.prompts_from):
            if r["prompt_id"] not in seen:
                seen.add(r["prompt_id"])
                prompts.append({"id": r["prompt_id"], "src": r["src"], "type": r.get("type"), "prompt": r["prompt"]})
    prompts = prompts[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # 指定しなければ、プロンプトを書き手に順番に割り当てる(--all-engines で全員に書かせる)
    jobs = [(e, p) for p in prompts for e in a.engines] if a.all_engines else \
        [(a.engines[i % len(a.engines)], p) for i, p in enumerate(prompts)]
    gem = [p for e, p in jobs if e == "gemini"]
    cli = [(e, p) for e, p in jobs if e != "gemini"]
    with ThreadPoolExecutor(2) as ex:
        fs = [ex.submit(run_gemini_batch, gem, out) if gem else None, ex.submit(run_cli, cli, out, a.parallel) if cli else None]
        for f in fs:
            if f:
                f.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--qiita", required=True, help="fetch_qiita.py で取り出した記事")
    b.add_argument("--llmjp", required=True, help="fetch_llmjp.py で取り出した文書")
    b.add_argument("--n-qiita", type=int, default=750)
    b.add_argument("--n-biz", type=int, default=750)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompts", help="build で作ったプロンプト")
    g.add_argument("--prompts-from", help="data/generated/generated-v1.jsonl.gz")
    r.add_argument("--out", required=True)
    r.add_argument("--engines", nargs="+", default=list(PROVENANCE), choices=list(PROVENANCE))
    r.add_argument("--all-engines", action="store_true")
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--limit", type=int)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else run(a)


if __name__ == "__main__":
    main()
