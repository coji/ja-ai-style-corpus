# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.44", "peft", "huggingface_hub", "fugashi", "unidic-lite"]
# ///
"""score.py — 検査用の文書の各語に差を付け、AI 以前と AI 時代の文章で、差が大きい語の量を比べる。

検査セット(data/eval/): AI 以前の Qiita 2022年8月と LLM-jp の文書、Qiita 2026年8月(各500件の ID)、
学習に使っていない生成記事100本。人間の文書の本文は fetch_qiita.py・fetch_llmjp.py で取り出す。
各セットから100文書、1文書あたり最大40文を無作為に選ぶ(公開した結果と同じ)。

使い方:
    uv run scripts/score.py --set qiita2022=work/qiita-pre2022.jsonl.gz qiita2026=work/qiita-2026.jsonl.gz \
        llmjp=work/llmjp.jsonl generated_holdout=data/eval/generated-holdout-v0.jsonl.gz --out work/scores
    uv run scripts/score.py --summary --out work/scores
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlm import ok_word  # noqa: E402
from textutil import sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def docs_of(name: str, path: str) -> list[str]:
    if name == "generated_holdout":
        return [r["text"] for r in read_jsonl(path)]
    ids = [l.strip() for l in open(ROOT / f"data/eval/test-{name}.txt") if l.strip()]
    by_id = {d.get("id") or d.get("docId"): d.get("body") or d.get("text") for d in read_jsonl(path)}
    return [by_id[i] for i in ids if i in by_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", default="work/scores")
    ap.add_argument("--models")
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--max-sents", type=int, default=40)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--threshold", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.set:
        from mlm import Pair
        pair = Pair(a.models)
        for spec in a.set:
            name, path = spec.split("=", 1)
            docs = docs_of(name, path)
            rng = random.Random(0)
            docs = docs if len(docs) <= a.docs else rng.sample(docs, a.docs)
            with open(out / f"scores-{name}.jsonl", "w", encoding="utf-8") as f:
                for di, d in enumerate(docs):
                    ss = sentences(d)
                    if len(ss) > a.max_sents:
                        ss = random.Random(di).sample(ss, a.max_sents)
                    for s in ss:
                        f.write(json.dumps({"doc": di, "s": s, "words": pair.score(s)}, ensure_ascii=False) + "\n")
            print(f"{name}: {len(docs)} 文書", flush=True)
    if a.summary or not a.set:
        print(f"| 検査セット | 文 | 差 > {a.threshold} の語(地の文1,000字あたり) |\n|---|---|---|")
        for p in sorted(out.glob("scores-*.jsonl")):
            rows = list(read_jsonl(p))
            chars = sum(len(r["s"]) for r in rows) or 1
            n = sum(1 for r in rows for w in r["words"] if ok_word(w) and w["d"] > a.threshold)
            print(f"| {p.stem.removeprefix('scores-')} | {len(rows)} | {n / chars * 1000:.2f} |")


if __name__ == "__main__":
    main()
