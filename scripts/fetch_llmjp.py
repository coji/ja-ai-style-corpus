# /// script
# requires-python = ">=3.11"
# ///
"""fetch_llmjp.py — 一覧に載った文書の本文を、LLM-jp コーパス v4 の配布元から取得する。

一覧(data/baseline/llmjp-explanatory-pre2022-v1.jsonl、data/generated/sources-v1.jsonl)には本文を含めていない。
該当するファイルを配布元から流し読みして、本文の SHA-256 が一覧と一致するものだけを書き出す(1ファイルで数分かかる)。
配布元から文書が削除されていれば取得できない(飛ばして件数を報告する)。

使い方:
    uv run scripts/fetch_llmjp.py --list data/baseline/llmjp-explanatory-pre2022-v1.jsonl --out work/llmjp.jsonl [--min-score 0.7]
    # 生成のプロンプトの元になった文書だけ
    uv run scripts/fetch_llmjp.py --list data/generated/sources-v1.jsonl --out work/llmjp-sources.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zlib
from collections import defaultdict

BASE = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4/-/raw/main/"


def stream_lines(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "ja-ai-style-corpus/1.0 (research)"})
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = ""
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += d.decompress(chunk).decode("utf-8", "ignore")
            *lines, buf = buf.split("\n")
            yield from lines
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    want = [json.loads(l) for l in open(a.list, encoding="utf-8")]
    want = [w for w in want if w.get("kind", "llmjp") == "llmjp" and w.get("source") and w.get("sha256")]
    want = [w for w in want if (w.get("score") or 1) >= a.min_score][: a.limit]
    by_source = defaultdict(dict)
    for w in want:
        by_source[w["source"].removeprefix("llm-jp-corpus-v4/")][w["docId"]] = w
    found = mismatch = 0
    with open(a.out, "w", encoding="utf-8") as fp:
        for src, docs in by_source.items():
            for line in stream_lines(BASE + src):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                w = docs.get(rec.get("meta", {}).get("docId"))
                if not w:
                    continue
                text = rec.get("text") or ""
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != w["sha256"]:
                    mismatch += 1
                    continue
                fp.write(json.dumps({**w, "text": text}, ensure_ascii=False) + "\n")
                found += 1
            print(f"{src}: 累計 {found} 件", file=sys.stderr)
    print(f"一覧 {len(want)} 件、取得 {found} 件、本文が一致しない {mismatch} 件、見つからない {len(want) - found - mismatch} 件")


if __name__ == "__main__":
    main()
