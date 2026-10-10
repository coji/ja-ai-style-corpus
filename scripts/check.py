# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.44", "peft", "huggingface_hub", "fugashi", "unidic-lite"]
# ///
"""check.py — 文章の中で、AI 時代のモデルだけが選ぶ語(AI 以前の人ならあまり書かない語)を示す。

語の表は使わない。2つのモデル(mlm.py)に各語を伏せて当てさせ、差 d = log P_B − log P_A が閾値を超えた語を出す。
示すのは「疑い」で、直すかどうかは文脈で決める。専門用語・新しい話題の語・文字どおりの意味で使った語も出ることがある。

直すときは、意味を保つ言い換えの候補を人か LLM が出し、--rank で並べる(候補をモデルに出させると質が低い)。

使い方:
    uv run scripts/check.py draft.md                 # 差 > 3 の語を文ごとに表示
    uv run scripts/check.py draft.md --threshold 4 --json
    uv run scripts/check.py --rank "この設定が効く。" 効く 影響する 有効だ 役に立つ
    cat draft.md | uv run scripts/check.py -
初回は2つのモデル(約420MB)と tokenizer を Hugging Face Hub からダウンロードする。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlm import Pair, ok_word  # noqa: E402
from textutil import sentences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="Markdown かテキスト。- で標準入力")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--models", help="手元のモデルのディレクトリか Hub のリポジトリ名(指定しなければ mlm.MODEL_REPO)")
    ap.add_argument("--rank", nargs="+", metavar=("SENTENCE", "OLD"), help="文、元の語、候補…")
    a = ap.parse_args()
    pair = Pair(a.models)
    if a.rank:
        if len(a.rank) < 3:
            sys.exit("--rank には 文・元の語・候補(1つ以上)を渡す")
        rows = pair.rank(a.rank[0], a.rank[1], a.rank[2:])
        if a.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
        else:
            print("候補\tA での自然さの変化\tAI 時代らしさの変化")
            for r in rows:
                print(f"{r['cand']}\t{r['dA']:+.2f}\t{r['d']:+.2f}")
        return
    if not a.file:
        ap.error("ファイルを指定する")
    text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(encoding="utf-8")
    hits = []
    for s in sentences(text):
        ws = [w for w in pair.score(s) if ok_word(w) and w["d"] > a.threshold]
        if ws:
            hits.append({"s": s, "words": [{k: w[k] for k in ("w", "lemma", "d", "topA")} for w in ws]})
    if a.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
        return
    for h in hits:
        marks = "、".join(f"{w['w']}({w['d']:.1f}" + (f" → A なら: {'・'.join(t for t in w['topA'][:3] if not t.startswith('##'))}" if w["topA"] else "") + ")"
                         for w in h["words"])
        print(f"{h['s']}\n  {marks}\n")
    print(f"{sum(len(h['words']) for h in hits)} 語(差 > {a.threshold})")


if __name__ == "__main__":
    main()
