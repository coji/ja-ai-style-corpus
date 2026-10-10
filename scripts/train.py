# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.44", "accelerate", "peft"]
# ///
"""train.py — AI 以前のモデル(pre-ai)と AI 時代のモデル(ai-era)を、同じ事前学習済みモデルから追加学習する。

    元       tohoku-nlp/bert-base-japanese-v3(CC-100 と 2023年1月の Wikipedia。生成AI以前の日本語)
    pre-ai   生成のプロンプトの元になった人間の文書(同じタイトルの Qiita 記事・同じ冒頭の LLM-jp の文書)
    ai-era   そのプロンプトで生成AIに書かせた文章(text_clean)。文を無作為に減らして pre-ai と同じ字数にそろえる
    検査用の文書(data/eval/test-*.txt)は学習から除く

話題をそろえるのは、そろえないと差に「最近の業務・話題の語」が混ざるため(docs/findings.md)。

段階:
    prepare  学習用の文を作る(<work>/train-pre-ai.txt、train-ai-era.txt)
    train    1つを追加学習する(<out>/pre-ai、<out>/ai-era)

使い方:
    uv run scripts/train.py prepare --qiita work/qiita-sources.jsonl.gz --llmjp work/llmjp-sources.jsonl --work work
    uv run scripts/train.py train pre-ai --work work --out models/local --lora 16
    uv run scripts/train.py train ai-era --work work --out models/local --lora 16
公開したモデルの設定: 2,500ステップ(約1.9周)、バッチ32、系列長256、語単位の伏せ字15%、
    全体を学習するとき学習率 5e-5、LoRA(r=16、alpha 32、dropout 0.05、query・key・value・dense)は 2e-4
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlm import BASE, SEQ  # noqa: E402
from textutil import sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def prepare(a) -> None:
    tests = {l.strip() for f in (ROOT / "data/eval").glob("test-*.txt") for l in open(f) if l.strip()}
    src = list(read_jsonl(ROOT / "data/generated/sources-v1.jsonl"))
    qid = {r["qiita_id"] for r in src if r["kind"] == "qiita"} - tests
    bid = {r["docId"] for r in src if r["kind"] == "llmjp"} - tests
    human = [d["body"] for d in read_jsonl(a.qiita) if d["id"] in qid] + \
            [d["text"] for d in read_jsonl(a.llmjp) if d["docId"] in bid]
    ai = [r["text_clean"] for r in read_jsonl(ROOT / "data/generated/generated-v1.jsonl.gz")]
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    a_lines = [s for d in human for s in sentences(d)]
    b_lines = [s for d in ai for s in sentences(d)]
    a_chars = sum(map(len, a_lines))
    # AI の文章は長いので、文を無作為に減らして同じ字数にそろえる(学習の周回数をそろえる)
    random.Random(0).shuffle(b_lines)
    kept, n = [], 0
    for l in b_lines:
        if n >= a_chars:
            break
        kept.append(l)
        n += len(l)
    (work / "train-pre-ai.txt").write_text("\n".join(a_lines) + "\n", encoding="utf-8")
    (work / "train-ai-era.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"pre-ai: {len(human)} 文書、{a_chars:,} 字 / ai-era: {len(ai)} 文書から {n:,} 字")


def train(a) -> None:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMaskedLM.from_pretrained(BASE)
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=a.lora, lora_alpha=a.lora * 2, lora_dropout=0.05,
                                                 target_modules=["query", "key", "value", "dense"]))
        model.print_trainable_parameters()
    lines = (Path(a.work) / f"train-{a.which}.txt").read_text(encoding="utf-8").split("\n")
    # 文をつなげて SEQ トークン前後の塊にする
    chunks, cur = [], ""
    for s in lines:
        if len(cur) + len(s) > SEQ * 1.3:
            chunks.append(cur)
            cur = ""
        cur += s
    if cur:
        chunks.append(cur)
    random.Random(0).shuffle(chunks)
    chunks = chunks[: a.steps * a.batch]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(chunks)

        def __getitem__(self, i):
            return tok(chunks[i], truncation=True, max_length=SEQ)

    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    cont = torch.tensor([t.startswith("##") for t in vocab])
    special = set(tok.all_special_ids)

    def collator(features):
        """語単位の伏せ字(whole word masking)。「##」で始まるトークンは前の語の続きとみなす。
        transformers の語単位の伏せ字は fast tokenizer が要り、このモデルの tokenizer では使えないので自前で書く。"""
        batch = tok.pad(features, return_tensors="pt")
        ids = batch["input_ids"]
        labels = ids.clone()
        mask = torch.zeros_like(ids, dtype=torch.bool)
        g = torch.Generator().manual_seed(int(time.time() * 1000) % (2**31))
        for r in range(ids.size(0)):
            words, cur = [], []
            for c, t in enumerate(ids[r].tolist()):
                if t in special:
                    if cur:
                        words.append(cur)
                    cur = []
                    continue
                if cont[t] and cur:
                    cur.append(c)
                else:
                    if cur:
                        words.append(cur)
                    cur = [c]
            if cur:
                words.append(cur)
            for wi in torch.randperm(len(words), generator=g)[: max(1, round(len(words) * 0.15))].tolist():
                mask[r, words[wi]] = True
        labels[~mask] = -100
        rnd = torch.rand(ids.shape, generator=g)
        ids[mask & (rnd < 0.8)] = tok.mask_token_id
        swap = mask & (rnd >= 0.8) & (rnd < 0.9)
        ids[swap] = torch.randint(len(tok), ids.shape, generator=g)[swap]
        batch["input_ids"], batch["labels"] = ids, labels
        return batch

    out = Path(a.out) / a.which
    args = TrainingArguments(output_dir=str(out) + "-ckpt", max_steps=a.steps, per_device_train_batch_size=a.batch,
                             learning_rate=a.lr or (2e-4 if a.lora else 5e-5), warmup_steps=int(a.steps * 0.06),
                             weight_decay=0.01, logging_steps=50, save_strategy="no", report_to=[],
                             dataloader_num_workers=0, seed=0)
    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=DS(), data_collator=collator).train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"{a.which}: {len(chunks)} 塊、{time.time() - t0:.0f} 秒 → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--qiita", required=True, help="fetch_qiita.py --only data/generated/sources-v1.jsonl の出力")
    p.add_argument("--llmjp", required=True, help="fetch_llmjp.py --list data/generated/sources-v1.jsonl の出力")
    p.add_argument("--work", default="work")
    t = sub.add_parser("train")
    t.add_argument("which", choices=["pre-ai", "ai-era"])
    t.add_argument("--work", default="work")
    t.add_argument("--out", default="models/local")
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, help="指定しなければ全体 5e-5、LoRA 2e-4")
    t.add_argument("--lora", type=int, default=0, help="LoRA のランク(0 なら全体を学習)")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else train(a)


if __name__ == "__main__":
    main()
