---
license: apache-2.0
language:
- ja
base_model: tohoku-nlp/bert-base-japanese-v3
library_name: transformers
pipeline_tag: fill-mask
tags:
- ai-generated-text
- stylometry
datasets:
- coji/ja-ai-style-corpus
---

# ja-ai-style-bert

生成AIが書く日本語に増えた言い回し(「この設定が効く」「静かに壊れる」「土台」「境界」「落とし穴」など)を、語の表なしに見つけるための2つのモデルです。

- `pre-ai/`: 生成AI以前の人間の文書で追加学習
- `ai-era/`: 同じ話題で生成AIに書かせた文章で追加学習

文の各語を伏せて両方に当てさせ、差 d = log P_ai-era(語) − log P_pre-ai(語) を見ます。大きいほど「AI 時代のモデルだけがそこにその語を置く」語です。どちらも単独では普通の穴埋めモデルで、2つを組にして使います。

## 使い方

```bash
git clone https://github.com/coji/ja-ai-style-corpus
cd ja-ai-style-corpus
uv run scripts/check.py draft.md        # 初回にこのモデルを自動でダウンロードする
```

自分で読み込む場合:

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
root = snapshot_download("coji/ja-ai-style-bert")
tok = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
A = AutoModelForMaskedLM.from_pretrained(f"{root}/pre-ai", dtype=torch.float32)
B = AutoModelForMaskedLM.from_pretrained(f"{root}/ai-era", dtype=torch.float32)
```

tokenizer には `fugashi` と `unidic-lite` が要ります。語は UniDic で切り、語の全トークンを同時に伏せます(`scripts/mlm.py`)。

## 学習

| | pre-ai | ai-era |
|---|---|---|
| 事前学習済みモデル | [tohoku-nlp/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)(CC-100 と 2023年1月の Wikipedia) | 同じ |
| データ | 生成のプロンプトの元になった人間の文書(Qiita 2019〜2021年8月の記事、LLM-jp コーパス v4 の AI 以前の説明的な文章)。1,331万字 | 同じプロンプトで3社の生成AI(claude-opus-5-5・gpt-6.1-sol・gemini-3.8-flash、effort low)に書かせた文章から、文を無作為に選んで同じ字数に合わせたもの |
| 設定 | 2,500ステップ(約1.8周)、バッチ32、系列長256、語単位の伏せ字15%(80% [MASK]・10% 置換・10% そのまま)、AdamW 学習率 5e-5、warmup 6%、weight decay 0.01、fp32、seed 0 | 同じ |
| 最終の学習損失 | 1.755 | 1.879 |

- 環境: Apple M5 Pro(64GB、MPS)、1つ約50分。Python 3.13、torch 2.14.1、transformers 5.19.0
- 保存は fp16(safetensors)。fp32 との差は、語ごとの差で最大0.04
- 検査用の文書(データセットの `data/eval/`)は学習から除いた
- 学習のコードは `scripts/train.py`、データは [coji/ja-ai-style-corpus](https://huggingface.co/datasets/coji/ja-ai-style-corpus)

## 評価

差 > 3 の語の量(カタカナ語・英字の語を除いた内容語、地の文1,000字あたり。各セット100文書から最大40文):

| 検査セット | 差 > 3 の語 |
|---|---|
| Qiita 2022年8月(AI 以前) | 1.17 |
| LLM-jp の AI 以前の文章 | 1.11 |
| Qiita 2026年8月 | 4.18 |
| 学習外の生成記事(7モデル) | 5.99 |

AI 時代の文章で差が大きく出る語の例: 構成・実務・書く・整理・確認・確かめる・迷う・躓く・揃える・手軽・手元・利く(効く)・土台・入り口・落とし穴・地味。

## 限界

- 差は「AI 以前の人がそこにその語を置くか」で、文章の良し悪しではありません。新しい技術用語や話題の語も大きく出ます
- 学習データは技術記事と Web の説明的な文章です。議事録やメールは入っていません
- 生成文の書き手は3モデルです。ほかのモデルの癖は学習していない可能性があります
- LoRA(差分だけの学習)では検出の力が大きく落ちたため、全体を学習したモデルを公開しています

詳しくは [docs/findings.md](https://github.com/coji/ja-ai-style-corpus/blob/main/docs/findings.md) を見てください。
