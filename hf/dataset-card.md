---
license: cc-by-4.0
language:
- ja
tags:
- synthetic
- ai-generated-text
- stylometry
size_categories:
- 1K<n<10K
dataset_info:
- config_name: generated
  features:
    - name: id
      dtype: string
    - name: prompt_id
      dtype: string
    - name: batch
      dtype: string
    - name: src
      dtype: string
    - name: type
      dtype: string
    - name: vendor
      dtype: string
    - name: model
      dtype: string
    - name: model_version
      dtype: string
    - name: effort
      dtype: string
    - name: harness
      dtype: string
    - name: harness_version
      dtype: string
    - name: settings
      dtype: string
    - name: prompt
      dtype: string
    - name: generated_at
      dtype: string
    - name: backfilled
      dtype: bool
    - name: text
      dtype: string
    - name: text_clean
      dtype: string
    - name: synthetic
      dtype: bool
    - name: notice
      dtype: string
- config_name: generated_holdout
  features:
    - name: id
      dtype: string
    - name: model
      dtype: string
    - name: effort
      dtype: string
    - name: harness
      dtype: string
    - name: harness_version
      dtype: string
    - name: settings
      dtype: string
    - name: prompt
      dtype: string
    - name: generated_at
      dtype: string
    - name: text
      dtype: string
    - name: text_clean
      dtype: string
    - name: synthetic
      dtype: bool
    - name: notice
      dtype: string
configs:
- config_name: generated
  data_files: data/generated/generated-v1.jsonl.gz
- config_name: generated_holdout
  data_files: data/eval/generated-holdout-v0.jsonl.gz
---

# ja-ai-style-corpus

生成AI以前の人間の文書と同じ話題で、3社の生成AIに書かせた日本語の文章 7,429本(2,444万字)と、人間の文書の一覧です。生成AIが書く日本語に増えた言い回しを調べるために作りました。

**生成文の内容は架空です。** 実在の組織名・人名・製品名が出てきても、日付・数値・発表・発言などは事実ではありません。全件に `synthetic: true` と注記を付けています。

## 中身

| ファイル | 中身 |
|---|---|
| `data/generated/generated-v1.jsonl.gz` | 生成文。`text`(手を加えていない出力)、`text_clean`(チャットの前置きと、伏せ字だけの署名行などを除いたもの)と出どころ |
| `data/generated/sources-v1.jsonl` | 各プロンプトの元になった人間の文書の所在(本文なし) |
| `data/baseline/*.jsonl` | AI 以前の Qiita 記事・2026年の Qiita 記事・LLM-jp コーパス v4 の説明的な文章の一覧(本文なし) |
| `data/eval/` | 検査に使った人間の文書の ID と、検査用の生成記事100本 |

## 出どころの項目

生成文の1行ごとに、ベンダーだけでなく次を記録しています。

| 項目 | 例 |
|---|---|
| `model` | claude-opus-5-5 / gpt-6.1-sol / gemini-3.8-flash |
| `effort` | low |
| `harness` | Claude Code (claude -p) / Codex CLI (codex exec) / Gemini API v1beta Batch |
| `harness_version` | 2.1.296 (Claude Code) / codex-cli 0.162.1 / v1beta |
| `settings` | 個人設定を読まない方法、ツールや Web 検索の扱い、追加した指示 |
| `prompt` | 渡したプロンプトそのもの |
| `model_version`・`usage` | Gemini の応答に含まれる値 |

プロンプトは、AI 以前の Qiita 記事のタイトル(「「{タイトル}」というタイトルで技術ブログの記事を書いて。本文は{字数}字程度で。」)か、LLM-jp コーパスの AI 以前の文章の冒頭3文(「次の内容を伝える{種類}を書いて。本文は{字数}字程度で。\n\n内容: {冒頭}」)です。

## ライセンス

- 生成文と一覧: CC BY 4.0
- プロンプトに含まれる人間の文書の一部(Qiita のタイトル、LLM-jp コーパスの文章の冒頭)の権利は、元の書き手にあります。LLM-jp コーパス由来の項目は、元のコーパスのライセンス(ODC-BY・CC BY 4.0)にも従ってください
- 生成文の利用は、各モデルの利用規約の範囲で行ってください

作り方・取得のスクリプト・わかったことは [GitHub](https://github.com/coji/ja-ai-style-corpus) にあります。
