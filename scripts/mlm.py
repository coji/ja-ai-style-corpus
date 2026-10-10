"""mlm.py — 2つの BERT(AI 以前・AI 時代)を読み込み、語ごとの差と、言い換え候補の並べ替えを計算する。

    A(pre-ai)   AI 以前の人間の文書で追加学習
    B(ai-era)   同じ話題で生成AIに書かせた文章で追加学習
    差 d        log P_B(語) − log P_A(語)。語を伏せて当てさせたとき、B だけがそこにその語を置くほど大きい

モデルは Hugging Face Hub(MODEL_REPO)から初回に自動でダウンロードする(2つで約420MB。キャッシュは
~/.cache/huggingface)。手元で学習したモデルを使うときは、ディレクトリを --models に渡す。
ディレクトリには pre-ai/ と ai-era/ を置き、それぞれ全体の重みか LoRA の差分(adapter_config.json)を入れる。
"""
from __future__ import annotations

import re
from pathlib import Path

BASE = "tohoku-nlp/bert-base-japanese-v3"
MODEL_REPO = "coji/ja-ai-style-bert"
SEQ = 256
CONTENT = {"動詞", "形容詞", "副詞", "名詞", "形状詞"}


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path: Path):
    """全体の重みか、事前学習済みモデルに足す LoRA の差分か(adapter_config.json の有無)で読み分ける。"""
    import torch
    from transformers import AutoModelForMaskedLM
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        return PeftModel.from_pretrained(AutoModelForMaskedLM.from_pretrained(BASE), path).merge_and_unload()
    # 公開したモデルは fp16 で保存してある。CPU でも動くよう fp32 で読む
    return AutoModelForMaskedLM.from_pretrained(path, dtype=torch.float32)


def resolve(models: str | None) -> Path:
    if models and Path(models).exists():
        return Path(models)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(models or MODEL_REPO))


class Pair:
    def __init__(self, models: str | None = None):
        import fugashi
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.dev = device()
        root = resolve(models)
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.A = load_model(root / "pre-ai").to(self.dev).eval()
        self.B = load_model(root / "ai-era").to(self.dev).eval()
        self.tagger = fugashi.Tagger()

    def words(self, s: str):
        return [(w.surface, w.feature.pos1, w.feature.pos2, w.feature.lemma or w.surface) for w in self.tagger(s)]

    def ids(self, ws):
        ids, spans = [self.tok.cls_token_id], []
        for surf, *_ in ws:
            sub = self.tok.convert_tokens_to_ids(self.tok.subword_tokenizer.tokenize(surf)) if surf.strip() else []
            spans.append((len(ids), len(ids) + len(sub)))
            ids += sub
        ids.append(self.tok.sep_token_id)
        return ids, spans

    def score(self, s: str, batch: int = 48) -> list[dict]:
        """文の内容語(固有名詞・数詞を除く)を1語ずつ伏せ(語の全トークンを同時に)、A と B での log 確率と差を返す。
        topA は A がその位置に予想する語(1トークンの語のときだけ)。top1p・entA は A の予想の自信。"""
        torch = self.torch
        ws = self.words(s)
        ids, spans = self.ids(ws)
        if len(ids) > SEQ:
            return []
        targets = [i for i, (surf, p1, p2, _) in enumerate(ws)
                   if p1 in CONTENT and p2 not in {"固有名詞", "数詞", "非自立可能"} and spans[i][1] > spans[i][0]
                   and re.search(r"[ぁ-んァ-ヶ一-龠]", surf)]
        out = []
        for k in range(0, len(targets), batch):
            tb = targets[k:k + batch]
            x = torch.tensor([ids] * len(tb))
            for r, i in enumerate(tb):
                a, b = spans[i]
                x[r, a:b] = self.tok.mask_token_id
            x = x.to(self.dev)
            with torch.no_grad():
                la = torch.log_softmax(self.A(x).logits, -1)
                lb = torch.log_softmax(self.B(x).logits, -1)
            for r, i in enumerate(tb):
                a, b = spans[i]
                pa = sum(la[r, a + j, t].item() for j, t in enumerate(ids[a:b]))
                pb = sum(lb[r, a + j, t].item() for j, t in enumerate(ids[a:b]))
                top = self.tok.convert_ids_to_tokens(torch.topk(la[r, a], 6).indices.tolist()) if b - a == 1 else []
                pr = la[r, a].exp()
                surf, p1, _, lem = ws[i]
                out.append({"w": surf, "lemma": lem, "pos": p1, "lpA": round(pa, 3), "lpB": round(pb, 3), "d": round(pb - pa, 3),
                            "topA": top, "top1p": round(pr.max().item(), 4), "entA": round(-(pr * la[r, a]).sum().item(), 3)})
        return out

    def pll(self, s: str) -> tuple[float, float]:
        """文全体の擬似対数尤度(各トークンを1つずつ伏せて当てた log 確率の合計)を A と B で返す。"""
        torch = self.torch
        ids, _ = self.ids(self.words(s))
        n = len(ids) - 2
        x = torch.tensor([ids] * n).to(self.dev)
        for r in range(n):
            x[r, r + 1] = self.tok.mask_token_id
        res = []
        with torch.no_grad():
            for m in (self.A, self.B):
                lp = torch.log_softmax(m(x).logits, -1)
                res.append(sum(lp[r, r + 1, ids[r + 1]].item() for r in range(n)))
        return res[0], res[1]

    def rank(self, sentence: str, old: str, candidates: list[str]) -> list[dict]:
        """old を各候補に置き換えた文を並べる。dA は A での自然さの変化(元の文との差、大きいほど自然)、
        d は B − A の差の変化(正なら候補のほうが AI 時代らしい)。dA − max(d, 0) の大きい順。
        候補は意味を保つ言い換え(人か LLM が出す)。モデルが見るのは「AI 以前の人がそう書くか」だけ。"""
        a0, b0 = self.pll(sentence)
        out = []
        for c in candidates:
            a1, b1 = self.pll(sentence.replace(old, c, 1))
            out.append({"cand": c, "dA": round(a1 - a0, 2), "d": round((b1 - a1) - (b0 - a0), 2)})
        out.sort(key=lambda r: -(r["dA"] - max(r["d"], 0)))
        return out


def ok_word(w: dict) -> bool:
    """カタカナ語・英字の語・複合の見出し語を除く(新しい技術用語は AI 以前のモデルが知らないだけなので)。"""
    return not re.fullmatch(r"[ァ-ヶー・]+", w["w"]) and "-" not in w["lemma"] and not re.search(r"[A-Za-z]", w["w"])
