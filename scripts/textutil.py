"""textutil.py — 文の切り出しと、生成文の掃除(各スクリプトから使う)。"""
from __future__ import annotations

import re

SENT_SPLIT = re.compile(r"(?<=[。！？!?])|\n")


def clean(text: str) -> str:
    """Markdown のコード・URL・画像・リンク・強調を除く。インラインコードは X に置き換える。"""
    text = re.sub(r"```.*?```|~~~.*?~~~", "\n", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "X", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*|__", "", text)
    return text


def sentences(text: str) -> list[str]:
    """10〜200字で、ひらがなを含み、表の行でない文だけを返す。"""
    out = []
    for s in SENT_SPLIT.split(clean(text)):
        s = s.strip().lstrip("#>-*|0123456789. ").strip()
        if 10 <= len(s) <= 200 and re.search(r"[ぁ-ん]", s) and s.count("|") < 2:
            out.append(s)
    return out


PREAMBLE_RE = re.compile(r"承知|かしこまり|執筆しました|作成しました|作成いたしました|ご提示いただいた|骨子をもとに|^以下は")
PLACEHOLDER_RE = re.compile(r"X{2,}|〇〇|○○|202X|20XX")
POSTSCRIPT_RE = re.compile(r"字数")
CONTACT_RE = re.compile(r"TEL|Tel|電話|FAX|ファックス|E-?mail|メール|〒|住所|所在地|URL")
SEPS = {"---", "***", "___"}


def clean_generated(text: str) -> str:
    """生成文から、本文でないものだけを除く(全件を見て決めた規則。除いた量は Gemini 2.4%、Claude 0.1%、GPT ほぼ0)。
    - 冒頭のチャットの返事(「承知いたしました。〜執筆しました。」「以下は〜の記事です。」)。最初の「---」か見出しまでを捨てる
    - 末尾の最後の「---」以降が、見出しのない4行以下の注記(字数など)なら捨てる。
      Gemini は「---」を節の区切りに使うので、本文の「まとめ」などは残す
    - 伏せ字(XX・○○・202X など)を含む行のうち、ひらがながほとんどない行(日付・署名)と、60字未満の連絡先・住所の行。
      「もし〇〇ならAをする」のように〇〇を変数として使う地の文は残す"""
    lines = text.strip().split("\n")
    if PREAMBLE_RE.search("\n".join(lines[:4])):
        cut = next((i for i, l in enumerate(lines[:12]) if l.strip() in SEPS or l.lstrip().startswith("#")), None)
        if cut is not None:
            lines = lines[cut + (1 if lines[cut].strip() in SEPS else 0):]
    seps = [i for i, l in enumerate(lines) if l.strip() in SEPS]
    if seps:
        tail = [l for l in lines[seps[-1] + 1:] if l.strip()]
        if len(tail) <= 4 and not any(l.lstrip().startswith("#") for l in tail) and \
                (any(POSTSCRIPT_RE.search(l) for l in tail) or all(PLACEHOLDER_RE.search(l) for l in tail)):
            lines = lines[:seps[-1]]
    lines = [l for l in lines if not (PLACEHOLDER_RE.search(l) and
                                      (len(re.findall(r"[ぁ-ん]", l)) < 5 or (CONTACT_RE.search(l) and len(l.strip()) < 60)))]
    return "\n".join(lines).strip()
