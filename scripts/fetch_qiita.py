# /// script
# requires-python = ">=3.11"
# ///
"""fetch_qiita.py — 一覧に載った Qiita 記事の本文を、Qiita API v2 で取得する。

一覧(data/baseline/qiita-*.jsonl)には本文を含めていない(著作権は書いた人にある)。
投稿日ごとに検索して(1回100件)、一覧の ID に当たる記事だけを書き出す。記事ごとに取りに行くより
リクエストがずっと少ない(4年分の8月で数百回)。

- アクセストークンが要る(環境変数 QIITA_TOKEN。https://qiita.com/settings/applications で発行。
  読み取りだけなので read_qiita の権限で足りる)。トークンがあると1時間に1,000回まで
- 一覧の body_sha256 と比べる。作者が後から記事を直していれば一致しない(件数を報告し、本文は書き出す)
- 削除・非公開になった記事は取得できない
- 途中で止まっても、もう一度実行すれば取得済みの日を飛ばす

使い方:
    QIITA_TOKEN=... uv run scripts/fetch_qiita.py --list data/baseline/qiita-pre2022.jsonl --out work/qiita-pre2022.jsonl.gz
    # 生成のプロンプトの元になった記事だけ
    QIITA_TOKEN=... uv run scripts/fetch_qiita.py --list data/baseline/qiita-pre2022.jsonl \
        --only data/generated/sources-v1.jsonl --out work/qiita-sources.jsonl.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://qiita.com/api/v2/items"


def fetch(query: str, page: int, token: str) -> tuple[list[dict], int]:
    url = f"{API}?{urllib.parse.urlencode({'query': query, 'per_page': 100, 'page': page})}"
    req = urllib.request.Request(url, headers={"User-Agent": "ja-ai-style-corpus/1.0", "Authorization": f"Bearer {token}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                remaining = int(res.headers.get("Rate-Remaining", "1000"))
                reset = int(res.headers.get("Rate-Reset", "0"))
                total = int(res.headers.get("Total-Count", "0"))
                items = json.loads(res.read().decode("utf-8"))
            if remaining < 5:
                wait = max(0, reset - int(time.time())) + 5
                print(f"  回数の上限に近いので {wait} 秒待つ", file=sys.stderr)
                time.sleep(wait)
            return items, total
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print("  回数の上限。10分待つ", file=sys.stderr)
                time.sleep(600)
                continue
            if e.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"取得できなかった: {query} page={page}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="+", required=True, help="data/baseline/qiita-*.jsonl")
    ap.add_argument("--only", help="この一覧(sources-v1.jsonl など)に qiita_id がある記事だけ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    token = os.environ.get("QIITA_TOKEN") or sys.exit("環境変数 QIITA_TOKEN を設定する")
    want = {}
    for p in a.list:
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            want[d["id"]] = d
    if a.only:
        only = {json.loads(l).get("qiita_id") for l in open(a.only, encoding="utf-8")}
        want = {k: v for k, v in want.items() if k in only}
    by_day = defaultdict(set)
    for k, v in want.items():
        by_day[v["created_at"][:10]].add(k)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = out.with_suffix(out.suffix + ".days")
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    found = mismatch = 0
    with gzip.open(out, "at", encoding="utf-8") as f:
        for day in sorted(by_day):
            if day in done:
                continue
            start = dt.date.fromisoformat(day)
            query = f"created:>={start} created:<{start + dt.timedelta(days=1)}"
            page = 1
            while True:
                items, total = fetch(query, page, token)
                for it in items:
                    if it["id"] not in by_day[day]:
                        continue
                    w = want[it["id"]]
                    ok = hashlib.sha256(it["body"].encode("utf-8")).hexdigest() == w["body_sha256"]
                    mismatch += not ok
                    found += 1
                    f.write(json.dumps({**w, "body": it["body"], "sha256_match": ok}, ensure_ascii=False) + "\n")
                if len(items) < 100 or page * 100 >= total:
                    break
                page += 1
            f.flush()
            with open(done_path, "a") as df:
                df.write(day + "\n")
            print(f"  {day}: 累計 {found} 件", file=sys.stderr)
    print(f"一覧 {len(want)} 件、取得 {found} 件(うち本文が一覧と違う {mismatch} 件)。今回の実行分のみの数")


if __name__ == "__main__":
    main()
