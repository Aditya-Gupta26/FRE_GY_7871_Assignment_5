"""Reddit collection via the Arctic Shift archive (free, no auth). Adapted from A4.

- posts: every post in the window (limit=auto pages of up to 1,000)
- comments: a time-stratified sample, the first N comments of every H-hour window
  (ascending; Arctic Shift times out on descending sort for big subs). Each window
  is cached to its own file, so an interrupted run resumes instead of starting over.
Subs run in parallel threads (one per sub) so the whole pull fits the deadline.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import requests

from src.config import (COLLECT_START, COMMENT_WINDOW_HOURS, COMMENTS_PER_WINDOW, RAW,
                        REDDIT_API, REDDIT_SUBS, WINDOW_END)

OUT = RAW / "reddit"
POST_FIELDS = "id,created_utc,subreddit,author,title,selftext,score,num_comments,url,link_flair_text"
COMMENT_FIELDS = "id,created_utc,subreddit,author,body,score,link_id,parent_id"


def _get(path: str, params: dict, retries: int = 8) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(f"{REDDIT_API}/{path}", params=params, timeout=90)
            if r.status_code == 200:
                js = r.json()
                if js.get("error"):
                    raise RuntimeError(js["error"])
                return js.get("data") or []
            if r.status_code == 429:
                reset = float(r.headers.get("X-RateLimit-Reset", 2 ** attempt))
                time.sleep(min(max(reset, 1), 60))
                continue
            if r.status_code >= 500 or (r.status_code == 422 and "Timeout" in r.text):
                time.sleep(3 * 2 ** min(attempt, 4))  # server asks us to slow down
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except (requests.RequestException, RuntimeError) as e:
            if "Timeout" not in str(e) and not isinstance(e, requests.RequestException):
                raise
            time.sleep(3 * 2 ** min(attempt, 4))
    raise RuntimeError(f"gave up on {path} {params}")


def pull_posts(sub: str, start: datetime = COLLECT_START, end: datetime = WINDOW_END) -> list[dict]:
    """Every post, paginated ascending by created_utc, cached per day."""
    cache = OUT / "post_cache" / sub
    cache.mkdir(parents=True, exist_ok=True)
    rows, seen, t = [], set(), start
    while t < end:
        u = min(t + timedelta(days=1), end)
        f = cache / f"{int(t.timestamp())}.json"
        if f.exists():
            batch = json.loads(f.read_text())
        else:
            batch, after = [], int(t.timestamp())
            while True:
                page = _get("posts/search", {"subreddit": sub, "after": after,
                                             "before": int(u.timestamp()), "limit": 100,
                                             "sort": "asc", "fields": POST_FIELDS})
                new = [p for p in page if p["id"] not in {b["id"] for b in batch}]
                batch += new
                if len(page) < 100 or not new:
                    break
                after = max(int(p["created_utc"]) for p in new)
                time.sleep(0.3)
            f.write_text(json.dumps(batch))
            time.sleep(0.3)
        for b in batch:
            if b["id"] not in seen:
                seen.add(b["id"])
                rows.append(b)
        t = u
    return rows


def pull_comments_stratified(sub: str, hours: int = COMMENT_WINDOW_HOURS,
                             per_window: int = COMMENTS_PER_WINDOW,
                             start: datetime = COLLECT_START, end: datetime = WINDOW_END) -> list[dict]:
    cache = OUT / "comment_cache" / sub
    cache.mkdir(parents=True, exist_ok=True)
    rows, seen, t = [], set(), start
    while t < end:
        u = min(t + timedelta(hours=hours), end)
        f = cache / f"{int(t.timestamp())}_{hours}h.json"
        if f.exists():
            batch = json.loads(f.read_text())
        else:
            batch = _get("comments/search", {"subreddit": sub, "after": int(t.timestamp()),
                                             "before": int(u.timestamp()), "limit": per_window,
                                             "sort": "asc", "fields": COMMENT_FIELDS})
            f.write_text(json.dumps(batch))
            time.sleep(0.3)
        for b in batch:
            if b["id"] not in seen:
                seen.add(b["id"])
                rows.append(b)
        t = u
    return rows


def _one_sub(sub: str) -> dict:
    out = {}
    for kind, fn in (("posts", pull_posts), ("comments", pull_comments_stratified)):
        path = OUT / f"{kind}_{sub}.jsonl"
        try:
            rows = fn(sub)
        except Exception as e:  # noqa: BLE001
            print(f"error {kind} {sub}: {e}", flush=True)
            out[f"{kind}_{sub}"] = f"error: {e}"
            continue
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out[f"{kind}_{sub}"] = len(rows)
        print(kind, sub, len(rows), flush=True)
    return out


def collect_all(workers: int = 6) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_one_sub, REDDIT_SUBS):
            summary.update(res)
    return summary


def load_raw() -> list[dict]:
    rows = []
    for p in sorted(OUT.glob("*.jsonl")):
        kind = p.stem.split("_")[0]
        for line in p.open():
            r = json.loads(line)
            r["kind"] = kind
            r["text"] = (r.get("title", "") + "\n" + (r.get("selftext") or "")).strip() \
                if kind == "posts" else r.get("body", "")
            rows.append(r)
    return rows
