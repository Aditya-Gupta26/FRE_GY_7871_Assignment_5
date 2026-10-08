"""Twitter/X collection through twitterapi.io advanced search. Adapted from A4.

- X's own API only searches the last 7 days on pay-per-use, so the historical window
  is bought from a data provider (twitterapi.io).
- Lean, time-stratified sampling: contiguous 90-minute windows from the Jun 30 close to the
  Oct 7 close, one "Latest" page (at most 20 tweets) per window, so the sample is spread
  evenly over time. 1,584 windows, at most 31,680 tweets.
- Billing (twitterapi.io/pricing, checked Oct 7, 2026): 15 credits per returned tweet, at
  least 15 credits per call, 1 USD = 100,000 credits. A hard budget guard counts the worst
  case of every call before it is made and stops before TWITTER_BUDGET_USD would be crossed.
- Every page is cached to disk, so re-runs are free and the pull is resumable.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from src.config import (COLLECT_START, CREDITS_PER_TWEET, CREDITS_PER_USD, MIN_CREDITS_PER_CALL, TWITTER_SUNK_USD,
                        RAW, ROOT, TWITTER_BUDGET_USD, TWITTER_ENDPOINT, TWITTER_QUERY,
                        TWITTER_SUBWINDOW_MIN, UTC, WINDOW_END)

load_dotenv(ROOT / ".env")
OUT = RAW / "twitter"
KEEP_FIELDS = ["id", "text", "createdAt", "lang", "likeCount", "retweetCount", "replyCount",
               "quoteCount", "viewCount", "bookmarkCount", "isReply", "inReplyToId",
               "conversationId", "url"]
MAX_PAGE = 20
_lock = threading.Lock()
_spent_credits = [0]          # worst-case credits committed in this run


def _api_key() -> str:
    key = os.getenv("TWITTERAPI_IO_KEY")
    if not key:
        raise SystemExit("TWITTERAPI_IO_KEY missing: put it in Assignment_5/.env")
    return key


def windows(start: datetime = COLLECT_START, end: datetime = WINDOW_END):
    t = start
    while t < end:
        u = min(t + timedelta(minutes=TWITTER_SUBWINDOW_MIN), end)
        yield t, u
        t = u


def _slim(tw: dict) -> dict:
    row = {k: tw.get(k) for k in KEEP_FIELDS}
    a = tw.get("author") or {}
    row.update({"author_user": a.get("userName"), "author_name": a.get("name"),
                "author_bio": a.get("description"), "author_followers": a.get("followers"),
                "author_blue": a.get("isBlueVerified"), "author_verified_type": a.get("verifiedType"),
                "author_created": a.get("createdAt")})
    return row


def account_credits(key: str) -> dict:
    r = requests.get("https://api.twitterapi.io/oapi/my/info", headers={"X-API-Key": key}, timeout=20)
    return r.json() if r.status_code == 200 else {}


def _reserve(budget_credits: int, already_credits: int) -> bool:
    """Reserve the worst case of one call (20 tweets). False = budget would be crossed."""
    worst = max(MIN_CREDITS_PER_CALL, MAX_PAGE * CREDITS_PER_TWEET)
    with _lock:
        if already_credits + _spent_credits[0] + worst > budget_credits:
            return False
        _spent_credits[0] += worst
        return True


def _settle(n_tweets: int):
    """Replace the reserved worst case by the actual charge of the call."""
    worst = max(MIN_CREDITS_PER_CALL, MAX_PAGE * CREDITS_PER_TWEET)
    actual = max(MIN_CREDITS_PER_CALL, n_tweets * CREDITS_PER_TWEET)
    with _lock:
        _spent_credits[0] += actual - worst


def fetch_page(since: datetime, until: datetime, key: str, retries: int = 8) -> dict:
    q = f"{TWITTER_QUERY} since_time:{int(since.timestamp())} until_time:{int(until.timestamp())}"
    params = {"query": q, "queryType": "Latest", "cursor": ""}
    for attempt in range(retries):
        try:
            r = requests.get(TWITTER_ENDPOINT, params=params, headers={"X-API-Key": key}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(max(5.2, 2 ** attempt))
                continue
            if r.status_code in (402, 403) or "credit" in r.text.lower():
                raise SystemExit(f"stopping: out of credits or not allowed ({r.status_code})")
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on window {since}")


def collect_window(since: datetime, until: datetime, key: str, budget_credits: int,
                   already_credits: int) -> dict | None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{since.astimezone(UTC):%Y%m%dT%H%M}.json"
    if path.exists():
        return json.loads(path.read_text())["meta"]
    if not _reserve(budget_credits, already_credits):
        return None
    data = fetch_page(since, until, key)
    tweets = [_slim(t) for t in data.get("tweets", [])]
    _settle(len(tweets))
    meta = {"since_utc": since.astimezone(UTC).isoformat(), "until_utc": until.astimezone(UTC).isoformat(),
            "n": len(tweets), "has_next_page": bool(data.get("has_next_page"))}
    path.write_text(json.dumps({"meta": meta, "tweets": tweets}))
    return meta


def spent_so_far_credits() -> int:
    """Credits already paid for pages cached on disk (actual, from the saved counts)."""
    tot = 0
    for p in OUT.glob("*.json"):
        n = json.loads(p.read_text())["meta"]["n"]
        tot += max(MIN_CREDITS_PER_CALL, n * CREDITS_PER_TWEET)
    return tot


def collect_all(limit: int | None = None, workers: int = 8) -> dict:
    key = _api_key()
    budget = int(TWITTER_BUDGET_USD * CREDITS_PER_USD)
    already = (spent_so_far_credits() if OUT.exists() else 0) + int(TWITTER_SUNK_USD * CREDITS_PER_USD)
    todo = list(windows())[: limit or None]
    metas, skipped, done = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(collect_window, s, u, key, budget, already) for s, u in todo]
        for fut in as_completed(futs):
            m = fut.result()
            if m is None:
                skipped += 1
            else:
                metas.append(m)
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(todo)} windows, {sum(x['n'] for x in metas)} tweets, "
                      f"run spend ${_spent_credits[0] / CREDITS_PER_USD:.2f}", flush=True)
    return {"windows": len(metas), "tweets": sum(m["n"] for m in metas), "skipped_budget": skipped,
            "cost_usd_this_run": _spent_credits[0] / CREDITS_PER_USD,
            "cost_usd_total_cached": (already + _spent_credits[0]) / CREDITS_PER_USD}


def load_raw() -> list[dict]:
    rows = []
    for p in sorted(OUT.glob("*.json")):
        blob = json.loads(p.read_text())
        m = blob["meta"]
        for t in blob["tweets"]:
            t = dict(t)
            t["win_since_utc"], t["win_until_utc"] = m["since_utc"], m["until_utc"]
            t["win_capped"] = m["has_next_page"] or m["n"] >= MAX_PAGE
            rows.append(t)
    return rows
