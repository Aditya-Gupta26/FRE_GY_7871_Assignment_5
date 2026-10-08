"""News collection through Google News RSS (free). Adapted from A4.

Two designs:
- named outlets: `site:` queries per outlet, 2-day windows, for the outlet-group
  comparison (US left / right / centre, business, international)
- US edition: one query per day with no site filter, for overall news salience
Google News RSS returns at most 100 items per query, so windows are kept short.
GDELT was dropped: it answered HTTP 429 from this IP on Oct 7, 2026 (as on A4).
"""
from __future__ import annotations

import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import feedparser
import requests

from src.config import (COLLECT_START, GOOGLE_NEWS_RSS, NEWS_QUERIES, OUTLETS, RAW, RSS_CAP,
                        RSS_DAYS_PER_WINDOW, WINDOW_END)

OUT = RAW / "news"
UA = {"User-Agent": "Mozilla/5.0 (academic research; NYU FRE-GY 7871 course project)"}


def _rss_get(q: str) -> list[dict] | None:
    url = f"{GOOGLE_NEWS_RSS}?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                feed = feedparser.parse(r.content)
                return [{"title": e.get("title"), "link": e.get("link"),
                         "published": e.get("published"),
                         "source": (e.get("source") or {}).get("title"),
                         "source_href": (e.get("source") or {}).get("href")} for e in feed.entries]
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return None  # None = failed (not cached, retried next run); [] = genuinely empty


def _site_job(site: str, topic: str) -> int:
    d = OUT / "rss_sites"
    d.mkdir(parents=True, exist_ok=True)
    total, t = 0, COLLECT_START.replace(hour=0)
    while t < WINDOW_END:
        u = t + timedelta(days=RSS_DAYS_PER_WINDOW)
        path = d / f"{topic}_{site.replace('/', '_')}_{t:%Y%m%d}.json"
        if not path.exists():
            q = f"{NEWS_QUERIES[topic]} site:{site} after:{t:%Y-%m-%d} before:{u:%Y-%m-%d}"
            items = _rss_get(q)
            if items is not None:
                path.write_text(json.dumps({"site": site, "topic": topic, "since": t.isoformat(),
                                            "until": u.isoformat(), "items": items}))
                total += len(items)
            time.sleep(1.2)
        t = u
    print("rss site", topic, site, total, flush=True)
    return total


def _edition_job(topic: str) -> int:
    d = OUT / "rss_us"
    d.mkdir(parents=True, exist_ok=True)
    total, t = 0, COLLECT_START.replace(hour=0)
    while t < WINDOW_END:
        u = t + timedelta(days=1)
        path = d / f"{topic}_US_{t:%Y%m%d}.json"
        if not path.exists():
            q = f"{NEWS_QUERIES[topic]} after:{t:%Y-%m-%d} before:{u:%Y-%m-%d}"
            items = _rss_get(q)
            if items is not None:
                path.write_text(json.dumps({"topic": topic, "since": t.isoformat(),
                                            "until": u.isoformat(), "items": items}))
                total += len(items)
            time.sleep(1.2)
        t = u
    print("rss US edition", topic, total, flush=True)
    return total


def collect_all(workers: int = 3) -> int:
    jobs = [(s, "ELECTION") for grp in OUTLETS.values() for s in grp]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_edition_job, t) for t in NEWS_QUERIES]
        futs += [ex.submit(_site_job, s, t) for s, t in jobs]
        return sum(f.result() for f in futs)


def _topup_one(path) -> int:
    """Re-pull one capped 2-day site window as two 1-day windows (written next to it)."""
    j = json.loads(path.read_text())
    t0 = datetime.fromisoformat(j["since"])
    got = 0
    for k in range(RSS_DAYS_PER_WINDOW):
        t, u = t0 + timedelta(days=k), t0 + timedelta(days=k + 1)
        out = path.parent.parent / "rss_topup" / f"{j['topic']}_{j['site']}_{t:%Y%m%d}_1d.json"
        if out.exists():
            continue
        q = f"{NEWS_QUERIES[j['topic']]} site:{j['site']} after:{t:%Y-%m-%d} before:{u:%Y-%m-%d}"
        items = _rss_get(q)
        if items is not None:
            out.write_text(json.dumps({"site": j["site"], "topic": j["topic"], "since": t.isoformat(),
                                       "until": u.isoformat(), "items": items}))
            got += len(items)
        time.sleep(1.2)
    return got


def topup_capped(workers: int = 3) -> int:
    """Windows that hit the 100-item cap are re-pulled by day, so fewer items are lost."""
    (OUT / "rss_topup").mkdir(parents=True, exist_ok=True)
    capped = [p for p in sorted((OUT / "rss_sites").glob("*.json"))
              if len(json.loads(p.read_text())["items"]) >= RSS_CAP]
    print("capped 2-day windows:", len(capped), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(_topup_one, capped))
