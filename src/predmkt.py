"""Prediction-market price history (free public APIs, no auth).

Polymarket: the Gamma API lists the markets of the "Balance of Power: 2026 Midterms"
event and their CLOB token ids; the CLOB `prices-history` endpoint returns the price
series of one token. Kalshi: trade-api v2 market candlesticks for the
KXBALANCEPOWERCOMBO-27FEB markets. Hourly data is pulled for the whole span (pre-window
included) so the daily series can be snapshotted at 16:00 ET, the stock close.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import requests

from src.config import (ET, KALSHI_API, KALSHI_EVENT, KALSHI_SERIES, POLY_CLOB,
                        POLY_EVENT_SLUG, POLY_GAMMA, POLY_HISTORY_START, PRICE_START, RAW, UTC)

OUT = RAW / "predmkt"
CHUNK_DAYS = 14   # Polymarket rejects startTs/endTs spans over 15 days (checked Oct 7)


def _get(url: str, params: dict | None = None) -> dict:
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {r.status_code} {url}: {r.text[:200]}")
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def _span(start_date: str = PRICE_START) -> tuple[int, int]:
    start = int(datetime.fromisoformat(start_date).replace(tzinfo=ET).timestamp())
    return start, int(time.time())


def polymarket() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = _get(f"{POLY_GAMMA}/events", {"slug": POLY_EVENT_SLUG})[0]
    (OUT / "polymarket_event.json").write_text(json.dumps(ev))
    start, end = _span(POLY_HISTORY_START)
    summary = {}
    for m in ev["markets"]:
        name = m["groupItemTitle"]
        token_yes = json.loads(m["clobTokenIds"])[0]
        pts = []
        t = max(start, int(datetime.fromisoformat(m["startDate"].replace("Z", "+00:00")).timestamp()))
        while t < end:
            u = min(t + CHUNK_DAYS * 86400, end)
            js = _get(f"{POLY_CLOB}/prices-history", {"market": token_yes, "startTs": t,
                                                       "endTs": u, "fidelity": 60})
            # the API appends one "current" point after endTs, so keep only [t, u]
            pts += [p for p in js.get("history", []) if t <= p["t"] <= u]
            t = u
            time.sleep(0.3)
        pts = sorted({p["t"]: p for p in pts}.values(), key=lambda p: p["t"])
        (OUT / f"poly_{name.replace(' ', '_').replace(',', '')}_hourly.json").write_text(json.dumps(pts))
        summary[name] = len(pts)
        print("polymarket", name, len(pts), flush=True)
    return summary


def kalshi() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = _get(f"{KALSHI_API}/events/{KALSHI_EVENT}", {"with_nested_markets": "true"})
    (OUT / "kalshi_event.json").write_text(json.dumps(ev))
    markets = ev.get("markets") or ev.get("event", {}).get("markets", [])
    start, end = _span()
    summary = {}
    for m in markets:
        tic = m["ticker"]
        opened = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        candles, t = [], max(start, opened)
        while t < end:
            u = min(t + CHUNK_DAYS * 86400, end)
            js = _get(f"{KALSHI_API}/series/{KALSHI_SERIES}/markets/{tic}/candlesticks",
                      {"start_ts": t, "end_ts": u, "period_interval": 60})
            candles += js.get("candlesticks", [])
            t = u
            time.sleep(0.3)
        candles = sorted({c["end_period_ts"]: c for c in candles}.values(),
                         key=lambda c: c["end_period_ts"])
        (OUT / f"kalshi_{tic}_hourly.json").write_text(json.dumps(candles))
        summary[tic] = len(candles)
        print("kalshi", tic, len(candles), flush=True)
    return summary


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()
