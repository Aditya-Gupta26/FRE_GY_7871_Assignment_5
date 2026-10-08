"""Prediction-market series at the daily 16:00 ET snapshot.

- Polymarket "Democrats Sweep": hourly CLOB `prices-history` points. The docs do not define
  `p`; a live check on Oct 7 matched it to the order-book midpoint (0.635), not the last trade
  (0.64), so it is treated as the displayed (mid) price. Points are stamped ~17 s past the
  hour, so the snapshot is the last point at or before 16:05 ET (SNAPSHOT_TOL_MIN).
- Kalshi KXBALANCEPOWERCOMBO-27FEB-DD: hourly candles; `end_period_ts` is the inclusive end of
  the hour. The last-trade field is empty in most hourly candles, so the series is the mid of
  the yes_bid and yes_ask closes ("offer price ... at the end of the candlestick period").
Each snapshot keeps the real timestamp of the price used, so timing can be asserted later.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ET, RAW, SNAPSHOT_HOUR_ET, SNAPSHOT_TOL_MIN

D = RAW / "predmkt"


def poly_hourly(name: str = "Democrats_Sweep") -> pd.DataFrame:
    pts = json.loads((D / f"poly_{name}_hourly.json").read_text())
    df = pd.DataFrame(pts)
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(ET)
    return df[["ts", "p"]].sort_values("ts").reset_index(drop=True)


def kalshi_hourly(ticker: str = "KXBALANCEPOWERCOMBO-27FEB-DD") -> pd.DataFrame:
    cs = json.loads((D / f"kalshi_{ticker}_hourly.json").read_text())

    def f(c, side):
        v = (c.get(side) or {}).get("close_dollars")
        return float(v) if v not in (None, "") else np.nan

    df = pd.DataFrame({"ts": pd.to_datetime([c["end_period_ts"] for c in cs], unit="s", utc=True),
                       "bid": [f(c, "yes_bid") for c in cs], "ask": [f(c, "yes_ask") for c in cs]})
    df["ts"] = df["ts"].dt.tz_convert(ET)
    df["p"] = (df["bid"] + df["ask"]) / 2
    return df[["ts", "p", "bid", "ask"]].sort_values("ts").reset_index(drop=True)


def snapshot(h: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each date: last hourly price with ts <= date 16:00 + tolerance. Returns p, price_ts."""
    h = h.dropna(subset=["p"])
    # build the local wall-clock time first, then localise, so a DST day is still 16:05 ET
    cuts = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {SNAPSHOT_HOUR_ET:02d}:{SNAPSHOT_TOL_MIN:02d}").tz_localize(ET)
                             for d in dates])
    pos = np.searchsorted(h["ts"].to_numpy(), cuts.to_numpy(), side="right") - 1
    out = pd.DataFrame(index=pd.DatetimeIndex([d.normalize().tz_localize(None) for d in dates]))
    ok = pos >= 0
    out["p"] = np.where(ok, h["p"].to_numpy()[np.clip(pos, 0, None)], np.nan)
    out["price_ts"] = pd.Series(np.where(ok, h["ts"].to_numpy()[np.clip(pos, 0, None)],
                                         np.datetime64("NaT")), index=out.index)
    nominal = cuts - pd.Timedelta(minutes=SNAPSHOT_TOL_MIN)
    out["stale_h"] = (nominal.tz_convert("UTC").tz_localize(None).to_numpy()
                      - pd.to_datetime(out["price_ts"], utc=True).dt.tz_localize(None).to_numpy()
                      ) / np.timedelta64(1, "h")
    return out


def sweep_daily(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Polymarket, Kalshi mid and their average at 16:00 ET for the given dates."""
    pm = snapshot(poly_hourly(), dates).add_prefix("poly_")
    ks = snapshot(kalshi_hourly(), dates).add_prefix("kal_")
    df = pd.concat([pm, ks], axis=1)
    df["avg_p"] = df[["poly_p", "kal_p"]].mean(axis=1, skipna=False)
    return df
