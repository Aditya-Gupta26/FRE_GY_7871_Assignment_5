"""Directional sentiment index (Q2), as pre-registered in config.PREREG.

Per doc: s in [-1, 1] (+ = pro-Democrat / anti-GOP), from the tool chosen by validation.
Per issue k, source g (Twitter and each subreddit) and bar t:
    m_{k,g,t} = mean of s over the docs,   m~_{k,g,t} = m_{k,g,t} - mean_t(m_{k,g,.})
so a source that always leans one way (r/Conservative, r/democrats) adds no level, only its
day-to-day moves. Sources are combined with fixed weights w_g (window doc shares):
    I_{k,t} = sum_g w_g m~_{k,g,t},   a source with no docs on t counts at its own mean (0).
An issue-day with fewer than MIN_DOCS_ISSUE_DAY docs is NaN, forward-filled for at most
FFILL_MAX_DAYS days. The composite is the salience-weighted mean over the key issues:
    C_t = sum_k w_k I_{k,t} / sum_k w_k,   w_k = issue k's voter-voice share in Q1.
Bars: calendar days [16:00 ET t-1, 16:00 ET t) for prediction markets (24/7), or trading
days [previous trading day 16:00, 16:00) for stocks, which pools weekends into Monday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FFILL_MAX_DAYS, MIN_DOCS_ISSUE_DAY


def to_trading_bar(bar_date: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day on or after the calendar bar date (its 16:00 close ends the bar)."""
    td = np.sort(trading_days.to_numpy())
    pos = np.searchsorted(td, bar_date.to_numpy(), side="left")
    out = pd.Series(pd.NaT, index=bar_date.index, dtype="datetime64[ns]")
    ok = pos < len(td)
    out[ok] = td[pos[ok]]
    return out


def issue_index(docs: pd.DataFrame, score: str, bar: str, issues: list[str], dates: pd.DatetimeIndex,
                source_col: str = "source") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (I: dates x issues, counts: dates x issues)."""
    d = docs[docs["issue"].isin(issues)]
    w_g = d[source_col].value_counts(normalize=True)
    cell = d.groupby(["issue", source_col, bar])[score].mean().rename("m").reset_index()
    cell["m_dm"] = cell["m"] - cell.groupby(["issue", source_col])["m"].transform("mean")
    cell["w"] = cell[source_col].map(w_g)
    cell["wm"] = cell["w"] * cell["m_dm"]
    I = cell.groupby(["issue", bar])["wm"].sum().unstack("issue").reindex(dates).reindex(columns=issues)
    n = d.groupby(["issue", bar]).size().unstack("issue").reindex(dates).reindex(columns=issues).fillna(0)
    I = I.where(n >= MIN_DOCS_ISSUE_DAY).ffill(limit=FFILL_MAX_DAYS)
    return I, n


def composite(I: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(I.columns).fillna(0)
    num = (I * w).sum(axis=1, min_count=1)
    den = I.notna().mul(w).sum(axis=1)
    return (num / den.replace(0, np.nan)).rename("C")


def split_half(docs: pd.DataFrame, score: str, bar: str, issues: list[str], dates, w: pd.Series,
               seed: int, source_col: str = "source") -> dict:
    """Random split of each day's docs into halves; correlate the two composites across days.
    Spearman-Brown step-up for the full-length reliability: 2r / (1 + r)."""
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 2, len(docs)).astype(bool)
    Ca = composite(issue_index(docs[half], score, bar, issues, dates, source_col)[0], w)
    Cb = composite(issue_index(docs[~half], score, bar, issues, dates, source_col)[0], w)
    r_lvl = float(Ca.corr(Cb))
    r_chg = float(Ca.diff().corr(Cb.diff()))
    return {"r_level": r_lvl, "r_change": r_chg, "sb_level": 2 * r_lvl / (1 + r_lvl),
            "sb_change": 2 * r_chg / (1 + r_chg) if r_chg > -1 else np.nan}
