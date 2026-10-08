"""Q1: issue salience ("what voters care about") from the topic model + LLM taxonomy.

Salience follows the class reading Bybee, Kelly, Manela and Xiu (2024), who measure news
"attention" as each topic's share of text. Here, for a set of docs D and issue k:
    share_k = (# docs in D whose topic maps to issue k) / (# docs in D whose topic maps to any issue)
so non-issue topics (horse race, media, noise) and HDBSCAN outliers are left out of the base.
Engagement-weighted version: each doc weighted by log(1 + engagement) (likes + retweets +
replies + quotes for tweets; score + comments for Reddit), so a viral post counts more but
cannot swamp the rest. Outlier-reduced version: outliers re-assigned by BERTopic's
reduce_outliers(strategy="embeddings").
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (COVERAGE_MIN, INTERIM, ISSUE_VOCAB, KEY_SHARE_MIN, LABELS, MIN_DOCS_ISSUE_DAY,
                        POLLS)

INDIA_RE = re.compile(r"\b(?:india|indian|bjp|modi|rahul gandhi|lok sabha|kerala|bengal|tamil nadu|"
                      r"karnataka|delhi|telangana|bihar)\b", re.I)


def doc_issues() -> pd.DataFrame:
    dt = pd.read_parquet(INTERIM / "doc_topics.parquet")
    tax = pd.read_csv(LABELS / "topic_taxonomy.csv")
    lab = dict(zip(tax["topic"], tax["label"]))
    conf = dict(zip(tax["topic"], tax["confidence"]))
    dt["issue"] = dt["topic"].map(lab).fillna("Outlier")
    dt["issue_ro"] = dt["topic_ro"].map(lab).fillna("Outlier")
    dt["conf"] = dt["topic"].map(conf).fillna("none")
    s = pd.read_parquet(INTERIM / "social.parquet")[["doc_id", "engagement", "sub", "clean"]]
    n = pd.read_parquet(INTERIM / "news.parquet")[["doc_id", "group", "site", "clean"]]
    dt = dt.merge(s, on="doc_id", how="left").merge(n, on="doc_id", how="left", suffixes=("", "_n"))
    dt["clean"] = dt["clean"].fillna(dt.pop("clean_n"))
    dt["india"] = dt["clean"].str.contains(INDIA_RE)
    return dt


def shares(d: pd.DataFrame, col: str = "issue", weight: str | None = None) -> pd.Series:
    x = d[d[col].isin(ISSUE_VOCAB)]
    w = np.log1p(x[weight].fillna(0).clip(lower=0)) if weight else pd.Series(1.0, index=x.index)
    s = w.groupby(x[col]).sum()
    return (s / s.sum()).reindex(ISSUE_VOCAB).fillna(0.0)


def coverage(d: pd.DataFrame, dates: pd.DatetimeIndex, col: str = "issue") -> pd.Series:
    x = d[d[col].isin(ISSUE_VOCAB)]
    n = x.groupby([col, "day"]).size().unstack(col).reindex(dates).fillna(0)
    return (n >= MIN_DOCS_ISSUE_DAY).mean().reindex(ISSUE_VOCAB).fillna(0.0)


def key_topics(share: pd.Series, cov: pd.Series) -> list[str]:
    """Pre-registered rule: share >= 5% and >= 20 docs on >= 90% of days; top 5 to 8 by share."""
    ok = share[(share >= KEY_SHARE_MIN) & (cov >= COVERAGE_MIN)].sort_values(ascending=False)
    keys = list(ok.index[:8])
    if len(keys) < 5:   # rule says at least 5: fill by share among issues meeting coverage
        extra = share[(cov >= COVERAGE_MIN) & ~share.index.isin(keys)].sort_values(ascending=False)
        keys += list(extra.index[: 5 - len(keys)])
    return keys


def poll_compare(voter: pd.Series, news: pd.Series) -> pd.DataFrame:
    rows = []
    for poll, cats in POLLS.items():
        p = pd.Series(cats, dtype=float)
        v, nw = voter.reindex(p.index) * 100, news.reindex(p.index) * 100
        rv = stats.spearmanr(p, v).statistic if len(p) > 2 else np.nan
        rn = stats.spearmanr(p, nw).statistic if len(p) > 2 else np.nan
        for k in p.index:
            rows.append({"poll": poll, "issue": k, "poll_pct": p[k], "voter_pct": v[k], "news_pct": nw[k],
                         "spearman_voter": rv, "spearman_news": rn, "n_categories": len(p)})
    return pd.DataFrame(rows)
