"""Build the cleaned social and news corpora, with the filter waterfall.

Social docs (Twitter + Reddit) pass the filters in the pre-set order and every step's
count is recorded (outputs/tables/filter_waterfall.csv). Each doc gets:
- bar_date: the daily 16:00 ET bar it feeds, [16:00 ET on t-1, 16:00 ET on t). A post at
  15:59 ET on Sept 14 feeds Sept 14; a post at 16:00 feeds Sept 15. Asserted below.
- voice: "voter" (comments, Reddit self-posts, tweets from non-media accounts) or
  "shared_media" (Reddit link posts, whose r/politics titles must be the exact headline,
  and tweets from news / media / organisation accounts).
News headlines carry a date only (Google News date-bounded feeds stamp most items 07:00 GMT),
so news gets a calendar date and is never used in a timed test.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src import reddit, twitter
from src.config import (COLLECT_START, ET, MAX_PER_AUTHOR_DAY, MIN_WORDS, OUTLETS, POLITICS_RE,
                        PROMO_RE, RAW, RD_MIN_SCORE, TW_MIN_ACCOUNT_AGE_DAYS, TW_MIN_FOLLOWERS,
                        TW_MIN_VIEWS, WINDOW_END)
from src.text_clean import ascii_ratio, dedup_key, is_spam, near_duplicate_mask, normalize

POL = re.compile(POLITICS_RE, re.I)
PROMO = re.compile(PROMO_RE, re.I)
MEDIA_RE = re.compile(r"\b(?:news|journalist|reporter|correspondent|editor|anchor|newsroom|"
                      r"newspaper|magazine|radio|tv|television|press|media|breaking|podcast|"
                      r"official account|wire|columnist)\b", re.I)
SITE_GROUP = {s: g for g, ss in OUTLETS.items() for s in ss}
DEAD_SITES = {"msnbc.com", "abcnews.go.com"}   # replaced by ms.now and abcnews.com (PROGRESS 6)


def bar_date(ts_utc: pd.Series) -> pd.Series:
    """Date of the first 16:00 ET at or after... strictly after ts: ts + 8 h, then the date."""
    et = ts_utc.dt.tz_convert(ET)
    return (et + pd.Timedelta(hours=8)).dt.tz_localize(None).dt.normalize()


def _assert_timing(df: pd.DataFrame):
    close = (df["bar_date"] + pd.Timedelta(hours=16)).dt.tz_localize(ET)
    assert (df["ts"].dt.tz_convert(ET) < close).all(), "a doc is at/after its bar's 16:00 close"
    assert (df["ts"].dt.tz_convert(ET) >= close - pd.Timedelta(days=1)).all(), "a doc predates its bar"


def load_twitter() -> pd.DataFrame:
    d = pd.DataFrame(twitter.load_raw())
    d["ts"] = pd.to_datetime(d["createdAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    d["author_created_ts"] = pd.to_datetime(d["author_created"], format="%a %b %d %H:%M:%S %z %Y",
                                            utc=True, errors="coerce")
    media = (d["author_verified_type"].fillna("").str.lower().isin(["business", "government"])
             | d["author_bio"].fillna("").str.contains(MEDIA_RE)
             | d["author_name"].fillna("").str.contains(MEDIA_RE))
    return pd.DataFrame({
        "doc_id": "tw_" + d["id"].astype(str), "platform": "twitter", "sub": "twitter", "kind": "tweet",
        "ts": d["ts"], "author": d["author_user"].fillna(""), "text": d["text"].fillna(""),
        "lang": d["lang"], "views": d["viewCount"], "likes": d["likeCount"],
        "followers": d["author_followers"], "acct_age_days": (d["ts"] - d["author_created_ts"]).dt.days,
        "score": np.nan, "engagement": d["likeCount"].fillna(0) + d["retweetCount"].fillna(0)
        + d["replyCount"].fillna(0) + d["quoteCount"].fillna(0),
        "voice": np.where(media, "shared_media", "voter"), "thread": d["conversationId"].astype(str)})


def load_reddit() -> pd.DataFrame:
    d = pd.DataFrame(reddit.load_raw())
    is_link = (d["kind"] == "posts") & ~d["url"].fillna("").str.contains("reddit.com/r/")
    thread = np.where(d["kind"] == "posts", "t3_" + d["id"].astype(str), d["link_id"].astype(str))
    return pd.DataFrame({
        "doc_id": "rd_" + d["id"].astype(str), "platform": "reddit", "sub": d["subreddit"],
        "kind": np.where(d["kind"] == "posts", np.where(is_link, "link_post", "self_post"), "comment"),
        "ts": pd.to_datetime(d["created_utc"].astype(int), unit="s", utc=True),
        "author": d["author"].fillna(""), "text": d["text"].fillna(""), "lang": None,
        "views": np.nan, "likes": np.nan, "followers": np.nan, "acct_age_days": np.nan,
        "score": d["score"], "engagement": d["score"].clip(lower=0) + d.get("num_comments", 0).fillna(0),
        "voice": np.where(is_link, "shared_media", "voter"), "thread": thread})


def social_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean social docs, waterfall table)."""
    raw = pd.concat([load_twitter(), load_reddit()], ignore_index=True)
    steps = []

    def log(name, df):
        steps.append({"step": name, **df.groupby("platform").size().to_dict(), "total": len(df)})

    d = raw
    log("0 downloaded", d)
    d = d[(d["ts"] >= COLLECT_START) & (d["ts"] < WINDOW_END)]
    log("1 inside Jun 30 16:00 to Oct 7 16:00 ET", d)
    removed = d["text"].str.strip().str.match(r"^\[(?:removed|deleted)\]") | (d["text"].str.strip() == "")
    d = d[~removed]
    log("2 not removed or deleted", d)
    d = d.assign(clean=d["text"].map(normalize))
    eng = np.where(d["platform"] == "twitter", d["lang"].eq("en"), d["clean"].map(ascii_ratio) >= 0.8)
    d = d[eng]
    log("3 English (tweet lang; Reddit ascii share >= 0.8)", d)
    d = d[d["clean"].str.contains(POL)]
    log("4 US-politics term in the text itself", d)
    spam = d["clean"].map(is_spam) | d["clean"].str.contains(PROMO)
    d = d[~spam]
    log("5 no spam or promo language", d)
    d = d.assign(n_words=d["clean"].str.split().str.len())
    d = d[d["n_words"] >= MIN_WORDS]
    log(f"6 at least {MIN_WORDS} words", d)
    tw = d["platform"] == "twitter"
    ok_tw = (d["views"] >= TW_MIN_VIEWS) & (d["followers"] >= TW_MIN_FOLLOWERS) & \
            (d["acct_age_days"] >= TW_MIN_ACCOUNT_AGE_DAYS)
    ok_rd = d["score"] >= RD_MIN_SCORE
    d = d[(tw & ok_tw) | (~tw & ok_rd)]
    log(f"7 engagement/account: tweets views>={TW_MIN_VIEWS}, followers>={TW_MIN_FOLLOWERS}, "
        f"age>={TW_MIN_ACCOUNT_AGE_DAYS}d; Reddit score>={RD_MIN_SCORE}", d)
    d = d.sort_values("ts")
    d = d.assign(dkey=d["clean"].map(dedup_key))
    d = d[d["dkey"].str.len() > 0].drop_duplicates("dkey")
    d = d[~pd.Series(near_duplicate_mask(d["dkey"].tolist()), index=d.index)]
    log("8 no exact or near duplicate (MinHash, Jaccard >= 0.8)", d)
    d = d.assign(bar_date=bar_date(d["ts"]))
    d = d[d.groupby(["platform", "author", "bar_date"]).cumcount() < MAX_PER_AUTHOR_DAY]
    log(f"9 at most {MAX_PER_AUTHOR_DAY} docs per author per day", d)
    d = d.drop(columns=["dkey"]).reset_index(drop=True)
    _assert_timing(d)
    assert d["doc_id"].is_unique
    return d, pd.DataFrame(steps).fillna(0)


def _domain(u: str | None) -> str:
    h = urlparse(u or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def news_corpus() -> tuple[pd.DataFrame, dict]:
    rows = []
    for sub in ("rss_sites", "rss_topup", "rss_us"):
        for p in sorted((RAW / "news" / sub).glob("*.json")):
            j = json.loads(p.read_text())
            site = j.get("site")
            if site in DEAD_SITES or j["topic"] != "ELECTION":
                continue   # POLICY ignored the date operators: 10,000 items, 104 unique links (PROGRESS 20)
            for it in j["items"]:
                rows.append({"title": it.get("title"), "link": it.get("link"), "published": it.get("published"),
                             "source": it.get("source"), "source_href": it.get("source_href"),
                             "query": j["topic"], "pull": sub, "site": site})
    d = pd.DataFrame(rows)
    n0 = len(d)
    d["date"] = pd.to_datetime(d["published"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    d["domain"] = d["source_href"].map(_domain)
    d["site"] = d["site"].fillna(d["domain"])
    d["group"] = d["site"].map(SITE_GROUP).fillna("US_OTHER")
    # Google News appends " - Source name" to every title
    d["clean"] = [normalize(re.sub(r"\s+-\s+" + re.escape(s or "") + r"\s*$", "", t or "")) if s else normalize(t)
                  for t, s in zip(d["title"], d["source"])]
    d = d[(d["date"] >= pd.Timestamp(COLLECT_START.date())) & (d["date"] <= pd.Timestamp(WINDOW_END.date()))]
    n1 = len(d)
    d = d.sort_values(["date", "pull"]).drop_duplicates("link")
    n2 = len(d)
    d = d.assign(dkey=d["clean"].map(dedup_key))
    d = d[d["dkey"].str.split().str.len() >= 4].drop_duplicates(["site", "dkey"])
    n3 = len(d)
    d = d.drop(columns=["dkey"]).reset_index(drop=True)
    d["doc_id"] = "nw_" + d.index.astype(str)
    stats = {"downloaded": n0, "dated_in_window": n1, "unique_link": n2, "unique_title_per_site_4plus_words": n3}
    return d, stats
