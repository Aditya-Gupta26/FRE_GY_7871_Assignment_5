"""Text cleaning shared by tweets, Reddit and news.

Copied from Assignment 4. Follows the preprocessing in the class readings on disk
(Bagheri; the Twitter sentiment overview slides; Yener's step-by-step guide):
- links are replaced with a token, and user mentions with @user (the
  cardiffnlp RoBERTa convention)
- HTML entities are unescaped (the class tweet_data.csv still had &amp;amp;)
- letter repetitions are squeezed to 2 (cooool -> cool)
- emojis are kept because VADER scores them
- exact and near duplicates are dropped (Jaccard on word 3-shingles, MinHash LSH)
"""
from __future__ import annotations

import html
import re

import pandas as pd
from datasketch import MinHash, MinHashLSH

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w{1,30}")
REPEAT_RE = re.compile(r"([A-Za-z])\1{2,}")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z][a-z'\-]+")
HASHTAG_RE = re.compile(r"#\w+")
CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}\b")
SPAM_TAGS = {"#airdrop", "#nft", "#nfts", "#crypto", "#bitcoin", "#btc", "#memecoin",
             "#giveaway", "#presale", "#solana", "#web3", "#altcoin", "#pump"}


def normalize(text: str | None) -> str:
    if not text:
        return ""
    t = html.unescape(html.unescape(str(text)))
    t = URL_RE.sub("http", t)
    t = MENTION_RE.sub("@user", t)
    t = REPEAT_RE.sub(r"\1\1", t)
    return WS_RE.sub(" ", t).strip()


def fix_gdelt_title(t: str | None) -> str:
    """GDELT titles come tokenised: 'Ex - Anthropic', '24 , 000', 'It s'."""
    if not t:
        return ""
    t = re.sub(r"(\d) , (\d{3})", r"\1,\2", t)
    t = re.sub(r"\s+([,.;:!?%])", r"\1", t)
    t = re.sub(r"(\w) - (\w)", r"\1-\2", t)
    t = re.sub(r"\b(\w+) s\b", lambda m: m.group(1) + "'s"
               if m.group(1).lower() not in {"it", "that", "there", "what", "he", "she", "who"}
               else m.group(1) + "'s", t)
    return WS_RE.sub(" ", t).strip()


def dedup_key(text: str) -> str:
    """Lowercase, drop link/mention tokens, punctuation and digits."""
    t = text.lower().replace("http", " ").replace("@user", " ")
    return " ".join(WORD_RE.findall(t))


def is_spam(text: str) -> bool:
    tags = [h.lower() for h in HASHTAG_RE.findall(text)]
    if len(tags) + len(CASHTAG_RE.findall(text)) >= 8:
        return True
    return len(SPAM_TAGS.intersection(tags)) >= 2


def ascii_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def near_duplicate_mask(texts: list[str], threshold: float = 0.8, num_perm: int = 64) -> list[bool]:
    """True for rows that are a near duplicate of an EARLIER row (keep the first)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    dup = []
    for i, t in enumerate(texts):
        toks = t.split()
        shingles = {" ".join(toks[j:j + 3]) for j in range(max(1, len(toks) - 2))}
        if not shingles or len(toks) < 4:
            dup.append(False)
            continue
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf8"))
        if lsh.query(m):
            dup.append(True)
        else:
            lsh.insert(str(i), m)
            dup.append(False)
    return dup


def clean_frame(df: pd.DataFrame, text_col: str = "text", time_col: str = "ts_utc",
                min_words: int = 3) -> pd.DataFrame:
    """Adds `clean`, drops junk, spam and duplicates. Returns a copy sorted by time."""
    d = df.copy()
    d["clean"] = d[text_col].map(normalize)
    d["n_words"] = d["clean"].str.split().str.len().fillna(0).astype(int)
    d = d[d["n_words"] >= min_words]
    d = d[d["clean"].map(ascii_ratio) >= 0.8]
    d["spam"] = d["clean"].map(is_spam)
    d = d[~d["spam"]]
    d = d.sort_values(time_col)
    d["dkey"] = d["clean"].map(dedup_key)
    d = d[d["dkey"].str.len() > 0].drop_duplicates("dkey")
    d = d[~pd.Series(near_duplicate_mask(d["dkey"].tolist()), index=d.index)]
    return d.drop(columns=["dkey"])
