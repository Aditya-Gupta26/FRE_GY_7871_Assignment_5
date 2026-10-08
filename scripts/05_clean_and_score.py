"""Clean and filter the corpora, then embed and score them.

  python scripts/05_clean_and_score.py corpus   # waterfall + data/interim/{social,news}.parquet
  python scripts/05_clean_and_score.py embed    # all-MiniLM-L6-v2 embeddings (.npy, row-aligned)
  python scripts/05_clean_and_score.py tone     # VADER + Twitter-RoBERTa on every doc
"""
import _bootstrap  # noqa: F401
import json
import sys

import numpy as np
import pandas as pd

from src.config import INTERIM, PROCESSED, TABLES

INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
part = sys.argv[1] if len(sys.argv) > 1 else "corpus"

if part == "corpus":
    from src.corpus import news_corpus, social_corpus
    soc, wf = social_corpus()
    wf.to_csv(TABLES / "filter_waterfall.csv", index=False)
    print(wf.to_string(index=False))
    soc.to_parquet(INTERIM / "social.parquet")
    print("\nsocial kept by platform/kind/voice:")
    print(soc.groupby(["platform", "kind", "voice"]).size().to_string())
    print("docs per bar day: min", soc.groupby("bar_date").size().min(), "median",
          soc.groupby("bar_date").size().median(), "days", soc["bar_date"].nunique())
    nw, st = news_corpus()
    nw.to_parquet(INTERIM / "news.parquet")
    (TABLES / "news_corpus_counts.json").write_text(json.dumps(st, indent=1))
    print("\nnews:", st)
    print(nw.groupby(["query", "group"]).size().to_string())

elif part == "embed":
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
    for name in ("social", "news"):
        d = pd.read_parquet(INTERIM / f"{name}.parquet")
        E = m.encode(d["clean"].tolist(), batch_size=256, show_progress_bar=False,
                     normalize_embeddings=True, convert_to_numpy=True)
        np.save(INTERIM / f"emb_{name}.npy", E.astype(np.float32))
        print(name, E.shape)

elif part == "tone":
    from src.sentiment import roberta_scores, vader_scores
    for name in ("social", "news"):
        d = pd.read_parquet(INTERIM / f"{name}.parquet")
        texts = d["clean"].tolist()
        out = pd.concat([d[["doc_id"]].reset_index(drop=True), vader_scores(texts), roberta_scores(texts)], axis=1)
        out.to_parquet(PROCESSED / f"tone_{name}.parquet")
        print(name, out.shape, out[["vader_compound", "rob_neg", "rob_pos"]].mean().round(3).to_dict())

elif part == "nli":
    # zero-shot NLI direction score on every tweet (the validated primary tool for tweets)
    from src.sentiment import NLIStance
    d = pd.read_parquet(INTERIM / "social.parquet")
    tw = d[d["platform"] == "twitter"].reset_index(drop=True)
    out = pd.concat([tw[["doc_id"]], NLIStance(batch=64, max_len=256)(tw["clean"].tolist())], axis=1)
    out.to_parquet(PROCESSED / "nli_twitter.parquet")
    print("nli twitter", out.shape, out[["nli_D", "nli_R", "nli_s"]].mean().round(3).to_dict())
