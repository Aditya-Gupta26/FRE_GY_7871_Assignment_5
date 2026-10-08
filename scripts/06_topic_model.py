"""Q1 topic model.

  python scripts/06_topic_model.py fit    # BERTopic on a 60k day x source stratified sample, applied to
                                          # every doc; writes doc topics + the sheet for the LLM taxonomy
  python scripts/06_topic_model.py diag   # 3-seed stability (ARI), BERTopic coherence, LDA sweep (Bybee)
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")   # UMAP.transform segfaulted on the full run (PROGRESS 21)
import _bootstrap  # noqa: F401
import json
import pickle
import sys

import numpy as np
import pandas as pd

from src import topics
from src.config import INTERIM, LABELS, RANDOM_SEED, TABLES

part = sys.argv[1] if len(sys.argv) > 1 else "fit"
LABELS.mkdir(parents=True, exist_ok=True)


def load_all():
    s = pd.read_parquet(INTERIM / "social.parquet")
    n = pd.read_parquet(INTERIM / "news.parquet")
    es, en = np.load(INTERIM / "emb_social.npy"), np.load(INTERIM / "emb_news.npy")
    assert len(es) == len(s) and len(en) == len(n)
    s = s.assign(src=s["platform"], day=s["bar_date"])
    n = n.assign(src="news", voice="news", day=n["date"], engagement=np.nan, sub=n["group"])
    cols = ["doc_id", "src", "voice", "sub", "day", "clean", "engagement"]
    allf = pd.concat([s[cols], n[cols]], ignore_index=True)
    return allf, np.vstack([es, en])


allf, E = load_all()
docs = allf["clean"].tolist()

if part == "fit":
    pos = topics.stratified_sample(allf, 60_000, ["src", "day"], RANDOM_SEED)
    min_cluster = int(round(0.004 * len(pos)))
    model, t_fit = topics.fit_bertopic([docs[i] for i in pos], E[pos], RANDOM_SEED, min_cluster)
    t_all = []
    for a in range(0, len(docs), 10_000):            # chunked transform (PROGRESS 21)
        t_c, _ = model.transform(docs[a:a + 10_000], embeddings=E[a:a + 10_000])
        t_all.extend(t_c)
        print("transformed", min(a + 10_000, len(docs)), "of", len(docs), flush=True)
    t_all = np.array(t_all)
    t_ro = np.array(model.reduce_outliers(docs, t_all, strategy="embeddings", embeddings=E))
    allf["topic"], allf["topic_ro"] = t_all, t_ro
    allf[["doc_id", "src", "voice", "day", "topic", "topic_ro"]].to_parquet(INTERIM / "doc_topics.parquet")
    with open(INTERIM / "bertopic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    words = topics.topic_words(model, 10)
    # topic sheet for the blind LLM taxonomy: words + 5 docs nearest the topic centroid
    rows = []
    for t, ws in words.items():
        idx = np.where(t_all == t)[0]
        c = E[idx].mean(0)
        near = idx[np.argsort(-(E[idx] @ c))[:5]]
        share = allf.loc[idx, "src"].value_counts(normalize=True)
        rows.append({"topic": t, "n_docs": len(idx), "top_words": ", ".join(ws),
                     "share_twitter": round(share.get("twitter", 0), 3), "share_reddit": round(share.get("reddit", 0), 3),
                     "share_news": round(share.get("news", 0), 3),
                     **{f"example_{k + 1}": docs[i][:220] for k, i in enumerate(near)}})
    sheet = pd.DataFrame(rows).sort_values("topic")
    sheet.to_csv(LABELS / "topic_sheet.csv", index=False)
    purity = sheet[["share_twitter", "share_reddit", "share_news"]].max(axis=1)
    summ = {"fit_sample": int(len(pos)), "min_cluster_size": min_cluster, "min_samples": 15,
            "n_topics": int(len(words)), "outlier_share_fit": float((t_fit == -1).mean()),
            "outlier_share_all": float((t_all == -1).mean()),
            "outlier_share_after_reduce": float((t_ro == -1).mean()),
            "source_purity_mean": float(purity.mean()),
            "source_purity_docweighted": float(np.average(purity, weights=sheet["n_docs"])),
            "topics_over_80pct_one_source": int((purity > 0.8).sum())}
    (TABLES / "q1_bertopic_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

elif part == "diag":
    pos = topics.stratified_sample(allf, 30_000, ["src", "day"], RANDOM_SEED + 1)
    sub_docs = [docs[i] for i in pos]
    toks = [topics.tokenize(d) for d in sub_docs]
    from sklearn.metrics import adjusted_rand_score
    labs = {}
    for sd in (1, 2, 3):
        m, lab = topics.fit_bertopic(sub_docs, E[pos], sd, int(round(0.004 * len(pos))))
        labs[sd] = lab
        coh = topics.coherence(topics.topic_words(m, 10), toks)
        print("seed", sd, "topics", len(set(lab)) - (1 if -1 in lab else 0), "outliers",
              round(float((lab == -1).mean()), 3), coh, flush=True)
        labs[f"coh{sd}"] = coh
    ari = {f"{a}-{b}": float(adjusted_rand_score(labs[a], labs[b])) for a, b in ((1, 2), (1, 3), (2, 3))}
    np.save(INTERIM / "seed_positions.npy", pos)
    for sd in (1, 2, 3):
        np.save(INTERIM / f"seed_labels_{sd}.npy", labs[sd])
    sweep, models, dic = topics.lda_sweep(toks)
    sweep.to_csv(TABLES / "q1_lda_sweep.csv", index=False)
    kbest = int(sweep.sort_values("c_npmi", ascending=False)["K"].iloc[0])
    lda, words = models[kbest]
    pd.DataFrame({"topic": list(words), "top_words": [", ".join(w) for w in words.values()]}).to_csv(
        TABLES / "q1_lda_topics.csv", index=False)
    out = {"subsample": int(len(pos)), "ari": ari,
           "bertopic_coherence": {str(s): labs[f"coh{s}"] for s in (1, 2, 3)},
           "lda_best_K_by_c_npmi": kbest, "lda_sweep": sweep.to_dict(orient="records")}
    (TABLES / "q1_topic_diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
