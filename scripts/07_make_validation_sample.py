"""Blind labelling sheets (make) and tool scoring against the labels (score).

  python scripts/07_make_validation_sample.py make
      validation: 150 items (60 tweets, 50 Reddit, 40 headlines)
      calibration: 1,500 items (600 tweets, 600 Reddit, 300 headlines), disjoint from validation
      50 calibration items are duplicated into a second sheet for a second, independent labeller
      Both are 2/3 docs naming a party side and 1/3 not, stratified by month, so the direction
      classes are not almost all "neither". Sheets hold only an item id and the text.
  python scripts/07_make_validation_sample.py score   (after labelling; see src/validate.py)
"""
import _bootstrap  # noqa: F401
import re
import sys

import numpy as np
import pandas as pd

from src.config import DEM_TERMS, GOP_TERMS, INTERIM, LABELS, RANDOM_SEED

part = sys.argv[1] if len(sys.argv) > 1 else "make"
DEM, GOP = re.compile(DEM_TERMS, re.I), re.compile(GOP_TERMS, re.I)


def frame() -> pd.DataFrame:
    s = pd.read_parquet(INTERIM / "social.parquet")
    n = pd.read_parquet(INTERIM / "news.parquet")
    s = s.assign(src=s["platform"], month=s["bar_date"].dt.month)
    n = n.assign(src="news", month=n["date"].dt.month)
    d = pd.concat([s[["doc_id", "src", "month", "clean"]], n[["doc_id", "src", "month", "clean"]]],
                  ignore_index=True)
    d["party"] = d["clean"].str.contains(DEM) | d["clean"].str.contains(GOP)
    return d


def draw(d: pd.DataFrame, n: int, rng, exclude: set) -> pd.DataFrame:
    d = d[~d["doc_id"].isin(exclude)]
    parts = []
    for flag, k in ((True, int(round(n * 2 / 3))), (False, n - int(round(n * 2 / 3)))):
        g = d[d["party"] == flag]
        per = g.groupby("month").size()
        alloc = (per / per.sum() * k).round().astype(int)
        alloc.iloc[-1] += k - alloc.sum()
        for m, km in alloc.items():
            gm = g[g["month"] == m]
            parts.append(gm.sample(n=min(km, len(gm)), random_state=int(rng.integers(1e9))))
    return pd.concat(parts)


if part == "make":
    rng = np.random.default_rng(RANDOM_SEED)
    d = frame()
    val = pd.concat([draw(d[d["src"] == s], k, rng, set()) for s, k in
                     (("twitter", 60), ("reddit", 50), ("news", 40))])
    cal = pd.concat([draw(d[d["src"] == s], k, rng, set(val["doc_id"])) for s, k in
                     (("twitter", 600), ("reddit", 600), ("news", 300))])
    assert not set(val["doc_id"]) & set(cal["doc_id"]), "validation and calibration overlap"
    for name, x in (("validation", val), ("calibration", cal)):
        x = x.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        x["item_id"] = [f"{name[0].upper()}{i:04d}" for i in range(len(x))]
        x[["item_id", "doc_id", "src", "party", "month"]].to_parquet(LABELS / f"{name}_key.parquet")
        sheet = x[["item_id", "clean"]].rename(columns={"clean": "text"})
        if name == "calibration":
            for b in range(5):
                sheet.iloc[b * 300:(b + 1) * 300].to_csv(LABELS / f"calibration_sheet_{b + 1}.csv", index=False)
            sheet.sample(n=50, random_state=RANDOM_SEED).to_csv(LABELS / "calibration_sheet_dup50.csv", index=False)
        else:
            sheet.to_csv(LABELS / "validation_sheet.csv", index=False)
        print(name, len(x), x.groupby(["src", "party"]).size().to_dict())

elif part == "score":
    import json
    import pickle
    import time

    from src import validate as V
    from src.config import PROCESSED, TABLES
    from src.sentiment import NLIStance

    def labels(name):
        fs = sorted(LABELS.glob(f"{name}_labels*.csv")) if name == "calibration" else [LABELS / f"{name}_labels.csv"]
        fs = [f for f in fs if "dup50" not in f.name]
        lab = pd.concat([pd.read_csv(f) for f in fs])
        lab["direction"] = lab["direction"].str.strip().str.upper()
        lab["tone"] = lab["tone"].str.strip().str.lower()
        assert lab["direction"].isin(["D", "R", "N"]).all() and lab["tone"].isin(["pos", "neg", "neu"]).all()
        assert lab["item_id"].is_unique
        return lab

    s = pd.read_parquet(INTERIM / "social.parquet")
    n = pd.read_parquet(INTERIM / "news.parquet")
    E = np.vstack([np.load(INTERIM / "emb_social.npy"), np.load(INTERIM / "emb_news.npy")])
    meta = pd.concat([s[["doc_id", "clean", "author"]], n[["doc_id", "clean", "site"]].rename(columns={"site": "author"})],
                     ignore_index=True)
    meta["row"] = np.arange(len(meta))
    tone = pd.concat([pd.read_parquet(PROCESSED / "tone_social.parquet"), pd.read_parquet(PROCESSED / "tone_news.parquet")])
    frames = {}
    for name in ("validation", "calibration"):
        key = pd.read_parquet(LABELS / f"{name}_key.parquet")
        lab = labels(name)
        assert set(lab["item_id"]) == set(key["item_id"]), f"{name}: labels do not cover the sheet"
        frames[name] = key.merge(lab, on="item_id").merge(meta, on="doc_id").merge(tone, on="doc_id")
    # zero-shot NLI on the 1,650 labelled docs only (speed benchmark recorded)
    allx = pd.concat(frames.values(), ignore_index=True)
    t0 = time.time()
    nli = NLIStance()(allx["clean"].tolist())
    secs = time.time() - t0
    allx = pd.concat([allx, nli], axis=1)
    allx.to_parquet(PROCESSED / "labelled_scored.parquet")
    # tool labels
    allx["tau"] = V.tau(allx["clean"])
    allx["lab_nli"] = V.nli_label(allx["nli_D"], allx["nli_R"])
    allx["lab_rob_tgt"] = V.target_label(allx["tau"].to_numpy(), V.rob_polarity(allx))
    allx["lab_vad_tgt"] = V.target_label(allx["tau"].to_numpy(), V.vader_polarity(allx["vader_compound"]))
    cal = allx[allx["item_id"].str.startswith("C")].reset_index(drop=True)
    val = allx[allx["item_id"].str.startswith("V")].reset_index(drop=True)
    Xc, Xv = E[cal["row"].to_numpy()], E[val["row"].to_numpy()]
    model, cv_pred = V.train_distilled(Xc, cal["direction"].to_numpy(), cal["author"].fillna("").to_numpy())
    val["lab_distil"] = V.distil_score(model, Xv)[1]
    with open(PROCESSED / "distilled_direction.pkl", "wb") as f:
        pickle.dump(model, f)
    rows = []
    for tool in ("lab_nli", "lab_rob_tgt", "lab_vad_tgt", "lab_distil"):
        for grp, g in [("all", val)] + list(val.groupby("src")):
            rows.append({"task": "direction", "tool": tool[4:], "texts": grp,
                         **V.metrics_ci(g["direction"].to_numpy(), g[tool].to_numpy())})
    for tool, pol in (("roberta", V.rob_polarity(val)), ("vader", V.vader_polarity(val["vader_compound"]))):
        for grp in ["all"] + sorted(val["src"].unique()):
            msk = np.ones(len(val), bool) if grp == "all" else (val["src"] == grp).to_numpy()
            rows.append({"task": "tone", "tool": tool, "texts": grp,
                         **V.metrics_ci(val.loc[msk, "tone"].to_numpy(), pol[msk])})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "q2_validation.csv", index=False)
    cvm = V.metrics_ci(cal["direction"].to_numpy(), cv_pred)
    # inter-labeller agreement on the 50 duplicated calibration items
    dup = pd.read_csv(LABELS / "calibration_labels_dup50.csv")
    dup["direction"] = dup["direction"].str.strip().str.upper()
    m = dup.merge(cal[["item_id", "direction"]], on="item_id", suffixes=("_b", "_a"))
    inter = {"N": len(m), "kappa": float(V.cohen_kappa_score(m["direction_a"], m["direction_b"])),
             "agreement": float((m["direction_a"] == m["direction_b"]).mean())}
    summary = {"nli_seconds_for_1650_docs": round(secs, 1), "distilled_cv_on_calibration": cvm,
               "interlabeller_direction": inter,
               "label_counts_validation": val["direction"].value_counts().to_dict(),
               "label_counts_calibration": cal["direction"].value_counts().to_dict(),
               "tau_nonzero_share_validation": float((val["tau"] != 0).mean())}
    (TABLES / "q2_validation_summary.json").write_text(json.dumps(summary, indent=1))
    print(res.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))
