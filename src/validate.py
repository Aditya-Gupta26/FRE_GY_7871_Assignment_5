"""Validation of the direction and tone tools against the blind labels.

Direction tools (each gives a score s in [-1, 1], + = pro-Democrat / anti-GOP, and a label D/R/N):
- nli:     s = P_entail(pro-D hypothesis) - P_entail(pro-R hypothesis), each scored on its own
           (multi_label); label = N if both P < 0.5, else the larger side.
- rob_tgt: target-signed Twitter-RoBERTa. tau = +1 if only Dem-side names, -1 if only GOP-side
           names, 0 otherwise; s = (P_pos - P_neg) * tau; label from the RoBERTa argmax:
           neutral or tau = 0 -> N, positive -> the named side, negative -> the other side.
- vad_tgt: same with VADER compound (cut-offs +-0.05, Hutto and Gilbert 2014).
- distil:  logistic regression on the 384-d sentence embeddings, trained on the 1,500 subagent
           calibration labels ("LLM-distilled"); s = P(D) - P(R); label = argmax.
Metrics: macro-F1 (scikit-learn f1_score, average="macro": unweighted mean of per-class F1) and
Cohen's kappa, kappa = (p_o - p_e) / (1 - p_e) (Cohen 1960), with 95% percentile bootstrap CIs
(2,000 resamples of items).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

from src.config import DEM_TERMS, GOP_TERMS, RANDOM_SEED

DEM, GOP = re.compile(DEM_TERMS, re.I), re.compile(GOP_TERMS, re.I)


def tau(texts: pd.Series) -> np.ndarray:
    d = texts.str.contains(DEM).to_numpy()
    g = texts.str.contains(GOP).to_numpy()
    return np.where(d & ~g, 1, np.where(g & ~d, -1, 0))


def nli_label(nD, nR) -> np.ndarray:
    nD, nR = np.asarray(nD), np.asarray(nR)
    return np.where(np.maximum(nD, nR) < 0.5, "N", np.where(nD >= nR, "D", "R"))


def target_label(t: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    """polarity in {pos, neg, neu}; t in {-1, 0, 1}."""
    out = np.full(len(t), "N", dtype=object)
    out[(t == 1) & (polarity == "pos")] = "D"
    out[(t == 1) & (polarity == "neg")] = "R"
    out[(t == -1) & (polarity == "pos")] = "R"
    out[(t == -1) & (polarity == "neg")] = "D"
    return out


def rob_polarity(df: pd.DataFrame) -> np.ndarray:
    P = df[["rob_neg", "rob_neu", "rob_pos"]].to_numpy()
    return np.array(["neg", "neu", "pos"])[P.argmax(1)]


def vader_polarity(c: pd.Series) -> np.ndarray:
    return np.where(c >= 0.05, "pos", np.where(c <= -0.05, "neg", "neu"))


def metrics_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_SEED) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    labels = sorted(set(y) | set(p))
    f1 = f1_score(y, p, average="macro", labels=labels, zero_division=0)
    k = cohen_kappa_score(y, p)
    rng = np.random.default_rng(seed)
    fs, ks = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        fs.append(f1_score(y[i], p[i], average="macro", labels=labels, zero_division=0))
        ks.append(cohen_kappa_score(y[i], p[i]) if len(set(y[i]) | set(p[i])) > 1 else np.nan)
    return {"N": int(len(y)), "macro_f1": float(f1), "f1_lo": float(np.nanpercentile(fs, 2.5)),
            "f1_hi": float(np.nanpercentile(fs, 97.5)), "kappa": float(k),
            "kappa_lo": float(np.nanpercentile(ks, 2.5)), "kappa_hi": float(np.nanpercentile(ks, 97.5)),
            "accuracy": float((y == p).mean())}


def train_distilled(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1.0) -> tuple:
    """Grouped 5-fold CV (no author/thread/site in both train and test), then a final fit."""
    cv_pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        m.fit(X[tr], y[tr])
        cv_pred[te] = m.predict(X[te])
    final = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, y)
    return final, cv_pred


def distil_score(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = model.predict_proba(X)
    cls = list(model.classes_)
    s = P[:, cls.index("D")] - P[:, cls.index("R")]
    return s, np.array(cls)[P.argmax(1)]
