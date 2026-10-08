"""Glue for Q2 to Q4 and the extra questions, shared by 08_run_analysis.py and the notebook.

Every test follows config.PREREG (and config.PREREG_DEVIATION for the issue-day minimum).
Formulas and sources are in each function's docstring and in the report's formula register.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

from src import granger, index, q1
from src.config import (EPISODE_DROP, FDR_Q, INTERIM, ISSUE_VOCAB, PERM_MIN_SHIFT, PROCESSED)
from src.validate import rob_polarity, tau

HAC_LAGS = 5


# ------------------------------------------------------------------ doc-level direction scores
def doc_scores() -> pd.DataFrame:
    """Social docs with issue (strict and outlier-reduced) and every direction score.
    s_primary: NLI for tweets, target-signed RoBERTa for Reddit (chosen by validation F1)."""
    s = pd.read_parquet(INTERIM / "social.parquet")
    tone = pd.read_parquet(PROCESSED / "tone_social.parquet")
    nli = pd.read_parquet(PROCESSED / "nli_twitter.parquet")
    di = q1.doc_issues()[["doc_id", "issue", "issue_ro", "conf", "india"]]
    d = s.merge(tone, on="doc_id").merge(di, on="doc_id").merge(nli, on="doc_id", how="left")
    d["tau"] = tau(d["clean"])
    d["s_rob_tgt"] = (d["rob_pos"] - d["rob_neg"]) * d["tau"]
    d["s_vad_tgt"] = d["vader_compound"] * d["tau"]
    d["s_tone"] = d["rob_pos"] - d["rob_neg"]            # non-directional baseline
    d["s_primary"] = np.where(d["platform"] == "twitter", d["nli_s"], d["s_rob_tgt"])
    d["source"] = np.where(d["platform"] == "twitter", "twitter", "r/" + d["sub"].astype(str))
    assert d.loc[d["platform"] == "twitter", "nli_s"].notna().all()
    return d


# ------------------------------------------------------------------ prediction-market changes
def dp_series(sw: pd.DataFrame, col: str = "poly_p", kind: str = "dp", h: int = 1) -> pd.Series:
    """kind dp: change in percentage points, 100 * (p_t - p_{t-h}) (Knight 2006; Snowberg et al.
    2007 use the change in the probability). kind dlogit: change in log-odds, logit p clipped to
    [0.001, 0.999] (Satopaa et al. 2016 use the same clipping), a robustness check only."""
    p = sw[col]
    if kind == "dlogit":
        q = p.clip(0.001, 0.999)
        return np.log(q / (1 - q)).diff(h)
    return 100 * p.diff(h)


def dow_dummies(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.get_dummies(pd.Series(idx.dayofweek, index=idx), prefix="dow", drop_first=True, dtype=float)


def stationarity(x: pd.Series) -> dict:
    """ADF (Said and Dickey 1984; null = unit root) and KPSS (Kwiatkowski et al. 1992;
    null = level stationary). 'stationary' needs ADF p < 0.05 and KPSS p > 0.05."""
    x = x.dropna()
    adf_p = float(adfuller(x, autolag="AIC")[1])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = float(kpss(x, regression="c", nlags="auto")[1])
    return {"N": len(x), "ADF_p": adf_p, "KPSS_p": kp, "stationary": bool(adf_p < 0.05 and kp > 0.05)}


def hac_reg(y: pd.Series, x: pd.Series, exog: pd.DataFrame | None = None, lags: int = HAC_LAGS) -> dict:
    """OLS y on x (+ exog) with Newey-West HAC SEs (Newey and West 1987)."""
    X = pd.concat([x.rename("x")] + ([exog] if exog is not None else []), axis=1)
    D = pd.concat([y.rename("y"), X], axis=1).dropna()
    f = sm.OLS(D["y"], sm.add_constant(D.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"coef": float(f.params["x"]), "se": float(f.bse["x"]), "t": float(f.tvalues["x"]),
            "p_hac": float(f.pvalues["x"]), "N": int(f.nobs)}


def family_a(sent: pd.Series, dp: pd.Series, exog: pd.DataFrame | None, label: str,
             perm: bool = True, names: tuple = ("sent", "dP")) -> pd.DataFrame:
    """Pre-registered family A: same-day relation + Granger lags 1-3 both directions (7 tests).
    BH (Benjamini and Hochberg 1995) at q = 0.10 on the HAC p-values."""
    rows = []
    both = pd.concat([sent.rename("s"), dp.rename("dp")], axis=1).dropna()
    r_p, p_p = stats.pearsonr(both["s"], both["dp"])
    r_s, p_s = stats.spearmanr(both["s"], both["dp"])
    h = hac_reg(both["dp"], both["s"], exog.reindex(both.index) if exog is not None else None)
    a_, b_ = names
    rows.append({"spec": label, "test": "same-day", "direction": f"{b_} ~ {a_}", "lag": 0, "N": h["N"],
                 "pearson_r": r_p, "spearman_r": r_s, "coef": h["coef"], "p_hac": h["p_hac"],
                 "p_perm": _perm_corr(both["s"], both["dp"]) if perm else np.nan})
    ex = exog.reindex(both.index) if exog is not None else None
    for lag in (1, 2, 3):
        for direction, y, x in ((f"{a_} -> {b_}", both["dp"], both["s"]), (f"{b_} -> {a_}", both["s"], both["dp"])):
            g = granger.granger_f(y, x, lag, ex)
            rows.append({"spec": label, "test": "granger", "direction": direction, "lag": lag, "N": g["N"],
                         "pearson_r": np.nan, "spearman_r": np.nan, "coef": g["sum_coef"],
                         "p_hac": g["p_hac"], "p_F": g["p"],
                         "p_perm": granger.permutation_p(y, x, lag, ex, min_shift=PERM_MIN_SHIFT) if perm else np.nan})
    out = pd.DataFrame(rows)
    out["q_bh"] = multipletests(out["p_hac"], alpha=FDR_Q, method="fdr_bh")[1]
    return out


def _perm_corr(x: pd.Series, y: pd.Series, min_shift: int = PERM_MIN_SHIFT) -> float:
    """Circular-shift p for a same-day correlation: |r| of x rolled by k vs the observed |r|.
    Keeps each series' autocorrelation (Yuan and Shou 2024 discuss the end-point wrap)."""
    xv, yv = x.to_numpy(), y.to_numpy()
    obs = abs(np.corrcoef(xv, yv)[0, 1])
    rs = [abs(np.corrcoef(np.roll(xv, k), yv)[0, 1]) for k in range(min_shift, len(xv) - min_shift)]
    return float((1 + sum(r >= obs for r in rs)) / (1 + len(rs)))


def leave_one_day_out(sent: pd.Series, dp: pd.Series, exog, test: str, lag: int, sent_leads: bool) -> dict:
    """Drop one day at a time. For Granger tests the lags are built on the FULL series and only
    the target row is removed (granger_f's `keep`), so a lag never jumps over a removed day."""
    vals = []
    both = pd.concat([sent.rename("s"), dp.rename("dp")], axis=1).dropna()
    for d in both.index:
        keep = both.index.drop(d)
        if test == "same-day":
            b = both.loc[keep]
            vals.append(hac_reg(b["dp"], b["s"], exog.reindex(keep) if exog is not None else None)["p_hac"])
        else:
            y, x = (both["dp"], both["s"]) if sent_leads else (both["s"], both["dp"])
            vals.append(granger.granger_f(y, x, lag, exog.reindex(both.index) if exog is not None else None, keep=keep)["p_hac"])
    return {"max_p": float(np.max(vals)), "share_p_lt_0.10": float(np.mean(np.array(vals) < 0.10))}


def episode_test(sent: pd.Series, dp: pd.Series, exog, test: str, lag: int, sent_leads: bool) -> dict:
    """Re-run one test without the Sept 10 to 26 episode (lags built on the full series)."""
    both = pd.concat([sent.rename("s"), dp.rename("dp")], axis=1).dropna()
    keep = drop_episode(both["s"]).index
    if test == "same-day":
        b = both.loc[keep]
        r = hac_reg(b["dp"], b["s"], exog.reindex(keep) if exog is not None else None)
        return {"p_hac": r["p_hac"], "coef": r["coef"]}
    y, x = (both["dp"], both["s"]) if sent_leads else (both["s"], both["dp"])
    g = granger.granger_f(y, x, lag, exog.reindex(both.index) if exog is not None else None, keep=keep)
    return {"p_hac": g["p_hac"], "coef": g["sum_coef"]}


def drop_episode(s: pd.Series) -> pd.Series:
    a, b = pd.Timestamp(EPISODE_DROP[0]), pd.Timestamp(EPISODE_DROP[1])
    return s[(s.index < a) | (s.index > b)]


def cross_corr(sent: pd.Series, dp: pd.Series, k: int = 5) -> pd.DataFrame:
    """corr(sent_{t-j}, dP_t) for j = -k..k; j > 0 means sentiment leads (descriptive only)."""
    return pd.DataFrame([{"lead_of_sent": j, "r": float(sent.shift(j).corr(dp))} for j in range(-k, k + 1)])


def mde_r(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable correlation, Fisher z: r* = tanh((z_{1-a/2} + z_{power}) / sqrt(N - 3))
    (Hulley et al. 2013, Designing Clinical Research, App. 6C, inverted)."""
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3)))
