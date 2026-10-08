"""Election betas and the mid-term basket.

Election beta, per name (Knight 2006, J. Public Econ. 90(4-5), eq. 3, run name by name):
    AR_it = a_i + gamma_i * dP_t + u_it
AR_it is the market-model abnormal return in percent (MacKinlay 1997 eq. 7) and dP_t the daily
change in P(Democratic sweep) in percentage points, both close to close at 16:00 ET, so gamma_i
is "% abnormal return per 1 pp rise in the sweep odds". Knight runs this pooled over firms with
firm effects; the pooled group version below does the same within each policy group, with
standard errors clustered by date (every firm shares the same dP_t on a date).
HAC standard errors (Newey and West 1987, Bartlett weights) use 5 lags = one trading week; the
lag count is a choice, not a formula. Snowberg, Wolfers and Zitzewitz (2007, QJE) warn that daily
pre-election regressions like this are biased (other news moves both series), so the Rigobon
and Sack (2003) heteroskedasticity estimator is run later as a robustness check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import GROUPS, NAME_GROUP

HAC_LAGS = 5


def hac_slope(y: pd.Series, x: pd.Series, lags: int = HAC_LAGS) -> dict:
    ok = y.notna() & x.notna()
    f = sm.OLS(y[ok], sm.add_constant(x[ok].rename("x"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b, se = float(f.params["x"]), float(f.bse["x"])
    z90 = stats.norm.ppf(0.95)
    return {"coef": b, "se": se, "t": b / se, "p": float(f.pvalues["x"]), "N": int(f.nobs),
            "ci90_lo": b - z90 * se, "ci90_hi": b + z90 * se}


def abnormal(ret: pd.DataFrame, names: list[str], bench: str, est_idx: pd.Index,
             use_alpha: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market-model ARs (percent) for `names` against `bench`, parameters from est_idx."""
    rows, ar = [], {}
    for t in names:
        r, m = ret[t], ret[bench]
        ok = r.loc[est_idx].notna() & m.loc[est_idx].notna()
        f = sm.OLS(r.loc[est_idx][ok], sm.add_constant(m.loc[est_idx][ok].rename("m"))).fit()
        a, b = float(f.params["const"]), float(f.params["m"])
        rows.append({"ticker": t, "alpha": a, "beta": b, "beta_N": int(f.nobs)})
        ar[t] = 100 * (r - (a if use_alpha else 0.0) - b * m)
    return pd.DataFrame(rows).set_index("ticker"), pd.DataFrame(ar)


def election_betas(ar: pd.DataFrame, dp: pd.Series, idx: pd.Index) -> pd.DataFrame:
    rows = []
    for t in ar.columns:
        g = NAME_GROUP[t]
        h = hac_slope(ar[t].loc[idx], dp.loc[idx])
        rows.append({"ticker": t, "group": g, "strength": GROUPS[g]["strength"],
                     "expected_sign": GROUPS[g]["sign"], **{f"gamma_{k}": v for k, v in h.items()},
                     "sign_match": bool(np.sign(h["coef"]) == GROUPS[g]["sign"])})
    return pd.DataFrame(rows).set_index("ticker")


def pooled_group_gamma(ar: pd.DataFrame, dp: pd.Series, idx: pd.Index) -> pd.DataFrame:
    """Knight-style pooled regression within each group: firm effects, date-clustered SEs."""
    rows = []
    for g, spec in GROUPS.items():
        long = []
        for t in spec["names"]:
            d = pd.DataFrame({"ar": ar[t].loc[idx], "dp": dp.loc[idx], "firm": t})
            long.append(d)
        L = pd.concat(long).dropna()
        L["date"] = L.index
        X = pd.get_dummies(L["firm"], drop_first=False, dtype=float)
        X["dp"] = L["dp"]
        f = sm.OLS(L["ar"], X).fit(cov_type="cluster",
                                   cov_kwds={"groups": pd.factorize(L["date"])[0]})
        b, se = float(f.params["dp"]), float(f.bse["dp"])
        rows.append({"group": g, "strength": spec["strength"], "expected_sign": spec["sign"],
                     "n_names": len(spec["names"]), "gamma": b, "se": se, "t": b / se,
                     "p": float(f.pvalues["dp"]), "N": int(f.nobs),
                     "sign_match": bool(np.sign(b) == spec["sign"])})
    return pd.DataFrame(rows).set_index("group")


def legs(ar: pd.DataFrame, strength: str | tuple = "strong", drop: tuple = ()) -> pd.DataFrame:
    """Equal-weighted long (+1) and short (-1) legs and L/S = long - short (MacKinlay eq. 13
    is the equal-weighted average AR). Each day averages the names with a return that day."""
    strengths = (strength,) if isinstance(strength, str) else strength
    names = [t for t in ar.columns if GROUPS[NAME_GROUP[t]]["strength"] in strengths and t not in drop]
    lo = [t for t in names if GROUPS[NAME_GROUP[t]]["sign"] > 0]
    sh = [t for t in names if GROUPS[NAME_GROUP[t]]["sign"] < 0]
    out = pd.DataFrame({"long": ar[lo].mean(axis=1, skipna=True),
                        "short": ar[sh].mean(axis=1, skipna=True)})
    out["ls"] = out["long"] - out["short"]
    return out


def rigobon_sack(dp: pd.Series, y: pd.Series, high: pd.Series, n_boot: int = 2000, seed: int = 7871) -> dict:
    """Heteroskedasticity-based estimate of the response of y to the sweep factor
    (Rigobon and Sack 2003, NBER WP 9609, eq. 10, read from the A3 class reading):
        d = [Cov_H(dP, y) - Cov_L(dP, y)] / [Var_H(dP) - Var_L(dP)]
    H = high election-news days, L = the other days. It only assumes the OTHER common factors
    are equally volatile on H and L days, so noise in dP and omitted factors cancel in the
    differences (the bias Snowberg et al. 2007 warn about for daily regressions).
    95% CI: bootstrap, resampling days within H and within L."""
    D = pd.concat([dp.rename("dp"), y.rename("y"), high.rename("H")], axis=1).dropna()
    H, L = D[D["H"].astype(bool)], D[~D["H"].astype(bool)]

    def est(h, l):
        num = np.cov(h["dp"], h["y"])[0, 1] - np.cov(l["dp"], l["y"])[0, 1]
        den = h["dp"].var() - l["dp"].var()
        return num / den, den

    d, den = est(H, L)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        hb = H.iloc[rng.integers(0, len(H), len(H))]
        lb = L.iloc[rng.integers(0, len(L), len(L))]
        bs.append(est(hb, lb)[0])
    return {"d": float(d), "var_shift": float(den), "N_H": len(H), "N_L": len(L),
            "ci95_lo": float(np.nanpercentile(bs, 2.5)), "ci95_hi": float(np.nanpercentile(bs, 97.5)),
            "var_ratio_H_over_L": float(H["dp"].var() / L["dp"].var())}
