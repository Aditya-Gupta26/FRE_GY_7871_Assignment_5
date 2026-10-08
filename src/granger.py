"""Granger causality tests (copied from Assignment 4, NaN handling fixed for A5).

Follows the class paper Souza et al. (2015): bivariate tests in BOTH directions, with
p-values reported by lag (1, 2, 3). The F-test is the standard restricted-vs-unrestricted
one, F = ((SSR_r - SSR_u) / q) / (SSR_u / df_u), the same as statsmodels
grangercausalitytests' ssr_ftest (Granger 1969 for the concept).

Additions for a short sample:
- stationarity checks (ADF + KPSS) before testing
- a VAR with lag order chosen by AIC/BIC, and an exogenous FOMC dummy
- the SIGN of the effect (Granger itself is sign-free): sum of lag coefficients,
  and orthogonalised impulse responses with Monte Carlo bands
- a circular-shift permutation p-value, which keeps each series' own
  autocorrelation but breaks their alignment, so it does not lean on the F
  distribution in small samples
- Benjamini-Hochberg FDR across the grid of tests
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def stationarity(x: pd.Series) -> dict:
    x = x.dropna()
    adf_p = adfuller(x, autolag="AIC")[1]
    try:
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    except Exception:  # noqa: BLE001
        kpss_p = np.nan
    return {"N": len(x), "ADF_p": adf_p, "KPSS_p": kpss_p,
            "stationary": bool(adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05))}


def _lagmat(s: pd.Series, p: int) -> pd.DataFrame:
    return pd.concat({f"{s.name}_L{k}": s.shift(k) for k in range(1, p + 1)}, axis=1)


def granger_f(y: pd.Series, x: pd.Series, p: int, exog: pd.DataFrame | None = None,
              keep: pd.Index | None = None) -> dict:
    """OLS Granger F-test: does x help predict y beyond y's own p lags (+ exog)?
    Returns F, p-value, N, and the sum of x-lag coefficients (the sign).
    `keep` restricts the TARGET rows after the lags are built on the full series
    (e.g. intraday bars only), so a lag never jumps over a removed bar."""
    y = y.rename("y")
    x = x.rename("x")
    Y = pd.concat([y, _lagmat(y, p), _lagmat(x, p)], axis=1)
    if exog is not None:
        Y = pd.concat([Y, exog], axis=1)
    Y = Y.dropna()
    if keep is not None:
        Y = Y.loc[Y.index.intersection(keep)]
        Y = Y.loc[:, Y.std() > 0] if "overnight" in Y else Y
    xcols = [c for c in Y.columns if c.startswith("x_L")]
    rcols = [c for c in Y.columns if c not in xcols and c != "y"]
    full = sm.OLS(Y["y"], sm.add_constant(Y[rcols + xcols])).fit()
    rest = sm.OLS(Y["y"], sm.add_constant(Y[rcols])).fit()
    q = len(xcols)
    df2 = full.df_resid
    F = ((rest.ssr - full.ssr) / q) / (full.ssr / df2)
    from scipy import stats
    pval = 1 - stats.f.cdf(F, q, df2)
    # HAC-robust Wald as a check (heteroskedastic, fat-tailed returns)
    hac = sm.OLS(Y["y"], sm.add_constant(Y[rcols + xcols])).fit(cov_type="HAC",
                                                               cov_kwds={"maxlags": p})
    R = np.zeros((q, len(hac.params)))
    for i, c in enumerate(xcols):
        R[i, list(hac.params.index).index(c)] = 1
    wald = hac.wald_test(R, scalar=True)
    return {"lag": p, "N": int(full.nobs), "F": float(F), "p": float(pval),
            "p_hac": float(wald.pvalue), "sum_coef": float(full.params[xcols].sum()),
            "coef_L1": float(full.params["x_L1"])}


def permutation_p(y: pd.Series, x: pd.Series, p: int, exog=None, min_shift: int | None = None,
                  keep=None) -> float:
    """Circular-shift test: share of shifts whose F is >= the observed F.
    min_shift keeps shifted series away from the true alignment; it scales with N
    (7 bars = one trading day for hourly data, 2 days for the 16-day daily sample)."""
    # align and drop NaNs FIRST, so a leading NaN from differencing is not rolled into the
    # middle of the shifted series (fix found in the A5 verification pass)
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    y, x = both["y"].rename(y.name), both["x"].rename(x.name)
    obs = granger_f(y, x, p, exog, keep)["F"]
    xv = x.to_numpy()
    n = len(xv)
    if min_shift is None:
        min_shift = 7 if n > 50 else 2
    Fs = []
    for k in range(min_shift, n - min_shift):
        xs = pd.Series(np.roll(xv, k), index=x.index, name=x.name)
        Fs.append(granger_f(y, xs, p, exog, keep)["F"])
    Fs = np.array(Fs)
    return float((1 + (Fs >= obs).sum()) / (1 + len(Fs)))


def both_directions(ret: pd.Series, sent: pd.Series, lags=(1, 2, 3), exog=None,
                    perm: bool = True, label: str = "", keep=None) -> pd.DataFrame:
    """Lag-by-lag table, in both directions (Souza et al. 2015)."""
    rows = []
    for p in lags:
        a = granger_f(ret, sent, p, exog, keep)
        b = granger_f(sent, ret, p, exog, keep)
        rows.append({"test": label, "direction": "sent -> ret", **a,
                     "p_perm": permutation_p(ret, sent, p, exog, keep=keep) if perm else np.nan})
        rows.append({"test": label, "direction": "ret -> sent", **b,
                     "p_perm": permutation_p(sent, ret, p, exog, keep=keep) if perm else np.nan})
    return pd.DataFrame(rows)


def var_analysis(df: pd.DataFrame, x: str, y: str, maxlags: int = 7, exog=None,
                 ic: str = "bic", irf_h: int = 7, fixed_p: int | None = None) -> dict:
    """VAR on [x, y]: lag choice, Granger both ways, IRF of y to an x shock.
    fixed_p pins the lag (we use 3, the max lag of the Granger table, so the IRF
    carries the same dynamics; AIC/BIC choices are still reported)."""
    data = df[[x, y]].dropna()
    ex = None if exog is None else exog.loc[data.index]
    model = VAR(data, exog=ex)
    sel = model.select_order(maxlags=maxlags)
    p = fixed_p if fixed_p else max(1, int(getattr(sel, ic)))
    res = model.fit(p)
    gc_xy = res.test_causality(y, [x], kind="f")
    gc_yx = res.test_causality(x, [y], kind="f")
    irf = res.irf(irf_h)
    try:
        lo, hi = irf.errband_mc(orth=True, repl=500, signif=0.1, seed=7871)
    except Exception:  # noqa: BLE001
        lo = hi = None
    ix, iy = data.columns.get_loc(x), data.columns.get_loc(y)
    return {
        "lag_aic": int(sel.aic), "lag_bic": int(sel.bic), "lag_used": p, "N": int(res.nobs),
        "p_x_to_y": float(gc_xy.pvalue), "p_y_to_x": float(gc_yx.pvalue),
        "irf": irf.orth_irfs[:, iy, ix],
        "irf_lo": None if lo is None else lo[:, iy, ix],
        "irf_hi": None if hi is None else hi[:, iy, ix],
        "cum_irf": float(irf.orth_cum_effects[irf_h, iy, ix]),
        "sum_coef_x_in_y": float(sum(res.params.loc[f"L{k}.{x}", y] for k in range(1, p + 1))),
    }


def fdr(pvals: pd.Series, alpha: float = 0.10) -> pd.Series:
    ok = pvals.notna()
    out = pd.Series(np.nan, index=pvals.index)
    out[ok] = multipletests(pvals[ok], alpha=alpha, method="fdr_bh")[1]
    return out
