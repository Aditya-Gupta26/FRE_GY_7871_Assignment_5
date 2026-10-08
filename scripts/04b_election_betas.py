"""Pre-window election betas for the basket checkpoint (no test-window data is touched).

Main pre-window: Aug 1, 2025 to Jun 30, 2026 with Polymarket (history from Jul 19, 2025).
Robustness: Jan 2 to Jun 30, 2026 with Polymarket, and with the Polymarket/Kalshi average.
Writes outputs/tables/q4_election_betas*.csv and prints the checkpoint summary.
"""
import _bootstrap  # noqa: F401
import json

import pandas as pd

from src import basket, market, pm
from src.config import BETA_END, BETA_ROBUST_START, BETA_START, CANDIDATES, TABLES, WINDOW_START

TABLES.mkdir(parents=True, exist_ok=True)
px = market.load_close()
ret = market.log_ret(px)
days = px.index
sw = pm.sweep_daily(days)
# timing check: every snapshot price is stamped no later than 16:05 ET of its own day
lag_ok = (sw["poly_stale_h"].dropna() > -5 / 60 - 1e-9).all() and (sw["kal_stale_h"].dropna() > -5 / 60 - 1e-9).all()
assert lag_ok, "a snapshot price is stamped after 16:05 ET"

out = {}
specs = {
    "main_poly": (BETA_START, BETA_END, "poly_p"),
    "robust_poly_jan": (BETA_ROBUST_START, BETA_END, "poly_p"),
    "robust_avg_jan": (BETA_ROBUST_START, BETA_END, "avg_p"),
}
for tag, (s, e, col) in specs.items():
    idx = days[(days >= s) & (days <= e)]
    assert idx.max() < pd.Timestamp(WINDOW_START.date()), "pre-window overlaps the test window"
    dp = (sw[col].diff() * 100)            # percentage points, trading day to trading day
    # the first day of the window needs the previous trading day's price: diff is on all days
    mm, ar = basket.abnormal(ret, list(CANDIDATES), "SPY", idx, use_alpha=True)
    eb = basket.election_betas(ar, dp, idx).join(mm)
    pg = basket.pooled_group_gamma(ar, dp, idx)
    lg = basket.legs(ar, "strong")
    lw = basket.legs(ar, "weak")
    bk = {"strong_ls": basket.hac_slope(lg["ls"].loc[idx], dp.loc[idx]),
          "strong_long": basket.hac_slope(lg["long"].loc[idx], dp.loc[idx]),
          "strong_short": basket.hac_slope(lg["short"].loc[idx], dp.loc[idx]),
          "weak_ls": basket.hac_slope(lw["ls"].loc[idx], dp.loc[idx]),
          "strong_ls_no_OSCR": basket.hac_slope(basket.legs(ar, "strong", drop=("OSCR",))["ls"].loc[idx], dp.loc[idx])}
    eb.to_csv(TABLES / f"q4_election_betas_{tag}.csv")
    pg.to_csv(TABLES / f"q4_group_gamma_{tag}.csv")
    pd.DataFrame(bk).T.to_csv(TABLES / f"q4_basket_gamma_{tag}.csv")
    dpi = dp.loc[idx]
    out[tag] = {"window": f"{idx.min().date()} to {idx.max().date()}", "N_days": len(idx),
                "dp_sd_pp": round(float(dpi.std()), 3), "dp_zero_share": round(float((dpi.abs() < 1e-9).mean()), 3),
                "p_start": float(sw[col].loc[idx[0]]), "p_end": float(sw[col].loc[idx[-1]]),
                "names_sign_match": int(eb["sign_match"].sum()), "names_total": len(eb),
                "basket": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in bk.items()}}
    print(f"\n=== {tag}: {out[tag]['window']} (N={len(idx)}), dP sd {out[tag]['dp_sd_pp']} pp, "
          f"zero-change share {out[tag]['dp_zero_share']}, P {out[tag]['p_start']:.3f} -> {out[tag]['p_end']:.3f}")
    print(eb[["group", "strength", "expected_sign", "beta", "gamma_coef", "gamma_se", "gamma_t",
              "gamma_ci90_lo", "gamma_ci90_hi", "sign_match"]].round(3).to_string())
    print(pg.round(3).to_string())
    print(pd.DataFrame(bk).T[["coef", "se", "t", "p", "N"]].round(3).to_string())
(TABLES / "q4_election_betas_summary.json").write_text(json.dumps(out, indent=1, default=str))
