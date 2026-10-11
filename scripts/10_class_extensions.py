"""Exploratory extensions from the in-class discussions (config.PREREG_CLASS, written Oct 10 19:27 ET
before any of these was run). Reuses the Q2 to Q4 objects exactly as 08_run_analysis.py builds them.

  python scripts/10_class_extensions.py      # writes outputs/tables/x*_*.csv and outputs/results_class.json

X1 scenario split of the odds (House vs Senate)       lecture slide 24
X2 issue-matched sentiment and attention tests         lecture slides 22-23
X3 attention vs the size of odds moves                 meme deck slides 2, 5
X4 Kalshi volume ("does trading confirm the story")    meme deck slides 5, 16
X5 placebos: random-sign baskets, Indian-politics text lecture slide 28
"""
import _bootstrap  # noqa: F401
import glob
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src import basket as B, index, market, pipeline as P, pm
from src.config import (BETA_END, BETA_START, CANDIDATES, ET, FDR_Q, FIGURES, GROUPS, INTERIM, NAME_GROUP,
                        OUTPUTS, PREREG, RANDOM_SEED, RAW, TABLES)
from src.corpus import bar_date

warnings.filterwarnings("ignore")
R = {}
CAL = pd.date_range("2026-07-01", "2026-10-07")
res = json.loads((OUTPUTS / "results.json").read_text())
keys = res["q1"]["key_topics"]
_sal = pd.read_csv(TABLES / "q1_issue_shares.csv", index_col=0)
w_k = _sal.loc[keys, "voter"] / _sal.loc[keys, "voter"].sum()      # exact weights (results.json rounds them)
SIGN = PREREG["expected_sign_index_vs_dp"]


def hac_multi(y: pd.Series, X: pd.DataFrame, lags: int = 5) -> dict:
    """OLS of y on several regressors with Newey-West HAC SEs (Newey and West 1987)."""
    D = pd.concat([y.rename("y"), X], axis=1).dropna()
    f = sm.OLS(D["y"], sm.add_constant(D[X.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"N": int(f.nobs)}
    for c in X.columns:
        out.update({f"{c}_coef": float(f.params[c]), f"{c}_se": float(f.bse[c]), f"{c}_p": float(f.pvalues[c])})
    return out


def claim_checks(fam: pd.DataFrame, s: pd.Series, y: pd.Series, ex, q_col: str = "q_bh") -> list:
    """PREREG claim rule: expected sign (+), BH q < 0.10, permutation p < 0.10, then
    leave-one-day-out and the Sept 10-26 episode drop for anything that passes the first three."""
    out = []
    for _, row in fam.iterrows():
        ok = bool(np.sign(row["coef"]) == SIGN and row[q_col] < FDR_Q and row["p_perm"] < 0.10)
        rec = {"test": row["test"], "direction": row["direction"], "lag": int(row["lag"]), "pre_checks_pass": ok}
        if ok:
            lead = row["direction"].split(" -> ")[0] == fam.attrs.get("lead_name", "sent")
            rec["lodo"] = P.leave_one_day_out(s, y, ex, row["test"], int(row["lag"]), lead)
            rec["drop_episode"] = P.episode_test(s, y, ex, row["test"], int(row["lag"]), lead)
            rec["claim"] = bool(rec["lodo"]["max_p"] < 0.10 and rec["drop_episode"]["p_hac"] < 0.10
                                and np.sign(rec["drop_episode"]["coef"]) == SIGN)
        out.append(rec)
    return out


def summarise(fam: pd.DataFrame, q_col: str = "q_bh") -> dict:
    sd = fam[fam["test"] == "same-day"].iloc[0]
    return {"same_day_r": float(sd["pearson_r"]), "same_day_p_hac": float(sd["p_hac"]),
            "min_p_hac": float(fam["p_hac"].min()), "min_q": float(fam[q_col].min()),
            "n_p_lt_0.10": int((fam["p_hac"] < 0.10).sum()), "n_tests": int(len(fam))}


# ---------------------------------------------------------------- shared objects (as in 08_run_analysis.py)
px = market.load_close()
ret = market.log_ret(px)
days = px.index
pre = days[(days >= BETA_START) & (days <= BETA_END)]
td = days[(days >= "2026-07-01") & (days <= "2026-10-07")]
names = list(CANDIDATES)
mm, ar_pre = B.abnormal(ret, names, "SPY", pre, use_alpha=True)          # pre-window ARs (with alpha)
AR = pd.DataFrame({t: 100 * (ret[t] - mm.loc[t, "beta"] * ret["SPY"]) for t in names}).loc[td]   # alpha = 0
LS_td = B.legs(AR, "strong")["ls"]
LS_pre = B.legs(ar_pre, "strong")["ls"]
d = P.doc_scores()
d = d[d["bar_date"].isin(CAL)]
dv = d[d["voice"] == "voter"].assign(issue=lambda x: x["issue_ro"])
C = index.composite(index.issue_index(dv, "s_primary", "bar_date", keys, CAL)[0], w_k)
dv_td = dv.assign(td_bar=index.to_trading_bar(dv["bar_date"], days))
C_td = index.composite(index.issue_index(dv_td, "s_primary", "td_bar", keys, td)[0], w_k)
assert np.allclose(C.dropna().to_numpy(),
                   pd.read_csv(TABLES / "q2_composite_daily.csv", index_col=0, parse_dates=True)["C"].dropna().to_numpy())
ex, ex_td = P.dow_dummies(CAL), P.dow_dummies(td)

# ================================================================ X1 scenario split
alld = pd.date_range("2025-07-19", "2026-10-07")
snap = {}
for nm in ("Democrats_Sweep", "R_Senate_D_House", "D_Senate_R_House", "Republicans_Sweep", "Other"):
    snap[f"poly_{nm}"] = pm.snapshot(pm.poly_hourly(nm), alld)["p"]
for tk in ("DD", "DR", "RD", "RR"):
    snap[f"kal_{tk}"] = pm.snapshot(pm.kalshi_hourly(f"KXBALANCEPOWERCOMBO-27FEB-{tk}"), alld)["p"]
sc = pd.DataFrame(snap)
sc["poly_sum5"] = sc[[c for c in sc if c.startswith("poly_")]].sum(axis=1, min_count=5)
sc["poly_house"] = sc["poly_Democrats_Sweep"] + sc["poly_R_Senate_D_House"]
sc["poly_senate"] = sc["poly_Democrats_Sweep"] + sc["poly_D_Senate_R_House"]
sc["kal_house"] = sc["kal_DD"] + sc["kal_DR"]
sc["kal_senate"] = sc["kal_DD"] + sc["kal_RD"]
sc.to_csv(TABLES / "x1_scenario_snapshots.csv")
cal_dp = {c: 100 * sc[c].diff().reindex(CAL) for c in ("poly_Democrats_Sweep", "poly_house", "poly_senate", "kal_house", "kal_senate")}
td_all = days[days >= "2025-07-21"]
td_dp = {c: 100 * sc[c].reindex(td_all).diff() for c in ("poly_Democrats_Sweep", "poly_house", "poly_senate", "kal_house", "kal_senate")}
x1 = {"poly_sum5_window": {"mean": float(sc["poly_sum5"].reindex(CAL).mean()), "min": float(sc["poly_sum5"].reindex(CAL).min()),
                           "max": float(sc["poly_sum5"].reindex(CAL).max())},
      "path": {c: {"Jul1": float(100 * sc[c].loc["2026-07-01"]), "Sep9": float(100 * sc[c].loc["2026-09-09"]),
                   "Sep26": float(100 * sc[c].loc["2026-09-26"]), "Oct7": float(100 * sc[c].loc["2026-10-07"])}
               for c in ("poly_Democrats_Sweep", "poly_house", "poly_senate", "kal_house", "kal_senate")},
      "corr_daily_window": {"house_senate": float(cal_dp["poly_house"].corr(cal_dp["poly_senate"])),
                            "house_sweep": float(cal_dp["poly_house"].corr(cal_dp["poly_Democrats_Sweep"])),
                            "senate_sweep": float(cal_dp["poly_senate"].corr(cal_dp["poly_Democrats_Sweep"])),
                            "house_poly_kalshi": float(cal_dp["poly_house"].corr(cal_dp["kal_house"])),
                            "senate_poly_kalshi": float(cal_dp["poly_senate"].corr(cal_dp["kal_senate"]))},
      "zero_change_share": {c: float((cal_dp[c].abs() < 1e-9).mean()) for c in ("poly_house", "poly_senate")}}
# (a) the basket on House and Senate odds, jointly and alone
rows = []
for win, y, idx in (("pre-window", LS_pre, pre), ("test window", LS_td, td)):
    for venue in ("poly", "kal"):
        if venue == "kal" and win == "pre-window":
            continue                      # Kalshi starts Dec 11, 2025
        X = pd.DataFrame({"dP_house": td_dp[f"{venue}_house"], "dP_senate": td_dp[f"{venue}_senate"]}).reindex(idx)
        rows.append({"window": win, "venue": venue, "model": "joint", **hac_multi(y.reindex(idx), X)})
        for c in X.columns:
            h = B.hac_slope(y.reindex(idx), X[c])
            rows.append({"window": win, "venue": venue, "model": c, "N": h["N"], f"{c}_coef": h["coef"],
                         f"{c}_se": h["se"], f"{c}_p": h["p"]})
    hs = B.hac_slope(y.reindex(idx), td_dp["poly_Democrats_Sweep"].reindex(idx))
    rows.append({"window": win, "venue": "poly", "model": "dP_sweep (reference)", "N": hs["N"],
                 "dP_sweep_coef": hs["coef"], "dP_sweep_se": hs["se"], "dP_sweep_p": hs["p"]})
x1_basket = pd.DataFrame(rows)
x1_basket.to_csv(TABLES / "x1_basket_house_senate.csv", index=False)
x1["corr_daily_pre_window_house_senate"] = float(td_dp["poly_house"].reindex(pre).corr(td_dp["poly_senate"].reindex(pre)))
# (b) sentiment C vs House and Senate odds: family A with permutation and the full claim rule
fams = []
for c in ("poly_house", "poly_senate"):
    f = P.family_a(C, cal_dp[c], ex, f"C vs dP {c}")
    f.attrs["lead_name"] = "sent"
    x1[f"family_{c}"] = summarise(f)
    x1[f"claims_{c}"] = claim_checks(f, C, cal_dp[c], ex)
    fams.append(f)
for c in ("kal_house", "kal_senate"):
    f = P.family_a(C, cal_dp[c], ex, f"C vs dP {c}", perm=False)
    x1[f"family_{c}"] = {k: v for k, v in summarise(f).items()}
    fams.append(f)
pd.concat(fams, ignore_index=True).to_csv(TABLES / "x1_family_C_house_senate.csv", index=False)
R["X1"] = x1
print("X1 paths:", json.dumps(x1["path"], indent=0)[:600])
print(x1_basket.round(3).to_string(index=False))
print({k: v for k, v in x1.items() if k.startswith("family")})

# ================================================================ X2 issue-matched tests
PAIRS = {"Health care": ["aca_insurers", "hospitals"], "Immigration & deportations": ["detention"],
         "Foreign policy & Iran war": ["defense"], "Tech & AI": ["crypto"], "Guns": ["guns"]}
issues5 = list(PAIRS)
I5, n5 = index.issue_index(dv_td, "s_primary", "td_bar", issues5, td)
cov5 = (n5 >= 5).mean()
gAR = {}
for k, groups in PAIRS.items():
    nm = [t for g in groups for t in GROUPS[g]["names"]]
    sign = GROUPS[groups[0]]["sign"]
    gAR[k] = sign * AR[nm].mean(axis=1)
# attention: each issue's share of voter issue docs on the trading-day bar (strict assignment, as in Q1)
from src import q1
from src.config import ISSUE_VOCAB
di = q1.doc_issues()
vo = di[(di["voice"] == "voter") & di["day"].isin(CAL) & di["issue"].isin(ISSUE_VOCAB)].copy()
vo["td_bar"] = index.to_trading_bar(pd.to_datetime(vo["day"]), days)
cnt = vo.groupby(["td_bar", "issue"]).size().unstack("issue").reindex(td).fillna(0)
share_td = cnt.div(cnt.sum(axis=1), axis=0)
x2_rows, x2_fams, att_rows = [], [], []
for k in issues5:
    tested = bool(cov5[k] >= 0.50)
    x2_rows.append({"issue": k, "groups": "+".join(PAIRS[k]), "coverage_ge5_td": float(cov5[k]), "tested": tested})
    if not tested:
        continue
    f = P.family_a(I5[k], gAR[k], ex_td, f"{k} -> {'+'.join(PAIRS[k])}", names=("sent", "AR"))
    f.attrs["lead_name"] = "sent"
    f["pair"] = k
    x2_fams.append(f)
    a, y = share_td[k], gAR[k].abs()
    h = P.hac_reg(y, a, ex_td)
    both = pd.concat([a.rename("a"), y.rename("y")], axis=1).dropna()
    att_rows.append({"pair": k, "r": float(both["a"].corr(both["y"])), "coef": h["coef"], "p_hac": h["p_hac"],
                     "p_perm": P._perm_corr(both["a"], both["y"]), "N": h["N"]})
x2f = pd.concat(x2_fams, ignore_index=True)
x2f["q_pooled"] = multipletests(x2f["p_hac"], alpha=FDR_Q, method="fdr_bh")[1]
x2f.to_csv(TABLES / "x2_issue_matched_direction.csv", index=False)
att = pd.DataFrame(att_rows)
att["q_bh"] = multipletests(att["p_hac"], alpha=FDR_Q, method="fdr_bh")[1]
att.to_csv(TABLES / "x2_issue_matched_attention.csv", index=False)
x2_claims = []
for k, f in x2f.groupby("pair"):
    f = f.reset_index(drop=True); f.attrs["lead_name"] = "sent"
    x2_claims += [{"pair": k, **c} for c in claim_checks(f, I5[k], gAR[k], ex_td, q_col="q_pooled")]
# (c) placebo matrix: every tested issue against every group (same-day r)
tested5 = [r["issue"] for r in x2_rows if r["tested"]]
mat = pd.DataFrame({g: {k: float(pd.concat([I5[k], gAR[g]], axis=1).dropna().corr().iloc[0, 1]) for k in tested5}
                    for g in tested5})
mat.to_csv(TABLES / "x2_placebo_matrix.csv")
diag = {k: float(mat.loc[k, k]) for k in tested5}
off = mat.to_numpy()[~np.eye(len(tested5), dtype=bool)]
R["X2"] = {"pairs": x2_rows, "direction_summary": {k: summarise(f, "q_pooled") for k, f in x2f.groupby("pair")},
           "n_direction_tests": int(len(x2f)), "n_q_lt_0.10": int((x2f["q_pooled"] < 0.10).sum()),
           "n_p_lt_0.10": int((x2f["p_hac"] < 0.10).sum()), "claims": x2_claims,
           "attention": att.round(4).to_dict(orient="records"),
           "placebo_matrix": {"matched_r": diag, "mean_abs_matched": float(np.mean(np.abs(list(diag.values())))),
                              "mean_abs_mismatched": float(np.mean(np.abs(off))),
                              "matched_rank_in_row_by_abs": {k: int((mat.loc[k].abs() >= abs(mat.loc[k, k])).sum()) for k in tested5}}}
print("\nX2 coverage:", cov5.round(2).to_dict())
print(x2f[["pair", "test", "direction", "lag", "N", "pearson_r", "coef", "p_hac", "p_perm", "q_pooled"]].round(4).to_string(index=False))
print(att.round(4).to_string(index=False)); print(mat.round(3).to_string())

# ================================================================ X3 attention vs |dP|
posts = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(str(RAW / "reddit" / "posts_*.jsonl")))],
                  ignore_index=True).drop_duplicates("id")
posts["bar"] = bar_date(pd.to_datetime(posts["created_utc"], unit="s", utc=True))
rd = posts.groupby("bar").agg(posts=("id", "size"),
                              authors=("author", lambda a: a[~a.isin(["[deleted]", "AutoModerator"])].nunique())).reindex(CAL)
pages = []
for f in sorted(glob.glob(str(RAW / "twitter" / "*.json"))):
    tw = json.load(open(f)).get("tweets", [])
    if len(tw) < 2:
        continue
    t = pd.to_datetime([x["createdAt"] for x in tw], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    span = (t.max() - t.min()).total_seconds() / 60
    if span > 0:
        pages.append({"newest": t.max(), "rate_per_min": (len(tw) - 1) / span})
pg = pd.DataFrame(pages)
pg["bar"] = bar_date(pd.Series(pg["newest"]))
tw_rate = pg.groupby("bar")["rate_per_min"].median().reindex(CAL)
z = lambda s: (s - s.mean()) / s.std()
att_df = pd.DataFrame({"reddit_posts": rd["posts"], "reddit_authors": rd["authors"], "tweet_rate": tw_rate})
att_df["A"] = (z(np.log(att_df["reddit_posts"])) + z(np.log(att_df["tweet_rate"]))) / 2
absdp = cal_dp["poly_Democrats_Sweep"].abs().rename("abs_dP_poly")
sw_cal = pm.sweep_daily(pd.date_range("2026-06-30", "2026-10-07"))
absdpk = P.dp_series(sw_cal, "kal_p").reindex(CAL).abs().rename("abs_dP_kal")
assert np.allclose(cal_dp["poly_Democrats_Sweep"].dropna(), P.dp_series(sw_cal, "poly_p").reindex(CAL).dropna())
A = att_df["A"]
famX3 = P.family_a(A, absdp, ex, "attention A vs |dP| Polymarket", names=("A", "absdP"))
famX3.attrs["lead_name"] = "A"
famX3.to_csv(TABLES / "x3_attention_vs_absdp.csv", index=False)
famX3k = P.family_a(A, absdpk, ex, "attention A vs |dP| Kalshi", perm=False, names=("A", "absdP"))
keep = CAL[CAL < "2026-10-06"]
# added check (not in PREREG_CLASS): A narrowly fails the ADF part of the stationarity rule, so its
# change is run too, as PREREG's stationarity_rule does for C
famX3d = P.family_a(A.diff(), absdp, ex, "change in attention vs |dP| Polymarket", names=("dA", "absdP"))
famX3d.attrs["lead_name"] = "dA"
famX3d.to_csv(TABLES / "x3_dattention_vs_absdp.csv", index=False)
x3 = {"stationarity_A": P.stationarity(A), "stationarity_dA": P.stationarity(A.diff()),
      "family_poly_dA": summarise(famX3d), "claims_dA": claim_checks(famX3d, A.diff(), absdp, ex),
      "family_poly": summarise(famX3), "claims": claim_checks(famX3, A, absdp, ex),
      "family_kalshi": summarise(famX3k),
      "drop_oct6_7_same_day": P.hac_reg(absdp.loc[keep], A.loc[keep], ex.loc[keep]),
      "component_same_day_r": {c: float(np.log(att_df[c]).corr(absdp)) for c in ("reddit_posts", "reddit_authors", "tweet_rate")},
      "components_corr": float(np.log(att_df["reddit_posts"]).corr(np.log(att_df["tweet_rate"]))),
      "levels": {"reddit_posts_median": float(att_df["reddit_posts"].median()), "tweet_rate_median": float(att_df["tweet_rate"].median()),
                 "pages_used": int(len(pg))},
      "A_event_vs_pre": {"A_mean_Sep10_26": float(A.loc["2026-09-10":"2026-09-26"].mean()),
                         "A_mean_Jul1_Sep9": float(A.loc[:"2026-09-09"].mean())}}
R["X3"] = x3
print("\nX3:", json.dumps({k: v for k, v in x3.items() if k != "claims"}, indent=0, default=str)[:1500])
print(famX3[["test", "direction", "lag", "N", "pearson_r", "coef", "p_hac", "p_perm", "q_bh"]].round(4).to_string(index=False))

# ================================================================ X4 Kalshi volume
def kalshi_bar_volume(tk: str) -> pd.DataFrame:
    cs = json.loads((RAW / "predmkt" / f"kalshi_KXBALANCEPOWERCOMBO-27FEB-{tk}_hourly.json").read_text())
    df = pd.DataFrame({"end": pd.to_datetime([c["end_period_ts"] for c in cs], unit="s", utc=True),
                       "vol": [float(c.get("volume_fp") or 0) for c in cs],
                       "oi": [float(c.get("open_interest_fp") or np.nan) for c in cs]})
    df["bar"] = bar_date(df["end"] - pd.Timedelta(minutes=1))     # candle ending 16:00 ET closes bar t
    return df


kv = {tk: kalshi_bar_volume(tk) for tk in ("DD", "DR", "RD", "RR")}
V = kv["DD"].groupby("bar")["vol"].sum().reindex(CAL).fillna(0.0)
Vev = sum(kv[tk].groupby("bar")["vol"].sum().reindex(CAL).fillna(0.0) for tk in kv)
oi = kv["DD"].groupby("bar")["oi"].last().reindex(CAL)
lv, lvev = np.log1p(V).rename("logV"), np.log1p(Vev).rename("logVev")
famX4 = P.family_a(A, lv, ex, "attention A vs log Kalshi DD volume", names=("A", "logV"))
famX4.attrs["lead_name"] = "A"
famX4.to_csv(TABLES / "x4_attention_vs_volume.csv", index=False)
x4 = {"vol_vs_absdp_kal": {**P.hac_reg(absdpk, lv, ex), "r": float(lv.corr(absdpk))},
      "vol_vs_absdp_poly": {**P.hac_reg(absdp, lv, ex), "r": float(lv.corr(absdp))},
      "event_vol_vs_absdp_kal": {**P.hac_reg(absdpk, lvev, ex), "r": float(lvev.corr(absdpk))},
      "family_A_vs_volume": summarise(famX4), "claims_A_vs_volume": claim_checks(famX4, A, lv, ex),
      "volume": {"mean_Jul1_Sep9": float(V.loc[:"2026-09-09"].mean()), "mean_Sep10_26": float(V.loc["2026-09-10":"2026-09-26"].mean()),
                 "mean_Sep27_Oct7": float(V.loc["2026-09-27":].mean()), "median_window": float(V.median()),
                 "zero_volume_days": int((V == 0).sum()), "total_window": float(V.sum())},
      "open_interest": {d_: float(oi.loc[d_]) for d_ in ("2026-07-01", "2026-09-09", "2026-09-26", "2026-10-07")}}
x4["volume"]["ratio_event_vs_before"] = x4["volume"]["mean_Sep10_26"] / x4["volume"]["mean_Jul1_Sep9"]
_b = pd.concat([lv, absdpk], axis=1).dropna()
x4["vol_vs_absdp_kal"]["p_perm"] = P._perm_corr(_b["logV"], _b["abs_dP_kal"])
R["X4"] = x4
pd.DataFrame({"A": A, "abs_dP_poly": absdp, "abs_dP_kal": absdpk, "kalshi_DD_volume": V, "kalshi_event_volume": Vev,
              "kalshi_DD_open_interest": oi, **att_df.drop(columns="A")}).to_csv(TABLES / "x3_x4_daily.csv")
print("\nX4:", json.dumps({k: v for k, v in x4.items() if not k.startswith("claims")}, indent=0, default=str)[:1500])

# ================================================================ X5 placebos
strong = [t for t in names if GROUPS[NAME_GROUP[t]]["strength"] == "strong"]
n_long = sum(CANDIDATES[t] > 0 for t in strong)
dp_td = P.dp_series(pm.sweep_daily(days[days >= "2026-06-29"]), "poly_p").reindex(td)
dp_pre = td_dp["poly_Democrats_Sweep"].reindex(pre)
A_te, A_pr = AR[strong].to_numpy(), ar_pre[strong].reindex(pre).to_numpy()
S_te = C_td.reindex(td).to_numpy()


def stats_for(long_mask: np.ndarray) -> tuple:
    ls_te = np.nanmean(A_te[:, long_mask], axis=1) - np.nanmean(A_te[:, ~long_mask], axis=1)
    ls_pr = np.nanmean(A_pr[:, long_mask], axis=1) - np.nanmean(A_pr[:, ~long_mask], axis=1)
    m1 = ~np.isnan(dp_td.to_numpy()) & ~np.isnan(ls_te)
    m2 = ~np.isnan(S_te) & ~np.isnan(ls_te)
    m3 = ~np.isnan(dp_pre.to_numpy()) & ~np.isnan(ls_pr)
    r_dp = np.corrcoef(ls_te[m1], dp_td.to_numpy()[m1])[0, 1]
    r_c = np.corrcoef(ls_te[m2], S_te[m2])[0, 1]
    x, y = dp_pre.to_numpy()[m3], ls_pr[m3]
    gam = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return r_dp, r_c, gam


real_mask = np.array([CANDIDATES[t] > 0 for t in strong])
real = stats_for(real_mask)
rng = np.random.default_rng(RANDOM_SEED)
draws = []
for _ in range(10_000):
    m = np.zeros(len(strong), bool); m[rng.choice(len(strong), n_long, replace=False)] = True
    draws.append(stats_for(m))
draws = np.array(draws)
pct = {nm_: float((draws[:, i] <= real[i]).mean()) for i, nm_ in enumerate(("r_LS_dP_test", "r_LS_C_test", "gamma_pre"))}
pd.DataFrame(draws, columns=["r_LS_dP_test", "r_LS_C_test", "gamma_pre"]).describe(percentiles=[.05, .5, .9, .95]).to_csv(TABLES / "x5_random_sign_placebo.csv")
# (b) placebo text: Indian Congress-BJP topic (topic 1, outlier-reduced) as a pseudo-issue
dt = pd.read_parquet(INTERIM / "doc_topics.parquet")[["doc_id", "topic_ro"]]
dpl = d[d["voice"] == "voter"].merge(dt, on="doc_id")
dpl = dpl[dpl["topic_ro"] == 1].assign(issue="India placebo")
Ipl, npl = index.issue_index(dpl, "s_primary", "bar_date", ["India placebo"], CAL)
famPl = P.family_a(Ipl["India placebo"], cal_dp["poly_Democrats_Sweep"], ex, "placebo: Indian-politics index vs dP")
famPl.attrs["lead_name"] = "sent"
famPl.to_csv(TABLES / "x5_placebo_text.csv", index=False)
R["X5"] = {"random_sign": {"real": dict(zip(("r_LS_dP_test", "r_LS_C_test", "gamma_pre"), map(float, real))),
                           "percentile_of_real": pct, "n_draws": 10_000, "n_long": int(n_long), "n_names": len(strong),
                           "draw_p90": dict(zip(("r_LS_dP_test", "r_LS_C_test", "gamma_pre"), map(float, np.percentile(draws, 90, axis=0))))},
           "placebo_text": {"docs": int(len(dpl)), "coverage_ge5": float((npl["India placebo"] >= 5).mean()),
                            "family": summarise(famPl), "claims": claim_checks(famPl, Ipl["India placebo"], cal_dp["poly_Democrats_Sweep"], ex)}}
print("\nX5 random-sign:", R["X5"]["random_sign"])
print("X5 placebo text:", R["X5"]["placebo_text"]["family"], "docs", len(dpl))

# ---------------------------------------------------------------- notebook figure B (not in the report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True)
ax[0].plot(A.index, A, color="#2a78d6", lw=1.2); ax[0].axhline(0, color="#898781", lw=0.6)
ax[0].set_ylabel("attention A\n(z-score)"); ax[0].set_title("(a) Attention: Reddit posts and tweet arrival rate", loc="left")
ax[1].bar(absdp.index, absdp, color="#eb6834", width=0.8); ax[1].set_ylabel("|dP| (pp)")
ax[1].set_title("(b) Size of the daily change in P(D sweep), Polymarket 16:00 ET", loc="left")
ax[2].bar(V.index, V / 1000, color="#1baf7a", width=0.8); ax[2].set_ylabel("contracts (k)")
ax[2].set_title("(c) Kalshi Democrats-sweep volume per 16:00-to-16:00 day", loc="left")
for a_ in ax:
    a_.axvspan(pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-26"), color="#f1efe8", lw=0, zorder=0)
ax[2].xaxis.set_major_locator(mdates.MonthLocator()); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.suptitle("Notebook Figure B. Attention, odds moves and trading volume (Jul 1 to Oct 7, 2026)", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "figB_attention_volume.png", dpi=200); plt.close(fig)

# ================================================================ checks ADDED after the independent review
# (Oct 10, ~19:50 ET). None of these was in PREREG_CLASS; they are labelled "added" wherever reported.
ADD = {}
# (a) X1: the one-variable House slope without days on which a component price is stale (> 0.5 h)
st = pd.DataFrame({nm: pm.snapshot(pm.poly_hourly(nm), alld)["stale_h"]
                   for nm in ("Democrats_Sweep", "R_Senate_D_House", "D_Senate_R_House")})
bad = st.reindex(pre).gt(0.5).any(axis=1)
bad_days = set(bad[bad].index) | {pre[pre.get_loc(d_) + 1] for d_ in bad[bad].index if pre.get_loc(d_) + 1 < len(pre)}
pre_ok = pre[~pre.isin(list(bad_days))]
ADD["x1_house_alone_drop_stale"] = {"n_dropped": int(len(pre) - len(pre_ok)), "n_stale_days": int(bad.sum()),
                                    **B.hac_slope(LS_pre.reindex(pre_ok), td_dp["poly_house"].reindex(pre_ok))}
# log decomposition: P(sweep) = P(House) x P(Senate | House), so dlog P(sweep) = dlog P(House) + dlog P(Senate | House)
def _dlog(c, a, b):
    return float(np.log(sc[c].loc[b] / sc[c].loc[a]))
ADD["x1_house_share_of_sweep_logrise"] = {
    "Jul1_Oct7": _dlog("poly_house", "2026-07-01", "2026-10-07") / _dlog("poly_Democrats_Sweep", "2026-07-01", "2026-10-07"),
    "Sep9_Sep26": _dlog("poly_house", "2026-09-09", "2026-09-26") / _dlog("poly_Democrats_Sweep", "2026-09-09", "2026-09-26")}
# P(D wins >= 1 chamber), with the 5 prices normalised to sum to 1
one = 1 - (sc["poly_Republicans_Sweep"] + sc["poly_Other"]) / sc["poly_sum5"]
ADD["p_at_least_one_chamber_norm"] = {"Jul1": float(100 * one.loc["2026-07-01"]), "Oct7": float(100 * one.loc["2026-10-07"]),
                                      "window_max": float(100 * one.reindex(CAL).max()), "window_min": float(100 * one.reindex(CAL).min())}
# (b) X3: the two attention components separately against |dP| (HAC with day-of-week dummies, permutation)
for c in ("tweet_rate", "reddit_posts"):
    x_ = np.log(att_df[c]).rename(c)
    b_ = pd.concat([x_, absdp], axis=1).dropna()
    ADD[f"x3_{c}_vs_absdp_poly"] = {**P.hac_reg(absdp, x_, ex), "r": float(b_[c].corr(b_["abs_dP_poly"])),
                                    "p_perm": P._perm_corr(b_[c], b_["abs_dP_poly"])}
ADD["x3_A_vs_absdp_kal_same_day"] = {"r": float(famX3k.loc[famX3k["test"] == "same-day", "pearson_r"].iloc[0]),
                                     "p_hac": float(famX3k.loc[famX3k["test"] == "same-day", "p_hac"].iloc[0])}
ADD["x3_tweet_rate_weekly_first_last"] = [float(att_df["tweet_rate"].iloc[:7].mean()), float(att_df["tweet_rate"].iloc[-7:].mean())]
# (c) X4: volume vs |dP| without the Sept 10-26 episode, and the Polymarket permutation p
noep = CAL[(CAL < "2026-09-10") | (CAL > "2026-09-26")]
ADD["x4_vol_vs_absdp_no_episode"] = {"kal": {**P.hac_reg(absdpk.loc[noep], lv.loc[noep], ex.loc[noep]), "r": float(lv.loc[noep].corr(absdpk.loc[noep]))},
                                     "poly": {**P.hac_reg(absdp.loc[noep], lv.loc[noep], ex.loc[noep]), "r": float(lv.loc[noep].corr(absdp.loc[noep]))}}
_bp = pd.concat([lv, absdp], axis=1).dropna()
ADD["x4_vol_vs_absdp_poly_p_perm"] = P._perm_corr(_bp["logV"], _bp["abs_dP_poly"])
ADD["x4_kal_mid_zero_change_share"] = float((absdpk.dropna() < 1e-9).mean())
# (d) X5: whole-GROUP sign flips (keeps the 6-group structure): every assignment with >= 1 long and >= 1 short group
import itertools
sg = [g for g, v in GROUPS.items() if v["strength"] == "strong"]
gstats = []
for signs in itertools.product((1, -1), repeat=len(sg)):
    if all(s_ == 1 for s_ in signs) or all(s_ == -1 for s_ in signs):
        continue
    m = np.array([dict(zip(sg, signs))[NAME_GROUP[t]] > 0 for t in strong])
    gstats.append(stats_for(m))
gstats = np.array(gstats)
ADD["x5_group_flips"] = {"n_assignments": int(len(gstats)),
                         **{f"pct_{k}": float((gstats[:, i] <= real[i]).mean()) for i, k in enumerate(("r_LS_dP_test", "r_LS_C_test", "gamma_pre"))},
                         "n_beating_real_gamma": int((gstats[:, 2] > real[2] + 1e-12).sum())}
ADD["x5_group_gamma_pre"] = {g: B.hac_slope(GROUPS[g]["sign"] * ar_pre[GROUPS[g]["names"]].reindex(pre).mean(axis=1), dp_pre)["coef"] for g in sg}
ADD["x5_real_LS_gamma_pre_p"] = B.hac_slope(LS_pre.reindex(pre), dp_pre)["p"]
ADD["x2_wrong_sign_hits"] = int(((x2f["p_hac"] < 0.10) & (np.sign(x2f["coef"]) != SIGN)).sum())
R["added_checks"] = ADD
print("\nADDED CHECKS:", json.dumps(ADD, indent=0, default=str)[:3000])

(OUTPUTS / "results_class.json").write_text(json.dumps(R, indent=1, default=str))
print("\nwrote outputs/results_class.json")
