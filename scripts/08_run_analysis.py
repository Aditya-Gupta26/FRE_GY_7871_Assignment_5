"""Every table and number in the report: Q1 to Q4 and the extra questions.

  python scripts/08_run_analysis.py            # writes outputs/tables/*.csv and outputs/results.json
Figures are made by scripts/09_make_figures.py from the saved tables.
"""
import _bootstrap  # noqa: F401
import json
import warnings

import numpy as np
import pandas as pd

from src import index, market, pipeline as P, pm, q1
from src.config import (COLLECT_START, GROUPS, INTERIM, ISSUE_VOCAB, LABELS, OUTPUTS, POLLS, PREREG,
                        RANDOM_SEED, TABLES, WINDOW_END)

warnings.filterwarnings("ignore")
TABLES.mkdir(parents=True, exist_ok=True)
R = {}
CAL = pd.date_range("2026-07-01", "2026-10-07")          # 99 calendar-day bars

# ====================================================================== Q1
di = q1.doc_issues()
vo = di[(di["voice"] == "voter") & di["day"].isin(CAL)]
me = di[(di["voice"] == "shared_media") & di["day"].isin(CAL)]
nw = di[(di["src"] == "news") & di["day"].isin(CAL)]
sal = pd.DataFrame({
    "voter": q1.shares(vo), "voter_eng": q1.shares(vo, weight="engagement"), "voter_ro": q1.shares(vo, "issue_ro"),
    "voter_no_lowconf": q1.shares(vo[vo["conf"] != "low"]), "voter_twitter": q1.shares(vo[vo["src"] == "twitter"]),
    "voter_reddit": q1.shares(vo[vo["src"] == "reddit"]), "shared_media": q1.shares(me), "news": q1.shares(nw),
    "cov5_ro": q1.coverage(vo.assign(issue=vo["issue_ro"]), CAL).values})
for g in ("US_LEFT", "US_RIGHT", "US_CENTER", "BUSINESS", "INTL"):
    sal[f"news_{g}"] = q1.shares(nw[nw["group"] == g])
sal = sal.sort_values("voter", ascending=False)
sal.to_csv(TABLES / "q1_issue_shares.csv")
keys = list(sal.index[(sal["voter"] >= 0.05)][:8])
w_k = sal.loc[keys, "voter"] / sal.loc[keys, "voter"].sum()
pc = q1.poll_compare(sal["voter"], sal["news"])
pc.to_csv(TABLES / "q1_poll_compare.csv", index=False)
# topic-assignment accuracy against the blind issue labels (validation items, social + news)
lab = pd.read_csv(LABELS / "validation_issue_labels.csv").rename(columns={"issue": "label"})
vk = pd.read_parquet(LABELS / "validation_key.parquet").merge(lab, on="item_id")
vk = vk.merge(di[["doc_id", "issue", "issue_ro"]], on="doc_id")
assert len(vk) == 150
nonout = vk["issue"] != "Outlier"
acc = {"N": int(len(vk)), "strict_agree_all": float((vk["label"] == vk["issue"]).mean()),
       "n_non_outlier": int(nonout.sum()),
       "strict_agree_non_outlier": float((vk.loc[nonout, "label"] == vk.loc[nonout, "issue"]).mean()),
       "ro_agree_all": float((vk["label"] == vk["issue_ro"]).mean()),
       "label_is_issue_share": float(vk["label"].isin(ISSUE_VOCAB).mean())}
# seed stability on what matters: map each seed's topics to the main fit's topics by centroid
# cosine, then to issues, and compare voter issue shares and the key-topic set (30k subsample)
from src import topics as T
Eall = np.vstack([np.load(INTERIM / "emb_social.npy"), np.load(INTERIM / "emb_news.npy")])
dt_all = pd.read_parquet(INTERIM / "doc_topics.parquet")
tax = pd.read_csv(LABELS / "topic_taxonomy.csv"); lab_of = dict(zip(tax["topic"], tax["label"]))
pos = np.load(INTERIM / "seed_positions.npy")
seed_rows = []
for sd in (1, 2, 3):
    lab = np.load(INTERIM / f"seed_labels_{sd}.npy")
    mp = T.match_topics(Eall[pos], dt_all["topic"].to_numpy()[pos], lab)
    iss = pd.Series([lab_of.get(mp.get(t, -1), "Outlier") if t != -1 else "Outlier" for t in lab], index=pos)
    sub = dt_all.iloc[pos].assign(issue=iss.values)
    shs = q1.shares(sub[sub["voice"] == "voter"])
    seed_rows.append(shs.rename(f"seed{sd}"))
main_sub = dt_all.iloc[pos].assign(issue=dt_all["topic"].iloc[pos].map(lab_of).fillna("Outlier").values)
seed_tab = pd.concat([q1.shares(main_sub[main_sub["voice"] == "voter"]).rename("main_fit")] + seed_rows, axis=1)
seed_tab.to_csv(TABLES / "q1_seed_stability.csv")
seed_keys = {c: list(seed_tab[c][seed_tab[c] >= 0.05].sort_values(ascending=False).index[:8]) for c in seed_tab}
seed_overlap = {c: len(set(v) & set(keys)) for c, v in seed_keys.items()}
seed_corr = seed_tab.corr(method="spearman").loc["main_fit"].round(3).to_dict()
daily = vo[vo["issue"].isin(ISSUE_VOCAB)].groupby(["day", "issue"]).size().unstack("issue").reindex(CAL).fillna(0)
(daily.div(daily.sum(axis=1), axis=0)).to_csv(TABLES / "q1_daily_issue_shares.csv")
diag = json.loads((TABLES / "q1_topic_diagnostics.json").read_text()) if (TABLES / "q1_topic_diagnostics.json").exists() else {}
R["q1"] = {"bertopic": json.loads((TABLES / "q1_bertopic_summary.json").read_text()), "diagnostics": diag,
           "n_docs": {"voter": len(vo), "voter_issue": int(vo["issue"].isin(ISSUE_VOCAB).sum()),
                      "shared_media": len(me), "news": len(nw), "news_issue": int(nw["issue"].isin(ISSUE_VOCAB).sum())},
           "key_topics": keys, "weights": w_k.round(4).to_dict(), "topic_assignment_accuracy": acc,
           "spearman": pc.groupby("poll")[["spearman_voter", "spearman_news", "n_categories"]].first().round(3).to_dict(orient="index"),
           "india_flagged_docs": int(di["india"].sum()),
           "seed_key_overlap_of_8": seed_overlap, "seed_share_spearman": seed_corr, "seed_keys": seed_keys}
print("Q1 key topics:", keys)
print((sal[["voter", "voter_ro", "voter_eng", "news"]] * 100).round(1).head(12).to_string())
print("assignment accuracy:", acc)

# ====================================================================== Q2
d = P.doc_scores()
d = d[d["bar_date"].isin(CAL)]
dv = d[d["voice"] == "voter"].assign(issue=lambda x: x["issue_ro"])          # deviation: outlier-reduced
I, n_kt = index.issue_index(dv, "s_primary", "bar_date", keys, CAL)
C = index.composite(I, w_k)
I.to_csv(TABLES / "q2_issue_index_daily.csv"); n_kt.to_csv(TABLES / "q2_issue_counts_daily.csv")
cov = (n_kt >= 5).mean()
alt = {}
alt["C_equal_w"] = index.composite(I, pd.Series(1.0, index=keys))
pooled = dv[dv["issue"].isin(keys)].groupby("bar_date")["s_primary"].mean().reindex(CAL)
alt["C_pooled"] = pooled - pooled.mean()
I_all, _ = index.issue_index(d.assign(issue=d["issue_ro"]), "s_primary", "bar_date", keys, CAL)
alt["C_all_social"] = index.composite(I_all, w_k)
I_rob, _ = index.issue_index(dv, "s_rob_tgt", "bar_date", keys, CAL)
alt["C_rob_tgt_only"] = index.composite(I_rob, w_k)
I_tone, _ = index.issue_index(dv, "s_tone", "bar_date", keys, CAL)
alt["C_tone_nondirectional"] = index.composite(I_tone, w_k)
# expanding-window salience weights (shares up to t-1, no look-ahead in the weights)
cum = vo[vo["issue"].isin(keys)].groupby(["day", "issue"]).size().unstack("issue").reindex(CAL).reindex(columns=keys).fillna(0).cumsum().shift(1)
wexp = cum.div(cum.sum(axis=1), axis=0)
alt["C_expanding_w"] = ((I * wexp).sum(axis=1, min_count=1) / (I.notna() * wexp).sum(axis=1).replace(0, np.nan))
import src.index as _ix
_old = _ix.MIN_DOCS_ISSUE_DAY; _ix.MIN_DOCS_ISSUE_DAY = 20
I20, _ = index.issue_index(dv, "s_primary", "bar_date", keys, CAL); _ix.MIN_DOCS_ISSUE_DAY = _old
alt["C_min20_prereg"] = index.composite(I20, w_k)
idx_df = pd.concat([C.rename("C")] + [v.rename(k) for k, v in alt.items()], axis=1)
idx_df.to_csv(TABLES / "q2_composite_daily.csv")
shs = pd.DataFrame([index.split_half(dv, "s_primary", "bar_date", keys, CAL, w_k, RANDOM_SEED + i) for i in range(50)])
sh = {"r_level_mean": float(shs["r_level"].mean()), "r_level_sd": float(shs["r_level"].std()),
      "r_level_min": float(shs["r_level"].min()), "r_level_max": float(shs["r_level"].max()),
      "r_change_mean": float(shs["r_change"].mean()), "sb_level_of_mean": float(2 * shs["r_level"].mean() / (1 + shs["r_level"].mean())),
      "n_seeds": 50}
R["q2"] = {"docs_in_index": int(dv["issue"].isin(keys).sum()), "docs_per_day_median": float(n_kt.sum(axis=1).median()),
           "coverage_ge5": cov.round(3).to_dict(), "tau_nonzero_share_reddit_voter": float((dv.loc[dv["platform"] == "reddit", "tau"] != 0).mean()),
           "split_half": sh, "C_sd": float(C.std()), "C_mean_level_first_half": float(C.iloc[:49].mean()),
           "C_mean_level_second_half": float(C.iloc[49:].mean()),
           "corr_C_with_alternatives": idx_df.corr().loc["C"].round(3).to_dict(),
           "validation": pd.read_csv(TABLES / "q2_validation.csv").round(3).to_dict(orient="records"),
           "validation_summary": json.loads((TABLES / "q2_validation_summary.json").read_text())}
print("\nQ2 coverage>=5:", cov.round(2).to_dict()); print("split-half:", sh)

# ====================================================================== Q3
sw = pm.sweep_daily(pd.date_range("2026-06-30", "2026-10-07"))
dp = P.dp_series(sw, "poly_p").reindex(CAL)
dpk = P.dp_series(sw, "kal_p").reindex(CAL)
dpa = P.dp_series(sw, "avg_p").reindex(CAL)
dlg = P.dp_series(sw, "poly_p", "dlogit").reindex(CAL)
# timing assert: the price that STARTS dP_t is stamped at/after the end of every bar t-1 doc,
# i.e. docs in bar t-1 are all before 16:00 ET of t-1 and that day's snapshot is at >= 15:00 ET
snap_ts = pd.to_datetime(sw["poly_price_ts"]).dt.tz_convert("America/New_York")
assert (snap_ts.dt.strftime("%H:%M") <= "16:05").all()
last_doc = d.groupby("bar_date")["ts"].max().dt.tz_convert("America/New_York")
snap_by_day = snap_ts.copy(); snap_by_day.index = pd.DatetimeIndex(snap_by_day.index)
chk = pd.concat([last_doc.rename("last_doc"), snap_by_day.rename("snap")], axis=1).dropna()
assert (chk["last_doc"] < chk["snap"]).all(), "a doc in bar t is not earlier than the bar-t price snapshot"
st = {"C": P.stationarity(C), "dC": P.stationarity(C.diff()), "p_level": P.stationarity(sw["poly_p"].reindex(CAL)),
      "dP": P.stationarity(dp)}
primary_is_level = st["C"]["stationary"]
S = C if primary_is_level else C.diff()
ex = P.dow_dummies(CAL)
famA = P.family_a(S, dp, ex, "primary: " + ("C" if primary_is_level else "dC") + " vs dP Polymarket")
famA.to_csv(TABLES / "q3_family_a.csv", index=False)
# robustness grid (no permutation, to keep run time reasonable; counts vs chance reported)
grid = []
specs = {"Kalshi mid": (S, dpk), "venue average": (S, dpa), "dlogit Polymarket": (S, dlg),
         "secondary dC" if primary_is_level else "secondary C": ((C.diff() if primary_is_level else C), dp)}
for k, v in alt.items():
    specs[k] = ((v if primary_is_level else v.diff()), dp)
for lab, (s_, d_) in specs.items():
    f = P.family_a(s_, d_, ex, lab, perm=False)
    grid.append(f)
for h in (3, 5):
    dph = P.dp_series(sw, "poly_p", h=h).reindex(CAL)
    s_h = (C.rolling(h).mean() if primary_is_level else C.diff(h))
    r = P.hac_reg(dph, s_h, None, lags=h - 1 if h > 1 else 1)
    grid.append(pd.DataFrame([{"spec": f"{h}-day overlapping changes", "test": "same-day", "direction": "sent ~ dP",
                               "lag": 0, "N": r["N"], "pearson_r": float(s_h.corr(dph)), "coef": r["coef"], "p_hac": r["p_hac"]}]))
gridf = pd.concat(grid, ignore_index=True)
gridf.to_csv(TABLES / "q3_robustness_grid.csv", index=False)
# claim checks on family A
claims = []
for _, row in famA.iterrows():
    sign_ok = np.sign(row["coef"]) == PREREG["expected_sign_index_vs_dp"] if row["direction"] != "dP -> sent" else True
    passed = bool(row["q_bh"] < 0.10 and row["p_perm"] < 0.10 and sign_ok)
    rec = {"test": row["test"], "direction": row["direction"], "lag": int(row["lag"]), "pre_checks_pass": passed}
    if passed:
        rec["lodo"] = P.leave_one_day_out(S, dp, ex, row["test"], int(row["lag"]), row["direction"])
        s2, d2 = P.drop_episode(S), P.drop_episode(dp)
        ff = P.family_a(s2, d2, ex.reindex(s2.index), "drop Sept 10-26", perm=False)
        m = ff[(ff["test"] == row["test"]) & (ff["lag"] == row["lag"]) & (ff["direction"] == row["direction"])].iloc[0]
        rec["drop_episode_p_hac"] = float(m["p_hac"]); rec["drop_episode_coef"] = float(m["coef"])
    claims.append(rec)
xc = P.cross_corr(S, dp); xc.to_csv(TABLES / "q3_cross_corr.csv", index=False)
var = None
try:
    from src.granger import var_analysis
    vd = pd.concat([S.rename("sent"), dp.rename("dP")], axis=1).dropna()
    v1 = var_analysis(vd, "sent", "dP", maxlags=5, fixed_p=3)
    v2 = var_analysis(vd[["dP", "sent"]], "dP", "sent", maxlags=5, fixed_p=3)
    var = {"irf_dP_to_sent_shock": [float(x) for x in v1["irf"]], "cum_irf_dP": v1["cum_irf"],
           "irf_lo": None if v1["irf_lo"] is None else [float(x) for x in v1["irf_lo"]],
           "irf_hi": None if v1["irf_hi"] is None else [float(x) for x in v1["irf_hi"]],
           "irf_sent_to_dP_shock_reverse_order": [float(x) for x in v2["irf"]], "lag_aic": v1["lag_aic"], "lag_bic": v1["lag_bic"]}
except Exception as e:  # noqa: BLE001
    var = {"error": str(e)}
venue_rel = float(dp.corr(dpk))
R["q3"] = {"N": int(pd.concat([S, dp], axis=1).dropna().shape[0]), "primary_is_level": bool(primary_is_level),
           "stationarity": st, "family_a": famA.round(4).to_dict(orient="records"), "claims": claims,
           "robust_count_p_lt_0.10": int((gridf["p_hac"] < 0.10).sum()), "robust_tests": int(len(gridf)),
           "mde_r_N": P.mde_r(int(pd.concat([S, dp], axis=1).dropna().shape[0])), "venue_daily_corr": venue_rel,
           "venue_gap_pp": {"min": float(((sw["poly_p"] - sw["kal_p"]) * 100).reindex(CAL).min()),
                            "max": float(((sw["poly_p"] - sw["kal_p"]) * 100).reindex(CAL).max()),
                            "mean": float(((sw["poly_p"] - sw["kal_p"]) * 100).reindex(CAL).mean())},
           "p_path": {"start": float(sw["poly_p"].loc["2026-07-01"]), "end": float(sw["poly_p"].loc["2026-10-07"]),
                      "max": float(sw["poly_p"].reindex(CAL).max()), "max_date": str(sw["poly_p"].reindex(CAL).idxmax().date())},
           "var": var, "cross_corr": xc.round(3).to_dict(orient="records")}
sw.to_csv(TABLES / "q3_sweep_daily.csv")
print("\nQ3 stationarity:", {k: (round(v["ADF_p"], 3), round(v["KPSS_p"], 3), v["stationary"]) for k, v in st.items()})
print(famA[["test", "direction", "lag", "N", "pearson_r", "coef", "p_hac", "p_perm", "q_bh"]].round(4).to_string(index=False))
print("claims:", claims)
print("robust p<0.10:", R["q3"]["robust_count_p_lt_0.10"], "of", R["q3"]["robust_tests"], "| MDE r:", round(R["q3"]["mde_r_N"], 3))


# ====================================================================== Q4
from src import basket as B
from src.config import BETA_END, BETA_START, CANDIDATES, NAME_GROUP
px = market.load_close()
ret = market.log_ret(px)
days = px.index
pre = days[(days >= BETA_START) & (days <= BETA_END)]
td = days[(days >= "2026-07-01") & (days <= "2026-10-07")]
names = list(CANDIDATES)
mm, _ = B.abnormal(ret, names, "SPY", pre, use_alpha=True)
AR = pd.DataFrame({t: 100 * (ret[t] - mm.loc[t, "beta"] * ret["SPY"]) for t in names}).loc[td]   # alpha = 0
fac = {}
for t in names:
    f = GROUPS[NAME_GROUP[t]]["factor"]
    if f:
        bf = B.abnormal(ret, [t], f, pre, use_alpha=True)[0].loc[t, "beta"]
        fac[t] = 100 * (ret[t] - bf * ret[f])
    else:
        fac[t] = AR[t] if t in AR else np.nan
ARf = pd.DataFrame(fac).loc[td]
pre_eb = pd.read_csv(TABLES / "q4_election_betas_main_poly.csv", index_col=0)
gate = [t for t in names if pre_eb.loc[t, "sign_match"]]
legs = {"strong": B.legs(AR, "strong"), "weak": B.legs(AR, "weak"), "strong_noOSCR": B.legs(AR, "strong", drop=("OSCR",)),
        "strong_factor": B.legs(ARf, "strong"), "weak_factor": B.legs(ARf, "weak"),
        "gate": B.legs(AR[gate], ("strong", "weak"))}
swt = pm.sweep_daily(days[days >= "2026-06-29"])
dpt = {c: P.dp_series(swt, c).reindex(td) for c in ("poly_p", "kal_p", "avg_p")}
val_rows = []
for lk, L in legs.items():
    for leg in ("ls", "long", "short"):
        for c, dpc in dpt.items():
            r = P.hac_reg(L[leg], dpc)
            val_rows.append({"basket": lk, "leg": leg, "venue": c, **r, "pearson_r": float(L[leg].corr(dpc))})
val = pd.DataFrame(val_rows); val.to_csv(TABLES / "q4_validation_vs_dp.csv", index=False)
LS = legs["strong"]["ls"]
cum = pd.DataFrame({k: L["ls"].cumsum() for k, L in legs.items()})
cum["long"], cum["short"] = legs["strong"]["long"].cumsum(), legs["strong"]["short"].cumsum()
cum.to_csv(TABLES / "q4_cum_ar.csv")
# trading-day sentiment bars: weekends and holidays pool into the next session
dv_td = dv.assign(td_bar=index.to_trading_bar(dv["bar_date"], days))
I_td, n_td = index.issue_index(dv_td, "s_primary", "td_bar", keys, td)
C_td = index.composite(I_td, w_k)
S_td = C_td if primary_is_level else C_td.diff()
ex_td = P.dow_dummies(td)
famB = P.family_a(S_td, LS, ex_td, "sentiment vs strong L/S AR (trading days)", names=("sent", "LS"))
famB.to_csv(TABLES / "q4_family_sent_vs_basket.csv", index=False)
famV = P.family_a(dpt["poly_p"], LS, ex_td, "dP vs strong L/S AR (trading days)", names=("dP", "LS"))
famV.to_csv(TABLES / "q4_family_dp_vs_basket.csv", index=False)
grid4 = []
for lk in ("weak", "weak_factor", "strong_noOSCR", "strong_factor", "gate"):
    grid4.append(P.family_a(S_td, legs[lk]["ls"], ex_td, f"sentiment vs {lk} L/S", perm=False, names=("sent", "LS")))
for leg in ("long", "short"):
    grid4.append(P.family_a(S_td, legs["strong"][leg], ex_td, f"sentiment vs strong {leg} leg", perm=False, names=("sent", leg)))
grid4 = pd.concat(grid4, ignore_index=True); grid4.to_csv(TABLES / "q4_robustness_grid.csv", index=False)


def _grid4_hits(k):
    roll = lambda x: pd.Series(np.roll(x.dropna().to_numpy(), k), index=x.dropna().index).reindex(x.index)
    St = roll(S_td); hits = 0
    for lk in ("weak", "weak_factor", "strong_noOSCR", "strong_factor", "gate"):
        hits += int((P.family_a(St, legs[lk]["ls"], ex_td, lk, perm=False)["p_hac"] < 0.10).sum())
    for leg in ("long", "short"):
        hits += int((P.family_a(St, legs["strong"][leg], ex_td, leg, perm=False)["p_hac"] < 0.10).sum())
    return hits


null4 = np.array([_grid4_hits(k) for k in range(7, len(td) - 7)])
obs4 = int((grid4["p_hac"] < 0.10).sum())
grid4_null = {"observed": obs4, "null_mean": float(null4.mean()), "null_p90": float(np.percentile(null4, 90)),
              "n_shifts": int(len(null4)), "p_value": float((1 + (null4 >= obs4).sum()) / (1 + len(null4)))}
print("Q4 grid joint null:", grid4_null)
highH = pd.Series(((td >= "2026-09-10") & (td <= "2026-09-26")), index=td)
rs = {lk: B.rigobon_sack(dpt["poly_p"], legs[lk]["ls"], highH) for lk in ("strong", "weak", "strong_factor", "weak_factor")}
rs.update({f"strong_{leg}": B.rigobon_sack(dpt["poly_p"], legs["strong"][leg], highH) for leg in ("long", "short")})
# H is the repricing window itself (picked from the odds, not from dated news) and contains the
# Sept 16 Fed hike, which breaks "other factors equally volatile on H and L": drop that day as a check
keep = td[td != pd.Timestamp("2026-09-16")]
rs["strong_drop_Sep16"] = B.rigobon_sack(dpt["poly_p"].loc[keep], legs["strong"]["ls"].loc[keep], highH.loc[keep])
pd.DataFrame(rs).T.to_csv(TABLES / "q4_rigobon_sack.csv")
famTD = P.family_a(S_td, dpt["poly_p"], ex_td, "trading-day calendar (C vs dP)", perm=False)
gridf = pd.concat([gridf, famTD], ignore_index=True)
gridf.to_csv(TABLES / "q3_robustness_grid.csv", index=False)
R["q3"]["robust_count_p_lt_0.10"] = int((gridf["p_hac"] < 0.10).sum()); R["q3"]["robust_tests"] = int(len(gridf))
R["q3"]["robust_hits_all_negative_sign"] = bool((gridf.loc[gridf["p_hac"] < 0.10, "coef"] < 0).all())


def _grid_hits(k):
    """Rebuild the robustness grid with every sentiment series circularly shifted by k days
    (breaks the alignment with dP, keeps each series' autocorrelation); count p < 0.10."""
    roll = lambda x: pd.Series(np.roll(x.to_numpy(), k), index=x.index)
    hits = 0
    for lab, (s_, d_) in specs.items():
        hits += int((P.family_a(roll(s_.dropna()).reindex(s_.index), d_, ex, lab, perm=False)["p_hac"] < 0.10).sum())
    for h in (3, 5):
        dph = P.dp_series(sw, "poly_p", h=h).reindex(CAL)
        s_h = (C.rolling(h).mean() if primary_is_level else C.diff(h))
        hits += int(P.hac_reg(dph, roll(s_h.dropna()).reindex(s_h.index), None, lags=h - 1)["p_hac"] < 0.10)
    hits += int((P.family_a(roll(S_td.dropna()).reindex(S_td.index), dpt["poly_p"], ex_td, "td", perm=False)["p_hac"] < 0.10).sum())
    return hits


null_hits = np.array([_grid_hits(k) for k in range(7, len(CAL) - 7)])
obs_hits = R["q3"]["robust_count_p_lt_0.10"]
R["q3"]["grid_joint_null"] = {"observed": obs_hits, "null_mean": float(null_hits.mean()), "null_sd": float(null_hits.std()),
                              "null_p90": float(np.percentile(null_hits, 90)), "n_shifts": int(len(null_hits)),
                              "p_value": float((1 + (null_hits >= obs_hits).sum()) / (1 + len(null_hits)))}
print("grid joint null:", R["q3"]["grid_joint_null"])
per_name = pd.DataFrame([{"ticker": t, "group": NAME_GROUP[t], "expected_sign": CANDIDATES[t],
                          **{f"test_{k}": v for k, v in P.hac_reg(AR[t], dpt["poly_p"]).items()},
                          "pre_gamma": pre_eb.loc[t, "gamma_coef"], "pre_gamma_se": pre_eb.loc[t, "gamma_se"],
                          "beta_spy": mm.loc[t, "beta"], "cum_ar_window": float(AR[t].sum())} for t in names])
per_name.to_csv(TABLES / "q4_per_name.csv", index=False)
R["q4"] = {"N_trading_days": int(len(td)), "gate_names": gate,
           "validation_vs_dp": val.round(4).to_dict(orient="records"),
           "family_sent_vs_basket": famB.round(4).to_dict(orient="records"),
           "family_dp_vs_basket": famV.round(4).to_dict(orient="records"),
           "robust_count_p_lt_0.10": int((grid4["p_hac"] < 0.10).sum()), "robust_tests": int(len(grid4)), "grid_joint_null": grid4_null,
           "cum_ar_end": cum.iloc[-1].round(3).to_dict(), "rigobon_sack": rs, "mde_r": P.mde_r(int(len(td))),
           "provisional_oct7": (OUTPUTS.parent / "data/raw/market/provisional_close.json").exists(),
           "pre_window": json.loads((TABLES / "q4_election_betas_summary.json").read_text())}
print("\nQ4 validation (strong L/S on dP):"); print(val[(val.basket == "strong")].round(3).to_string(index=False))
print(famB[["test", "direction", "lag", "N", "pearson_r", "coef", "p_hac", "p_perm", "q_bh"]].round(4).to_string(index=False))
print("cum AR end:", R["q4"]["cum_ar_end"])

# ====================================================================== extra questions
# E1: media vs voters agenda (Sacerdote et al. outlet groups)
def tvd(a, b):
    return float(0.5 * (a - b).abs().sum())
from scipy import stats as _st
e1 = []
for col in ["shared_media", "news", "news_US_LEFT", "news_US_RIGHT", "news_US_CENTER", "news_BUSINESS", "news_INTL"]:
    e1.append({"group": col, "tvd_vs_voter": tvd(sal[col], sal["voter"]),
               "spearman_vs_voter": float(_st.spearmanr(sal[col], sal["voter"]).statistic),
               "economy_share": float(sal.loc["Economy & cost of living", col]),
               "iran_share": float(sal.loc["Foreign policy & Iran war", col])})
e1 = pd.DataFrame(e1); e1.to_csv(TABLES / "e1_agenda_distance.csv", index=False)
# E2: which single issue tracks the sweep odds (same-day and lag-1 Granger, BH across issues)
e2 = []
for k in keys:
    s_k = I[k] if primary_is_level else I[k].diff()
    h = P.hac_reg(dp, s_k, ex)
    g = __import__("src.granger", fromlist=["x"]).granger_f(dp, s_k, 1, ex)
    g2 = __import__("src.granger", fromlist=["x"]).granger_f(s_k, dp, 1, ex)
    e2.append({"issue": k, "same_day_coef": h["coef"], "same_day_p_hac": h["p_hac"], "N": h["N"],
               "granger_sent_to_dP_p_hac": g["p_hac"], "granger_dP_to_sent_p_hac": g2["p_hac"]})
e2 = pd.DataFrame(e2)
from statsmodels.stats.multitest import multipletests as _mt
allp = np.r_[e2["same_day_p_hac"], e2["granger_sent_to_dP_p_hac"], e2["granger_dP_to_sent_p_hac"]]
q = _mt(allp, alpha=0.10, method="fdr_bh")[1]
nk = len(e2)
e2["same_day_q"], e2["g_sent_dP_q"], e2["g_dP_sent_q"] = q[:nk], q[nk:2 * nk], q[2 * nk:]
e2.to_csv(TABLES / "e2_issue_vs_sweep.csv", index=False)
# E3: the Sept 10-26 repricing vs the 17 days before it
pre_w, ev_w = (CAL >= "2026-08-24") & (CAL <= "2026-09-09"), (CAL >= "2026-09-10") & (CAL <= "2026-09-26")
sh_daily = pd.read_csv(TABLES / "q1_daily_issue_shares.csv", index_col=0, parse_dates=True)
e3 = pd.DataFrame({"pre_mean_share": sh_daily[pre_w].mean(), "event_mean_share": sh_daily[ev_w].mean()})
e3["change_pp"] = 100 * (e3["event_mean_share"] - e3["pre_mean_share"])
e3 = e3.sort_values("change_pp", ascending=False); e3.to_csv(TABLES / "e3_issue_shift.csv")
ev_corr = {"C_pre_mean": float(C[pre_w].mean()), "C_event_mean": float(C[ev_w].mean()),
           "dP_event_sum_pp": float(dp[ev_w].sum()), "dP_pre_sum_pp": float(dp[pre_w].sum()),
           "xcorr_event": P.cross_corr(S[ev_w], dp[ev_w], 3).round(3).to_dict(orient="records")}
R["extras"] = {"e1": e1.round(4).to_dict(orient="records"), "e2": e2.round(4).to_dict(orient="records"),
               "e3_shift": e3.round(4).reset_index().to_dict(orient="records"), "e3": ev_corr}
print("\nE1:"); print(e1.round(3).to_string(index=False))
print("E2:"); print(e2.round(3).to_string(index=False))
print("E3:"); print(e3.round(3).head(8).to_string()); print(ev_corr["C_pre_mean"], ev_corr["C_event_mean"], ev_corr["dP_event_sum_pp"])

(OUTPUTS / "results.json").write_text(json.dumps(R, indent=1, default=str))
