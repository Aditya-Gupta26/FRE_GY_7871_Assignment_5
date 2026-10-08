"""Report figures, drawn only from the saved tables in outputs/tables (no recomputation).

Style: one validated categorical palette (slots 1-3 of the reference palette, checked with the
dataviz validator: CVD and normal-vision separation pass; aqua < 3:1 contrast, so every line is
directly labelled), thin 1.6 px lines, recessive grid, no second y-axis anywhere (a second
measure gets its own panel), units on every axis.
"""
import _bootstrap  # noqa: F401
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import textwrap

import numpy as np
import pandas as pd

from src.config import FIGURES, OUTPUTS, POLLS, TABLES

FIGURES.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5, "axes.titlesize": 7.8,
                     "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.fontsize": 6.6, "legend.frameon": False,
                     "lines.linewidth": 1.6, "savefig.dpi": 220, "figure.dpi": 110})
R = json.loads((OUTPUTS / "results.json").read_text())
keys = R["q1"]["key_topics"]
EV = {"2026-09-13": "Sweep odds jump\n(Sept 13)", "2026-09-16": "Fed hike\n(Sept 16)"}


def datefmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def shade(ax):
    ax.axvspan(pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-26"), color="#f1efe8", lw=0, zorder=0)


# ---------------------------------------------------------------- Fig 1: shares vs poll
sal = pd.read_csv(TABLES / "q1_issue_shares.csv", index_col=0)
poll = pd.Series(POLLS["Reuters/Ipsos Sep 2026 (RV)"], dtype=float) / 100
show = sal.index[(sal[["voter", "news"]].max(axis=1) >= 0.02) | sal.index.isin(poll.index)]
t = sal.loc[show, ["voter", "news"]].assign(poll=poll.reindex(show)).sort_values("voter")
fig, ax = plt.subplots(figsize=(7.2, 2.35))
y = np.arange(len(t)); h = 0.26
for i, (col, c, lab) in enumerate([("voter", BLUE, "Voter voice (social, issue docs)"),
                                    ("news", ORANGE, "News (election headlines)"),
                                    ("poll", AQUA, "Reuters/Ipsos Sept 17 to 20 (RV, most important problem)")]):
    ax.barh(y + (1 - i) * h, 100 * t[col].fillna(0), height=h - 0.03, color=c, label=lab)
ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_xlabel("share of issue docs / of respondents (%)")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 1. What voters talk about vs what media cover vs what a poll asks (Jul 1 to Oct 7, 2026)", loc="left")
fig.tight_layout(); fig.savefig(FIGURES / "fig1_issue_shares.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: salience over time
ds = pd.read_csv(TABLES / "q1_daily_issue_shares.csv", index_col=0, parse_dates=True)
fig, axs = plt.subplots(2, 4, figsize=(7.2, 2.5), sharex=True)
for ax, k in zip(axs.flat, keys):
    shade(ax)
    ax.plot(ds.index, 100 * ds[k].rolling(7, min_periods=4, center=True).mean(), color=BLUE)
    ax.set_title(textwrap.fill(k, 26), loc="left", fontsize=7.0); datefmt(ax)
fig.supylabel("share of voter issue docs (%), centred 7-day mean", fontsize=7.0)
fig.suptitle("Notebook Figure A. Daily share of voter-voice issue docs per key issue (grey band: Sept 10 to 26 repricing)",
             x=0.01, ha="left", fontsize=7.8)
fig.tight_layout(); fig.savefig(FIGURES / "figA_salience_over_time.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: directional index
I = pd.read_csv(TABLES / "q2_issue_index_daily.csv", index_col=0, parse_dates=True)
C = pd.read_csv(TABLES / "q2_composite_daily.csv", index_col=0, parse_dates=True)["C"]
fig, axs = plt.subplots(2, 5, figsize=(7.2, 2.45), sharex=True)
for ax, k in zip(axs.flat, keys + ["Composite C (salience-weighted)"]):
    shade(ax)
    s = C if k.startswith("Composite") else I[k]
    ax.axhline(0, color=INK2, lw=0.6)
    ax.plot(s.index, s.rolling(7, min_periods=4, center=True).mean(), color=INK if k.startswith("Composite") else BLUE)
    ax.set_title(textwrap.fill(k.replace("Composite C (salience-weighted)", "Composite C"), 20), loc="left", fontsize=6.6); datefmt(ax)
fig.supylabel("direction (demeaned),\ncentred 7-day mean", fontsize=6.8)
for ax in list(axs.flat)[len(keys) + 1:]:
    ax.set_visible(False)
fig.suptitle("Figure 2. Directional index per key issue (+ = more pro-Democrat / anti-GOP than the issue's own window average)",
             x=0.01, ha="left", fontsize=7.8)
fig.tight_layout(); fig.savefig(FIGURES / "fig2_directional_index.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: sweep odds, index, lead-lag
sw = pd.read_csv(TABLES / "q3_sweep_daily.csv", index_col=0, parse_dates=True).loc["2026-07-01":]
xc = pd.read_csv(TABLES / "q3_cross_corr.csv")
fig = plt.figure(figsize=(7.2, 2.6))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.28)
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1); a3 = fig.add_subplot(gs[:, 1])
for ax in (a1, a2):
    shade(ax); datefmt(ax)
lp, = a1.plot(sw.index, 100 * sw["poly_p"], color=BLUE, label="Polymarket")
lk, = a1.plot(sw.index, 100 * sw["kal_p"], color=ORANGE, label="Kalshi (bid/ask mid)")
arw = dict(arrowstyle="-", color=INK2, lw=0.6)
a1.annotate("Polymarket", (pd.Timestamp("2026-09-13"), 53.5), xytext=(pd.Timestamp("2026-08-20"), 58), fontsize=6.6, color=INK, arrowprops=arw)
a1.annotate("Kalshi", (pd.Timestamp("2026-09-13"), 47.5), xytext=(pd.Timestamp("2026-09-20"), 42.0), fontsize=6.6, color=INK, arrowprops=arw)
a1.legend(handles=[lp, lk], loc="upper left")   # explicit handles: the shaded band is not a series
a1.set_ylabel("P(Democratic sweep), %"); a1.set_title("(a) Sweep odds at 16:00 ET", loc="left")
a1.tick_params(labelbottom=False)   # shared x axis; labels only under panel (b)
a2.axhline(0, color=INK2, lw=0.6); a2.plot(C.index, C, color=INK, lw=1.0, alpha=0.5)
a2.plot(C.index, C.rolling(7, min_periods=4, center=True).mean(), color=INK)
a2.set_ylabel("composite C"); a2.set_title("(b) Directional index C (thin: daily; thick: centred 7-day mean)", loc="left")
a3.bar(xc["lead_of_sent"], xc["r"], color=[BLUE if j > 0 else (ORANGE if j < 0 else INK2) for j in xc["lead_of_sent"]], width=0.7)
n = R["q3"]["N"]; band = 1.96 / np.sqrt(n)
a3.axhline(band, color=INK2, lw=0.6, ls="--"); a3.axhline(-band, color=INK2, lw=0.6, ls="--")
a3.set_xticks(range(-5, 6))
from matplotlib.patches import Patch
a3.legend(handles=[Patch(color=BLUE, label="index leads (j > 0)"), Patch(color=ORANGE, label="dP leads (j < 0)"),
                   Patch(color=INK2, label="same day")], loc="upper left")
a3.set_xlabel("lag j (days)"); a3.set_ylabel("corr(index on day t-j, dP on day t)")
a3.set_title("(c) Lead-lag (dashed: +-1.96/sqrt(N))", loc="left")
fig.suptitle("Figure 3. Sweep odds and the directional index, in separate panels (no shared axis)", x=0.01, y=1.02, ha="left", fontsize=7.8)
fig.savefig(FIGURES / "fig3_sweep_index.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: basket
cum = pd.read_csv(TABLES / "q4_cum_ar.csv", index_col=0, parse_dates=True)
cum = pd.concat([pd.DataFrame(0.0, index=[pd.Timestamp("2026-06-30")], columns=cum.columns), cum])
fig, (b1, b2) = plt.subplots(2, 1, figsize=(7.2, 2.55), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
for ax in (b1, b2):
    shade(ax); datefmt(ax)
b1.axhline(0, color=INK2, lw=0.6)
hs = []
for col, c, lab in (("long", AQUA, "Long leg (Dem-sweep winners)"), ("short", ORANGE, "Short leg (Dem-sweep losers)"),
                    ("strong", BLUE, "Long minus short")):
    hs.append(b1.plot(cum.index, cum[col], color=c, label=lab)[0])
    b1.text(cum.index[-1], cum[col].iloc[-1], " " + lab.split(" (")[0], color=INK, fontsize=6.4, va="center")
b1.set_ylabel("cumulative market-adj.\nreturn (%)")
fig.legend(handles=hs, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.97))
b1.set_title("(a) Primary basket, equal weights, AR = r - beta x SPY (beta from Aug 2025 to Jun 2026)", loc="left")
b2.plot(sw.index, 100 * sw["poly_p"], color=BLUE); b2.set_ylabel("P(sweep), %")
b2.set_title("(b) Polymarket P(Democratic sweep), its own panel", loc="left")
fig.suptitle("Figure 4. Mid-term basket market-adjusted performance and the sweep odds", x=0.01, y=0.995, ha="left", fontsize=7.8)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(FIGURES / "fig4_basket.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))
