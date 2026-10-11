"""Extra markdown cells for analysis.ipynb. The report sections themselves are copied verbatim
from REPORT.md by make_notebook.py, so the notebook text and the report cannot drift apart."""

MD = {}
MD["intro"] = """# Assignment 5: US midterm issues, sentiment, sweep odds and a mid-term basket

This notebook reproduces every table and figure in `REPORT.pdf`. It does not pull data itself: the collectors and scorers are `scripts/01` to `07`, and they write to `data/`. The first code cell re-runs `scripts/08_run_analysis.py` (all tables and `outputs/results.json`) and `scripts/09_make_figures.py` (all figures) on the saved data, so everything shown below is computed here, the same way as for the report.

Each section below starts with the report's own text for that section (copied from `REPORT.md` when the notebook is built), and then shows the tables behind it. After each figure block there is a short "How these figures are made" cell."""

MD["how_q1"] = """**How these figures are made (Q1).** Every doc has a BERTopic topic (or -1, an outlier). The LLM taxonomy (`data/labels/topic_taxonomy.csv`) maps each of the 61 topics to an issue, horse race, media or noise. Figure 1 takes the voter-voice social docs (comments, self-posts, non-media tweets) dated Jul 1 to Oct 7 whose topic is an issue, and divides the count per issue by the count over all issues; the news bars do the same for election headlines; the green bars are the Reuters/Ipsos percentages. Notebook Figure A (moved out of the report to keep it at 6 pages with legible figures) is the same share computed day by day, with a centred 7-day mean. No sentiment score is used in either, and both are descriptive, not predictions."""

MD["how_q2"] = """**How Figure 2 is made (Q2).** Each voter doc gets a direction score: NLI s = P(pro-D) - P(pro-R) for tweets, (P_pos - P_neg) x tau for Reddit. For each key issue, source (Twitter and each subreddit) and day, the mean score is demeaned within that issue and source, then the sources are combined with fixed weights (their share of docs over the window). That gives the issue lines. The composite C weights the 8 issues by their Q1 voter shares. Lines are centred 7-day means; the tests use the daily values. Descriptive."""

MD["how_q3"] = """**How Figure 3 is made (Q3).** Panel (a) is the last hourly price at or before 16:05 ET each day (Polymarket's price; Kalshi's bid/ask mid). Panel (b) is the daily composite C from Q2 in its own panel, so there is no second y-axis. Panel (c) is the correlation of C shifted by j days with the daily change in the Polymarket price; blue bars (j > 0) are the index leading, orange bars the odds leading. It is a test, not a prediction."""

MD["how_q4"] = """**How Figure 4 is made (Q4).** For each name, beta on SPY comes from daily log returns Aug 1, 2025 to Jun 30, 2026. In the test window AR = 100 x (r - beta x r_SPY), with no alpha. The long leg is the equal-weighted mean AR of the 9 names expected to gain from a sweep, the short leg of the 8 expected to lose, and the lines are cumulative sums from 0 on Jun 30. Panel (b) is the Polymarket price in its own panel. Descriptive."""

MD["score_where"] = """**Which score is used where, and why (same as REPORT Section 3).** Tweets use zero-shot NLI, the best tool on the 60 validation tweets (macro-F1 0.59, κ 0.40), because it reads stance without needing a party name. Reddit uses target-signed Twitter-RoBERTa, the best on the 50 Reddit items (0.58, κ 0.32). Together, as the index uses them, they score macro-F1 0.60 and κ 0.38 on the 110 social items, against 0.22 for always guessing "neither". Headlines are not in the index at all, because they have a date but no time. VADER and the distilled classifier are robustness or comparison tools only. The table below is the full validation."""

MD["how_class"] = """**How these extensions are made (Section 10).** Every test here was written down before it was run (`config.PREREG_CLASS`), but after the main results, so all of it is exploratory. (1) House and Senate odds come from the mutually exclusive Polymarket outcomes: P(D House) = P(D sweep) + P(R Senate, D House), P(D Senate) = P(D sweep) + P(D Senate, R House), snapped at 16:00 ET like the sweep odds. (2) Each strong basket group is paired with its own issue before looking: health care with insurers and hospitals, immigration with GEO and CXW, foreign policy with defense, tech and AI with COIN and HOOD, guns with RGR and SWBI. The group's market-adjusted return is multiplied by its expected sign, so + is always the expected direction. (3) Attention combines Reddit post counts (collected in full) with the tweet arrival rate, (n - 1)/T for a page of n tweets spanning T minutes. Tweet counts themselves are fixed by the one-page design, so they say nothing about attention. (4) Kalshi volume is summed per 16:00-to-16:00 day; an hour with no candle counts as zero. (5) The random-sign placebo keeps the same 17 names and the 9 long / 8 short split, and reshuffles which names are long. Notebook Figure B shows attention, the size of odds moves and Kalshi volume. It is descriptive.

**What the independent review added (all labelled "added" in the report and under `added_checks` in `results_class.json`).**
- **House slope with clean prices:** the House-alone basket slope without the 25 pre-window days with a stale component price.
- **House share of the rise:** the House share of the sweep's log-rise.
- **Normalised probability:** P(Democrats win at least one chamber), with the five prices normalised to sum to 1.
- **Attention components:** each attention component tested separately against the size of odds moves.
- **Volume without September:** the volume relation re-run without Sept 10 to 26.
- **Group-level sign flips:** sign flips of whole policy groups (62 assignments). The name-level splits break the six-group structure, so they give too narrow a null.

**Three details the review noted.**
- **Source weights in (2):** in the issue-matched tests, the source weights come from the five paired issues' own doc mix, not the eight Q2 key issues. The two versions correlate at least 0.988, and foreign policy to defense at lag 1 has p below 0.01 either way.
- **Tweet-rate drift:** the tweet rate drifts up over the window, from about 8 to about 13 tweets a minute in the first and last weeks. That is why A is close to a unit root.
- **Reddit's weekly cycle:** Reddit posts have a strong day-of-week cycle. The regressions remove it with dummies; the circular-shift test does not."""
