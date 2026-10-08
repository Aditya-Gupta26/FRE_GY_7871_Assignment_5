"""Extra markdown cells for analysis.ipynb. The report sections themselves are copied verbatim
from REPORT.md by make_notebook.py, so the notebook text and the report cannot drift apart."""

MD = {}
MD["intro"] = """# Assignment 5: US midterm issues, sentiment, sweep odds and a mid-term basket

This notebook reproduces every table and figure in `REPORT.pdf`. It does not pull data itself: the collectors and scorers are `scripts/01` to `07`, and they write to `data/`. The first code cell re-runs `scripts/08_run_analysis.py` (all tables and `outputs/results.json`) and `scripts/09_make_figures.py` (all figures) on the saved data, so everything shown below is computed here, the same way as for the report.

Each section below starts with the report's own text for that section (copied from `REPORT.md` when the notebook is built), and then shows the tables behind it. After each figure block there is a short "How these figures are made" cell."""

MD["how_q1"] = """**How these figures are made (Q1).** Every doc has a BERTopic topic (or -1, an outlier). The LLM taxonomy (`data/labels/topic_taxonomy.csv`) maps each of the 61 topics to an issue, horse race, media or noise. Figure 1 takes the voter-voice social docs (comments, self-posts, non-media tweets) dated Jul 1 to Oct 7 whose topic is an issue, and divides the count per issue by the count over all issues; the news bars do the same for election headlines; the green bars are the Reuters/Ipsos percentages. Figure 2 is the same share computed day by day, with a 7-day mean. No sentiment score is used in either, and both are descriptive, not predictions."""

MD["how_q2"] = """**How this figure is made (Q2).** Each voter doc gets a direction score: NLI s = P(pro-D) - P(pro-R) for tweets, (P_pos - P_neg) x tau for Reddit. For each key issue, source (Twitter and each subreddit) and day, the mean score is demeaned within that issue and source, then the sources are combined with fixed weights (their share of docs over the window). That gives the issue lines. The composite C weights the 8 issues by their Q1 voter shares. Lines are 7-day means; the tests use the daily values. Descriptive."""

MD["how_q3"] = """**How this figure is made (Q3).** Panel (a) is the last hourly price at or before 16:05 ET each day (Polymarket's price; Kalshi's bid/ask mid). Panel (b) is the daily composite C from Q2 in its own panel, so there is no second y-axis. Panel (c) is the correlation of C shifted by j days with the daily change in the Polymarket price; blue bars (j > 0) are the index leading, orange bars the odds leading. It is a test, not a prediction."""

MD["how_q4"] = """**How this figure is made (Q4).** For each name, beta on SPY comes from daily log returns Aug 1, 2025 to Jun 30, 2026. In the test window AR = 100 x (r - beta x r_SPY), with no alpha. The long leg is the equal-weighted mean AR of the 9 names expected to gain from a sweep, the short leg of the 8 expected to lose, and the lines are cumulative sums from 0 on Jun 30. Panel (b) is the Polymarket price in its own panel. Descriptive."""

MD["score_where"] = """**Which score is used where, and why (same as REPORT Section 3).** Tweets use zero-shot NLI, the best tool on the 60 validation tweets (macro-F1 0.59, κ 0.40), because it reads stance without needing a party name. Reddit uses target-signed Twitter-RoBERTa, the best on the 50 Reddit items (0.58, κ 0.32). Headlines are not in the index at all, because they have a date but no time. VADER and the distilled classifier are robustness or comparison tools only. The table below is the full validation."""
