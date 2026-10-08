# AI use disclosure

**Tools used:** Claude (Opus 5.5), through Claude Code, including Claude Code subagents.

**What I used them for:** The implementation for this assignment was done by Claude Code in this repository, under my direction. That covers:
- the Twitter, Reddit, news and prediction-market collectors and the market data scripts
- cleaning, the filter waterfall and deduplication
- the topic model (BERTopic, plus LDA as the baseline) and its diagnostics
- the direction tools, the validation code, the directional index, the tests (Granger, HAC, permutation, FDR, VAR, Rigobon-Sack) and the figures
- the notebook, the README and the first draft of REPORT.md

I directed it step by step: first a written plan, then a full verification pass of every fact and formula in that plan, then one piece at a time with a check of the actual numbers before the next piece was built.

The "super intelligence tools" part of the question was done by Claude Code subagents, and every one of these steps is disclosed in the report:
- **Topic taxonomy.** A fresh subagent with no context saw only each topic's top words and example posts (no sentiment scores) and mapped the 61 BERTopic topics to voter issues (`data/labels/topic_taxonomy.csv`).
- **Validation labels.** A separate blind subagent labelled the 150 validation items for direction and tone, seeing only the labelling guide and the text, never any tool's output. Another one gave the blind issue labels. **These are LLM labels, not human labels.**
- **Calibration labels.** Five subagents labelled 1,500 other items, which trained the LLM-distilled classifier. A sixth relabelled 50 of them to measure agreement (κ = 0.96).
- **Verification and review.**
  - Before building, three subagents checked the plan's facts: the policy facts, the API and data facts, and the method and formula sources.
  - An independent Plan agent critiqued the design.
  - Two independent reviewer agents recomputed the data step, the betas and Q1 to Q4 from the raw files with their own code.
  - At the end, a fresh agent did the full scrutiny check: the brief word by word, my decisions, every number in the report, README, AI_USE and notebook against the output files, the captions against the figures, and the build.

No explainer artifact was used for this assignment.

**What I decided and did myself:**
- **The window and the deadline.** Jul 1 to the Oct 7 close, finishing by Oct 8.
- **The sources and the money.** Reddit and Twitter together, with Twitter capped at $5.50 (it was charged $4.48), and the `min_faves:5` query after the probe showed only half of the `min_faves:2` tweets pass the views rule.
- **The LLM route.** No paid API; Claude Code subagents plus local models.
- **The validation design.** Blind subagent labels, as in A4.
- **The contracts.** Polymarket "Democrats Sweep" as the primary series, with Kalshi as the cross-check.
- **The readings.** Adding Bybee et al. (A1) and Rigobon and Sack (A3) once they turned out to fit this question, and dropping the three A4 readings that are not on disk.
- **The basket.** Extending the election-beta pre-window to Aug 2025 to Jun 2026 (from Jan to Jun 2026) after the review showed the shorter window had too little power; the strong-mechanism groups as the primary basket and the weak ones as a sub-basket; approving the basket from pre-window betas only, before any test-window return was looked at; and keeping the pre-registered basket even when the short leg had no pre-window support.
- **BERTopic as the main model,** with Bybee's LDA as the baseline.
- **The rules of the work:** every result explained, nothing claimed before it is verified, every formula cited and checked, and a separate agent reviewing every step.

**Anything the model got wrong that I had to correct:** All of these are from the running bug log. They were caught by the checks at every step: counts per source and day, timestamps against known events, recomputation by independent agents, and looking at every figure.

The most serious ones would have changed the results or misled the reader:
- **The Polymarket "16:00" snapshot was really the 15:00 price.** Every hourly point is stamped about 16 seconds past the hour, so "last point at or before 16:00" picked the previous hour. That would have let posts from 15:00 to 16:00 sit after the start of the price change they were tested against. Fixed by snapping at 16:05 and asserting every post is earlier than the price used.
- **The brief's "14-point gap" between the two venues was wrong.** Their own histories show -1.0 to +6.0 points; the 48% was Kalshi's early-September price compared with Polymarket's late-September one.
- **The POLICY news query was useless.** Google ignored its date filters and returned the same ~100 articles every day (10,000 items, 104 unique). It was dropped.
- **News headlines carry a date only** (78% of the final headlines are stamped 07:00 GMT), so news was kept out of every timed test.
- **The pre-registered key-topic rule could not be met.** Under the main topic assignment no issue had 20 docs a day on 90% of days (under the outlier-reduced one only two did, too few for 5 to 8 key topics). The deviation (5 docs, outlier-reduced topics) was written down before any index existed, and the report shows the lag-2 pattern disappears under the original rule.
- **The first draft overstated things.** It said the Q1 ranking was robust in its top half, but in all three model seeds "abortion and social issues" falls from 11% to 3% to 6%; that issue is also mostly race and religion debates. Split-half reliability was reported from one lucky split (0.16; the average over 50 splits is 0.08). The robustness hits were compared with chance as if the tests were independent; a joint shifted null shows they are what chance gives (P = 0.30).
- **Several unverified sentences were caught and rewritten:** that each hospital cut guidance (Tenet raised it), that "company news" drove single-name moves, that the repricing "came from" polls, that the ethics spike was the Epstein files (the data say mostly the PAC-money topic).

Data and code bugs:
- **Two all-NaN stock rows.** In the first download these were Memorial Day and Labor Day, the same yfinance bug as in A4. Fixed by keeping only the days SPY traded.
- **Collector bugs:**
  - Arctic Shift rejects `domain` as a field.
  - abcnews.go.com returned 0 items and msnbc.com 1 (now abcnews.com and ms.now).
  - The Polymarket API appends a "current" point after the end time.
  - On the evening of Oct 7, Yahoo served an empty Oct 7 bar; it was filled from the last hourly bar and flagged provisional, then replaced by the official close in a later re-pull.
- **BERTopic segfaulted** while transforming 138k docs (numba threads). It now runs single-threaded in chunks.
- **gensim's LdaMulticore crashed** on macOS, so the single-process LDA is used instead.
- **A faster NLI setting would have silently changed scores for long tweets.** 11% of tweets are over 120 tokens; a 0.49 difference showed up against the validated scores, so the validated 256-token limit is kept.
- **The sweep-odds figure's legend was mislabelled** (now Figure 3). The shaded band was named "Polymarket" and the Polymarket line "Kalshi". It was caught by looking at the figure, along with overlapping axis labels and a legend covering a line.
- **Missing checks and small inconsistencies:**
  - A missing pre-registered robustness check (the trading-day calendar).
  - A timing assert weaker than the plan's rule.
  - Q4 tables labelled with the wrong variable.
  - Expanding weights computed on a different basis.
  - A detectable-r computed with N = 68 instead of 69.
  - A Rigobon-Sack high-news window that contains the Fed hike (a version without Sept 16 is now reported).

**The final scrutiny check found more**, all fixed before hand-in:
- **The Q1 headline mislabelled its category.** "Iran war 20.3%" is the whole foreign-policy label, whose biggest part is Israel aid and AIPAC (9.5 points; the Iran war is 6.9). A sentence also said the Iran topics hold gas prices, but the gas-price topic is in the economy.
- **Four rounding errors.** One p-value, one γ, one correlation and one band limit.
- **Results left out of the report.** Two single names (RUN, DLTR) move with the odds after correction; no insurer or hospital confirmed its pre-window γ; and the pre-registered test in changes is null.
- **Smaller gaps.** Missing "In short" and Readings lines on Q5, floating labels in the sweep-odds figure (now Figure 3), captions that did not explain the grey band, log returns written as if they were simple returns, and one statement ("record gas prices") that the sources did not support.

**A second full re-check, on my request after the first push,** used three fresh agents (one on the question and method, one recomputing every number, one on the code) and found more, all fixed:
- **The one correlation the question asks for in Q4** (sentiment index vs the basket's market-adjusted return) was tested but never written as a number; it is r = -0.02.
- **The basket's +6.4% "right direction" was an endpoint effect:** the basket was flat (-0.3%) on Oct 1 after the whole rise in the odds, and the gain came in the last four days while the odds stayed within 63.5% to 66.5%.
- **The short leg had no price support** (1 of 8 names with the expected pre-window sign), and the Q1 ranking depended on the topic assignment, so the robust claims were narrowed.
- **Smaller ones:** "Kalshi candles mostly have no trade" (it is 26%), the 72,688 headlines are not all from the 30 named outlets, a few rounding errors, and several code issues that did not change any result.

These got caught because nothing was accepted just because the code ran. The reviewer agents reproduced every headline number to rounding, and the final check matched every number in the report against `outputs/results.json` and the tables.
