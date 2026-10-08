# Assignment 5: US Midterm Issues, Sentiment, Democratic Sweep Odds and a Mid-Term Basket (Jul 1 to Oct 7, 2026)

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026

**Question in one line:** which issues do US voters care about for the Nov 3, 2026 midterms (topic modelling on news and social media, with an LLM in the loop), does a validated sentiment index on those issues move with the probability of a Democratic sweep on Polymarket and Kalshi, and does a basket of stocks exposed to the result move with that index?

Repo: https://github.com/Aditya-Gupta26/FRE_GY_7871_Assignment_5

`REPORT.pdf` is the write-up (built from `REPORT.md`). `analysis.ipynb` reproduces every table and figure in it. `AI_USE.md` is the AI disclosure.

---

## What this gives you

Scripts, run in this order:

| Script | Does | Writes | Time |
|---|---|---|---|
| `01_collect_twitter.py [--probe]` | twitterapi.io advanced search, one page per 90-minute window (1,584 windows), hard budget guard at $5.50 | `data/raw/twitter/` | ~5 min |
| `02_collect_reddit.py` | Arctic Shift: every post plus the first 50 comments of each 6-hour window, 6 political subs | `data/raw/reddit/` | ~50 min |
| `03_collect_news.py` then `03_collect_news.py topup` | Google News RSS `site:` queries for 30 outlets (2-day windows), then capped windows re-pulled by day | `data/raw/news/` | ~45 min |
| `04_get_market_data.py stocks` / `predmkt` | yfinance daily prices (27 names, 10 ETFs or futures); Polymarket and Kalshi hourly sweep prices | `data/raw/market/`, `data/raw/predmkt/` | ~2 min |
| `04b_election_betas.py` | pre-window (Aug 2025 to Jun 2026) election betas, the basket checkpoint | `outputs/tables/q4_election_betas_*.csv` | ~10 s |
| `05_clean_and_score.py corpus` / `embed` / `tone` / `nli` | cleaning, filter waterfall, voter vs media tags; MiniLM embeddings; VADER and Twitter-RoBERTa; zero-shot NLI on tweets | `data/interim/`, `data/processed/` | ~25 min on Apple GPU |
| `06_topic_model.py fit` / `diag` | BERTopic (fit on a 60k sample, applied to all docs) and the topic sheet for the LLM; seeds, coherence and the LDA sweep | `data/interim/doc_topics.parquet`, `data/labels/topic_sheet.csv` | ~5 min + ~10 min |
| `07_make_validation_sample.py make` / `score` | blind labelling sheets (150 validation, 1,500 calibration, 50 double-labelled); tool scores, F1 and κ with bootstrap CIs, the distilled classifier | `data/labels/`, `outputs/tables/q2_validation*` | ~5 min |
| `08_run_analysis.py` | Q1 to Q4 and the extra questions | `outputs/tables/`, `outputs/results.json` | ~10 s |
| `09_make_figures.py` | the 4 report figures and notebook Figure A from the saved tables | `outputs/figures/` | ~10 s |
| `make_notebook.py` | builds `analysis.ipynb` from `REPORT.md` text plus code cells; then run it with nbconvert | `analysis.ipynb` | ~1 min |
| `build_report.py`, `99_lint_prose.py` | REPORT.md to PDF (markdown-it, WeasyPrint); checks for em dashes | `REPORT.pdf` | ~5 s |

The steps that need an LLM were done by Claude Code subagents and their output is committed in `data/labels/`: the topic taxonomy (`topic_taxonomy.csv`), the blind validation labels, the calibration labels and the blind issue labels. The labelling guide is `data/labels/LABELLING_GUIDE.md`.

The modules:

| Module | What it does |
|---|---|
| `src/config.py` | Window, sources, queries, outlets, basket groups, the **pre-registration** (`PREREG`, written before any test) and the one disclosed deviation (`PREREG_DEVIATION`), the issue vocabulary and poll crosswalk. All dates in ET. |
| `src/twitter.py`, `src/reddit.py`, `src/news.py`, `src/predmkt.py` | Collectors, all cached and resumable. |
| `src/corpus.py`, `src/text_clean.py` | The filter waterfall, near-duplicate removal (MinHash, Jaccard 0.8), the 16:00 ET bars (asserted), voter vs shared-media tags. |
| `src/topics.py`, `src/q1.py` | BERTopic and LDA, coherence, seed matching; issue shares and the poll comparison. |
| `src/sentiment.py`, `src/validate.py` | VADER, Twitter-RoBERTa, zero-shot NLI; the direction tools, metrics and the distilled classifier. |
| `src/index.py` | The directional index (issue x source demeaning, salience weights) and split-half reliability. |
| `src/pm.py`, `src/market.py`, `src/basket.py` | Sweep odds at 16:00 ET; prices and abnormal returns; election betas, legs and the Rigobon-Sack estimator. |
| `src/granger.py`, `src/pipeline.py` | Granger (F, HAC, permutation), BH, VAR; family A tests and robustness. |

**The 30 news outlets** (grouping follows Sacerdote et al. 2020; the lean labels are only for grouping):
- US left: cnn.com, ms.now (MSNBC since Nov 15, 2025), nytimes.com, washingtonpost.com, npr.org, nbcnews.com, huffpost.com
- US right: foxnews.com, nypost.com, washingtonexaminer.com, dailywire.com, breitbart.com, newsmax.com
- US centre: apnews.com, reuters.com, thehill.com, politico.com, usatoday.com, cbsnews.com, abcnews.com, axios.com, newsweek.com
- Business: wsj.com, bloomberg.com, cnbc.com, marketwatch.com
- International: theguardian.com, bbc.com, ft.com, economist.com

### If a download will not run

- **Twitter:** you need your own twitterapi.io key in `.env` as `TWITTERAPI_IO_KEY=...` (chmod 600). Price checked Oct 7, 2026: $0.15 per 1,000 tweets, at least 15 credits per call, 1 USD = 100,000 credits, credits never expire, the smallest top-up on the payment page is $10. Run `--probe` first (6 windows, about $0.02). The full pull is at most 31,680 tweets, at most $4.75; the collector refuses any call that would take total spend past $5.50. My run was charged $4.48 by the provider.
- **Arctic Shift:** "Timeout. Maybe slow down a bit" is normal on busy subs; the client backs off and resumes. `domain` and `is_self` are not valid `fields` names (the API returns 400).
- **Google News RSS:** a feed returns at most 100 items, and in date-bounded feeds most items are stamped 07:00 GMT (date only). The POLICY query in `config.py` is kept only as a record: Google ignored its date operators and served the same ~100 articles every day, so it is not used.
- **GDELT** (429 from my IP) and **Bluesky** (public search returns 403) were not used.
- **Polymarket:** `prices-history` with `startTs`/`endTs` rejects spans over 15 days, and appends one "current" point after `endTs`; the collector uses 14-day chunks and filters on `t`.
- **yfinance:** on the evening of Oct 7, Yahoo's Oct 7 daily bar was empty for every ticker; the script then fills that day from the 15:30 to 16:00 hourly bar and writes `provisional_close.json`. Re-run `04_get_market_data.py stocks` later to get the official close.
- **UMAP transform** segfaulted on the full 138k docs with numba's parallel threads; `06_topic_model.py` sets `NUMBA_NUM_THREADS=1` and transforms in 10k chunks.

## Setup

```bash
cd Assignment_5
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "TWITTERAPI_IO_KEY=your_key_here" > .env && chmod 600 .env
python scripts/04_get_market_data.py stocks && python scripts/04_get_market_data.py predmkt
python scripts/01_collect_twitter.py --probe && python scripts/01_collect_twitter.py
python scripts/02_collect_reddit.py
python scripts/03_collect_news.py && python scripts/03_collect_news.py topup
python scripts/04b_election_betas.py
python scripts/05_clean_and_score.py corpus && python scripts/05_clean_and_score.py embed
python scripts/05_clean_and_score.py tone && python scripts/05_clean_and_score.py nli
python scripts/06_topic_model.py fit && python scripts/06_topic_model.py diag
python scripts/07_make_validation_sample.py score    # the label CSVs in data/labels/ are committed
python scripts/08_run_analysis.py && python scripts/09_make_figures.py
python scripts/make_notebook.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
python scripts/build_report.py && python scripts/99_lint_prose.py
```

Note: `07_make_validation_sample.py make` is seeded, so on the same data it rebuilds the same text sheets (they are not committed because they hold raw post text); the committed `*_key.parquet` files tie the labels to doc ids.

## Reading the data you end up with

```python
import pandas as pd
soc = pd.read_parquet("data/interim/social.parquet")        # 65,891 filtered social docs
soc[["doc_id", "platform", "sub", "kind", "voice", "ts", "bar_date", "clean"]]
top = pd.read_parquet("data/interim/doc_topics.parquet")    # topic and outlier-reduced topic per doc
tone = pd.read_parquet("data/processed/tone_social.parquet")  # vader_*, rob_neg / rob_neu / rob_pos
nli = pd.read_parquet("data/processed/nli_twitter.parquet")   # nli_D, nli_R, nli_s for every tweet
```

`bar_date` is the 16:00 ET bar a post feeds: a post at 15:59 ET on a day feeds that day, a post at 16:00 feeds the next. The direction score used in the index is NLI `nli_s` for tweets and `(rob_pos - rob_neg) x tau` for Reddit (tau = +1 if only Democrat-side names appear, -1 if only GOP-side names); REPORT Section 3 explains why.

## Submitting

1. **GitHub:** the notebook with outputs saved, all code, the figures in `outputs/figures/` (so REPORT.md shows them on GitHub), `AI_USE.md`, and the label files and item keys in `data/labels/` (not the text sheets, which hold raw post text). No other data, no `.env`, no `PROGRESS.md`, no `Materials/`.
2. **Brightspace:** `REPORT.pdf`, plus the repo URL.
