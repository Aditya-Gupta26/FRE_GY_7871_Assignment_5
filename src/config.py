"""Central config: dates, sources, queries, tickers, the pre-registration and the event calendar.

Every timestamp is stored in UTC and converted to US/Eastern (ET) only when text
is aligned to the daily 16:00 ET snapshot. The window ends at the Oct 7, 2026 close
because the report is due Oct 10 and had to be finished by Oct 8.
Facts quoted in comments were verified on Oct 7, 2026 (sources in the plan file and REPORT.md).
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LABELS = DATA / "labels"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------- window and clocks
# Test window: Jul 1 to the Oct 7 close. Text collection starts at the Jun 30 close,
# because the Jul 1 daily bar is [Jun 30 16:00 ET, Jul 1 16:00 ET).
COLLECT_START = datetime(2026, 6, 30, 16, 0, tzinfo=ET)
WINDOW_START = datetime(2026, 7, 1, 0, 0, tzinfo=ET)
WINDOW_END = datetime(2026, 10, 7, 16, 0, tzinfo=ET)
SNAPSHOT_HOUR_ET = 16      # daily clock for stocks, prediction markets and text
SNAPSHOT_TOL_MIN = 5       # Polymarket's 16:00 point is stamped ~16:00:17 ET (checked on disk)

# Pre-window for election betas (no look-ahead). Polymarket history starts Jul 19, 2025,
# so the main pre-window is Aug 1, 2025 to Jun 30, 2026; Jan 2 to Jun 30, 2026 (both venues)
# is the robustness window.
BETA_START = "2025-08-01"
BETA_END = "2026-06-30"
BETA_ROBUST_START = "2026-01-02"
PRICE_START = "2025-07-01"   # a little earlier, so the first pre-window return exists
PRICE_END = "2026-10-08"     # yfinance end is exclusive
POLY_HISTORY_START = "2025-07-19"

# ---------------------------------------------------------------- Reddit (Arctic Shift)
REDDIT_API = "https://arctic-shift.photon-reddit.com/api"
REDDIT_SUBS = ["politics", "Conservative", "democrats", "Republican",
               "PoliticalDiscussion", "moderatepolitics"]
# Lean design: every post, plus a time-stratified comment sample
# (first N comments of each H-hour window, ascending; descending times out on big subs).
COMMENT_WINDOW_HOURS = 6
COMMENTS_PER_WINDOW = 50

# ---------------------------------------------------------------- Twitter / X (twitterapi.io)
# Pricing checked Oct 7, 2026 (twitterapi.io/pricing): $0.15 per 1k tweets, 1 USD = 100,000
# credits, minimum 15 credits per call, at most 20 tweets per advanced-search page.
TWITTER_ENDPOINT = "https://api.twitterapi.io/twitter/tweet/advanced_search"
TWITTER_QUERY = ('(midterms OR midterm OR "2026 election" OR Congress OR Democrats OR Republicans '
                 'OR GOP OR "vote blue" OR "vote red" OR "House race" OR "Senate race") '
                 'lang:en -filter:retweets min_faves:5')
TWITTER_SUBWINDOW_MIN = 90       # 16 windows a day, one page each: 1,584 calls, <= $4.75
TWITTER_BUDGET_USD = 5.50        # Aditya's hard cap; the collector stops before crossing it
TWITTER_SUNK_USD = 0.024         # probe (6 windows, min_faves:2, deleted) + a 2-call min_faves:5 test
CREDITS_PER_USD = 100_000
CREDITS_PER_TWEET = 15
MIN_CREDITS_PER_CALL = 15

# ---------------------------------------------------------------- social filters (waterfall)
TW_MIN_VIEWS = 100
TW_MIN_FOLLOWERS = 10
TW_MIN_ACCOUNT_AGE_DAYS = 30
MAX_PER_AUTHOR_DAY = 5
MIN_WORDS = 5
RD_MIN_SCORE = 2                 # Reddit: at least one upvote beyond the author's own
PROMO_RE = (r"giveaway|airdrop|whitelist|presale|follow (?:and|&|\+) (?:rt|retweet)|\bdm me\b|"
            r"promo code|use my code|join (?:my|our) (?:discord|telegram)|onlyfans|"
            r"100x|to the moon|free (?:trial|crypto)")
# "Topic term in the text itself" (brief 1.4): a broad US-politics lexicon. It is a relevance
# filter, not a topic list; the topics come from the topic model.
POLITICS_RE = (r"\b(?:congress\w*|senat\w*|house|democrat\w*|dems?|republican\w*|gop|maga|trump\w*|"
               r"vance|biden|harris|schumer|jeffries|pelosi|thune|speaker|mcconnell|election\w*|"
               r"midterms?|vot\w+|ballot\w*|campaign\w*|candidate\w*|poll\w*|primar\w+|governor|"
               r"president\w*|white house|administration|policy|policies|bill|law\w*|legislat\w+|"
               r"tariff\w*|inflation|econom\w+|prices?|cost of living|afford\w+|jobs?|wages?|"
               r"immigra\w+|ice|deport\w*|border|asylum|abortion|roe|healthcare|health care|medicaid|"
               r"medicare|obamacare|aca|insurance|premiums?|social security|guns?|crime|police|"
               r"iran|war|israel|ukraine|military|shutdown|budget|tax\w*|deficit|gas|epstein|"
               r"corrupt\w*|impeach\w*|court|scotus|justice|fed|rates?|liberal\w*|conservative\w*|"
               r"leftis\w+|right-wing|far-right|far-left|progressive\w*|socialis\w+|fascis\w+|democracy|constitution\w*)\b")

# Party targets for the target-signed tools (tau = +1 only Dem side named, -1 only GOP side).
DEM_TERMS = (r"\b(?:democrats?|dems?|democratic party|schumer|jeffries|pelosi|kamala|harris|biden|"
             r"newsom|aoc|ocasio-cortez|bernie|sanders|pritzker|whitmer|buttigieg|walz|obama|"
             r"liberals?|libs|leftists?|progressives?)\b")
GOP_TERMS = (r"\b(?:republicans?|gop|maga|trump\w*|vance|speaker johnson|mike johnson|thune|"
             r"mcconnell|hegseth|rubio|desantis|noem|bondi|conservatives?|right-wing|rinos?)\b")

# ---------------------------------------------------------------- news (Google News RSS)
# Checked Oct 7: one feed returns at most 100 items, and date-bounded feeds stamp most items
# 07:00 GMT, so headlines carry a DATE only. They are used for daily shares, never timed tests.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
NEWS_QUERIES = {
    # issue-neutral election coverage: news topic SHARES come from this query only
    "ELECTION": ('(midterm OR midterms OR "2026 election" OR "House race" OR "Senate race" '
                 'OR "control of Congress" OR "balance of power" OR "campaign trail")'),
    # DROPPED: Google News ignored after:/before: for this nested query and returned the same
    # ~100 articles every day (10,000 items, 104 unique links). Kept here only as a record.
    "POLICY": ('(Congress OR "White House" OR Republicans OR Democrats) '
               '(economy OR inflation OR tariffs OR immigration OR healthcare OR abortion '
               'OR "gas prices" OR Iran OR shutdown OR crime OR Medicaid OR jobs)'),
}
# Named outlets, grouped Sacerdote-style. Lean labels are for grouping only.
# msnbc.com -> ms.now (rebrand Nov 15, 2025) and abcnews.go.com -> abcnews.com (0 items before).
OUTLETS = {
    "US_LEFT": ["cnn.com", "ms.now", "nytimes.com", "washingtonpost.com", "npr.org",
                "nbcnews.com", "huffpost.com"],
    "US_RIGHT": ["foxnews.com", "nypost.com", "washingtonexaminer.com", "dailywire.com",
                 "breitbart.com", "newsmax.com"],
    "US_CENTER": ["apnews.com", "reuters.com", "thehill.com", "politico.com", "usatoday.com",
                  "cbsnews.com", "abcnews.com", "axios.com", "newsweek.com"],
    "BUSINESS": ["wsj.com", "bloomberg.com", "cnbc.com", "marketwatch.com"],
    "INTL": ["theguardian.com", "bbc.com", "ft.com", "economist.com"],
}
RSS_DAYS_PER_WINDOW = 2
RSS_CAP = 100                    # a capped 2-day window is re-pulled as two 1-day windows

# ---------------------------------------------------------------- prediction markets
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
POLY_EVENT_SLUG = "balance-of-power-2026-midterms"
POLY_SWEEP = "Democrats Sweep"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXBALANCEPOWERCOMBO"
KALSHI_EVENT = "KXBALANCEPOWERCOMBO-27FEB"
KALSHI_SWEEP = "KXBALANCEPOWERCOMBO-27FEB-DD"

# ---------------------------------------------------------------- equities
# Groups, the sign expected under a Democratic sweep (+1 gains, -1 loses), and whether the
# mechanism runs through a lever Congress controls in 2026 (strong) or one already fixed by
# statute or executive action, so a sweep cannot move it over a veto (weak). Approved by Aditya
# on Oct 7: strong groups form the primary basket, weak groups a separate sub-basket.
GROUPS = {
    "detention":  {"names": ["GEO", "CXW"], "sign": -1, "strength": "strong", "factor": None},
    "defense":    {"names": ["LMT", "NOC", "GD", "HII"], "sign": -1, "strength": "strong", "factor": "ITA"},
    "crypto":     {"names": ["COIN", "HOOD"], "sign": -1, "strength": "strong", "factor": "IBIT"},
    "aca_insurers": {"names": ["OSCR", "CNC", "MOH"], "sign": +1, "strength": "strong", "factor": "XLV"},
    "hospitals":  {"names": ["HCA", "THC", "UHS", "CYH"], "sign": +1, "strength": "strong", "factor": "XLV"},
    "guns":       {"names": ["RGR", "SWBI"], "sign": +1, "strength": "strong", "factor": None},
    "oil_gas":    {"names": ["XOM", "OXY", "DVN"], "sign": -1, "strength": "weak", "factor": "USO"},
    "clean_energy": {"names": ["FSLR", "ENPH", "RUN", "NEE"], "sign": +1, "strength": "weak", "factor": "ICLN"},
    "tariff_retail": {"names": ["DLTR", "FIVE", "BBY"], "sign": +1, "strength": "weak", "factor": "XRT"},
}
AMBIGUOUS_NAMES = {"OSCR": "gained ACA members after the subsidy expiry (2.03M to 2.96M), so its sign is unclear"}
CANDIDATES = {t: g["sign"] for g in GROUPS.values() for t in g["names"]}
NAME_GROUP = {t: k for k, g in GROUPS.items() for t in g["names"]}
BENCHMARKS = ["SPY", "ITA", "XLV", "XLU", "ICLN", "XRT", "IBIT", "USO", "CL=F", "XLE"]

# ---------------------------------------------------------------- pre-registration
# Written Oct 7, 2026, 20:45 ET, BEFORE any sentiment index or test-window test was computed.
PREREG = {
    "primary_index": "C_t: social-only, issue-weighted, source-demeaned daily mean direction score",
    "primary_test": "C_t vs dP_t (Polymarket Democrats Sweep, 16:00 ET snapshots, calendar days)",
    "expected_sign_index_vs_dp": +1,
    "secondary_test": "dC_t vs dP_t",
    "stationarity_rule": "if C_t fails ADF (p>=0.05) or KPSS (p<=0.05), dC_t becomes primary",
    "family_A": "same-day corr + Granger lags 1-3 both directions (7 tests), BH q=0.10 on HAC p",
    "claim_rule": "expected sign AND BH-adjusted HAC p<0.10 AND permutation p<0.10 AND survives "
                  "leave-one-day-out AND survives dropping Sept 10-26",
    "robustness": "Kalshi mid; venue average; 3- and 5-day overlapping changes (NW lags h-1); "
                  "dlogit(p); expanding/equal/pooled weights; trading-day calendar",
    "null_meaning": "daily dP has low SNR (venues agree at r~0.4 daily), so a null means "
                    "'not detectable at a daily horizon in 99 days', not 'sentiment is irrelevant'",
    "basket_primary": "all names in strong groups (equal weight, long +1 minus short -1)",
    "basket_robust": "weak-group sub-basket; sign(gamma)==rationale gate basket",
    "expected_sign_basket_vs_dp": +1,
    "key_topic_rule": "voter-voice social share >= 5%, top 5 to 8, >= 20 docs on >= 90% of days",
    # added Oct 7, 21:28 ET, still before any index or test was computed:
    "index_docs": "voter-voice social docs (comments, self-posts, non-media tweets); all social docs incl. shared media is a sensitivity check",
    "index_construction": "mean s per issue x source x day, demeaned within issue x source over the window; sources (Twitter + 6 subs) combined with fixed window doc-share weights, a source with no docs that day counts at its own mean (0); issue-day with < 20 docs -> NaN, forward-filled <= 2 days",
}
GRANGER_LAGS = (1, 2, 3)
FDR_Q = 0.10
PERM_ALPHA = 0.10
PERM_MIN_SHIFT = 7
EPISODE_DROP = ("2026-09-10", "2026-09-26")
MIN_DOCS_ISSUE_DAY = 5         # deviation: was 20 (see PREREG_DEVIATION)
MIN_DOCS_ISSUE_DAY_PREREG = 20
FFILL_MAX_DAYS = 2
COVERAGE_MIN = 0.90
KEY_SHARE_MIN = 0.05

# ---------------------------------------------------------------- events (ET), verified Oct 7
EVENTS = [
    {"date": "2026-07-24", "label": "Sec. 122 tariffs lapse, Sec. 301 tariffs start", "kind": "policy"},
    {"date": "2026-09-02", "label": "CR to Dec 11 signed (no Oct 1 shutdown)", "kind": "policy"},
    {"date": "2026-09-13", "label": "Polymarket sweep 49.5% to 53.5% (Sun)", "kind": "market"},
    {"date": "2026-09-15", "label": "NYT/Siena D+8; CLARITY cloture fails 49-50", "kind": "poll"},
    {"date": "2026-09-16", "label": "Fed hikes 25bp", "kind": "macro"},
    {"date": "2026-09-21", "label": "Reuters/Ipsos Trump approval 32%", "kind": "poll"},
    {"date": "2026-09-23", "label": "Cook shifts GA, KS, SC toward D", "kind": "poll"},
]

RANDOM_SEED = 7871

# ---------------------------------------------------------------- Q1 issue vocabulary and poll crosswalk
# Written Oct 7, 2026, 21:30 ET, BEFORE any topic output was looked at. The LLM taxonomy step is
# asked to use these issue names where they fit (it may add others); the poll categories map to
# them as below. Poll numbers verified on Oct 7 (sources in REPORT.md).
ISSUE_VOCAB = [
    "Economy & cost of living", "Tariffs & trade", "Taxes, budget & spending", "Health care",
    "Immigration & deportations", "Foreign policy & Iran war", "Democracy & rule of law",
    "Government ethics & corruption", "Trump & the administration", "Political violence & extremism",
    "Crime & policing", "Abortion & social issues", "Guns", "Education", "Energy & environment",
    "Tech & AI",
]
NON_ISSUE = ["Horse race & candidates", "Media & press", "Noise / other"]
POLLS = {
    # Pew Research Center, Jul 6-12, 2026, open-ended, registered voters (Form 1): issue most wanted
    # from House and Senate candidates. Only categories >= 4% are published in the report chart.
    "Pew Jul 2026 (RV)": {"Economy & cost of living": 29, "Government ethics & corruption": 9,
                          "Immigration & deportations": 7, "Health care": 5,
                          "Foreign policy & Iran war": 4, "Trump & the administration": 4},
    # Gallup "most important problem", Sept 2026 (field dates inferred Sept 1-17), all adults.
    "Gallup Sep 2026": {"Economy & cost of living": 31, "Trump & the administration": 30,
                        "Immigration & deportations": 10, "Foreign policy & Iran war": 7},
    # Reuters/Ipsos, Sept 17-20, 2026, registered voters, most important problem (closed list).
    "Reuters/Ipsos Sep 2026 (RV)": {"Economy & cost of living": 22, "Democracy & rule of law": 15,
                                    "Government ethics & corruption": 13, "Foreign policy & Iran war": 8,
                                    "Immigration & deportations": 6, "Political violence & extremism": 6,
                                    "Health care": 4, "Tech & AI": 3},
}
# Gallup's "Government/poor leadership" is mapped to "Trump & the administration" (dissatisfaction
# with the government in power); this is a judgement call, stated in the report.

# ---------------------------------------------------------------- deviation from the pre-registration
# Written Oct 7, 2026, 21:50 ET, after seeing only DOC COUNTS per issue-day (no sentiment index, no test).
PREREG_DEVIATION = {
    "why": "No issue meets '>= 20 docs on >= 90% of days': strict topic assignment gives 127 voter issue "
           "docs a day (plan assumed ~1,000 social docs a day; 40% of docs are HDBSCAN outliers and many sit "
           "in horse-race topics). Best coverage at 20 docs was 69% (Iran war).",
    "key_topics": "salience part of the rule only: voter-voice share >= 5% (strict assignment), top 8",
    "index_assignment": "index docs use the outlier-reduced topic assignment (reduce_outliers, embeddings), "
                        "about 230 voter issue docs a day",
    "min_docs": "issue-day minimum lowered from 20 to 5 docs; each key issue's coverage is reported",
}
