"""Collect news headlines from Google News RSS: named outlets (site: queries) and the US edition.

  python scripts/03_collect_news.py          # 2-day site windows + daily US edition (resumable)
  python scripts/03_collect_news.py topup    # re-pull every 2-day window that hit the 100 cap, by day
"""
import _bootstrap  # noqa: F401
import sys

from src import news

if len(sys.argv) > 1 and sys.argv[1] == "topup":
    print("topup items:", news.topup_capped())
else:
    print("rss items:", news.collect_all())
