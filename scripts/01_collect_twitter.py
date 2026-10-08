"""Collect tweets from twitterapi.io (90-minute windows, one page each, hard budget guard).

Usage:
  python scripts/01_collect_twitter.py --probe     # 6 windows only, check key, format and cost
  python scripts/01_collect_twitter.py             # full run (cached, resumable)
"""
import _bootstrap  # noqa: F401
import argparse
import json

from src.config import CREDITS_PER_USD, TWITTER_BUDGET_USD
from src.twitter import _api_key, account_credits, collect_all, windows

ap = argparse.ArgumentParser()
ap.add_argument("--probe", action="store_true")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

planned = list(windows())
max_usd = len(planned) * 20 * 15 / CREDITS_PER_USD
print(f"{len(planned)} windows planned, at most {len(planned) * 20:,} tweets = at most ${max_usd:.2f} "
      f"(budget cap ${TWITTER_BUDGET_USD:.2f})")
cred = account_credits(_api_key())
have = (cred.get("recharge_credits") or 0) + (cred.get("total_bonus_credits") or 0)
print(f"account balance: {have:,} credits = ${have / CREDITS_PER_USD:.2f}")

res = collect_all(limit=6, workers=2) if args.probe else collect_all(workers=args.workers)
print(json.dumps(res, indent=1))
cred2 = account_credits(_api_key())
have2 = (cred2.get("recharge_credits") or 0) + (cred2.get("total_bonus_credits") or 0)
print(f"balance after: {have2:,} credits = ${have2 / CREDITS_PER_USD:.2f}; "
      f"provider-side spend this run ${(have - have2) / CREDITS_PER_USD:.4f}")
