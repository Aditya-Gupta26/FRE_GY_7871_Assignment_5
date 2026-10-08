"""Download daily stock prices (yfinance) and hourly prediction-market prices (Polymarket, Kalshi).

  python scripts/04_get_market_data.py stocks
  python scripts/04_get_market_data.py predmkt
"""
import _bootstrap  # noqa: F401
import json
import sys
import warnings

import pandas as pd
import yfinance as yf

from src import predmkt
from src.config import BENCHMARKS, CANDIDATES, PRICE_END, PRICE_START, RAW, WINDOW_END

warnings.filterwarnings("ignore")
part = sys.argv[1] if len(sys.argv) > 1 else "all"

if part in ("stocks", "all"):
    out = RAW / "market"
    out.mkdir(parents=True, exist_ok=True)
    tickers = list(CANDIDATES) + BENCHMARKS
    daily = yf.download(tickers, start=PRICE_START, end=PRICE_END, interval="1d",
                        auto_adjust=True, progress=False, group_by="column")
    # Yahoo sometimes serves an all-NaN daily bar for the latest session for a few hours after
    # the close (seen on Oct 7, 2026). Fill that day from the last regular-session 60-minute bar
    # (15:30 to 16:00) and record it as provisional, so the next re-pull replaces it.
    last = pd.Timestamp(WINDOW_END.date())
    flag = out / "provisional_close.json"
    if last in daily.index and daily.loc[last, ("Close", "SPY")] != daily.loc[last, ("Close", "SPY")]:
        hb = yf.download(tickers, start=str(last.date()), end=str((last + pd.Timedelta(days=1)).date()),
                         interval="60m", auto_adjust=True, progress=False, prepost=False)["Close"]
        hb = hb[hb.index.tz_convert("America/New_York").strftime("%H:%M") == "15:30"]
        for t in hb.columns:
            daily.loc[last, ("Close", t)] = float(hb[t].iloc[-1])
        flag.write_text(json.dumps({"date": str(last.date()), "source": "last 60m bar (15:30-16:00)",
                                    "tickers": list(hb.columns)}))
        print("filled", last.date(), "from the 15:30 hourly bar (provisional)")
    elif flag.exists():
        flag.unlink()
    daily.to_parquet(out / "daily.parquet")
    print("daily", daily.shape, daily.index.min().date(), daily.index.max().date())
    close = daily["Close"]
    print("NaN count per ticker (non-zero only):")
    print(close.isna().sum()[close.isna().sum() > 0].to_string())

if part in ("predmkt", "all"):
    print(json.dumps(predmkt.polymarket(), indent=1))
    print(json.dumps(predmkt.kalshi(), indent=1))
