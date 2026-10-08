"""Daily stock prices, returns, market-model betas and abnormal returns.

Market model (MacKinlay 1997, JEL 35(1), eq. 3 and eq. 7):
    R_it = alpha_i + beta_i R_mt + e_it ;  AR_it = R_it - alpha_hat_i - beta_hat_i R_mt
In the pre-window the full market model is used (Knight 2006 eq. 1 and 2 do the same).
In the test window the brief's definition is used, AR = r - beta_hat x SPY with alpha = 0,
so a pre-window drift is not extrapolated into the cumulative-return plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import RAW


def load_close() -> pd.DataFrame:
    """Close prices on days SPY traded only (yfinance adds holiday rows when an index like
    ^VIX or a future prints on an exchange holiday: bug 2 in PROGRESS.md)."""
    px = pd.read_parquet(RAW / "market" / "daily.parquet")["Close"]
    px = px[px["SPY"].notna()].copy()
    assert px["SPY"].notna().all()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    return px


def log_ret(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff()


def market_model(r: pd.Series, m: pd.Series) -> dict:
    X = sm.add_constant(m.rename("mkt"))
    ok = r.notna() & m.notna()
    f = sm.OLS(r[ok], X[ok]).fit()
    return {"alpha": float(f.params["const"]), "beta": float(f.params["mkt"]), "N": int(f.nobs)}
