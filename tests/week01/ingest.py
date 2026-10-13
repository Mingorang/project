"""Canonical OHLCV ingestion; keep raw data immutable."""
import pandas as pd
REQUIRED=["Date","Open","High","Low","Close"]
def load_candles(path):
    x=pd.read_csv(path)
    if not set(REQUIRED).issubset(x): raise ValueError("Missing OHLC fields")
    x["Date"]=pd.to_datetime(x["Date"],utc=True)
    x=x.sort_values("Date")
    if x.Date.duplicated().any(): raise ValueError("Duplicate candles")
    if not ((x.High>=x[["Open","Close","Low"]].max(axis=1)) & (x.Low<=x[["Open","Close","High"]].min(axis=1))).all(): raise ValueError("Invalid OHLC")
    if (x[REQUIRED[1:]]<=0).any().any(): raise ValueError("Nonpositive prices")
    return x
