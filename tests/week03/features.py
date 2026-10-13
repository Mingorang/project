"""Causal trailing indicators; rows are evaluated at candle close."""
import numpy as np
import pandas as pd
def make_features(c):
    x=c.copy(); close=x.Close.astype(float); ret=np.log(close/close.shift())
    delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi14"]=100-100/(1+rs)
    e12=close.ewm(span=12,adjust=False).mean(); e26=close.ewm(span=26,adjust=False).mean()
    macd=e12-e26; x["macd"]=macd; x["macd_signal"]=macd.ewm(span=9,adjust=False).mean(); x["macd_hist"]=x.macd-x.macd_signal
    x["ret1"]=ret; x["ret5"]=np.log(close/close.shift(5))
    x["vol14"]=ret.rolling(14).std()
    x["bull_pct14"]=(x.Close>x.Open).astype(float).rolling(14).mean()
    x["sma5"]=close.rolling(5).mean(); x["sma12"]=close.rolling(12).mean()
    x["bb_width20"]=4*close.rolling(20).std()/close.rolling(20).mean()
    prev=close.shift(); tr=pd.concat([x.High-x.Low,(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x["atr14"]=tr.rolling(14).mean()
    x["body_ratio"]=(x.Close-x.Open).abs()/(x.High-x.Low).replace(0,np.nan)
    if "Volume" in x: x["relative_volume20"]=x.Volume/x.Volume.rolling(20).mean()
    return x
