"""
Technical-signal composite per ticker.

Six 1/0/-1 sub-signals are summed into a [-3, +3] composite, then bucketed:

    >= +1.5  STRONG BUY
    >=  +1   BUY
    >=  +0.5 LEAN BULL
    > -0.5   NEUTRAL
    <= -0.5  LEAN BEAR
    <= -1    SELL
    <= -1.5  STRONG SELL

Each sub-signal weighs 0.5 in the composite (matches the reference site's
+2.00 / +1.50 / +1.00 / +0.50 / 0.00 / -0.50 ... ladder).

Sub-signals (0/+0.5/-0.5):
    rsi    : +0.5 if 45<=RSI<70, -0.5 if RSI>75 or RSI<30, else 0
    macd   : sign(MACD - signal_line) * 0.5
    trend  : +0.5 if price > 50DMA > 200DMA, -0.5 if reverse, else 0
    bb     : +0.5 if close > middle band, -0.5 if < middle band, 0 if outside ±2σ
    vol    : +0.5 if last-5d avg vol > 30d avg vol, -0.5 if much lower
    range  : +0.5 if close in upper third of 20D range, -0.5 if lower third
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def compute_signal(df: pd.DataFrame) -> dict:
    """
    df has columns Open High Low Close Volume with a DatetimeIndex.
    Returns a dict with each sub-signal score (+/-0.5/0), the composite score,
    and a label.
    """
    close = df["Close"]
    if len(close) < 60:
        return _empty()
    last = float(close.iloc[-1])

    # RSI
    r = float(rsi(close).iloc[-1])
    if 45 <= r < 70:
        s_rsi = 0.5
    elif r > 75 or r < 30:
        s_rsi = -0.5
    else:
        s_rsi = 0.0

    # MACD
    line, sig = macd(close)
    s_macd = 0.5 if line.iloc[-1] > sig.iloc[-1] else -0.5

    # Trend (50/200 DMA)
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    if pd.notna(sma50) and pd.notna(sma200):
        if last > sma50 > sma200:
            s_trend = 0.5
        elif last < sma50 < sma200:
            s_trend = -0.5
        else:
            s_trend = 0.0
    elif pd.notna(sma50):
        s_trend = 0.5 if last > sma50 else -0.5
    else:
        s_trend = 0.0

    # Bollinger
    mid, upper, lower = bollinger(close)
    mid_v = mid.iloc[-1]
    if pd.notna(mid_v):
        if last > upper.iloc[-1] or last < lower.iloc[-1]:
            s_bb = 0.0  # extreme — treat as exhaustion, neutral
        elif last > mid_v:
            s_bb = 0.5
        else:
            s_bb = -0.5
    else:
        s_bb = 0.0

    # Volume regime
    vol = df["Volume"]
    avg5 = vol.tail(5).mean()
    avg30 = vol.tail(30).mean()
    if avg30 and not np.isnan(avg30):
        ratio = avg5 / avg30
        if ratio > 1.2:
            s_vol = 0.5
        elif ratio < 0.7:
            s_vol = -0.5
        else:
            s_vol = 0.0
    else:
        s_vol = 0.0

    # Range position over last 20 days
    hi20 = df["High"].tail(20).max()
    lo20 = df["Low"].tail(20).min()
    if hi20 > lo20:
        pos = (last - lo20) / (hi20 - lo20)
        if pos > 0.66:
            s_range = 0.5
        elif pos < 0.33:
            s_range = -0.5
        else:
            s_range = 0.0
    else:
        s_range = 0.0

    composite = round(s_rsi + s_macd + s_trend + s_bb + s_vol + s_range, 2)
    label = _label_for(composite)

    return {
        "composite": composite,
        "label": label,
        "rsi": round(r, 1),
        "subs": {
            "rsi": s_rsi, "macd": s_macd, "trend": s_trend,
            "bb": s_bb, "vol": s_vol, "range": s_range,
        },
    }


def _empty():
    return {"composite": 0.0, "label": "NEUTRAL", "rsi": None,
            "subs": {"rsi": 0, "macd": 0, "trend": 0, "bb": 0, "vol": 0, "range": 0}}


def _label_for(c: float) -> str:
    if c >= 1.75:
        return "STRONG BUY"
    if c >= 1.0:
        return "BUY"
    if c >= 0.5:
        return "LEAN BULL"
    if c <= -1.75:
        return "STRONG SELL"
    if c <= -1.0:
        return "SELL"
    if c <= -0.5:
        return "LEAN BEAR"
    return "NEUTRAL"
