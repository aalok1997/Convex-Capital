"""
Volatility analytics:
  - 20-day realized historical volatility (annualized)
  - GARCH(1,1) one-day-ahead conditional vol forecast (annualized)
  - regime tag (LOW / NORMAL / HIGH / EXTREME) based on the 1-year HV percentile

The "expansion" metric is (GARCH - HV20) / HV20, matching the reference site.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

ANN = math.sqrt(252)


def hv_20(close: pd.Series) -> float:
    """20-day realized vol, annualized, in decimal form (e.g. 0.46 = 46%)."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 21:
        return float("nan")
    return float(rets.tail(20).std() * ANN)


def garch_forecast(close: pd.Series) -> float:
    """
    One-day-ahead conditional volatility from a GARCH(1,1), annualized.
    Returns NaN if the series is too short or the model fails to converge.
    """
    try:
        from arch import arch_model
    except Exception:
        return float("nan")
    rets = np.log(close / close.shift(1)).dropna() * 100  # arch wants %-returns
    if len(rets) < 252:
        return float("nan")
    try:
        am = arch_model(rets, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=1, reindex=False)
        var_pct = float(f.variance.values[-1, 0])  # in %^2
        daily_sigma = math.sqrt(var_pct) / 100.0
        return float(daily_sigma * ANN)
    except Exception:
        return float("nan")


def regime(close: pd.Series, current_hv: float) -> str:
    """Tag the current HV against the past year's distribution."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 252 or math.isnan(current_hv):
        return "NORMAL"
    rolling = rets.rolling(20).std() * ANN
    rolling = rolling.dropna().tail(252)
    if rolling.empty:
        return "NORMAL"
    pct = (rolling < current_hv).mean()
    if pct < 0.25:
        return "LOW"
    if pct < 0.65:
        return "NORMAL"
    if pct < 0.90:
        return "HIGH"
    return "EXTREME"


def vol_panel(close: pd.Series) -> dict:
    h = hv_20(close)
    g = garch_forecast(close)
    if math.isnan(h) or math.isnan(g):
        return {"hv20": h, "garch": g, "expansion": None,
                "regime": regime(close, h), "signal": "WEAK"}
    expansion = (g - h) / h if h > 0 else 0.0
    if expansion > 0.20:
        signal = "SIGNAL"
    elif expansion > 0.10:
        signal = "BASE"
    else:
        signal = "WEAK"
    return {"hv20": h, "garch": g, "expansion": expansion,
            "regime": regime(close, h), "signal": signal}
