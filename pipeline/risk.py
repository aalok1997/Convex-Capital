"""
Cross-asset risk analytics:
  - Pairwise return correlation matrix
  - Historical scenario stress test (computed via per-ticker beta to SPY ×
    historical SPY scenario move). This is a transparent, reproducible
    methodology — all inputs are in the snapshot and the math is documented
    so the user can verify any number.
  - Portfolio-level beta, annualized return, vol, Sharpe, Sortino, max DD,
    historical 1-day VaR(95).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_matrix(closes: Dict[str, pd.Series], lookback: int = 252) -> pd.DataFrame:
    rets = {}
    for tk, s in closes.items():
        if s is None or s.empty:
            continue
        r = np.log(s / s.shift(1)).dropna().tail(lookback)
        rets[tk] = r
    if not rets:
        return pd.DataFrame()
    df = pd.DataFrame(rets).dropna()
    if df.empty:
        return pd.DataFrame()
    return df.corr()


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

def beta_to(close: pd.Series, bench: pd.Series, lookback: int = 252) -> float:
    """OLS beta of `close` returns on `bench` returns over `lookback` days."""
    if close is None or bench is None or close.empty or bench.empty:
        return float("nan")
    r = np.log(close / close.shift(1)).dropna().tail(lookback)
    b = np.log(bench / bench.shift(1)).dropna().tail(lookback)
    aligned = pd.concat([r, b], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if cov[1, 1] == 0:
        return float("nan")
    return float(cov[0, 1] / cov[1, 1])


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

# Historical SPY moves for each scenario, peak-to-trough.
# Sources (publicly verifiable):
#   COVID (2/19/2020 - 3/23/2020):     SPY -33.9%   (Yahoo Finance)
#   2022 Rate Shock (1/3 - 10/12/2022):SPY -25.2%   (Yahoo Finance)
#   GFC (10/9/2007 - 3/9/2009):        SPY -56.5%   (Yahoo Finance)
#   Dot-Com Bust (3/24/2000 - 10/9/2002): SPY -49.1% (Yahoo Finance)
#   Black Monday (10/19/1987 single day): -20.4%    (S&P 500, widely cited)
#   2024 Yen Carry Unwind (8/1 - 8/5/2024): SPY -8.4% (Yahoo Finance)
#
# These are static reference numbers (the historical past doesn't change).
# Portfolio impact = portfolio_beta × scenario_SPY_move (linear approximation).
HISTORICAL_SCENARIOS: List[Tuple[str, float, str]] = [
    ("COVID Crash (Feb-Mar 2020)", -0.339,
     "Rapid 33.9% selloff in SPY over 23 trading days as COVID-19 spread."),
    ("2022 Rate Shock (Jan-Oct)", -0.252,
     "Grinding 9-month decline as the Fed hiked rates aggressively."),
    ("GFC 2008-09", -0.565,
     "Credit crisis bear market; SPY drew down 56.5% from Oct 2007 peak."),
    ("Dot-Com Bust (2000-02)", -0.491,
     "Tech bubble burst; SPY -49.1% peak-to-trough over 30 months."),
    ("Black Monday (Oct 1987)", -0.204,
     "Flash crash; -20.4% on a single trading day."),
    ("Aug 2024 Yen Carry Unwind", -0.084,
     "Yen carry unwind shock; SPY -8.4% in one week."),
]


def stress_tests(portfolio_beta: float, current_nav: float) -> List[dict]:
    out = []
    if math.isnan(portfolio_beta) or current_nav <= 0:
        portfolio_beta = 1.0
    for name, mkt_move, descr in HISTORICAL_SCENARIOS:
        impact = portfolio_beta * mkt_move
        stressed_nav = current_nav * (1 + impact)
        pnl = stressed_nav - current_nav
        out.append({
            "scenario": name,
            "market_move": mkt_move,
            "portfolio_impact": impact,
            "stressed_nav": stressed_nav,
            "pnl": pnl,
            "description": descr,
        })
    return out


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(nav: pd.Series, bench: pd.Series = None) -> dict:
    """
    Annualized return, vol, Sharpe (rf=0), Sortino, max drawdown,
    1-day historical VaR at 95%, and beta to bench.
    """
    if nav is None or len(nav) < 5:
        return {}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {}
    days = (nav.index[-1] - nav.index[0]).days or 1
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    ann_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0.0
    ann_vol = float(rets.std() * math.sqrt(252))
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0
    downside = rets[rets < 0]
    dd_vol = float(downside.std() * math.sqrt(252)) if not downside.empty else 0.0
    sortino = (ann_return / dd_vol) if dd_vol > 0 else 0.0
    cummax = nav.cummax()
    max_dd = float(((nav - cummax) / cummax).min())
    var95 = float(np.percentile(rets, 5))  # left-tail 5%
    metrics = {
        "annualized_return": float(ann_return),
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "var_95_1d": var95,
    }
    if bench is not None and len(bench) > 5:
        metrics["beta_spy"] = beta_to(nav, bench)
    return metrics
