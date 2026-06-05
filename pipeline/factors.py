"""
Equity factor model — 5 academic risk factors implemented with liquid ETF
proxies. For each portfolio holding we OLS-regress its 252-day log returns
on the factor returns to get factor loadings (betas), idiosyncratic
volatility, and an R-squared "explained-by-factors" score. Aggregating the
loadings by current weight gives a portfolio-level factor exposure.

Factor construction:

    Market   = SPY                         (broad equity market return)
    Size     = IWM  - SPY                  (small minus large)
    Value    = IWD  - IWF                  (value minus growth)
    Momentum = MTUM - SPY                  (momentum minus market)
    Quality  = QUAL - SPY                  (quality minus market)

This is a practitioner's analogue of the Fama-French / Carhart factor zoo,
built from ETF returns instead of academic factor portfolios so all data
comes from one source (yfinance). Loadings are interpretable directly:

    Market beta > 1   = leveraged to broad market
    Size > 0          = small-cap tilt
    Value > 0         = value tilt (vs growth)
    Momentum > 0      = momentum-positive
    Quality > 0       = quality tilt (vs junk)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


FACTOR_ETF_INPUTS: Dict[str, List[str]] = {
    # Factor name → [long-leg ETF, short-leg ETF] (long minus short = factor)
    # Single-element list means the factor is just that ETF's return.
    #
    # Tuned for a small/mid-cap-focused portfolio:
    #   - Value uses Russell 2000 small-cap value/growth (IWN/IWO) instead
    #     of the large-cap IWD/IWF — the right value factor for SMID names.
    #   - Profitability split out from Quality (small caps are bimodal:
    #     wildly profitable cash-cows vs cash-burners).
    #   - Liquidity uses microcap-minus-smallcap (IWC − IWM) — captures the
    #     illiquidity premium that drives a lot of SMID returns.
    #   - Credit (HYG − LQD) catches small-cap sensitivity to credit cycles
    #     even when not directly held on credit.
    "Market":        ["SPY"],
    "Size":          ["IWM", "SPY"],
    "Value (SMID)":  ["IWN", "IWO"],
    "Momentum":      ["MTUM", "SPY"],
    "Quality":       ["QUAL", "SPY"],
    "Profitability": ["COWZ", "SPY"],
    "Liquidity":     ["IWC", "IWM"],
    "Credit":        ["HYG", "LQD"],
}

# All ETFs we need to fetch to build the factor returns. The pipeline pulls
# these once and reuses across all per-ticker regressions.
FACTOR_PROXY_TICKERS: List[str] = sorted({tk for legs in FACTOR_ETF_INPUTS.values()
                                          for tk in legs})


def compute_factor_returns(closes: Dict[str, pd.Series],
                           lookback: int = 252) -> pd.DataFrame:
    """Build the daily factor return series from ETF proxies.

    `closes` must contain a Close series for every ticker listed in
    FACTOR_PROXY_TICKERS. Missing proxies cause that factor to be skipped.
    """
    rets = {}
    for tk, s in closes.items():
        if s is None or s.empty:
            continue
        rets[tk] = np.log(s / s.shift(1)).dropna()
    if not rets:
        return pd.DataFrame()
    rets_df = pd.DataFrame(rets).dropna()
    if rets_df.empty:
        return pd.DataFrame()
    rets_df = rets_df.tail(lookback)
    out = {}
    for factor, legs in FACTOR_ETF_INPUTS.items():
        if any(leg not in rets_df.columns for leg in legs):
            continue
        if len(legs) == 1:
            out[factor] = rets_df[legs[0]]
        else:
            out[factor] = rets_df[legs[0]] - rets_df[legs[1]]
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def regress_loadings(ticker_returns: pd.Series,
                     factor_returns: pd.DataFrame) -> Optional[dict]:
    """OLS regress ticker returns on factor returns.

    Returns a dict of factor loadings plus R-squared and idiosyncratic
    (residual) volatility annualized. None if there isn't enough data or
    the linear system is degenerate.
    """
    if ticker_returns is None or ticker_returns.empty or factor_returns.empty:
        return None
    df = pd.concat([ticker_returns.rename("y"), factor_returns],
                   axis=1, join="inner").dropna()
    if len(df) < 60:
        return None
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    X = np.column_stack([np.ones(len(X)), X])  # alpha intercept
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    idio_vol = float(np.std(resid) * math.sqrt(252))
    factor_names = list(factor_returns.columns)
    out = {
        "alpha_daily": float(beta[0]),
        "alpha_annual": float(beta[0] * 252),
        "loadings": {fn: float(b) for fn, b in zip(factor_names, beta[1:])},
        "r_squared": float(max(0.0, min(1.0, r_squared))),
        "idio_vol_annual": idio_vol,
        "obs": int(len(df)),
    }
    return out


def factor_correlation(factor_returns: pd.DataFrame) -> dict:
    """Pearson correlation matrix between the factor return series.

    Useful sanity-check: well-constructed factors should have low
    cross-correlations. High off-diagonals indicate factor overlap.
    """
    if factor_returns.empty:
        return {"factors": [], "matrix": []}
    corr = factor_returns.corr()
    factors = list(corr.columns)
    matrix = []
    for i, fi in enumerate(factors):
        row = []
        for j, fj in enumerate(factors):
            v = corr.iloc[i, j]
            row.append(None if pd.isna(v) else round(float(v), 3))
        matrix.append(row)
    return {"factors": factors, "matrix": matrix}


def portfolio_factor_exposure(holdings: List[dict],
                              loadings_by_ticker: Dict[str, dict]) -> dict:
    """Weighted-average factor exposure across the portfolio.

    Each holding contributes its loadings × (market_value / total_equity).
    Cash positions are excluded (they have no equity factor exposure).
    """
    if not holdings:
        return {}
    total_equity = sum(h.get("market_value", 0) for h in holdings)
    if total_equity <= 0:
        return {}
    out = {"loadings": {}, "alpha_annual": 0.0, "r_squared_weighted": 0.0}
    for h in holdings:
        tk = h["ticker"]
        l = loadings_by_ticker.get(tk)
        if not l:
            continue
        w = h.get("market_value", 0) / total_equity
        out["alpha_annual"] += w * l.get("alpha_annual", 0.0)
        out["r_squared_weighted"] += w * l.get("r_squared", 0.0)
        for fname, fval in l.get("loadings", {}).items():
            out["loadings"][fname] = out["loadings"].get(fname, 0.0) + w * fval
    return out
