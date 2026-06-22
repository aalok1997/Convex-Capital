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
# Monte Carlo: bootstrap historical daily NAV returns over a forward horizon.
# Honest scenario analysis: future is sampled with replacement from past, so
# the shape of the distribution (fat tails included) is preserved without
# assuming normality.
# ---------------------------------------------------------------------------

def monte_carlo(nav: pd.Series,
                horizon_days: int = 21,
                n_paths: int = 10_000,
                seed: int = 42) -> dict:
    if nav is None or len(nav) < 30:
        return {}
    rets = nav.pct_change().dropna().values
    if len(rets) < 20:
        return {}
    rng = np.random.default_rng(seed)
    sims = rng.choice(rets, size=(n_paths, horizon_days), replace=True)
    # Compound each path's daily returns to a terminal-return multiple.
    path_mult = np.cumprod(1.0 + sims, axis=1)
    terminal = path_mult[:, -1] - 1.0
    # Worst intra-horizon drawdown per path.
    running_max = np.maximum.accumulate(path_mult, axis=1)
    dd = (path_mult - running_max) / running_max
    max_dd = dd.min(axis=1)
    current_nav = float(nav.iloc[-1])
    pct = lambda a, p: float(np.percentile(a, p))
    return {
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "current_nav": current_nav,
        "terminal_return": {
            "p5": pct(terminal, 5), "p25": pct(terminal, 25),
            "p50": pct(terminal, 50), "p75": pct(terminal, 75),
            "p95": pct(terminal, 95),
            "prob_negative": float((terminal < 0).mean()),
        },
        "terminal_nav": {
            "p5": current_nav * (1 + pct(terminal, 5)),
            "p50": current_nav * (1 + pct(terminal, 50)),
            "p95": current_nav * (1 + pct(terminal, 95)),
        },
        "max_drawdown": {
            "p5": pct(max_dd, 5), "p50": pct(max_dd, 50),
            "p95": pct(max_dd, 95),
        },
        "method": "Bootstrap of historical daily NAV returns. "
                  "10,000 paths, 21-trading-day horizon (~1 month).",
    }


# ---------------------------------------------------------------------------
# Factor stress: pre-defined sector + geography shocks applied to current
# positions. Transparent and documented — every shock is in the snapshot.
# ---------------------------------------------------------------------------

# (scenario_name, description, {sector: shock_pct}, {country: shock_pct})
FACTOR_SCENARIOS: List[dict] = [
    {
        "name": "Tech Selloff -15%",
        "description": "Concentrated drawdown in Tech and Communication Services.",
        "sector_shocks": {"Technology": -0.15, "Communication Services": -0.10},
        "country_shocks": {},
    },
    {
        "name": "Rate Shock (+100bp)",
        "description": "Long-duration growth and rate-sensitive sectors hit; banks benefit.",
        "sector_shocks": {"Technology": -0.08, "Consumer Cyclical": -0.10,
                          "Financial Services": +0.05, "Real Estate": -0.12,
                          "Utilities": -0.06},
        "country_shocks": {},
    },
    {
        "name": "Energy Spike (+30%)",
        "description": "Oil shock — producers rally, downstream consumers pinched.",
        "sector_shocks": {"Energy": +0.30, "Industrials": -0.05,
                          "Consumer Cyclical": -0.07, "Consumer Defensive": -0.03},
        "country_shocks": {},
    },
    {
        "name": "Recession (broad -20%)",
        "description": "Cyclicals and credit-sensitive names down hard; defensives flat.",
        "sector_shocks": {"Technology": -0.15, "Consumer Cyclical": -0.22,
                          "Financial Services": -0.18, "Industrials": -0.18,
                          "Energy": -0.20, "Materials": -0.20,
                          "Communication Services": -0.12, "Healthcare": -0.04,
                          "Consumer Defensive": -0.02, "Utilities": 0.0,
                          "Real Estate": -0.15},
        "country_shocks": {},
    },
    {
        "name": "China Decoupling",
        "description": "China-listed and China-revenue names take a 25% hit.",
        "sector_shocks": {},
        "country_shocks": {"China": -0.25, "Hong Kong": -0.20},
    },
    {
        "name": "Risk-On Rally",
        "description": "Growth and cyclicals rip; defensives lag.",
        "sector_shocks": {"Technology": +0.12, "Consumer Cyclical": +0.10,
                          "Financial Services": +0.06, "Communication Services": +0.08,
                          "Healthcare": +0.02, "Consumer Defensive": +0.01,
                          "Utilities": -0.01},
        "country_shocks": {},
    },
]


def factor_stress(holdings: List[dict], current_nav: float) -> List[dict]:
    """
    Apply pre-defined factor scenarios to current holdings.

    Each position is shocked by its sector AND country shock (additively when
    both apply). Cash is untouched.
    """
    if not holdings or current_nav <= 0:
        return []
    equity = sum(h.get("market_value", 0) for h in holdings)
    out = []
    for scen in FACTOR_SCENARIOS:
        ssh = scen["sector_shocks"]
        csh = scen["country_shocks"]
        pnl = 0.0
        for h in holdings:
            shock = ssh.get(h.get("sector"), 0.0) + csh.get(h.get("country"), 0.0)
            pnl += h.get("market_value", 0) * shock
        impact_pct = pnl / current_nav  # impact as fraction of NAV (cash unchanged)
        out.append({
            "scenario": scen["name"],
            "description": scen["description"],
            "shocks": {**{f"sector:{k}": v for k, v in ssh.items()},
                       **{f"country:{k}": v for k, v in csh.items()}},
            "pnl": float(pnl),
            "stressed_nav": float(current_nav + pnl),
            "portfolio_impact": float(impact_pct),
            "equity_at_risk_pct": float(equity / current_nav) if current_nav else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Correlation tail risk: portfolio volatility under normal correlation vs.
# stressed correlation where every cross-correlation collapses to 1
# (the "diversification fails when you need it most" scenario).
# ---------------------------------------------------------------------------

def risk_budget(holdings: List[dict],
                closes: Dict[str, pd.Series],
                current_nav: float,
                lookback: int = 252) -> List[dict]:
    """Marginal and total risk contribution per position.

    Decomposes portfolio variance into per-position contributions:
        TRC_i = w_i × (Σ × w)_i / σ_p
    where Σ is the asset covariance matrix, w the weight vector, and σ_p
    portfolio volatility. Sum of TRC across positions = σ_p (by definition).
    The risk-share column (TRC_i / σ_p) is what tells you which position is
    driving overall risk — often very different from raw market-value weight
    when a few high-vol names dominate.
    """
    if not holdings or current_nav <= 0:
        return []
    weights = {}
    for h in holdings:
        weights[h["ticker"]] = h.get("market_value", 0) / current_nav
    tickers = [t for t in weights if t in closes and not closes[t].empty]
    if len(tickers) < 2:
        return []
    rets_df = pd.DataFrame({t: np.log(closes[t] / closes[t].shift(1))
                            for t in tickers}).dropna().tail(lookback)
    if rets_df.empty:
        return []
    w = np.array([weights[t] for t in tickers])
    cov = rets_df.cov().values * 252  # annualize covariance
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return []
    port_vol = math.sqrt(port_var)
    marginal = cov @ w / port_vol         # marginal contribution to vol
    total_contrib = w * marginal          # total contribution to vol
    out = []
    for i, tk in enumerate(tickers):
        out.append({
            "ticker": tk,
            "weight": float(w[i]),
            "marginal_risk": float(marginal[i]),
            "total_risk_contribution": float(total_contrib[i]),
            "risk_share": float(total_contrib[i] / port_vol) if port_vol > 0 else 0.0,
        })
    out.sort(key=lambda r: -r["risk_share"])
    return out


def correlation_tail_stress(holdings: List[dict],
                            closes: Dict[str, pd.Series],
                            current_nav: float,
                            lookback: int = 252) -> dict:
    if not holdings or current_nav <= 0:
        return {}
    weights = {}
    for h in holdings:
        weights[h["ticker"]] = h.get("market_value", 0) / current_nav
    tickers = [t for t in weights if t in closes and not closes[t].empty]
    if len(tickers) < 2:
        return {}
    rets_df = pd.DataFrame({t: np.log(closes[t] / closes[t].shift(1)) for t in tickers}).dropna().tail(lookback)
    if rets_df.empty:
        return {}
    w = np.array([weights[t] for t in tickers])
    vols = rets_df.std().values * math.sqrt(252)
    corr = rets_df.corr().values
    # Portfolio annual vol under current correlation
    cov = np.outer(vols, vols) * corr
    port_vol_current = float(math.sqrt(max(w @ cov @ w, 0.0)))
    # Stressed: all off-diagonal correlations forced to 1.0
    corr_stressed = np.ones_like(corr)
    cov_stressed = np.outer(vols, vols) * corr_stressed
    port_vol_stressed = float(math.sqrt(max(w @ cov_stressed @ w, 0.0)))
    # 3-sigma shock under each regime, expressed as dollar P&L on NAV
    equity_share = sum(weights.values())
    pnl_3sigma_current = -current_nav * equity_share * port_vol_current / math.sqrt(252) * 3
    pnl_3sigma_stressed = -current_nav * equity_share * port_vol_stressed / math.sqrt(252) * 3
    return {
        "current_correlation_avg": float(np.mean(corr[np.triu_indices_from(corr, k=1)])) if len(tickers) > 1 else 0.0,
        "port_vol_annual_current": port_vol_current,
        "port_vol_annual_stressed_corr1": port_vol_stressed,
        "vol_expansion_pct": float((port_vol_stressed - port_vol_current) / port_vol_current) if port_vol_current > 0 else 0.0,
        "pnl_3sigma_1d_current": float(pnl_3sigma_current),
        "pnl_3sigma_1d_stressed": float(pnl_3sigma_stressed),
        "n_positions": len(tickers),
        "method": "Portfolio variance recomputed with all cross-correlations forced "
                  "to 1.0, holding individual vols constant. 3σ 1-day P&L uses "
                  "stressed annualized vol / √252.",
    }


# ---------------------------------------------------------------------------
# Liquidity: days to exit each position at 10% participation of ADV.
# ---------------------------------------------------------------------------

PARTICIPATION_RATE = 0.10  # max % of daily volume we'd take without moving the market


def liquidity_stress(holdings: List[dict],
                     histories: Dict[str, 'PriceHistory']) -> List[dict]:
    out = []
    for h in holdings:
        tk = h["ticker"]
        ph = histories.get(tk)
        if ph is None or ph.df.empty:
            continue
        vol20 = float(ph.df["Volume"].tail(20).mean()) if "Volume" in ph.df.columns else 0.0
        last = float(ph.df["Close"].iloc[-1]) if not ph.df.empty else 0.0
        adv_dollar = vol20 * last
        if adv_dollar <= 0:
            continue
        share_capacity = vol20 * PARTICIPATION_RATE
        days_to_exit = h["shares"] / share_capacity if share_capacity > 0 else float("inf")
        # Classification matches mispriced-style tags.
        if days_to_exit < 0.25:
            tag = "DEEP"
        elif days_to_exit < 1.0:
            tag = "LIQUID"
        elif days_to_exit < 3.0:
            tag = "OK"
        elif days_to_exit < 10.0:
            tag = "THIN"
        else:
            tag = "ILLIQUID"
        out.append({
            "ticker": tk,
            "shares": float(h["shares"]),
            "position_value": float(h.get("market_value", 0)),
            "avg_daily_volume_20d": float(vol20),
            "avg_daily_dollar_volume": float(adv_dollar),
            "days_to_exit": float(days_to_exit),
            "liquidity_tag": tag,
        })
    out.sort(key=lambda r: -r["days_to_exit"])
    return out


# ---------------------------------------------------------------------------
# Drawdown curve from NAV.
# ---------------------------------------------------------------------------

def drawdown_curve(nav: pd.Series) -> pd.Series:
    if nav is None or nav.empty:
        return pd.Series(dtype=float)
    cummax = nav.cummax()
    return (nav - cummax) / cummax


# ---------------------------------------------------------------------------
# Synthetic portfolio history
#
# A freshly-launched portfolio doesn't have enough live NAV history for
# statistical metrics. We backfill by reconstructing daily portfolio returns
# from each holding's actual price history, weighted by *current* portfolio
# weights. The interpretation: "if I had held today's exact weights for the
# last 252 trading days, what would the daily return series have looked like?"
# This gives meaningful Sharpe, vol, Sortino, max DD, and Monte Carlo inputs
# from the very first day of the live portfolio.
# ---------------------------------------------------------------------------

def synthetic_portfolio_returns(holdings: List[dict],
                                closes: Dict[str, pd.Series],
                                lookback: int = 252) -> pd.Series:
    """Daily portfolio log-returns over `lookback` days using current weights.

    Each holding contributes (weight × ticker_log_return). Cash is excluded
    from the weighting since it contributes ~zero return. The dataframe is
    forward-filled to a common business-day index so mixed listing calendars
    (e.g. US + Kazakhstan + HK) don't drop the join to nothing.
    """
    if not holdings:
        return pd.Series(dtype=float)
    total_equity = sum(h.get("market_value", 0) for h in holdings)
    if total_equity <= 0:
        return pd.Series(dtype=float)
    series = {}
    for h in holdings:
        tk = h["ticker"]
        s = closes.get(tk)
        if s is None or s.empty:
            continue
        r = np.log(s / s.shift(1)).dropna()
        series[tk] = r.tail(lookback * 2)  # generous tail, we'll align later
    if not series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(series)
    # Align on a union business-day index, ffill (carry forward last return = 0)
    # so missing days don't drop the whole row when one ticker doesn't trade.
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B"))
    df = df.fillna(0.0).tail(lookback)
    weights = {h["ticker"]: h.get("market_value", 0) / total_equity for h in holdings}
    cols = [t for t in df.columns if t in weights]
    if not cols:
        return pd.Series(dtype=float)
    w = np.array([weights[t] for t in cols])
    port_rets = (df[cols].values @ w)
    return pd.Series(port_rets, index=df.index, name="portfolio_returns")


def _downside_deviation(rets: pd.Series, mar_daily: float = 0.0,
                        periods_per_year: int = 252) -> float:
    """Proper Sortino downside deviation per Frank Sortino's original spec.

    DD = sqrt( mean( min(R - MAR, 0)^2 ) ) * sqrt(periods_per_year)

    Crucially, the mean is over ALL observations (not just below-target ones)
    — that's what distinguishes downside deviation from "std of negatives."
    Above-target periods contribute zero, but they're still counted in N.
    """
    shortfall = (rets - mar_daily).clip(upper=0)
    if len(shortfall) == 0:
        return 0.0
    rms = math.sqrt(float((shortfall ** 2).mean()))
    return rms * math.sqrt(periods_per_year)


def _information_ratio(port_rets: pd.Series, bench_rets: pd.Series) -> dict:
    """IR = annualized active return / tracking error.

    Tracking error = std(R_p - R_b) annualized.
    Active return  = mean(R_p - R_b) annualized.
    The metric tells you risk-adjusted active performance against a benchmark.
    """
    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {}
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    mean_active = float(active.mean())
    ann_active = (1 + mean_active) ** 252 - 1
    te = float(active.std() * math.sqrt(252))
    if te <= 0:
        return {"information_ratio": 0.0, "tracking_error": 0.0,
                "annualized_active_return": ann_active}
    return {"information_ratio": ann_active / te,
            "tracking_error": te,
            "annualized_active_return": ann_active}


def _up_down_capture(port_rets: pd.Series, bench_rets: pd.Series) -> dict:
    """Capture ratios in up- and down-benchmark periods.

    Up Capture   = (geometric product of port returns when bench up) /
                   (geometric product of bench returns when bench up)
    Down Capture = same for bench-down periods.
    A skilled long-only manager wants Up > 1 and Down < 1.
    """
    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {}
    p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    up_mask = b > 0
    down_mask = b < 0
    out = {}
    if up_mask.any():
        bp = float((1 + b[up_mask]).prod() - 1)
        pp = float((1 + p[up_mask]).prod() - 1)
        out["up_capture"] = (pp / bp) if bp != 0 else None
    if down_mask.any():
        bp = float((1 + b[down_mask]).prod() - 1)
        pp = float((1 + p[down_mask]).prod() - 1)
        out["down_capture"] = (pp / bp) if bp != 0 else None
    return out


def synthetic_portfolio_metrics(holdings: List[dict],
                                closes: Dict[str, pd.Series],
                                bench_close: pd.Series = None,
                                lookback: int = 252,
                                risk_free_rate: float = 0.0) -> dict:
    """Annualized vol, Sharpe, Sortino, max DD, VaR computed from a synthetic
    portfolio return series. Beta vs SPY computed by regressing synthetic
    portfolio returns on SPY returns over the same window.

    risk_free_rate: annualized risk-free rate as a fraction (e.g. 0.045 for
    4.5%). Sharpe and Sortino use (return - rf) in the numerator. Default 0
    so callers that don't supply a rate fall back to the simplified formula.
    """
    rets = synthetic_portfolio_returns(holdings, closes, lookback)
    if rets is None or len(rets) < 30:
        return {}
    rets_clean = rets[rets != 0.0]  # exclude ffilled non-trading days from stats
    if len(rets_clean) < 30:
        rets_clean = rets
    ann_vol = float(rets_clean.std() * math.sqrt(252))
    mean_daily = float(rets_clean.mean())
    ann_return = float((1 + mean_daily) ** 252 - 1)
    excess_return = ann_return - risk_free_rate

    # Sharpe: classic (R - RF) / sigma
    sharpe = (excess_return / ann_vol) if ann_vol > 0 else 0.0

    # Sortino: PROPER Frank Sortino formula. MAR = daily-equivalent of RF.
    # (R - MAR) / DD, where DD = RMS-of-shortfalls × √252.
    mar_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    dd_vol = _downside_deviation(rets_clean, mar_daily=mar_daily)
    sortino = (excess_return / dd_vol) if dd_vol > 0 else 0.0

    # Drawdown + VaR
    synthetic_nav = (1 + rets).cumprod()
    cummax = synthetic_nav.cummax()
    max_dd = float(((synthetic_nav - cummax) / cummax).min())
    var95 = float(np.percentile(rets_clean, 5))

    # Calmar: annualized return / |max drawdown|.
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    out = {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "var_95_1d": var95,
        "risk_free_rate": float(risk_free_rate),
        "method": "Synthetic: current weights × each holding's 252-day return history. "
                  "Sortino uses Frank Sortino downside deviation (RMS of shortfalls vs MAR=RF).",
    }

    # Benchmark-relative metrics — these require a benchmark series.
    # Caller chooses which benchmark (IWM for SMID funds, SPY for large-cap).
    if bench_close is not None and not bench_close.empty:
        bench_rets = np.log(bench_close / bench_close.shift(1)).dropna().tail(lookback)
        aligned = pd.concat([rets, bench_rets], axis=1, join="inner").dropna()
        aligned = aligned[(aligned.iloc[:, 0] != 0) & (aligned.iloc[:, 1] != 0)]
        if len(aligned) >= 30:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if cov[1, 1] > 0:
                beta = float(cov[0, 1] / cov[1, 1])
                out["beta_benchmark"] = beta

                # Benchmark annualized return for CAPM and Jensen's Alpha
                bench_mean = float(aligned.iloc[:, 1].mean())
                bench_ann_return = (1 + bench_mean) ** 252 - 1
                out["benchmark_annualized_return"] = bench_ann_return

                # Treynor ratio: (R - RF) / Beta — excess return per unit of
                # systematic risk
                if beta != 0:
                    out["treynor"] = excess_return / beta

                # Jensen's Alpha (CAPM-based): R_p - [RF + β(R_m - RF)]
                expected = risk_free_rate + beta * (bench_ann_return - risk_free_rate)
                out["jensens_alpha"] = float(ann_return - expected)

        # Information Ratio + Tracking Error
        ir = _information_ratio(rets, bench_rets)
        out.update(ir)
        cap = _up_down_capture(rets, bench_rets)
        out.update(cap)
    return out


def monte_carlo_from_returns(returns: pd.Series,
                             current_nav: float,
                             horizon_days: int = 21,
                             n_paths: int = 10_000,
                             seed: int = 42) -> dict:
    """Same Monte Carlo as monte_carlo() but operating on an arbitrary daily
    return series (e.g. synthetic portfolio returns)."""
    if returns is None or len(returns) < 30 or current_nav <= 0:
        return {}
    rets = returns[returns != 0.0].values
    if len(rets) < 30:
        rets = returns.values
    rng = np.random.default_rng(seed)
    sims = rng.choice(rets, size=(n_paths, horizon_days), replace=True)
    path_mult = np.cumprod(1.0 + sims, axis=1)
    terminal = path_mult[:, -1] - 1.0
    running_max = np.maximum.accumulate(path_mult, axis=1)
    dd = (path_mult - running_max) / running_max
    max_dd = dd.min(axis=1)
    pct = lambda a, p: float(np.percentile(a, p))
    return {
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "current_nav": current_nav,
        "terminal_return": {
            "p5": pct(terminal, 5), "p25": pct(terminal, 25),
            "p50": pct(terminal, 50), "p75": pct(terminal, 75),
            "p95": pct(terminal, 95),
            "prob_negative": float((terminal < 0).mean()),
        },
        "terminal_nav": {
            "p5": current_nav * (1 + pct(terminal, 5)),
            "p50": current_nav * (1 + pct(terminal, 50)),
            "p95": current_nav * (1 + pct(terminal, 95)),
        },
        "max_drawdown": {
            "p5": pct(max_dd, 5), "p50": pct(max_dd, 50),
            "p95": pct(max_dd, 95),
        },
        "method": "Bootstrap of synthetic portfolio daily returns "
                  "(current weights × each holding's 252-day history). "
                  "10,000 paths, 21-trading-day horizon (~1 month).",
    }


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(nav: pd.Series, bench: pd.Series = None,
                      risk_free_rate: float = 0.0) -> dict:
    """
    Annualized return, vol, Sharpe (excess of risk_free_rate), Sortino,
    max drawdown, 1-day historical VaR at 95%, and beta to bench.

    risk_free_rate: annualized risk-free rate as a fraction (e.g. 0.045 for
    4.5%). Default 0 for backwards compatibility.
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
    excess_return = ann_return - risk_free_rate
    sharpe = (excess_return / ann_vol) if ann_vol > 0 else 0.0
    # Proper Sortino downside deviation: RMS of shortfalls against MAR=RF.
    mar_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    dd_vol = _downside_deviation(rets, mar_daily=mar_daily)
    sortino = (excess_return / dd_vol) if dd_vol > 0 else 0.0
    cummax = nav.cummax()
    max_dd = float(((nav - cummax) / cummax).min())
    var95 = float(np.percentile(rets, 5))  # left-tail 5%
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    metrics = {
        "annualized_return": float(ann_return),
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "var_95_1d": var95,
        "risk_free_rate": float(risk_free_rate),
    }
    if bench is not None and len(bench) > 5:
        beta = beta_to(nav, bench)
        metrics["beta_benchmark"] = beta
        # Bench-relative metrics
        bench_rets = bench.pct_change().dropna()
        bench_ann_return = (1 + float(bench_rets.mean())) ** 252 - 1
        metrics["benchmark_annualized_return"] = float(bench_ann_return)
        if beta and not math.isnan(beta) and beta != 0:
            metrics["treynor"] = float(excess_return / beta)
            expected = risk_free_rate + beta * (bench_ann_return - risk_free_rate)
            metrics["jensens_alpha"] = float(ann_return - expected)
        ir = _information_ratio(rets, bench_rets)
        metrics.update(ir)
        cap = _up_down_capture(rets, bench_rets)
        metrics.update(cap)
    return metrics
