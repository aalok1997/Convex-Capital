"""
Entry point. Run from the project root:

    python -m pipeline.run                 # live yfinance
    python -m pipeline.run --sample-data   # synthetic fixtures (no network)

Writes the dashboard snapshot to data/snapshot.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import datetime as dt
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .data_source import YFinanceSource, SampleSource, PriceHistory, TickerInfo, NewsItem
from .portfolio import (load_trades, replay, daily_nav_curve, benchmark_curves,
                        PortfolioState)
from .signals import compute_signal
from .volatility import vol_panel
from .risk import (correlation_matrix, beta_to, stress_tests, portfolio_metrics,
                   monte_carlo, factor_stress, correlation_tail_stress,
                   liquidity_stress, drawdown_curve)


BENCHMARKS = ("SPY", "IWM")


def main(argv=None):
    p = argparse.ArgumentParser(description="Convex Capital pipeline")
    p.add_argument("--sample-data", action="store_true",
                   help="Use synthetic fixtures (no network access required)")
    p.add_argument("--trades", default="trades.csv",
                   help="Path to trades.csv (default: trades.csv at project root)")
    p.add_argument("--out", default="docs/data/snapshot.json",
                   help="Output JSON path (default: docs/data/snapshot.json)")
    p.add_argument("--sample-portfolio", action="store_true",
                   help="With --sample-data, seed a demo portfolio for the snapshot")
    args = p.parse_args(argv)

    src = SampleSource() if args.sample_data else YFinanceSource(sleep_seconds=0.15)

    trades_path = args.trades
    if args.sample_data and args.sample_portfolio:
        trades_path = _write_sample_trades()

    if not os.path.exists(trades_path):
        print(f"trades file not found: {trades_path}", file=sys.stderr)
        return 2

    trades = load_trades(trades_path)
    state = replay(trades)

    print(f"Loaded {len(trades)} trades, {len(state.positions)} open positions, "
          f"cash=${state.cash:,.2f}")

    tickers = sorted(state.positions.keys())
    histories: Dict[str, PriceHistory] = {}
    infos: Dict[str, TickerInfo] = {}
    news: Dict[str, List[NewsItem]] = {}

    for tk in tickers:
        print(f"  fetching {tk}...")
        histories[tk] = src.history(tk, period="2y")
        infos[tk] = src.info(tk)
        news[tk] = src.news(tk, limit=5)

    # Benchmarks
    bench_hist: Dict[str, PriceHistory] = {}
    for b in BENCHMARKS:
        bench_hist[b] = src.history(b, period="2y")

    # NAV curve
    start = trades["date"].iloc[0] if not trades.empty else pd.Timestamp.today().normalize()
    end = pd.Timestamp.today().normalize()
    nav = daily_nav_curve(trades, histories, start=start, end=end)
    spy_close = bench_hist["SPY"].df["Close"] if not bench_hist["SPY"].df.empty else pd.Series(dtype=float)
    bench_curves = benchmark_curves(start, end, bench_hist)

    # Holdings panel
    holdings = []
    sector_value = defaultdict(float)
    geo_value = defaultdict(float)
    total_equity = 0.0
    for tk, pos in state.positions.items():
        h = histories.get(tk)
        last = float(h.df["Close"].iloc[-1]) if h and not h.df.empty else 0.0
        mkt_value = pos.shares * last
        unrealized = (last - pos.cost_basis) * pos.shares
        info = infos.get(tk)
        sector = info.sector if info else "Unknown"
        country = info.country if info else "Unknown"
        sector_value[sector] += mkt_value
        geo_value[country] += mkt_value
        total_equity += mkt_value
        holdings.append({
            "ticker": tk,
            "company": info.name if info else tk,
            "sector": sector,
            "country": country,
            "shares": pos.shares,
            "price": last,
            "cost_basis": pos.cost_basis,
            "market_value": mkt_value,
            "unrealized_pnl": unrealized,
        })
    holdings.sort(key=lambda h: -h["market_value"])

    nav_now = state.cash + total_equity

    # Sector / Geographic exposure as % of NAV (cash counted as its own slice
    # so the pies always sum to 100% of total portfolio value).
    if state.cash > 0:
        sector_value["Cash"] += state.cash
        geo_value["Cash"] += state.cash
    sector_exposure = _pct_breakdown(sector_value, nav_now)
    geo_exposure = _pct_breakdown(geo_value, nav_now)

    # Per-ticker signals + vol panels
    signals_panel = []
    vol_rows = []
    for tk in tickers:
        h = histories.get(tk)
        if h is None or h.df.empty:
            continue
        sig = compute_signal(h.df)
        v = vol_panel(h.df["Close"])
        signals_panel.append({
            "ticker": tk,
            "label": sig["label"],
            "composite": sig["composite"],
            "rsi": sig["rsi"],
            "subs": sig["subs"],
        })
        vol_rows.append({
            "ticker": tk,
            "hv20": _none_if_nan(v["hv20"]),
            "garch": _none_if_nan(v["garch"]),
            "expansion": _none_if_nan(v["expansion"]),
            "regime": v["regime"],
            "signal": v["signal"],
        })

    # Correlation matrix (only for tickers with non-empty history)
    closes = {tk: h.df["Close"] for tk, h in histories.items() if not h.df.empty}
    cm = correlation_matrix(closes)

    # Portfolio metrics + stress tests
    metrics = portfolio_metrics(nav, spy_close)
    portfolio_beta = metrics.get("beta_spy", float("nan"))
    if math.isnan(portfolio_beta):
        # Fall back to weighted-average ticker beta
        if total_equity > 0:
            wb = 0.0
            for tk, pos in state.positions.items():
                h = histories.get(tk)
                if h is None or h.df.empty or spy_close.empty:
                    continue
                b = beta_to(h.df["Close"], spy_close)
                if not math.isnan(b):
                    last = float(h.df["Close"].iloc[-1])
                    w = (pos.shares * last) / total_equity
                    wb += w * b
            portfolio_beta = wb if wb else 1.0
        else:
            portfolio_beta = 0.0
    metrics["beta_spy"] = portfolio_beta
    stress = stress_tests(portfolio_beta, nav_now)
    mc = monte_carlo(nav)
    factor = factor_stress(holdings, nav_now)
    closes_for_corr = {tk: h.df["Close"] for tk, h in histories.items() if not h.df.empty}
    corr_tail = correlation_tail_stress(holdings, closes_for_corr, nav_now)
    liquidity = liquidity_stress(holdings, histories)
    dd_series = drawdown_curve(nav)

    # Income / dividend tracker
    income_rows = []
    total_dividend_income = 0.0
    for tk, pos in state.positions.items():
        info = infos.get(tk)
        if info is None or not info.dividend_rate:
            continue
        last = next((h["price"] for h in holdings if h["ticker"] == tk), 0.0)
        annual_inc = pos.shares * info.dividend_rate
        total_dividend_income += annual_inc
        income_rows.append({
            "ticker": tk,
            "shares": pos.shares,
            "div_rate": info.dividend_rate,
            "yield": info.dividend_yield,
            "yld_on_cost": (info.dividend_rate / pos.cost_basis) if pos.cost_basis else None,
            "annual_income": annual_inc,
            "ex_dividend_date": info.ex_dividend_date,
            "last_dividend_amount": info.last_dividend_amount,
        })
    income_rows.sort(key=lambda r: -(r["annual_income"] or 0))
    cash_yield = 0.04  # assumed money-market yield on idle cash
    cash_income = state.cash * cash_yield
    total_income = total_dividend_income + cash_income

    # Fundamentals — split into Valuation and Profitability sub-tabs in the UI
    fundamentals = []
    for tk in tickers:
        info = infos.get(tk)
        if info is None:
            continue
        fundamentals.append({
            "ticker": tk,
            "market_cap": info.market_cap,
            "trailing_pe": info.trailing_pe,
            "forward_pe": info.forward_pe,
            "price_to_sales": info.price_to_sales,
            "price_to_book": info.price_to_book,
            "ev_to_ebitda": info.ev_to_ebitda,
            "profit_margin": info.profit_margin,
            "operating_margin": info.operating_margin,
            "return_on_equity": info.return_on_equity,
            "return_on_assets": info.return_on_assets,
            "revenue_growth": info.revenue_growth,
            "earnings_growth": info.earnings_growth,
        })

    # Trade blotter
    blotter = []
    for _, t in trades.sort_values("date", ascending=False).iterrows():
        if t["action"] == "DEPOSIT" or t["action"] == "WITHDRAW":
            continue
        blotter.append({
            "date": t["date"].strftime("%Y-%m-%d"),
            "ticker": t["ticker"],
            "action": t["action"],
            "shares": float(t["shares"]),
            "price": float(t["price"]),
            "value": float(t["shares"] * t["price"]),
        })

    # News (flatten with ticker tag)
    news_flat = []
    for tk in tickers:
        for n in news.get(tk, []):
            news_flat.append({
                "ticker": tk,
                "title": n.title,
                "publisher": n.publisher,
                "link": n.link,
                "published": n.published,
            })
    news_flat.sort(key=lambda n: n["published"], reverse=True)

    snapshot = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "fund_name": "Convex Capital",
        "tagline": "Lose small. Win big. — Paper-traded portfolio. Hypothetical performance for educational purposes only.",
        "summary": {
            "nav": nav_now,
            "cash": state.cash,
            "equity": total_equity,
            "deposits": state.deposits,
            "total_return": (nav_now / state.deposits - 1) if state.deposits > 0 else 0.0,
            "realized_pnl": state.realized_pnl,
            "open_positions": len(state.positions),
        },
        "metrics": _scrub(metrics),
        "nav_curve": _series_to_records(nav),
        "drawdown_curve": _series_to_records(dd_series),
        "benchmarks": {b: _series_to_records(s) for b, s in bench_curves.items()},
        "holdings": holdings,
        "sector_exposure": sector_exposure,
        "geographic_exposure": geo_exposure,
        "signals": signals_panel,
        "volatility": vol_rows,
        "correlation": _corr_to_dict(cm),
        "stress_tests": stress,
        "monte_carlo": _scrub(mc),
        "factor_stress": _scrub(factor),
        "correlation_tail_risk": _scrub(corr_tail),
        "liquidity": liquidity,
        "income": {
            "total_annual_income": total_income,
            "dividend_income": total_dividend_income,
            "cash_income": cash_income,
            "cash_yield": cash_yield,
            "rows": income_rows,
        },
        "fundamentals": fundamentals,
        "trade_blotter": blotter,
        "news": news_flat[:50],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_scrub(snapshot), f, indent=2, default=_default)

    print(f"Wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import math


def _pct_breakdown(d: dict, total: float) -> List[dict]:
    if total <= 0:
        return []
    items = [{"name": k, "value": v, "pct": v / total} for k, v in d.items() if v > 0]
    items.sort(key=lambda x: -x["pct"])
    return items


def _series_to_records(s: pd.Series) -> List[dict]:
    if s is None or s.empty:
        return []
    out = []
    for d, v in s.items():
        if pd.isna(v):
            continue
        out.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "value": float(v)})
    return out


def _corr_to_dict(cm: pd.DataFrame) -> dict:
    if cm is None or cm.empty:
        return {"tickers": [], "matrix": []}
    tickers = list(cm.columns)
    matrix = []
    for i, ti in enumerate(tickers):
        row = []
        for j, tj in enumerate(tickers):
            v = cm.iloc[i, j]
            row.append(None if pd.isna(v) else round(float(v), 3))
        matrix.append(row)
    return {"tickers": tickers, "matrix": matrix}


def _scrub(o):
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    return o


def _none_if_nan(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _write_sample_trades() -> str:
    """Seed a small demo portfolio so --sample-data --sample-portfolio renders
    a populated dashboard for visual QA."""
    path = "data/sample_trades.csv"
    os.makedirs("data", exist_ok=True)
    rows = [
        ("2026-01-02", "CASH", "DEPOSIT", 1000000, 1.00, "seed"),
        ("2026-01-15", "AAPL", "BUY", 500, 175.00, ""),
        ("2026-01-15", "MSFT", "BUY", 300, 410.00, ""),
        ("2026-02-03", "NVDA", "BUY", 200, 720.00, ""),
        ("2026-02-20", "GOOGL", "BUY", 400, 145.00, ""),
        ("2026-03-05", "BABA", "BUY", 800, 80.00, ""),
        ("2026-03-15", "AAPL", "SELL", 100, 195.00, "trim"),
        ("2026-04-02", "JPM", "BUY", 250, 195.00, ""),
        ("2026-04-20", "JNJ", "BUY", 200, 155.00, ""),
    ]
    df = pd.DataFrame(rows, columns=["date", "ticker", "action", "shares", "price", "notes"])
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    sys.exit(main())
