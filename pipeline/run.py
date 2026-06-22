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

from .data_source import (YFinanceSource, SampleSource, PriceHistory,
                          TickerInfo, NewsItem)
from .portfolio import (load_trades, replay, daily_nav_curve, benchmark_curves,
                        PortfolioState)
from .signals import compute_signal
from .volatility import vol_panel
from .risk import (correlation_matrix, beta_to, stress_tests, portfolio_metrics,
                   monte_carlo, factor_stress, correlation_tail_stress,
                   liquidity_stress, drawdown_curve,
                   synthetic_portfolio_returns, synthetic_portfolio_metrics,
                   monte_carlo_from_returns, risk_budget)
from .factors import (FACTOR_PROXY_TICKERS, compute_factor_returns,
                      regress_loadings, factor_correlation,
                      residual_correlation, portfolio_factor_exposure)


BENCHMARKS = ("SPY", "IWM")
RISK_FREE_TICKER = "^TNX"  # CBOE 10-year Treasury yield index


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

    # Hard safety: refuse to produce a snapshot that has negative cash. The
    # paper-traded fund can't be overdrawn — if the staged trades.csv would
    # cause it, the user is supposed to fix the trade ledger, not silently
    # let the dashboard show negative cash.
    if state.cash < -1.0:  # tiny tolerance for sub-dollar rounding artifacts
        print(f"ERROR: Trades would result in NEGATIVE cash (${state.cash:,.2f}).",
              file=sys.stderr)
        print(f"  Total deposits: ${state.deposits:,.2f}", file=sys.stderr)
        print(f"  Open positions: {len(state.positions)}", file=sys.stderr)
        print(f"  Adjust trades.csv so total purchases <= deposits, then re-run.",
              file=sys.stderr)
        return 3

    print(f"Loaded {len(trades)} trades, {len(state.positions)} open positions, "
          f"cash=${state.cash:,.2f}")

    tickers = sorted(state.positions.keys())
    histories: Dict[str, PriceHistory] = {}
    infos: Dict[str, TickerInfo] = {}
    news: Dict[str, List[NewsItem]] = {}

    filings: Dict[str, List[NewsItem]] = {}
    analyst: Dict[str, List[NewsItem]] = {}
    insider: Dict[str, List[NewsItem]] = {}
    earnings_dates: List[dict] = []
    for tk in tickers:
        print(f"  fetching {tk}...")
        histories[tk] = src.history(tk, period="2y")
        infos[tk] = src.info(tk)
        news[tk] = src.news(tk, limit=8)
        filings[tk] = src.sec_filings(tk, since_days=120, limit=6)
        analyst[tk] = src.analyst_actions(tk, since_days=90, limit=4)
        insider[tk] = src.sec_insider_trades(tk, since_days=60, limit=6)
        nxt = src.next_earnings(tk)
        if nxt:
            earnings_dates.append(nxt)

    # Benchmarks
    bench_hist: Dict[str, PriceHistory] = {}
    for b in BENCHMARKS:
        bench_hist[b] = src.history(b, period="2y")

    # Risk-free rate from 10-year Treasury yield (^TNX). Returned by yfinance
    # as a percent value (4.487 = 4.487%). We use the 10y for two reasons:
    # (1) duration matches a long-only equity portfolio's intended holding
    # horizon better than the 3-month bill, (2) it's the convention most
    # equity-strategy practitioners and CAPM frameworks use. Converted to a
    # fraction so Sharpe/Sortino can subtract it from annualized return.
    risk_free_rate = 0.0
    try:
        tnx_hist = src.history(RISK_FREE_TICKER, period="10d")
        if tnx_hist is not None and not tnx_hist.df.empty:
            last = float(tnx_hist.df["Close"].dropna().iloc[-1])
            risk_free_rate = last / 100.0  # 4.487 → 0.04487
    except Exception:
        pass

    # Factor ETF proxies (Market/Size/Value/Momentum/Quality). Reuse what
    # we already have from BENCHMARKS to skip duplicate API calls.
    factor_proxy_hist: Dict[str, PriceHistory] = {}
    for ftk in FACTOR_PROXY_TICKERS:
        if ftk in bench_hist:
            factor_proxy_hist[ftk] = bench_hist[ftk]
        else:
            print(f"  fetching factor proxy {ftk}...")
            factor_proxy_hist[ftk] = src.history(ftk, period="2y")

    # Live quote policy (matches the fund's pricing rule):
    #   PRE-market  → backfill any missing PRIOR days; do not advance to today
    #   OPEN        → live intraday tick (fast_info.last_price)
    #   AFTER-hours → today's official regular-session close (regularMarketPrice)
    #   CLOSED      → most recent regular-session close
    #
    # In all sessions we query regularMarketPrice (or fast_info during OPEN)
    # so we can backfill the daily history when yfinance's history endpoint
    # is lagging. During PRE-market we still call the quote — at that hour
    # the regularMarketPrice IS yesterday's close — but the injection step
    # below caps the target date at yesterday so today's bar isn't created.
    session_now = _market_session()
    live_quotes: Dict[str, float] = {}
    # During PRE-market, treat the quote as a "closed-session" lookup so we
    # get yesterday's close (regularMarketPrice) rather than a pre-market tick.
    quote_session = "CLOSED" if session_now == "PRE" else session_now
    for tk in state.positions.keys():
        lp = src.live_price(tk, session=quote_session)
        if lp is not None:
            live_quotes[tk] = lp

    # Inject the most recent live price as a daily bar where yfinance's
    # daily-history endpoint is lagging. yfinance often returns the current
    # bar as NaN and we drop it in data_source.history(), but the live
    # quote endpoints (regularMarketPrice / fast_info) usually have the
    # finalized close hours before the daily-history catches up. We bridge
    # that gap here so the NAV curve, drawdown chart, signals, and vol
    # panels reflect the most recent close.
    #
    # Target date logic: the bar we inject corresponds to the NEXT trading
    # day after each ticker's last-known close (one trading day at a time;
    # repeat-call fills weekend / multi-day gaps).
    try:
        import zoneinfo as _zi
        today_et = dt.datetime.now(_zi.ZoneInfo("America/New_York")).date()
    except Exception:
        today_et = (dt.datetime.utcnow() - dt.timedelta(hours=4)).date()

    def _next_trading_day(d):
        nxt = d + dt.timedelta(days=1)
        while nxt.weekday() >= 5:  # skip Sat/Sun
            nxt += dt.timedelta(days=1)
        return nxt

    def _inject(ph, live_p):
        if ph is None or ph.df.empty or live_p is None:
            return False
        last_date = ph.df.index[-1].normalize().date()
        target = _next_trading_day(last_date)
        # Only inject up through today.
        if target > today_et:
            return False
        # During PRE-market we backfill PRIOR trading days only — today's
        # bar must wait until the regular session so the displayed price
        # stays at the prior close. After 9:30 AM ET, today's bar is fair game.
        if session_now == "PRE" and target >= today_et:
            return False
        target_ts = pd.Timestamp(target)
        # Open/High/Low echo the close — only Close matters for NAV / curves;
        # the others keep range / Bollinger from seeing a NaN spike.
        new_row = pd.DataFrame(
            {"Open": [live_p], "High": [live_p], "Low": [live_p],
             "Close": [live_p], "Volume": [0.0]},
            index=[target_ts])
        ph.df = pd.concat([ph.df, new_row])
        return True

    # Inject for every session — _inject() itself blocks today's bar
    # during PRE-market, so the gating doesn't need to repeat that here.
    for tk, lp in live_quotes.items():
        _inject(histories.get(tk), lp)
    # Backfill benchmarks too so the NAV-vs-SPY/IWM chart aligns
    for bench in ("SPY", "IWM"):
        ph = bench_hist.get(bench)
        if ph is None or ph.df.empty:
            continue
        blive = src.live_price(bench, session=quote_session)
        if blive is not None:
            _inject(ph, blive)

    # NAV curve — computed AFTER the live-price injection so the curve
    # picks up today's close. Benchmarks reuse the same backfilled histories.
    start = trades["date"].iloc[0] if not trades.empty else pd.Timestamp.today().normalize()
    end = pd.Timestamp.today().normalize()
    nav = daily_nav_curve(trades, histories, start=start, end=end)
    spy_close = bench_hist["SPY"].df["Close"] if not bench_hist["SPY"].df.empty else pd.Series(dtype=float)
    iwm_close = bench_hist["IWM"].df["Close"] if not bench_hist["IWM"].df.empty else pd.Series(dtype=float)
    # Primary fund benchmark: Russell 2000 (IWM) — appropriate for a SMID
    # focused portfolio. All benchmark-relative metrics (IR, Treynor,
    # Jensen's alpha, Up/Down Capture, Beta) compute against this index.
    primary_bench_close = iwm_close if not iwm_close.empty else spy_close
    primary_bench_name = "IWM" if not iwm_close.empty else "SPY"
    bench_curves = benchmark_curves(start, end, bench_hist)

    # Holdings panel
    holdings = []
    sector_value = defaultdict(float)
    geo_value = defaultdict(float)
    total_equity = 0.0
    for tk, pos in state.positions.items():
        h = histories.get(tk)
        # yfinance sometimes includes today's not-yet-finalized bar with NaN
        # values during/after the trading day. Skip trailing NaNs so the last
        # price we use is always a valid float.
        last_clean = h.df["Close"].dropna() if h and not h.df.empty else None
        daily_close = float(last_clean.iloc[-1]) if last_clean is not None and not last_clean.empty else 0.0
        last = live_quotes.get(tk, daily_close)
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
            "short_percent_of_float": info.short_percent_of_float if info else None,
            "short_ratio": info.short_ratio if info else None,
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

    # Build a one-line ticker → Close-series map used by several risk
    # calculations downstream (synthetic returns, correlation, tail stress).
    closes_for_corr = {tk: h.df["Close"] for tk, h in histories.items() if not h.df.empty}

    # Portfolio metrics — prefer realized NAV if we have >= 30 days; fall back
    # to synthetic-history metrics (current weights × each holding's 252-day
    # return history) so newly-launched portfolios still show meaningful
    # vol / Sharpe / Sortino / max DD instead of empty cells.
    metrics_live = (portfolio_metrics(nav, primary_bench_close, risk_free_rate=risk_free_rate)
                    if len(nav) >= 30 else {})
    metrics_synth = synthetic_portfolio_metrics(holdings, closes_for_corr, primary_bench_close,
                                                risk_free_rate=risk_free_rate)
    metrics = metrics_live or metrics_synth or {}
    metrics["data_basis"] = "realized_nav" if metrics_live else "synthetic_252d"
    metrics["days_of_live_nav"] = int(len(nav))
    metrics["benchmark_ticker"] = primary_bench_name
    metrics["benchmark_name"] = "Russell 2000" if primary_bench_name == "IWM" else "S&P 500"

    # Beta vs primary benchmark (IWM) — used for IR/Treynor/Jensen/UI display
    portfolio_beta = metrics.get("beta_benchmark", metrics.get("beta_spy", float("nan")))
    if math.isnan(portfolio_beta) or portfolio_beta == 0:
        if total_equity > 0 and not primary_bench_close.empty:
            wb = 0.0
            for tk, pos in state.positions.items():
                h = histories.get(tk)
                if h is None or h.df.empty:
                    continue
                b = beta_to(h.df["Close"], primary_bench_close)
                if not math.isnan(b):
                    last = float(h.df["Close"].iloc[-1])
                    w = (pos.shares * last) / total_equity
                    wb += w * b
            portfolio_beta = wb if wb else 1.0
        else:
            portfolio_beta = 0.0
    metrics["beta_benchmark"] = portfolio_beta
    metrics["beta_spy"] = portfolio_beta  # alias for UI compat

    # Beta vs SPY — separately computed for historical stress tests, where
    # the crash percentages (COVID -33.9%, GFC -56.5%, etc.) are SPY moves.
    # Using beta_vs_IWM with SPY moves would mismatch the underlying math.
    beta_for_stress = float("nan")
    if total_equity > 0 and not spy_close.empty:
        wb = 0.0
        for tk, pos in state.positions.items():
            h = histories.get(tk)
            if h is None or h.df.empty:
                continue
            b = beta_to(h.df["Close"], spy_close)
            if not math.isnan(b):
                last = float(h.df["Close"].iloc[-1])
                w = (pos.shares * last) / total_equity
                wb += w * b
        beta_for_stress = wb if wb else 1.0
    if math.isnan(beta_for_stress):
        beta_for_stress = 1.0
    metrics["beta_spy_stress"] = float(beta_for_stress)
    stress = stress_tests(beta_for_stress, nav_now)
    # Monte Carlo: same precedence rule — live NAV bootstrap if long enough,
    # otherwise bootstrap from synthetic portfolio returns.
    synth_rets = synthetic_portfolio_returns(holdings, closes_for_corr)
    mc = monte_carlo(nav) if len(nav) >= 30 else monte_carlo_from_returns(synth_rets, nav_now)
    factor = factor_stress(holdings, nav_now)
    corr_tail = correlation_tail_stress(holdings, closes_for_corr, nav_now)
    liquidity = liquidity_stress(holdings, histories)
    risk_budget_rows = risk_budget(holdings, closes_for_corr, nav_now)
    dd_series = drawdown_curve(nav)

    # 5-factor risk model — Market / Size / Value / Momentum / Quality
    factor_closes = {tk: ph.df["Close"] for tk, ph in factor_proxy_hist.items()
                     if not ph.df.empty}
    factor_rets = compute_factor_returns(factor_closes)
    factor_corr = factor_correlation(factor_rets)
    per_ticker_loadings: Dict[str, dict] = {}
    factor_exposures_table = []
    for tk in tickers:
        ph = histories.get(tk)
        if ph is None or ph.df.empty:
            continue
        tk_rets = np.log(ph.df["Close"] / ph.df["Close"].shift(1)).dropna()
        l = regress_loadings(tk_rets, factor_rets)
        if l is None:
            continue
        per_ticker_loadings[tk] = l
        row = {"ticker": tk, "r_squared": l["r_squared"],
               "alpha_annual": l["alpha_annual"],
               "idio_vol_annual": l["idio_vol_annual"]}
        row.update(l["loadings"])
        factor_exposures_table.append(row)
    portfolio_factors = portfolio_factor_exposure(holdings, per_ticker_loadings)
    # Residual correlation — what's left after stripping the 8-factor exposure
    ticker_returns_for_resid = {}
    for tk in tickers:
        ph = histories.get(tk)
        if ph is None or ph.df.empty:
            continue
        ticker_returns_for_resid[tk] = np.log(ph.df["Close"] / ph.df["Close"].shift(1)).dropna()
    residual_corr = residual_correlation(ticker_returns_for_resid, factor_rets)

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

    # Trade blotter — newest first, with execution time when available
    blotter = []
    sort_col = "datetime" if "datetime" in trades.columns else "date"
    for _, t in trades.sort_values(sort_col, ascending=False).iterrows():
        if t["action"] == "DEPOSIT" or t["action"] == "WITHDRAW":
            continue
        blotter.append({
            "date": t["date"].strftime("%Y-%m-%d"),
            "time": t["datetime"].strftime("%H:%M") if "datetime" in trades.columns else "",
            "ticker": t["ticker"],
            "action": t["action"],
            "shares": float(t["shares"]),
            "price": float(t["price"]),
            "value": float(t["shares"] * t["price"]),
        })

    # Merged news feed: Yahoo headlines + SEC 8-K material events +
    # analyst upgrades/downgrades. Each item carries category + priority so
    # the UI can surface fundamental events (M&A, earnings, leadership)
    # ahead of general headlines.
    pri_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
    news_flat = []
    seen = set()  # dedupe by (ticker, title) — Yahoo sometimes reposts
    for tk in tickers:
        for n in (list(insider.get(tk, [])) + list(filings.get(tk, [])) +
                  list(analyst.get(tk, [])) + list(news.get(tk, []))):
            key = (n.ticker, (n.title or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            news_flat.append({
                "ticker": n.ticker,
                "title": n.title,
                "publisher": n.publisher,
                "link": n.link,
                "published": n.published,
                "category": n.category,
                "priority": n.priority,
                "source": n.source,
            })
    news_flat.sort(key=lambda n: (pri_rank.get(n["priority"], 9),
                                  -_published_to_epoch(n["published"])))

    earnings_dates.sort(key=lambda r: r["earnings_date"])

    snapshot = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "market_session": _market_session(),
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
        "residual_correlation": residual_corr,
        "factor_correlation": factor_corr,
        "factor_exposures": factor_exposures_table,
        "portfolio_factor_exposure": portfolio_factors,
        "stress_tests": stress,
        "monte_carlo": _scrub(mc),
        "factor_stress": _scrub(factor),
        "correlation_tail_risk": _scrub(corr_tail),
        "liquidity": liquidity,
        "risk_budget": risk_budget_rows,
        "income": {
            "total_annual_income": total_income,
            "dividend_income": total_dividend_income,
            "cash_income": cash_income,
            "cash_yield": cash_yield,
            "rows": income_rows,
        },
        "fundamentals": fundamentals,
        "trade_blotter": blotter,
        "news": news_flat[:80],
        "upcoming_earnings": earnings_dates[:30],
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


def _market_session() -> str:
    """Classify the current US Eastern time into a market session tag.

    Returns one of: PRE | OPEN | AFTER | CLOSED. Used by the dashboard to
    pick the correct freshness-pill color (LIVE vs EXTENDED vs CLOSED).
    """
    try:
        import zoneinfo
        now_et = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        # Crude fallback: assume EDT (UTC-4) — close enough for the pill color.
        now_et = dt.datetime.utcnow() - dt.timedelta(hours=4)
    if now_et.weekday() >= 5:  # Sat/Sun
        return "CLOSED"
    minutes = now_et.hour * 60 + now_et.minute
    if 240 <= minutes < 570:    # 4:00 AM – 9:30 AM ET
        return "PRE"
    if 570 <= minutes < 960:    # 9:30 AM – 4:00 PM ET
        return "OPEN"
    if 960 <= minutes < 1200:   # 4:00 PM – 8:00 PM ET
        return "AFTER"
    return "CLOSED"


def _published_to_epoch(s: str) -> float:
    """Parse a published-timestamp (ISO or date-only) into a Unix epoch for
    sorting; returns 0 if unparseable so undated items sink to the bottom."""
    if not s:
        return 0.0
    try:
        return pd.Timestamp(s).timestamp()
    except Exception:
        return 0.0


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
