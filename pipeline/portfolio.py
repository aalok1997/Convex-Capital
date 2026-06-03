"""
Portfolio reconstruction from trades.csv.

trades.csv schema:
    date,ticker,action,shares,price,notes

Actions:
    DEPOSIT      ticker=CASH, increases cash balance
    WITHDRAW     ticker=CASH, decreases cash balance
    BUY          decreases cash by shares*price, increases position
    SELL         increases cash by shares*price, decreases position

The reconstruction yields a daily NAV time series (cash + sum of
position_shares * close_price) and current holdings.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from .data_source import PriceHistory


@dataclass
class Lot:
    """A single buy lot, used for FIFO realized-P&L tracking."""
    shares: float
    price: float
    date: pd.Timestamp


@dataclass
class Position:
    ticker: str
    shares: float = 0.0
    cost_basis: float = 0.0  # weighted-avg cost per share (informational)
    lots: List[Lot] = field(default_factory=list)


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position]
    realized_pnl: float
    deposits: float


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Optional intraday `time` column (HH:MM, market timezone). If present,
    # it's combined with `date` into a full datetime so replay order matches
    # actual execution sequence within a day. If absent, all same-day trades
    # default to 16:00 (market close) and are processed alphabetically by ticker.
    if "time" in df.columns:
        df["time"] = df["time"].fillna("16:00").astype(str).str.strip()
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])
    else:
        df["datetime"] = pd.to_datetime(df["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["shares"] = df["shares"].astype(float)
    df["price"] = df["price"].astype(float)
    return df


def replay(trades: pd.DataFrame) -> PortfolioState:
    cash = 0.0
    deposits = 0.0
    realized = 0.0
    positions: Dict[str, Position] = {}

    for _, t in trades.iterrows():
        action, ticker, sh, px = t["action"], t["ticker"], t["shares"], t["price"]
        if action == "DEPOSIT":
            cash += sh * px
            deposits += sh * px
        elif action == "WITHDRAW":
            cash -= sh * px
            deposits -= sh * px
        elif action == "BUY":
            cash -= sh * px
            pos = positions.setdefault(ticker, Position(ticker=ticker))
            pos.lots.append(Lot(shares=sh, price=px, date=t["date"]))
            total_cost = pos.cost_basis * pos.shares + sh * px
            pos.shares += sh
            pos.cost_basis = total_cost / pos.shares if pos.shares else 0.0
        elif action == "SELL":
            cash += sh * px
            pos = positions.setdefault(ticker, Position(ticker=ticker))
            remaining = sh
            while remaining > 1e-9 and pos.lots:
                lot = pos.lots[0]
                used = min(lot.shares, remaining)
                realized += (px - lot.price) * used
                lot.shares -= used
                remaining -= used
                if lot.shares <= 1e-9:
                    pos.lots.pop(0)
            pos.shares -= sh
            if pos.shares <= 1e-9:
                pos.shares = 0.0
                pos.cost_basis = 0.0

    # Drop zeroed positions for cleanliness
    positions = {k: v for k, v in positions.items() if v.shares > 1e-9}
    return PortfolioState(cash=cash, positions=positions,
                          realized_pnl=realized, deposits=deposits)


def daily_nav_curve(trades: pd.DataFrame,
                    price_histories: Dict[str, PriceHistory],
                    start: pd.Timestamp = None,
                    end: pd.Timestamp = None) -> pd.Series:
    """
    Reconstruct the daily NAV curve.

    For each business day, NAV = cash_at_eod + sum(position_shares_eod * close).
    Closes come from the per-ticker price histories. Days with missing closes
    forward-fill the last known price.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    start = start or trades["date"].iloc[0]
    end = end or pd.Timestamp.today().normalize()
    if end < start:
        end = start

    # Build a unified business-day index from all price histories that
    # actually overlap our window — fall back to a simple bdate_range if none.
    all_idx = []
    for ph in price_histories.values():
        if not ph.df.empty:
            all_idx.append(ph.df.index)
    if all_idx:
        idx = sorted(set().union(*[set(i) for i in all_idx]))
        idx = pd.DatetimeIndex([d for d in idx if start <= d <= end])
    else:
        idx = pd.bdate_range(start=start, end=end)
    if len(idx) == 0:
        idx = pd.bdate_range(start=start, end=end)

    # Pre-index closes per ticker forward-filled to idx
    closes = {}
    for tk, ph in price_histories.items():
        if ph.df.empty:
            continue
        s = ph.df["Close"].reindex(idx, method="ffill")
        closes[tk] = s

    cash = 0.0
    positions: Dict[str, float] = {}
    nav = pd.Series(index=idx, dtype=float)

    trade_idx = 0
    trades_sorted = trades.reset_index(drop=True)
    n = len(trades_sorted)
    for d in idx:
        # Apply all trades on or before this day that haven't been applied yet
        while trade_idx < n and trades_sorted.loc[trade_idx, "date"] <= d:
            t = trades_sorted.loc[trade_idx]
            sh, px = float(t["shares"]), float(t["price"])
            if t["action"] == "DEPOSIT":
                cash += sh * px
            elif t["action"] == "WITHDRAW":
                cash -= sh * px
            elif t["action"] == "BUY":
                cash -= sh * px
                positions[t["ticker"]] = positions.get(t["ticker"], 0.0) + sh
            elif t["action"] == "SELL":
                cash += sh * px
                positions[t["ticker"]] = positions.get(t["ticker"], 0.0) - sh
            trade_idx += 1
        equity = 0.0
        for tk, sh in positions.items():
            if sh == 0:
                continue
            s = closes.get(tk)
            if s is not None and d in s.index and not pd.isna(s.loc[d]):
                equity += sh * float(s.loc[d])
            elif s is not None and not s.empty:
                # Forward-fill: use most recent available close <= d
                pre = s.loc[:d]
                if not pre.empty and not pd.isna(pre.iloc[-1]):
                    equity += sh * float(pre.iloc[-1])
        nav.loc[d] = cash + equity

    return nav.dropna()


def benchmark_curves(start: pd.Timestamp,
                     end: pd.Timestamp,
                     bench_histories: Dict[str, PriceHistory]) -> Dict[str, pd.Series]:
    """
    Normalize each benchmark to start at 0% on `start` so it can be plotted as
    cumulative-return % alongside portfolio.
    """
    out = {}
    for name, ph in bench_histories.items():
        if ph.df.empty:
            continue
        s = ph.df["Close"]
        s = s[(s.index >= start) & (s.index <= end)]
        if s.empty:
            continue
        out[name] = (s / s.iloc[0]) - 1.0
    return out
