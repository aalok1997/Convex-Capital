"""
Data source abstraction.

Default backend is yfinance. The interface is intentionally narrow so a future
Polygon / Tiingo backend can be dropped in without touching the rest of the
pipeline.

The --sample-data flag in run.py swaps in deterministic fixture data so
computations and JSON shape can be verified without network access.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class TickerInfo:
    ticker: str
    name: str
    sector: str
    country: str
    market_cap: Optional[float]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    price_to_sales: Optional[float]
    ev_to_ebitda: Optional[float] = None
    price_to_book: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    dividend_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    ex_dividend_date: Optional[str] = None
    last_dividend_amount: Optional[float] = None


@dataclass
class NewsItem:
    ticker: str
    title: str
    publisher: str
    link: str
    published: str  # ISO timestamp


@dataclass
class PriceHistory:
    """Daily OHLCV with a DatetimeIndex."""
    ticker: str
    df: pd.DataFrame  # columns: Open High Low Close Volume

    @property
    def close(self) -> pd.Series:
        return self.df["Close"]


# ---------------------------------------------------------------------------
# yfinance backend
# ---------------------------------------------------------------------------

class YFinanceSource:
    """Live data via yfinance."""

    def __init__(self, sleep_seconds: float = 0.0):
        # Tiny pause between calls helps when iterating many tickers.
        self.sleep = sleep_seconds

    def _yf(self):
        import yfinance as yf  # imported lazily so sample mode doesn't need it
        return yf

    def history(self, ticker: str, period: str = "2y") -> PriceHistory:
        yf = self._yf()
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if self.sleep:
            time.sleep(self.sleep)
        if df.empty:
            return PriceHistory(ticker=ticker, df=pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]))
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return PriceHistory(ticker=ticker, df=df[["Open", "High", "Low", "Close", "Volume"]])

    def info(self, ticker: str) -> TickerInfo:
        yf = self._yf()
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        if self.sleep:
            time.sleep(self.sleep)

        ex_div = info.get("exDividendDate")
        if isinstance(ex_div, (int, float)) and not math.isnan(ex_div):
            ex_div = dt.datetime.utcfromtimestamp(ex_div).date().isoformat()
        else:
            ex_div = None

        last_div = None
        try:
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                last_div = float(divs.iloc[-1])
        except Exception:
            pass

        return TickerInfo(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            sector=info.get("sector") or "Unknown",
            country=info.get("country") or "Unknown",
            market_cap=_safe_float(info.get("marketCap")),
            trailing_pe=_safe_float(info.get("trailingPE")),
            forward_pe=_safe_float(info.get("forwardPE")),
            price_to_sales=_safe_float(info.get("priceToSalesTrailing12Months")),
            ev_to_ebitda=_safe_float(info.get("enterpriseToEbitda")),
            price_to_book=_safe_float(info.get("priceToBook")),
            profit_margin=_safe_float(info.get("profitMargins")),
            operating_margin=_safe_float(info.get("operatingMargins")),
            return_on_equity=_safe_float(info.get("returnOnEquity")),
            return_on_assets=_safe_float(info.get("returnOnAssets")),
            revenue_growth=_safe_float(info.get("revenueGrowth")),
            earnings_growth=_safe_float(info.get("earningsGrowth")),
            avg_volume_20d=_safe_float(info.get("averageVolume10days") or info.get("averageVolume")),
            dividend_rate=_safe_float(info.get("dividendRate")),
            dividend_yield=_safe_float(info.get("dividendYield")),
            ex_dividend_date=ex_div,
            last_dividend_amount=last_div,
        )

    def news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        yf = self._yf()
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception:
            raw = []
        if self.sleep:
            time.sleep(self.sleep)
        out: List[NewsItem] = []
        for item in raw[:limit]:
            # yfinance has changed its news shape a few times; handle both.
            content = item.get("content") if isinstance(item, dict) else None
            if content:  # newer shape
                ts = content.get("pubDate") or content.get("displayTime") or ""
                title = content.get("title", "")
                publisher = (content.get("provider") or {}).get("displayName", "") or ""
                link = (content.get("canonicalUrl") or {}).get("url") or content.get("clickThroughUrl", {}).get("url", "")
                published = ts
            else:  # older shape
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                link = item.get("link", "")
                pt = item.get("providerPublishTime")
                published = dt.datetime.utcfromtimestamp(pt).isoformat() if pt else ""
            out.append(NewsItem(
                ticker=ticker, title=title, publisher=publisher,
                link=link, published=published,
            ))
        return out


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sample / fixture backend (no network)
# ---------------------------------------------------------------------------

class SampleSource:
    """
    Deterministic synthetic backend used by --sample-data. Generates plausible
    OHLCV histories from seeded random walks so RSI/MACD/GARCH/correlation
    have realistic-looking inputs to operate on.
    """

    SECTORS = {
        "AAPL": ("Apple Inc.", "Technology", "United States"),
        "MSFT": ("Microsoft Corp.", "Technology", "United States"),
        "GOOGL": ("Alphabet Inc.", "Communication Services", "United States"),
        "NVDA": ("NVIDIA Corp.", "Technology", "United States"),
        "BABA": ("Alibaba Group", "Consumer Cyclical", "China"),
        "JPM":  ("JPMorgan Chase", "Financial Services", "United States"),
        "XOM":  ("Exxon Mobil", "Energy", "United States"),
        "JNJ":  ("Johnson & Johnson", "Healthcare", "United States"),
        "SPY":  ("SPDR S&P 500 ETF", "Index", "United States"),
        "IWM":  ("iShares Russell 2000", "Index", "United States"),
    }

    def __init__(self, seed: int = 42, days: int = 504):
        self.seed = seed
        self.days = days
        self._cache: Dict[str, PriceHistory] = {}

    def _gen(self, ticker: str) -> PriceHistory:
        if ticker in self._cache:
            return self._cache[ticker]
        rng = np.random.default_rng(self.seed + sum(ord(c) for c in ticker))
        # Annualized vol between 18% and 65% depending on ticker
        ann_vol = 0.18 + (sum(ord(c) for c in ticker) % 47) / 100.0
        daily_vol = ann_vol / math.sqrt(252)
        drift = 0.08 / 252  # 8% annual drift
        rets = rng.normal(loc=drift, scale=daily_vol, size=self.days)
        # Add an occasional vol cluster to make GARCH meaningful
        for i in range(0, self.days, 90):
            j = min(i + 10, self.days)
            rets[i:j] *= 2.0
        start_price = 50 + (sum(ord(c) for c in ticker) % 200)
        prices = start_price * np.exp(np.cumsum(rets))
        end = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(end=end, periods=self.days)
        close = pd.Series(prices, index=idx)
        high = close * (1 + np.abs(rng.normal(0, 0.005, self.days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, self.days)))
        open_ = close.shift(1).fillna(close.iloc[0])
        vol = (rng.integers(1_000_000, 20_000_000, self.days)).astype(float)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                           "Close": close, "Volume": vol})
        self._cache[ticker] = PriceHistory(ticker=ticker, df=df)
        return self._cache[ticker]

    def history(self, ticker: str, period: str = "2y") -> PriceHistory:
        return self._gen(ticker)

    def info(self, ticker: str) -> TickerInfo:
        name, sector, country = self.SECTORS.get(
            ticker, (ticker, "Technology", "United States"))
        rng = np.random.default_rng(self.seed + sum(ord(c) for c in ticker))
        return TickerInfo(
            ticker=ticker, name=name, sector=sector, country=country,
            market_cap=float(rng.uniform(0.3e9, 800e9)),
            trailing_pe=float(rng.uniform(8, 60)),
            forward_pe=float(rng.uniform(7, 45)),
            price_to_sales=float(rng.uniform(0.5, 15)),
            ev_to_ebitda=float(rng.uniform(5, 30)),
            price_to_book=float(rng.uniform(0.8, 18)),
            profit_margin=float(rng.uniform(0.02, 0.35)),
            operating_margin=float(rng.uniform(0.05, 0.40)),
            return_on_equity=float(rng.uniform(0.05, 0.45)),
            return_on_assets=float(rng.uniform(0.02, 0.20)),
            revenue_growth=float(rng.uniform(-0.10, 0.40)),
            earnings_growth=float(rng.uniform(-0.20, 0.60)),
            avg_volume_20d=float(rng.integers(1_000_000, 30_000_000)),
            dividend_rate=float(rng.uniform(0, 4)) if ticker not in ("BABA", "GOOGL") else 0.0,
            dividend_yield=float(rng.uniform(0, 0.04)),
            ex_dividend_date=(dt.date.today() + dt.timedelta(days=int(rng.integers(-90, 90)))).isoformat(),
            last_dividend_amount=float(rng.uniform(0.1, 2.0)),
        )

    def news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        templates = [
            ("{t} beats Q1 estimates on strong demand", "MarketBeat"),
            ("{t} stock climbs after analyst upgrade", "Yahoo Finance"),
            ("Why {t} could be the trade of the quarter", "Barron's"),
            ("{t} announces buyback program", "Reuters"),
        ]
        out = []
        now = dt.datetime.utcnow()
        for i, (tmpl, pub) in enumerate(templates[:limit]):
            out.append(NewsItem(
                ticker=ticker,
                title=tmpl.format(t=ticker),
                publisher=pub,
                link=f"https://example.com/{ticker.lower()}/{i}",
                published=(now - dt.timedelta(hours=i * 6)).isoformat() + "Z",
            ))
        return out
