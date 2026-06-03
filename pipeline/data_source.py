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
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# News categorization
#
# The same shape is applied to every news source (Yahoo, SEC 8-K, analyst
# actions, calendar). HIGH-priority items are surfaced first in the UI.
# ---------------------------------------------------------------------------

# (category, priority, regex)
NEWS_KEYWORD_RULES: List[Tuple[str, str, "re.Pattern"]] = [
    ("M_AND_A",            "HIGH", re.compile(r"\b(acquir(?:es|ed|ing|ition)|merger|merges?|merging|"
                                              r"takeover|spin[- ]?off|"
                                              r"divest(?:s|ed|iture)?|joint venture|"
                                              r"strategic alliance|all[- ]cash deal|"
                                              r"definitive agreement)\b", re.I)),
    ("EARNINGS",           "HIGH", re.compile(r"\b(earnings (?:call|report|release|transcript|beat|miss|preview)|"
                                              r"Q[1-4]\s*\d{2,4}?\s*(?:results|earnings|call|transcript)|"
                                              r"quarterly results?|beats? (?:Q[1-4]|estimates)|"
                                              r"misses? (?:Q[1-4]|estimates)|raises? guidance|cuts? guidance|"
                                              r"lifts? guidance|lowers? guidance|"
                                              r"reports? Q[1-4]|reports? (?:earnings|results)|"
                                              r"EPS (?:beat|miss|of))\b", re.I)),
    ("LEADERSHIP",         "HIGH", re.compile(r"\b(CEO|CFO|COO|chairman|chairperson|resigns?|"
                                              r"steps? down|appoint(?:s|ed|ment)|hires?|"
                                              r"new (?:chief|president))\b", re.I)),
    ("REGULATORY",         "HIGH", re.compile(r"\b(FDA (?:approval|approves|grants|rejects)|"
                                              r"phase\s*[123]|clinical trial|EU approval|"
                                              r"DOJ probe|antitrust|investigation|"
                                              r"recall|halt)\b", re.I)),
    ("LEGAL",              "MED",  re.compile(r"\b(lawsuit|sued|settle(?:s|d|ment)?|"
                                              r"SEC charges|fraud|class action|verdict)\b", re.I)),
    ("ANALYST",            "MED",  re.compile(r"\b(upgrade[ds]?|downgrade[ds]?|"
                                              r"reiterate[ds]?|initiate[ds]? coverage|"
                                              r"price target|rating|buy rating|sell rating|"
                                              r"outperform|underperform|overweight|underweight)\b", re.I)),
    ("PRODUCT",            "MED",  re.compile(r"\b(launch(?:es|ed|ing)?|unveil(?:s|ed)?|"
                                              r"introduces?|announces? (?:new |a )?(?:product|service)|"
                                              r"rollout|partnership with)\b", re.I)),
    ("BUYBACK_DIVIDEND",   "MED",  re.compile(r"\b(buyback|share repurchase|dividend (?:hike|increase|cut)|"
                                              r"special dividend|stock split)\b", re.I)),
]


def categorize_headline(title: str) -> Tuple[str, str]:
    """Return (category, priority) by matching the title against rules in order.
    First matching rule wins; falls back to ('OTHER', 'LOW')."""
    if not title:
        return ("OTHER", "LOW")
    for cat, pri, pat in NEWS_KEYWORD_RULES:
        if pat.search(title):
            return (cat, pri)
    return ("OTHER", "LOW")


# ---------------------------------------------------------------------------
# SEC 8-K item code → (human label, category, priority)
#
# A current 8-K is the SEC's mandated "material event" filing. Each filing
# tags one or more Item numbers indicating the nature of the event. We map
# the high-signal items to our news categories so the headline carries the
# correct badge in the UI.
# ---------------------------------------------------------------------------

SEC_8K_ITEMS: Dict[str, Tuple[str, str, str]] = {
    "1.01": ("Material definitive agreement signed", "MATERIAL_AGREEMENT", "HIGH"),
    "1.02": ("Material agreement terminated",         "MATERIAL_AGREEMENT", "HIGH"),
    "1.03": ("Bankruptcy or receivership",            "LEGAL",              "HIGH"),
    "2.01": ("Completion of acquisition or disposition", "M_AND_A",         "HIGH"),
    "2.02": ("Earnings release / results announced",  "EARNINGS",           "HIGH"),
    "2.03": ("Material financial obligation incurred","MATERIAL_AGREEMENT", "MED"),
    "2.05": ("Costs from exit or disposal activity",  "MATERIAL_AGREEMENT", "MED"),
    "2.06": ("Material impairment recognized",        "EARNINGS",           "HIGH"),
    "3.01": ("Notice of delisting / non-compliance",  "REGULATORY",         "HIGH"),
    "3.02": ("Unregistered equity sale",              "MATERIAL_AGREEMENT", "MED"),
    "3.03": ("Material modification to security rights","MATERIAL_AGREEMENT","MED"),
    "4.01": ("Auditor change",                        "REGULATORY",         "MED"),
    "4.02": ("Non-reliance on prior financials",      "REGULATORY",         "HIGH"),
    "5.01": ("Change in control",                     "M_AND_A",            "HIGH"),
    "5.02": ("Officer / director change",             "LEADERSHIP",         "HIGH"),
    "5.03": ("Amendments to bylaws / charter",        "MATERIAL_AGREEMENT", "MED"),
    "5.07": ("Shareholder vote results",              "DISCLOSURE",         "MED"),
    "5.08": ("Shareholder director nominations",      "DISCLOSURE",         "LOW"),
    "7.01": ("Regulation FD disclosure",              "DISCLOSURE",         "MED"),
    "8.01": ("Other material event",                  "FILING",             "MED"),
    "9.01": ("Financial statements and exhibits",     "FILING",             "LOW"),
}


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
    category: str = "OTHER"   # EARNINGS | M_AND_A | ANALYST | LEADERSHIP |
                              # MATERIAL_AGREEMENT | REGULATORY | PRODUCT |
                              # LEGAL | DISCLOSURE | FILING | OTHER
    priority: str = "LOW"     # HIGH | MED | LOW
    source: str = "yahoo"     # yahoo | sec | analyst | calendar


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
    """Live data via yfinance.

    News pipeline is multi-source: Yahoo news + SEC 8-K filings + analyst
    upgrade/downgrade actions + earnings calendar. Every item is normalized
    to NewsItem shape with category and priority.
    """

    # SEC EDGAR's fair-access policy requires a User-Agent identifying the
    # caller. Email here is the project's builder email.
    SEC_HEADERS = {"User-Agent": "Convex Capital (aalok.develops@gmail.com)"}

    def __init__(self, sleep_seconds: float = 0.0):
        # Tiny pause between calls helps when iterating many tickers.
        self.sleep = sleep_seconds
        # Lazy-loaded ticker → 10-digit CIK lookup, used for SEC filings.
        self._cik_map: Optional[Dict[str, str]] = None

    def _yf(self):
        import yfinance as yf  # imported lazily so sample mode doesn't need it
        return yf

    # ----------------------------------------------------------------------
    # SEC EDGAR — 8-K material event filings
    # ----------------------------------------------------------------------

    def _load_cik_map(self) -> Dict[str, str]:
        """Fetch and cache the SEC's ticker→CIK mapping (one HTTP call per run)."""
        if self._cik_map is not None:
            return self._cik_map
        import requests
        try:
            r = requests.get("https://www.sec.gov/files/company_tickers.json",
                             headers=self.SEC_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            self._cik_map = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                             for v in data.values()}
        except Exception:
            self._cik_map = {}
        return self._cik_map

    def sec_filings(self, ticker: str, since_days: int = 120,
                    limit: int = 5) -> List[NewsItem]:
        """Return recent 8-K filings as NewsItem records, parsed into our
        category/priority shape. Empty list if the ticker is foreign-listed
        (not in EDGAR) or the lookup fails."""
        cik = self._load_cik_map().get(ticker.upper())
        if not cik:
            return []
        import requests
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = requests.get(url, headers=self.SEC_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []
        if self.sleep:
            time.sleep(self.sleep)
        recent = data.get("filings", {}).get("recent", {}) or {}
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        prims = recent.get("primaryDocument", [])
        items_all = recent.get("items", [])
        cutoff = (dt.date.today() - dt.timedelta(days=since_days)).isoformat()
        out: List[NewsItem] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            filing_date = dates[i] if i < len(dates) else ""
            if filing_date < cutoff:
                continue
            items_raw = items_all[i] if i < len(items_all) else ""
            items = [s.strip() for s in re.split(r"[,\s]+", items_raw) if s.strip()]
            # Convert "5.02" -> human label + classify as the highest-priority item
            labels: List[str] = []
            best = ("FILING", "LOW")
            for code in items:
                meta = SEC_8K_ITEMS.get(code)
                if not meta:
                    continue
                label, cat, pri = meta
                labels.append(f"{code} {label}")
                if _priority_rank(pri) < _priority_rank(best[1]):
                    best = (cat, pri)
            if not labels:
                continue
            title = "8-K · " + " · ".join(labels)
            acc = accs[i].replace("-", "") if i < len(accs) else ""
            doc = prims[i] if i < len(prims) else ""
            link = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{acc}/{doc}") if acc and doc else ""
            out.append(NewsItem(
                ticker=ticker, title=title, publisher="SEC EDGAR",
                link=link, published=filing_date,
                category=best[0], priority=best[1], source="sec",
            ))
            if len(out) >= limit:
                break
        return out

    # ----------------------------------------------------------------------
    # Analyst upgrades / downgrades (via yfinance)
    # ----------------------------------------------------------------------

    def analyst_actions(self, ticker: str, since_days: int = 60,
                        limit: int = 5) -> List[NewsItem]:
        yf = self._yf()
        try:
            df = yf.Ticker(ticker).upgrades_downgrades
        except Exception:
            df = None
        if self.sleep:
            time.sleep(self.sleep)
        if df is None or df.empty:
            return []
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=since_days)
        df = df[df.index >= cutoff].sort_index(ascending=False)
        out: List[NewsItem] = []
        for ts, row in df.iterrows():
            firm = str(row.get("Firm", "") or "")
            to_g = str(row.get("ToGrade", "") or "")
            from_g = str(row.get("FromGrade", "") or "")
            action = str(row.get("Action", "") or "").lower()
            if action == "up":
                verb = "Upgrade"; pri = "MED"
            elif action == "down":
                verb = "Downgrade"; pri = "MED"
            elif action == "init":
                verb = "Coverage initiated"; pri = "MED"
            elif action == "reit":
                verb = "Reiterated"; pri = "LOW"
            else:
                verb = action.title() or "Rating change"; pri = "LOW"
            title = f"{firm}: {verb} — {from_g} → {to_g}" if from_g else f"{firm}: {verb} {to_g}".strip()
            out.append(NewsItem(
                ticker=ticker, title=title.strip(" —"),
                publisher=firm or "Analyst", link="",
                published=ts.isoformat(),
                category="ANALYST", priority=pri, source="analyst",
            ))
            if len(out) >= limit:
                break
        return out

    # ----------------------------------------------------------------------
    # Earnings calendar (next-scheduled date per ticker)
    # ----------------------------------------------------------------------

    def next_earnings(self, ticker: str) -> Optional[dict]:
        yf = self._yf()
        try:
            cal = yf.Ticker(ticker).calendar
        except Exception:
            return None
        if self.sleep:
            time.sleep(self.sleep)
        if not cal:
            return None
        # yfinance returns a dict in newer versions, DataFrame in older.
        if isinstance(cal, dict):
            edt = cal.get("Earnings Date")
            if not edt:
                return None
            if isinstance(edt, (list, tuple)) and edt:
                edt = edt[0]
            try:
                d = pd.Timestamp(edt).date().isoformat()
            except Exception:
                return None
            return {"ticker": ticker, "earnings_date": d}
        try:
            edt = cal.loc["Earnings Date"].iloc[0]
            d = pd.Timestamp(edt).date().isoformat()
            return {"ticker": ticker, "earnings_date": d}
        except Exception:
            return None

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

    def live_price(self, ticker: str) -> Optional[float]:
        """Latest available traded price, including pre-market and after-hours.

        Tries `fast_info.last_price` first (Yahoo's normalized intraday quote);
        falls back to `info['currentPrice']`. Returns None if neither is
        available — caller should fall back to the most recent daily close.
        """
        yf = self._yf()
        try:
            t = yf.Ticker(ticker)
            try:
                fi = t.fast_info
                lp = getattr(fi, "last_price", None)
                if lp is None and isinstance(fi, dict):
                    lp = fi.get("last_price") or fi.get("lastPrice")
                if lp is not None and not math.isnan(float(lp)) and float(lp) > 0:
                    return float(lp)
            except Exception:
                pass
            try:
                inf = t.info or {}
                for key in ("currentPrice", "regularMarketPrice",
                            "postMarketPrice", "preMarketPrice"):
                    v = inf.get(key)
                    if v is not None and not math.isnan(float(v)) and float(v) > 0:
                        return float(v)
            except Exception:
                pass
        finally:
            if self.sleep:
                time.sleep(self.sleep)
        return None

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
            cat, pri = categorize_headline(title)
            out.append(NewsItem(
                ticker=ticker, title=title, publisher=publisher,
                link=link, published=published,
                category=cat, priority=pri, source="yahoo",
            ))
        return out


def _priority_rank(p: str) -> int:
    """Lower rank = higher priority (used to pick the most-important item code
    when a single 8-K covers multiple items)."""
    return {"HIGH": 0, "MED": 1, "LOW": 2}.get(p, 3)


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
            title = tmpl.format(t=ticker)
            cat, pri = categorize_headline(title)
            out.append(NewsItem(
                ticker=ticker,
                title=title,
                publisher=pub,
                link=f"https://example.com/{ticker.lower()}/{i}",
                published=(now - dt.timedelta(hours=i * 6)).isoformat() + "Z",
                category=cat, priority=pri, source="yahoo",
            ))
        return out

    def sec_filings(self, ticker: str, since_days: int = 120,
                    limit: int = 5) -> List[NewsItem]:
        return []

    def analyst_actions(self, ticker: str, since_days: int = 60,
                        limit: int = 5) -> List[NewsItem]:
        return []

    def next_earnings(self, ticker: str) -> Optional[dict]:
        return None

    def live_price(self, ticker: str) -> Optional[float]:
        # Sample source returns None so run.py falls back to the daily close.
        return None
