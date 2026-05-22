# Convex Capital

A public, paper-traded portfolio dashboard inspired by [mispricedassets.io](https://mispricedassets.io/). Starts with $1,000,000 in virtual cash; trades are added by editing `trades.csv`.

> **Hypothetical / paper-traded performance.** For educational and tracking purposes only. Not investment advice. Not an offer to manage assets.

## What's in the box

- `pipeline/` — Python data pipeline (yfinance + pandas + arch). Reads `trades.csv`, fetches prices/fundamentals/news, computes signals/volatility/correlation/stress, writes `docs/data/snapshot.json`.
- `docs/` — Single-file static dashboard (HTML + Chart.js via CDN) that renders the snapshot. This is the deployable folder (served by GitHub Pages).
- `.github/workflows/refresh.yml` — Optional GitHub Actions cron that re-runs the pipeline each weekday after market close and commits the new snapshot.
- `trades.csv` — Your trade ledger (the only file you edit day-to-day).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

Live data (yfinance, requires internet):

```bash
python -m pipeline.run
```

Sample / fixture mode for offline testing of the layout:

```bash
python -m pipeline.run --sample-data --sample-portfolio
```

Both modes write `docs/data/snapshot.json`.

## Viewing the dashboard locally

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

## Adding a trade

Append a row to `trades.csv`:

```csv
date,ticker,action,shares,price,notes
2026-05-09,CASH,DEPOSIT,1000000,1.00,Initial paper-trade funding
2026-05-12,AAPL,BUY,500,193.50,Initial position
2026-05-20,AAPL,SELL,100,201.75,Trim
```

Supported actions: `DEPOSIT`, `WITHDRAW` (use `ticker=CASH`), `BUY`, `SELL`.

Then re-run the pipeline. The dashboard auto-refreshes the next time it's loaded.

## Deploying

The `docs/` folder is a static site. Any of these will work for free:

- **GitHub Pages** (used here) — Settings → Pages → Source: `main`, folder: `/docs`.
- **Vercel** — `vercel --prod docs` (or connect the repo and set the output dir to `docs`).
- **Netlify** — drag-and-drop `docs/`, or connect the repo with `docs` as the publish directory.
- **Cloudflare Pages** — same; build output `docs`.

For automatic daily refresh, enable the included GitHub Actions workflow (`.github/workflows/refresh.yml`). It runs the pipeline at 21:30 UTC on weekdays (≈ 5:30 PM ET, after US market close), commits the updated snapshot, and your static host re-deploys automatically. No API keys needed for the yfinance backend.

## Methodology (for verification)

Every number on the dashboard is derived from publicly verifiable inputs and documented math. Where the reference site uses opaque scores, this implementation makes the formula explicit:

- **Technical signals** (`pipeline/signals.py`) — Six sub-signals each contribute ±0.5 or 0 to a composite score in [−3, +3]: RSI 14-period, MACD(12,26,9), 50/200-DMA trend stack, Bollinger middle-band position, 5d-vs-30d volume ratio, position in 20-day high/low range.
- **Volatility regime** (`pipeline/volatility.py`) — 20-day annualized historical volatility vs. a GARCH(1,1) one-day-ahead conditional vol forecast, fit on log returns. Regime tag is the percentile of current HV20 within the trailing-1-year HV20 distribution.
- **Stress tests** (`pipeline/risk.py`) — Linear approximation: `portfolio_beta × historical SPY peak-to-trough move`. Beta is OLS-regressed over the trailing 252 trading days. Historical SPY moves are static and citable: COVID Crash (2020-02-19 → 2020-03-23) −33.9%, 2022 Rate Shock −25.2%, GFC −56.5%, Dot-Com Bust −49.1%, Black Monday 1987 −20.4% (S&P 500 single-day), Aug 2024 Yen Carry Unwind −8.4%. All from Yahoo Finance daily closes.
- **Correlation matrix** (`pipeline/risk.py`) — Pearson correlation of trailing-252-day log returns.
- **Portfolio metrics** — Annualized return CAGR, annualized vol = `std(daily_returns) × √252`, Sharpe (risk-free = 0), Sortino (downside-only vol), max drawdown, 1-day historical VaR(95) = 5th percentile of daily NAV returns.

## Caveats and known limitations

- **yfinance** is an unofficial scraper. It's free and reliable enough for a daily-update personal tracker, but Yahoo can change its endpoints; if the pipeline starts failing, swap to Polygon or Tiingo by adding a new class to `pipeline/data_source.py` with the same interface.
- **Options panel deferred to v2.** The reference site's "Best Short-Term Calls" panel needs an options chain feed; revisit once the rest is live.
- **News quality** is whatever Yahoo surfaces — adequate but uneven. Adding Tiingo News (free tier) gives richer headlines.
- **Empty-state quirk:** if `trades.csv` only contains a deposit dated on a weekend, the NAV chart starts empty until the next business day rolls in. Cosmetic only.

## Sources cited in methodology

- SEC final rule on hypothetical performance: <https://www.sec.gov/rules/final/2020/ia-5653.pdf>
- FINRA Rule 2210 (communications with the public): <https://www.finra.org/rules-guidance/rulebooks/finra-rules/2210>
- yfinance library: <https://github.com/ranaroussi/yfinance>
- arch (GARCH) library: <https://arch.readthedocs.io/>

## Disclosure note

The dashboard footer carries a "hypothetical / not investment advice" disclaimer. As long as the site is not used to solicit investors or imply a real fund, this is appropriate framing. If at any point the goal becomes to attract investor capital, the SEC Marketing Rule and FINRA 2210 hypothetical-performance disclosure requirements likely apply — consult a securities lawyer before that pivot.
