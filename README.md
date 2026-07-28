# Pairs trader

Cointegration-based pairs trading backtest framework. Screens a synthetic S&P 500 universe for stationary spreads, runs a walk-forward backtest with realistic transaction costs, and produces a performance report.

## Setup

```bash
pip install -r requirements.txt
```

## Run the backtest

```bash
make backtest
```

Outputs written to `outputs/`:
- `equity_curve.png` — equity curve, drawdown, rolling Sharpe, spread z-score
- `tearsheet.png` — full tear sheet
- `metrics.json` — in-sample and out-of-sample stats
- `report.pdf` — method, results, and failure analysis

## Run tests

```bash
make test
```

## Project structure

```
pairs_trader/
  data.py          # synthetic price generator and CSV loader
  cointegration.py # Engle-Granger screen, ADF, half-life
  signals.py       # z-score signal with entry/exit thresholds
  backtest.py      # walk-forward engine
  costs.py         # commission and slippage model
  metrics.py       # Sharpe, Sortino, max drawdown, turnover
  significance.py  # permutation test vs random pair baseline
  report.py        # tear sheet and PDF report generator
  kalman.py        # Kalman filter dynamic hedge ratio
  portfolio.py     # multi-pair risk parity sizing
scripts/
  run_backtest.py  # entry point wired to make backtest
tests/
  test_cointegration.py
  test_signals.py
  test_backtest.py
  test_metrics.py
```

## Data

The default run uses a synthetic 50-ticker universe generated from correlated geometric Brownian motion with embedded OU-process spread dynamics. To use real prices, pass a CSV with columns `[date, ticker, close]` via `--prices path/to/prices.csv`.

## Methodology notes

- Cointegration test: Engle-Granger two-step (ADF on OLS residuals), p < 0.05 threshold
- Walk-forward: 252-day estimation window, 63-day out-of-sample trading window
- Hedge ratio: OLS (default) or Kalman filter (`--hedge kalman`)
- Entry: z-score > 2.0, exit: z-score < 0.5, stop: z-score > 3.5
- Costs: 5 bps commission + 2.5 bps half-spread slippage per leg
- Significance: Monte Carlo permutation over 1000 random pairs

## Limitations and caveats

- Synthetic data does not capture real market microstructure, corporate events, or regime changes.
- Engle-Granger is sensitive to look-ahead bias; cointegration parameters are re-estimated each window.
- Transaction costs are stylized; real execution slippage scales with position size.
- Out-of-sample results on synthetic data are not predictive of live trading performance.
