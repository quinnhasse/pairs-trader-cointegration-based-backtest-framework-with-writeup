"""Price data loading and synthetic universe generation.

The synthetic generator produces a universe of tickers whose prices follow
correlated geometric Brownian motion (GBM) with embedded Ornstein-Uhlenbeck
spread dynamics between selected pairs. This ensures some pairs are genuinely
cointegrated and some are not, giving the screening step real signal to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def generate_universe(
    n_tickers: int = 50,
    n_days: int = 756,
    n_cointegrated_pairs: int = 8,
    seed: int = 42,
    start_date: str = "2019-01-02",
) -> pd.DataFrame:
    """Generate a synthetic price universe with embedded cointegrated pairs.

    Parameters
    ----------
    n_tickers:
        Total number of tickers in the universe.
    n_days:
        Number of trading days to generate.
    n_cointegrated_pairs:
        Number of pairs to embed OU-process spread dynamics in. The rest
        follow independent GBM.
    seed:
        Random seed for reproducibility.
    start_date:
        Start date string (used only to build a DatetimeIndex).

    Returns
    -------
    pd.DataFrame
        Shape (n_days, n_tickers). Index is DatetimeIndex, columns are ticker
        strings like "T00", "T01", etc.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # --- common market factor ---
    market_returns = rng.normal(0.0003, 0.01, size=n_days)

    # --- per-ticker idiosyncratic parameters ---
    betas = rng.uniform(0.6, 1.4, size=n_tickers)
    idio_vols = rng.uniform(0.005, 0.012, size=n_tickers)

    # start all prices at 100
    log_prices = np.zeros((n_days, n_tickers))
    idio_returns = rng.normal(0, 1, size=(n_days, n_tickers)) * idio_vols

    for t in range(1, n_days):
        log_prices[t] = (
            log_prices[t - 1]
            + betas * market_returns[t]
            + idio_returns[t]
            - 0.5 * idio_vols**2  # Ito correction
        )

    prices = np.exp(log_prices) * 100.0

    # --- embed cointegrated pairs via OU spread ---
    # For pairs (0,1), (2,3), ... up to n_cointegrated_pairs
    for k in range(n_cointegrated_pairs):
        i, j = 2 * k, 2 * k + 1
        if j >= n_tickers:
            break
        hedge = rng.uniform(0.8, 1.2)
        # OU parameters: mean-reversion speed (kappa), long-run mean (mu), vol (sigma_ou)
        kappa = rng.uniform(0.03, 0.08)  # daily mean-reversion speed
        sigma_ou = rng.uniform(0.005, 0.015)

        spread = np.zeros(n_days)
        for t in range(1, n_days):
            spread[t] = spread[t - 1] + kappa * (0.0 - spread[t - 1]) + rng.normal(0, sigma_ou)

        # Overwrite ticker j so that log(Pj) = hedge * log(Pi) + spread + noise
        log_prices[:, j] = hedge * log_prices[:, i] + spread
        prices[:, j] = np.exp(log_prices[:, j]) * 100.0

    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load prices from a CSV file.

    Expected CSV columns: date, ticker, close.
    Returns a wide DataFrame: rows = dates, columns = tickers.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Wide price DataFrame indexed by date.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    wide = df.pivot(index="date", columns="ticker", values="close")
    wide.index.name = "date"
    return wide


def split_universe(
    prices: pd.DataFrame,
    estimation_days: int = 252,
    oos_days: int = 63,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split prices into walk-forward (estimation, oos) window pairs.

    Parameters
    ----------
    prices:
        Full price DataFrame.
    estimation_days:
        Number of days in each in-sample estimation window.
    oos_days:
        Number of days in each out-of-sample trading window.

    Returns
    -------
    List of (in_sample, out_of_sample) DataFrame pairs.
    """
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    n = len(prices)
    start = 0
    while start + estimation_days + oos_days <= n:
        is_df = prices.iloc[start : start + estimation_days]
        oos_df = prices.iloc[start + estimation_days : start + estimation_days + oos_days]
        windows.append((is_df, oos_df))
        start += oos_days
    return windows
