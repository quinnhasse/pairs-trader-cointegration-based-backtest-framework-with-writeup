"""Engle-Granger cointegration screening and pair selection.

The two-step Engle-Granger procedure:
  1. Regress log(P_y) on log(P_x) via OLS to estimate the hedge ratio.
  2. Run ADF on the OLS residuals; reject unit root at p < threshold.

Half-life of mean reversion is estimated from the AR(1) coefficient of the
residual series: hl = -log(2) / log(|phi|).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller


@dataclass
class PairResult:
    """Result of a cointegration test for one pair."""

    ticker_x: str
    ticker_y: str
    hedge_ratio: float          # OLS coefficient of log(P_x) predicting log(P_y)
    intercept: float
    adf_stat: float
    adf_pvalue: float
    half_life_days: float       # estimated mean-reversion half-life (trading days)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_cointegrated(self) -> bool:
        """True when residuals are stationary at the 5% level."""
        return self.adf_pvalue < 0.05


def _ols_hedge_ratio(log_px: np.ndarray, log_py: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit log(P_y) ~ beta * log(P_x) + alpha via OLS.

    Returns (beta, alpha, residuals).
    """
    x = add_constant(log_px)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = OLS(log_py, x).fit()
    beta = float(result.params[1])
    alpha = float(result.params[0])
    residuals = np.asarray(result.resid)
    return beta, alpha, residuals


def _half_life(residuals: np.ndarray) -> float:
    """Estimate mean-reversion half-life from an AR(1) fit on the residual series.

    Returns half-life in days. Returns inf if the series appears non-stationary.
    """
    lag = residuals[:-1]
    delta = residuals[1:] - residuals[:-1]
    x = add_constant(lag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = OLS(delta, x).fit()
    phi = float(result.params[1])  # AR coefficient on lag
    if phi >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + phi))


def eg_test(
    prices: pd.DataFrame,
    ticker_x: str,
    ticker_y: str,
    log_transform: bool = True,
) -> PairResult:
    """Run the Engle-Granger cointegration test on one pair.

    Parameters
    ----------
    prices:
        Wide price DataFrame (rows = dates, columns = tickers).
    ticker_x:
        The independent variable in the OLS regression.
    ticker_y:
        The dependent variable (we model P_y as a function of P_x).
    log_transform:
        If True, apply log before regression (standard for equity prices).

    Returns
    -------
    PairResult with hedge ratio, ADF statistics, and half-life.
    """
    px = prices[ticker_x].values.astype(float)
    py = prices[ticker_y].values.astype(float)

    if log_transform:
        px = np.log(px)
        py = np.log(py)

    beta, alpha, residuals = _ols_hedge_ratio(px, py)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_result = adfuller(residuals, autolag="AIC", maxlag=10)

    adf_stat = float(adf_result[0])
    adf_pvalue = float(adf_result[1])
    hl = _half_life(residuals)

    return PairResult(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        hedge_ratio=beta,
        intercept=alpha,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        half_life_days=hl,
        residuals=residuals,
    )


def screen_universe(
    prices: pd.DataFrame,
    max_half_life_days: float = 120.0,
    min_half_life_days: float = 2.0,
    pvalue_threshold: float = 0.05,
    max_pairs: int | None = None,
) -> list[PairResult]:
    """Screen all pairs in a universe for cointegration.

    Tests every ordered pair (x, y) where x < y in column order. Returns only
    pairs that pass the ADF threshold and have a plausible mean-reversion speed.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    max_half_life_days:
        Upper bound on half-life — pairs mean-reverting too slowly are excluded.
    min_half_life_days:
        Lower bound — half-life under this suggests noise rather than a true spread.
    pvalue_threshold:
        ADF p-value cutoff for the stationarity test.
    max_pairs:
        If set, return at most this many pairs (sorted by p-value).

    Returns
    -------
    List of PairResult objects passing all filters, sorted by ADF p-value.
    """
    tickers = list(prices.columns)
    n = len(tickers)
    passing: list[PairResult] = []

    for i in range(n):
        for j in range(i + 1, n):
            tx, ty = tickers[i], tickers[j]
            try:
                result = eg_test(prices, tx, ty)
            except Exception:
                continue

            if result.adf_pvalue >= pvalue_threshold:
                continue
            if result.half_life_days < min_half_life_days:
                continue
            if result.half_life_days > max_half_life_days:
                continue

            passing.append(result)

    passing.sort(key=lambda r: r.adf_pvalue)

    if max_pairs is not None:
        passing = passing[:max_pairs]

    return passing


def compute_spread(
    prices: pd.DataFrame,
    pair: PairResult,
    log_transform: bool = True,
) -> pd.Series:
    """Compute the hedge-ratio-adjusted spread for a pair.

    spread_t = log(P_y_t) - hedge_ratio * log(P_x_t) - intercept

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    pair:
        PairResult with fitted hedge ratio and intercept.
    log_transform:
        If True, apply log before computing the spread.

    Returns
    -------
    pd.Series of spread values indexed like prices.
    """
    px = prices[pair.ticker_x].astype(float)
    py = prices[pair.ticker_y].astype(float)

    if log_transform:
        px = np.log(px)
        py = np.log(py)

    spread = py - pair.hedge_ratio * px - pair.intercept
    spread.name = f"{pair.ticker_y}_{pair.ticker_x}_spread"
    return spread
