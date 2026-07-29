"""Z-score signal generator for pairs trading.

Computes a rolling z-score of the spread and generates discrete position
signals based on configurable entry, exit, and stop thresholds.

Signal convention:
  +1  long Y, short X (spread below -entry_z, expecting reversion upward)
  -1  short Y, long X (spread above +entry_z, expecting reversion downward)
   0  flat

State machine transitions:
  Flat -> Long:   z < -entry_z
  Flat -> Short:  z >  entry_z
  Long -> Flat:   z > -exit_z  (normal mean reversion)
  Long -> Flat:   z < -stop_z  (stop loss, spread moved further against)
  Short -> Flat:  z <  exit_z  (normal mean reversion)
  Short -> Flat:  z >  stop_z  (stop loss)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """Parameters controlling signal generation."""

    entry_z: float = 2.0        # z-score magnitude to open a position
    exit_z: float = 0.5         # z-score magnitude to close a position (mean-reversion)
    stop_z: float = 3.5         # z-score magnitude stop loss
    lookback: int = 60          # rolling window for mean and std estimation (days)
    min_periods: int = 30       # minimum observations before generating any signal


def compute_zscore(spread: pd.Series, lookback: int = 60, min_periods: int = 30) -> pd.Series:
    """Compute rolling z-score of a spread series.

    z_t = (spread_t - mu_t) / sigma_t

    where mu_t and sigma_t are the rolling mean and standard deviation over
    the previous `lookback` observations.

    Parameters
    ----------
    spread:
        Time series of spread values.
    lookback:
        Rolling window size.
    min_periods:
        Minimum number of non-NaN values required to produce a result.

    Returns
    -------
    pd.Series of z-scores, NaN during the burn-in period.
    """
    mu = spread.rolling(window=lookback, min_periods=min_periods).mean()
    sigma = spread.rolling(window=lookback, min_periods=min_periods).std()
    # Avoid division by near-zero std; set to NaN when spread is constant
    sigma = sigma.where(sigma > 1e-10, other=np.nan)
    zscore = (spread - mu) / sigma
    zscore.name = "zscore"
    return zscore


def generate_signals(
    zscore: pd.Series,
    config: SignalConfig | None = None,
) -> pd.Series:
    """Convert a z-score series into a discrete {-1, 0, +1} signal.

    Uses a stateful machine: once in a position, stay until the exit or stop
    condition fires. Positions are never opened when z-score is NaN.

    Parameters
    ----------
    zscore:
        Rolling z-score series as returned by compute_zscore.
    config:
        SignalConfig; uses defaults if None.

    Returns
    -------
    pd.Series of int8 signals indexed like zscore.
    """
    if config is None:
        config = SignalConfig()

    z = zscore.values
    n = len(z)
    signals = np.zeros(n, dtype=np.int8)
    position = 0  # current position: -1, 0, or +1

    for t in range(n):
        if np.isnan(z[t]):
            signals[t] = 0
            continue

        if position == 0:
            if z[t] > config.entry_z:
                position = -1  # spread too high -> short spread
            elif z[t] < -config.entry_z:
                position = 1   # spread too low  -> long spread
        elif position == 1:
            # Long spread: exit when z reverts above -exit_z, stop when z < -stop_z
            if z[t] > -config.exit_z or z[t] < -config.stop_z:
                position = 0
        elif position == -1:
            # Short spread: exit when z reverts below exit_z, stop when z > stop_z
            if z[t] < config.exit_z or z[t] > config.stop_z:
                position = 0

        signals[t] = position

    return pd.Series(signals, index=zscore.index, name="signal", dtype=np.int8)


def signals_from_spread(
    spread: pd.Series,
    config: SignalConfig | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Compute z-score and signals from a raw spread series.

    Convenience wrapper combining compute_zscore and generate_signals.

    Parameters
    ----------
    spread:
        Raw spread series.
    config:
        SignalConfig; uses defaults if None.

    Returns
    -------
    Tuple of (zscore, signals) pd.Series.
    """
    if config is None:
        config = SignalConfig()
    zscore = compute_zscore(spread, lookback=config.lookback, min_periods=config.min_periods)
    signals = generate_signals(zscore, config)
    return zscore, signals


def count_trades(signals: pd.Series) -> int:
    """Count the number of complete round-trip trades in a signal series.

    A trade is counted each time the signal transitions from non-zero to zero.

    Parameters
    ----------
    signals:
        Discrete signal series {-1, 0, 1}.

    Returns
    -------
    Integer count of completed trades.
    """
    # count each transition from in-position to flat
    transitions = (signals == 0) & (signals.shift(1).fillna(0) != 0)
    return int(transitions.sum()) + int((signals.iloc[-1] != 0))
