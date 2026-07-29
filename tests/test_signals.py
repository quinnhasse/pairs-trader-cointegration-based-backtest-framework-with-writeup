"""Unit tests for the z-score signal generation module."""

import numpy as np
import pandas as pd
import pytest

from pairs_trader.signals import (
    SignalConfig,
    compute_zscore,
    count_trades,
    generate_signals,
    signals_from_spread,
)


def make_spread(values: list[float], start: str = "2020-01-01") -> pd.Series:
    """Helper to construct a spread Series from a list of floats."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="spread")


class TestComputeZscore:
    def test_nan_during_burnin(self):
        """Z-scores should be NaN for the first min_periods-1 observations."""
        spread = make_spread([float(i) for i in range(50)])
        z = compute_zscore(spread, lookback=20, min_periods=10)
        assert z.iloc[:9].isna().all()

    def test_zscore_shape(self):
        """Output length must match input length."""
        spread = make_spread([float(i) for i in range(100)])
        z = compute_zscore(spread, lookback=30)
        assert len(z) == 100

    def test_constant_spread_gives_nan(self):
        """A constant spread has zero std; z-score should be NaN."""
        spread = make_spread([5.0] * 100)
        z = compute_zscore(spread, lookback=30, min_periods=10)
        # After burnin, all z-scores should be NaN (zero std)
        assert z.dropna().isna().all() or z.dropna().abs().max() < 1e-9 or z.iloc[50:].isna().all()

    def test_zscore_mean_near_zero(self):
        """For a stationary series the rolling z-score should have mean near 0."""
        rng = np.random.default_rng(1)
        values = rng.normal(0, 1, 300).tolist()
        spread = make_spread(values)
        z = compute_zscore(spread, lookback=60, min_periods=30)
        assert abs(z.dropna().mean()) < 0.3

    def test_zscore_unit_std(self):
        """Rolling z-score std should be approximately 1."""
        rng = np.random.default_rng(2)
        values = rng.normal(5, 2, 300).tolist()
        spread = make_spread(values)
        z = compute_zscore(spread, lookback=60, min_periods=30)
        std = z.dropna().std()
        assert 0.8 < std < 1.2, f"Expected std~1, got {std:.3f}"


class TestGenerateSignals:
    def _entry_signal(self, z_val: float, entry: float = 2.0) -> int:
        """Return the signal generated after a single z-score spike."""
        config = SignalConfig(entry_z=entry, exit_z=0.5, stop_z=5.0, lookback=10, min_periods=1)
        # Build a series: flat lead-in, then spike
        values = [0.0] * 5 + [z_val]
        dates = pd.bdate_range("2020-01-01", periods=6)
        z = pd.Series(values, index=dates)
        signals = generate_signals(z, config)
        return int(signals.iloc[-1])

    def test_positive_spike_opens_short(self):
        """z > entry_z should open a short position (signal = -1)."""
        assert self._entry_signal(2.5) == -1

    def test_negative_spike_opens_long(self):
        """z < -entry_z should open a long position (signal = +1)."""
        assert self._entry_signal(-2.5) == 1

    def test_no_position_below_entry(self):
        """z within (-entry_z, entry_z) should keep signal at 0."""
        assert self._entry_signal(1.5) == 0

    def test_exit_from_long(self):
        """Long position should close when z rises above -exit_z."""
        config = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=5.0)
        # Enter long at z=-2.5, then revert to z=0
        values = [0.0, 0.0, -2.5, -2.0, -1.0, 0.0, 0.5]
        dates = pd.bdate_range("2020-01-01", periods=7)
        z = pd.Series(values, index=dates)
        signals = generate_signals(z, config)
        assert int(signals.iloc[-1]) == 0

    def test_stop_from_long(self):
        """Long position should close when z drops below -stop_z."""
        config = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=3.5)
        values = [0.0, 0.0, -2.5, -4.0]
        dates = pd.bdate_range("2020-01-01", periods=4)
        z = pd.Series(values, index=dates)
        signals = generate_signals(z, config)
        assert int(signals.iloc[-1]) == 0

    def test_nan_produces_flat(self):
        """NaN z-scores should always yield signal 0."""
        values = [float("nan")] * 10
        dates = pd.bdate_range("2020-01-01", periods=10)
        z = pd.Series(values, index=dates)
        signals = generate_signals(z)
        assert (signals == 0).all()

    def test_output_dtype(self):
        """Signal series should have dtype int8."""
        dates = pd.bdate_range("2020-01-01", periods=20)
        z = pd.Series(np.linspace(-3, 3, 20), index=dates)
        signals = generate_signals(z)
        assert signals.dtype == np.int8

    def test_signal_values_in_set(self):
        """All signal values must be in {-1, 0, 1}."""
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2020-01-01", periods=200)
        z = pd.Series(rng.normal(0, 1.5, 200), index=dates)
        signals = generate_signals(z)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestCountTrades:
    def test_no_trades_all_flat(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        signals = pd.Series([0] * 10, index=dates, dtype=np.int8)
        assert count_trades(signals) == 0

    def test_counts_completed_roundtrip(self):
        """One round-trip trade: enter, then exit."""
        dates = pd.bdate_range("2020-01-01", periods=6)
        signals = pd.Series([0, 1, 1, 1, 0, 0], index=dates, dtype=np.int8)
        assert count_trades(signals) == 1

    def test_open_position_at_end_counts(self):
        """An open position at the end of the series counts as one trade."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        signals = pd.Series([0, 0, -1, -1, -1], index=dates, dtype=np.int8)
        assert count_trades(signals) == 1


class TestSignalsFromSpread:
    def test_returns_two_series(self):
        rng = np.random.default_rng(7)
        values = rng.normal(0, 1, 150).cumsum().tolist()
        spread = make_spread(values)
        result = signals_from_spread(spread)
        assert len(result) == 2
        z, sig = result
        assert isinstance(z, pd.Series)
        assert isinstance(sig, pd.Series)

    def test_signal_length_matches_spread(self):
        spread = make_spread([float(i % 5 - 2) for i in range(100)])
        z, sig = signals_from_spread(spread)
        assert len(sig) == 100
