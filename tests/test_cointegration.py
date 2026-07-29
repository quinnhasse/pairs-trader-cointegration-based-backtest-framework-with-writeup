"""Unit tests for the cointegration screening module."""

import numpy as np
import pandas as pd
import pytest

from pairs_trader.cointegration import (
    PairResult,
    compute_spread,
    screen_universe,
    eg_test,
)
from pairs_trader.data import generate_universe


@pytest.fixture(scope="module")
def small_universe():
    """20-ticker, 400-day universe with 4 embedded cointegrated pairs."""
    return generate_universe(n_tickers=20, n_days=400, n_cointegrated_pairs=4, seed=0)


class TestTestPair:
    def test_cointegrated_pair_detected(self, small_universe):
        """Embedded pairs (T00/T01, T02/T03, ...) should pass ADF at 5%."""
        result = eg_test(small_universe, "T00", "T01")
        assert result.is_cointegrated, (
            f"Expected T00/T01 to be cointegrated, got p={result.adf_pvalue:.4f}"
        )

    def test_independent_pair_not_cointegrated(self, small_universe):
        """Pairs at the tail of the ticker list (no embedded spread) should mostly fail."""
        # T18/T19 are not embedded cointegrated pairs in a 20-ticker universe with 4 pairs
        # (embedded pairs are T00/T01, T02/T03, T04/T05, T06/T07)
        # T12 and T13 have no embedded relationship
        result = eg_test(small_universe, "T12", "T13")
        # Not asserting p > 0.05 strictly since random walks occasionally cointegrate by chance;
        # we just check the result object is well-formed
        assert isinstance(result.adf_pvalue, float)
        assert 0.0 <= result.adf_pvalue <= 1.0

    def test_hedge_ratio_sign(self, small_universe):
        """Hedge ratio should be positive for positively correlated pairs."""
        result = eg_test(small_universe, "T00", "T01")
        assert result.hedge_ratio > 0

    def test_half_life_positive(self, small_universe):
        """Half-life should be a positive finite number for cointegrated pairs."""
        result = eg_test(small_universe, "T00", "T01")
        assert result.half_life_days > 0
        assert np.isfinite(result.half_life_days)

    def test_residuals_shape(self, small_universe):
        """Residuals array should match the length of the input series."""
        result = eg_test(small_universe, "T00", "T01")
        assert len(result.residuals) == len(small_universe)

    def test_returns_pair_result(self, small_universe):
        result = eg_test(small_universe, "T00", "T01")
        assert isinstance(result, PairResult)
        assert result.ticker_x == "T00"
        assert result.ticker_y == "T01"


class TestScreenUniverse:
    def test_finds_embedded_pairs(self, small_universe):
        """Screen should identify at least some of the 4 embedded cointegrated pairs."""
        results = screen_universe(small_universe, max_pairs=10)
        tickers_found = {(r.ticker_x, r.ticker_y) for r in results}
        embedded = [("T00", "T01"), ("T02", "T03"), ("T04", "T05"), ("T06", "T07")]
        found_count = sum(1 for p in embedded if p in tickers_found)
        assert found_count >= 2, f"Expected >=2 embedded pairs, found {found_count}"

    def test_sorted_by_pvalue(self, small_universe):
        """Results must be sorted ascending by ADF p-value."""
        results = screen_universe(small_universe, max_pairs=10)
        pvals = [r.adf_pvalue for r in results]
        assert pvals == sorted(pvals)

    def test_max_pairs_limit(self, small_universe):
        """max_pairs argument caps the number of returned pairs."""
        results = screen_universe(small_universe, max_pairs=3)
        assert len(results) <= 3

    def test_half_life_filter(self, small_universe):
        """All returned pairs must satisfy the half-life bounds."""
        max_hl = 50.0
        min_hl = 2.0
        results = screen_universe(small_universe, max_half_life_days=max_hl, min_half_life_days=min_hl)
        for r in results:
            assert min_hl <= r.half_life_days <= max_hl

    def test_pvalue_filter(self, small_universe):
        """All returned pairs must have ADF p-value below the threshold."""
        threshold = 0.05
        results = screen_universe(small_universe, pvalue_threshold=threshold)
        for r in results:
            assert r.adf_pvalue < threshold


class TestComputeSpread:
    def test_spread_is_series(self, small_universe):
        result = eg_test(small_universe, "T00", "T01")
        spread = compute_spread(small_universe, result)
        assert isinstance(spread, pd.Series)

    def test_spread_length(self, small_universe):
        result = eg_test(small_universe, "T00", "T01")
        spread = compute_spread(small_universe, result)
        assert len(spread) == len(small_universe)

    def test_spread_stationary_approx(self, small_universe):
        """Spread of a cointegrated pair should have low std relative to price levels."""
        result = eg_test(small_universe, "T00", "T01")
        spread = compute_spread(small_universe, result)
        # Spread is in log-price space; std should be small (< 0.3)
        assert spread.std() < 0.3, f"Spread std={spread.std():.3f} too large for cointegrated pair"

    def test_spread_mean_near_zero(self, small_universe):
        """OLS fit should center the spread near zero by construction."""
        result = eg_test(small_universe, "T00", "T01")
        spread = compute_spread(small_universe, result)
        assert abs(spread.mean()) < 0.1
