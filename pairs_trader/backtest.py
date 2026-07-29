"""Walk-forward backtest engine for pairs trading.

Architecture:
  1. Split the full price history into rolling (estimation, oos) windows.
  2. For each window:
       a. Screen the estimation period for cointegrated pairs.
       b. For the top-N pairs, compute signals on the OOS period using
          hedge ratios estimated in-sample (no look-ahead).
       c. Compute PnL net of transaction costs.
  3. Concatenate all OOS PnL windows to produce a full equity curve.

The engine is deliberately single-threaded and deterministic given a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pairs_trader.cointegration import PairResult, compute_spread, eg_test, screen_universe
from pairs_trader.costs import CostModel
from pairs_trader.data import split_universe
from pairs_trader.signals import SignalConfig, signals_from_spread


@dataclass
class BacktestConfig:
    """Configuration for the walk-forward backtest."""

    estimation_days: int = 252      # in-sample estimation window
    oos_days: int = 63              # out-of-sample trading window
    max_pairs_per_window: int = 5   # top-N pairs per estimation period
    position_notional: float = 1.0  # dollar notional per leg (normalized)
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    cost_model: CostModel = field(default_factory=CostModel)
    pvalue_threshold: float = 0.05
    max_half_life_days: float = 120.0
    min_half_life_days: float = 2.0
    use_kalman: bool = False         # if True, use Kalman filter hedge ratio


@dataclass
class WindowResult:
    """Output of running one walk-forward window."""

    window_idx: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    pairs_found: int
    pairs_traded: list[str]         # e.g. ["T04/T05", "T06/T07"]
    daily_pnl: pd.Series            # OOS daily PnL (dollar-normalized, net of costs)
    daily_returns: pd.Series        # daily_pnl as returns (fraction of portfolio notional)
    n_trades: int


@dataclass
class BacktestResult:
    """Aggregated result across all walk-forward windows."""

    window_results: list[WindowResult]
    equity_curve: pd.Series           # cumulative sum of daily PnL
    daily_returns: pd.Series          # concatenated daily returns
    is_daily_returns: pd.Series       # in-sample daily returns (for IS Sharpe)
    is_equity_curve: pd.Series
    total_trades: int
    config: BacktestConfig


def _compute_pair_pnl(
    oos_prices: pd.DataFrame,
    pair: PairResult,
    config: BacktestConfig,
) -> tuple[pd.Series, int]:
    """Compute daily PnL for one pair over the OOS window.

    Parameters
    ----------
    oos_prices:
        Out-of-sample price DataFrame.
    pair:
        PairResult with in-sample hedge ratio.
    config:
        BacktestConfig.

    Returns
    -------
    (daily_pnl, n_trades): daily PnL Series and trade count.
    """
    spread = compute_spread(oos_prices, pair)
    _, signals = signals_from_spread(spread, config.signal_config)

    px = oos_prices[pair.ticker_x].values.astype(float)
    py = oos_prices[pair.ticker_y].values.astype(float)
    sig = signals.values

    n = len(sig)
    pnl = np.zeros(n, dtype=float)
    trade_flags = np.zeros(n, dtype=bool)

    prev_sig = 0
    for t in range(1, n):
        curr_sig = int(sig[t - 1])
        if curr_sig == 0:
            pnl[t] = 0.0
        else:
            # Long signal (+1): long Y, short X
            # PnL = returns_y - hedge * returns_x  (log-price differences in ratio)
            # Using simple returns for PnL calculation: (P_t / P_{t-1} - 1)
            ret_x = (px[t] - px[t - 1]) / px[t - 1]
            ret_y = (py[t] - py[t - 1]) / py[t - 1]
            pnl[t] = curr_sig * (ret_y - pair.hedge_ratio * ret_x) * config.position_notional

        # Detect trades (signal changes)
        if int(sig[t]) != prev_sig:
            trade_flags[t] = True
        prev_sig = int(sig[t])

    # Deduct transaction costs on trade days
    cost_per_trade_day = (px + py) * config.cost_model.one_way_fraction * config.position_notional
    pnl -= cost_per_trade_day * trade_flags.astype(float)

    n_trades = int(trade_flags.sum())
    pnl_series = pd.Series(pnl, index=oos_prices.index, name=f"{pair.ticker_x}/{pair.ticker_y}")
    return pnl_series, n_trades


def _compute_is_pair_pnl(
    is_prices: pd.DataFrame,
    pair: PairResult,
    config: BacktestConfig,
) -> pd.Series:
    """Compute in-sample daily PnL for one pair (for IS Sharpe estimation)."""
    spread = compute_spread(is_prices, pair)
    _, signals = signals_from_spread(spread, config.signal_config)

    px = is_prices[pair.ticker_x].values.astype(float)
    py = is_prices[pair.ticker_y].values.astype(float)
    sig = signals.values

    n = len(sig)
    pnl = np.zeros(n, dtype=float)

    for t in range(1, n):
        curr_sig = int(sig[t - 1])
        if curr_sig != 0:
            ret_x = (px[t] - px[t - 1]) / px[t - 1]
            ret_y = (py[t] - py[t - 1]) / py[t - 1]
            pnl[t] = curr_sig * (ret_y - pair.hedge_ratio * ret_x) * config.position_notional

    return pd.Series(pnl, index=is_prices.index)


def run_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the full walk-forward backtest.

    Parameters
    ----------
    prices:
        Full universe price DataFrame.
    config:
        BacktestConfig; uses defaults if None.

    Returns
    -------
    BacktestResult with equity curve, daily returns, and per-window results.
    """
    if config is None:
        config = BacktestConfig()

    if config.use_kalman:
        from pairs_trader.kalman import kalman_hedge_ratio as _kalman  # noqa: F401

    windows = split_universe(prices, config.estimation_days, config.oos_days)
    window_results: list[WindowResult] = []

    all_oos_pnl: list[pd.Series] = []
    all_is_pnl: list[pd.Series] = []

    for idx, (is_prices, oos_prices) in enumerate(windows):
        # --- in-sample cointegration screen ---
        pairs = screen_universe(
            is_prices,
            pvalue_threshold=config.pvalue_threshold,
            max_half_life_days=config.max_half_life_days,
            min_half_life_days=config.min_half_life_days,
            max_pairs=config.max_pairs_per_window,
        )

        window_pnl_list: list[pd.Series] = []
        window_is_pnl_list: list[pd.Series] = []
        traded_pairs: list[str] = []
        total_window_trades = 0

        for pair in pairs:
            # Refit hedge ratio on in-sample (already done in screen_universe via eg_test)
            # Optionally re-estimate with Kalman on OOS
            if config.use_kalman:
                try:
                    from pairs_trader.kalman import KalmanHedge

                    kh = KalmanHedge()
                    kh.fit(is_prices[pair.ticker_x], is_prices[pair.ticker_y])
                    pair_with_kalman = kh.to_pair_result(pair, oos_prices)
                    pair_to_use = pair_with_kalman
                except Exception:
                    pair_to_use = pair
            else:
                pair_to_use = pair

            oos_pnl, n_trades = _compute_pair_pnl(oos_prices, pair_to_use, config)
            is_pnl = _compute_is_pair_pnl(is_prices, pair_to_use, config)

            window_pnl_list.append(oos_pnl)
            window_is_pnl_list.append(is_pnl)
            traded_pairs.append(f"{pair.ticker_x}/{pair.ticker_y}")
            total_window_trades += n_trades

        if window_pnl_list:
            # Equal-weight across pairs in this window
            window_total_pnl = sum(window_pnl_list) / len(window_pnl_list)
            window_total_is_pnl = sum(window_is_pnl_list) / len(window_is_pnl_list)
        else:
            window_total_pnl = pd.Series(
                np.zeros(len(oos_prices)), index=oos_prices.index
            )
            window_total_is_pnl = pd.Series(
                np.zeros(len(is_prices)), index=is_prices.index
            )

        all_oos_pnl.append(window_total_pnl)
        all_is_pnl.append(window_total_is_pnl)

        wr = WindowResult(
            window_idx=idx,
            is_start=is_prices.index[0],
            is_end=is_prices.index[-1],
            oos_start=oos_prices.index[0],
            oos_end=oos_prices.index[-1],
            pairs_found=len(pairs),
            pairs_traded=traded_pairs,
            daily_pnl=window_total_pnl,
            daily_returns=window_total_pnl,  # normalized, so pnl == returns here
            n_trades=total_window_trades,
        )
        window_results.append(wr)

    if not all_oos_pnl:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            window_results=[],
            equity_curve=empty,
            daily_returns=empty,
            is_daily_returns=empty,
            is_equity_curve=empty,
            total_trades=0,
            config=config,
        )

    daily_returns = pd.concat(all_oos_pnl).sort_index()
    equity_curve = daily_returns.cumsum()
    equity_curve.name = "equity_curve"

    # IS returns: deduplicate overlapping windows by keeping last occurrence
    is_daily_returns = pd.concat(all_is_pnl).groupby(level=0).mean().sort_index()
    is_equity_curve = is_daily_returns.cumsum()
    is_equity_curve.name = "is_equity_curve"

    total_trades = sum(wr.n_trades for wr in window_results)

    return BacktestResult(
        window_results=window_results,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        is_daily_returns=is_daily_returns,
        is_equity_curve=is_equity_curve,
        total_trades=total_trades,
        config=config,
    )
