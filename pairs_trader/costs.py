"""Transaction cost and slippage model.

Costs are applied on each leg independently. For a pairs trade, you trade two
legs simultaneously, so total cost is 2 * leg_cost.

Cost model:
  execution_price = mid_price * (1 + direction * (commission_bps + half_spread_bps) / 10_000)

where direction = +1 for buys, -1 for sells.

In practice this means:
  - Buying at price P costs P * (1 + total_bps/10_000)
  - Selling at price P receives P * (1 - total_bps/10_000)

The PnL impact per leg per trade is roughly:
  slippage_pct = (commission_bps + half_spread_bps) / 10_000
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """Parameters for the transaction cost model."""

    commission_bps: float = 5.0       # round-trip commission, basis points per leg
    half_spread_bps: float = 2.5      # one-way bid-ask half-spread per leg

    @property
    def one_way_bps(self) -> float:
        """Total one-way cost in basis points (commission + half-spread)."""
        return self.commission_bps + self.half_spread_bps

    @property
    def one_way_fraction(self) -> float:
        """Total one-way cost as a decimal fraction."""
        return self.one_way_bps / 10_000.0

    def execution_price(self, mid: float, is_buy: bool) -> float:
        """Return the all-in execution price for one leg.

        Parameters
        ----------
        mid:
            Mid-market price.
        is_buy:
            True for a buy order (pays spread + commission), False for a sell.

        Returns
        -------
        float: execution price including costs.
        """
        direction = 1.0 if is_buy else -1.0
        return mid * (1.0 + direction * self.one_way_fraction)

    def round_trip_cost(self, mid: float) -> float:
        """Cost of entering and exiting one leg at the same price level.

        Parameters
        ----------
        mid:
            Representative mid price for the leg.

        Returns
        -------
        float: total round-trip cost in dollar terms per unit.
        """
        buy_price = self.execution_price(mid, is_buy=True)
        sell_price = self.execution_price(mid, is_buy=False)
        return buy_price - sell_price

    def apply_to_pnl(
        self,
        pnl: np.ndarray,
        trade_flags: np.ndarray,
        prices_x: np.ndarray,
        prices_y: np.ndarray,
        position_size: float = 1.0,
    ) -> np.ndarray:
        """Deduct transaction costs from a raw PnL array on trade days.

        Costs are applied on both legs (x and y) on every day where a trade
        occurs (i.e., position changes).

        Parameters
        ----------
        pnl:
            Raw daily PnL array, shape (n_days,).
        trade_flags:
            Boolean array (n_days,) — True on days when a new trade is opened
            or an existing trade is closed.
        prices_x:
            Prices for the x leg, shape (n_days,).
        prices_y:
            Prices for the y leg, shape (n_days,).
        position_size:
            Dollar notional per leg.

        Returns
        -------
        np.ndarray: PnL after deducting costs, same shape as input.
        """
        pnl_net = pnl.copy().astype(float)
        cost_per_leg_x = prices_x * self.one_way_fraction * position_size
        cost_per_leg_y = prices_y * self.one_way_fraction * position_size
        # Each trade event hits two legs; buy + sell on entry + exit = 2 events,
        # but we count entry as one event and exit as another separately.
        trade_cost = (cost_per_leg_x + cost_per_leg_y) * trade_flags.astype(float)
        pnl_net -= trade_cost
        return pnl_net


def compute_slippage_series(
    prices: np.ndarray,
    signals: np.ndarray,
    cost_model: CostModel,
) -> np.ndarray:
    """Compute per-day slippage cost for a signal series on a single leg.

    Parameters
    ----------
    prices:
        Price array for the leg, shape (n_days,).
    signals:
        Signal array {-1, 0, 1}, shape (n_days,).
    cost_model:
        CostModel instance.

    Returns
    -------
    np.ndarray: slippage cost (positive = drag on PnL), shape (n_days,).
    """
    n = len(signals)
    slippage = np.zeros(n, dtype=float)

    for t in range(1, n):
        prev = int(signals[t - 1])
        curr = int(signals[t])
        if curr == prev:
            continue  # no change, no cost
        # A transition means either opening, closing, or reversing a position
        # Each leg trade incurs one-way cost
        slippage[t] = prices[t] * cost_model.one_way_fraction

    return slippage
