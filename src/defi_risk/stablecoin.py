import numpy as np
import pandas as pd
from typing import Optional



def simulate_mean_reverting_peg(
    n_paths: int = 1000,
    n_steps: int = 365,
    T: float = 1.0,
    kappa: float = 2.0,
    sigma: float = 0.02,
    p0: float = 1.0,
    mu_level: float = 1.0,
    random_seed: Optional[int] = 42,   
) -> pd.DataFrame:
    ...

    """
    Simulate a mean-reverting stablecoin price around `mu_level` (≈1.0)
    using a discrete-time Ornstein-Uhlenbeck-style process.

    dP_t = kappa * (mu_level - P_t) dt + sigma dW_t

    Returns:
        DataFrame of shape (n_steps+1, n_paths)
        index: time grid from 0 to T
        columns: path index
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    # paths[p, t]
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = paths[:, t - 1]
        # Euler step
        drift = kappa * (mu_level - prev) * dt
        diffusion = sigma * np.sqrt(dt) * z
        paths[:, t] = prev + drift + diffusion

    df = pd.DataFrame(paths.T, index=times, columns=[f"path_{i}" for i in range(n_paths)])
    return df


def constant_product_slippage(
    reserve_stable: float,
    reserve_collateral: float,
    trade_size: float,
    direction: str = "sell_stable",
    fee_bps: float = 0.0,
) -> dict:
    """
    Simple x*y = k AMM to approximate a stablecoin pool.

    Args:
        reserve_stable: current stablecoin reserve in the pool
        reserve_collateral: reserve of the 'other' asset (e.g. USDC)
        trade_size: size of the trade in stablecoin units
        direction: "sell_stable" or "buy_stable"
        fee_bps: fee in basis points (e.g. 4 = 0.04%)

    Returns:
        dict with:
            - amount_out
            - new_reserve_stable
            - new_reserve_collateral
            - execution_price
            - mid_price
            - price_impact_pct
    """
    fee = fee_bps / 10_000.0

    if direction == "sell_stable":
        amount_in = trade_size * (1.0 - fee)
        x0, y0 = reserve_stable, reserve_collateral
        k = x0 * y0
        x1 = x0 + amount_in
        y1 = k / x1
        amount_out = y0 - y1
        new_reserve_stable, new_reserve_collateral = x1, y1
    elif direction == "buy_stable":
        # user sells collateral to buy stablecoin
        amount_in = trade_size * (1.0 - fee)
        x0, y0 = reserve_collateral, reserve_stable
        k = x0 * y0
        x1 = x0 + amount_in
        y1 = k / x1
        amount_out = y0 - y1
        # invert mapping back
        new_reserve_collateral, new_reserve_stable = x1, y1
    else:
        raise ValueError("direction must be 'sell_stable' or 'buy_stable'")

    # approximate mid-price before trade (stable in terms of collateral)
    mid_price = reserve_collateral / reserve_stable
    execution_price = trade_size / amount_out if amount_out > 0 else np.inf
    price_impact_pct = (execution_price / mid_price - 1.0) * 100.0

    return {
        "amount_out": amount_out,
        "new_reserve_stable": new_reserve_stable,
        "new_reserve_collateral": new_reserve_collateral,
        "execution_price": execution_price,
        "mid_price": mid_price,
        "price_impact_pct": price_impact_pct,
    }


def slippage_curve(
    reserve_stable: float,
    reserve_collateral: float,
    max_trade_pct: float = 0.5,
    n_points: int = 30,
    fee_bps: float = 0.0,
    direction: str = "sell_stable",
) -> pd.DataFrame:
    """
    Compute price impact for a range of trade sizes expressed
    as a fraction of the pool stablecoin reserve.

    Returns a DataFrame with columns:
        - trade_size
        - trade_size_pct
        - price_impact_pct
    """
    trade_sizes = np.linspace(0.0, max_trade_pct * reserve_stable, n_points)
    impacts = []

    for size in trade_sizes:
        if size <= 0:
            impacts.append(0.0)
            continue

        res = constant_product_slippage(
            reserve_stable=reserve_stable,
            reserve_collateral=reserve_collateral,
            trade_size=size,
            direction=direction,
            fee_bps=fee_bps,
        )
        impacts.append(res["price_impact_pct"])

    df = pd.DataFrame(
        {
            "trade_size": trade_sizes,
            "trade_size_pct": trade_sizes / reserve_stable,
            "price_impact_pct": impacts,
        }
    )
    return df
