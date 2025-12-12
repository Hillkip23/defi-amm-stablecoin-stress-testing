import numpy as np
from typing import Tuple
import pandas as pd




def constant_product_price(x_reserve: float, y_reserve: float) -> float:
    """Return the AMM price y/x."""
    if x_reserve <= 0:
        raise ValueError("x_reserve must be positive")
    return y_reserve / x_reserve


def pool_reserves_from_price(k: float, price: float):
    """Given invariant k = x*y and target price p = y/x, solve for (x, y)."""
    if price <= 0:
        raise ValueError("price must be positive")
    x = np.sqrt(k / price)
    y = np.sqrt(k * price)
    return x, y




def impermanent_loss(price_rel):
    """Impermanent loss (no fees) as a function of relative price change R = P_t / P_0.

    Works with scalars, numpy arrays, and pandas Series/DataFrames.
    Formula: IL = 2*sqrt(R)/(1+R) - 1
    """
    return 2 * np.sqrt(price_rel) / (1 + price_rel) - 1


def lp_value_relative(price_rel: float) -> float:
    """LP position value relative to HODL, assuming no fees."""
    return 1.0 + impermanent_loss(price_rel)


def uniswap_v3_position_value(price: float, p_lower: float, p_upper: float, L: float = 1.0) -> float:
    """
    Value of a Uniswap v3 position (in token1 units) for a given price and range [p_lower, p_upper].

    We assume price is token1 per token0, and use the standard Uniswap v3 formulas.
    L is the liquidity parameter; since we only care about *relative* value, we can set L = 1.
    """
    if p_lower <= 0 or p_upper <= 0:
        raise ValueError("Price bounds must be positive")
    if p_lower >= p_upper:
        raise ValueError("p_lower must be < p_upper")

    sqrtP = np.sqrt(price)
    sqrtPa = np.sqrt(p_lower)
    sqrtPb = np.sqrt(p_upper)

    if price <= p_lower:
        # Entirely in token0
        amount0 = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount1 = 0.0
    elif price >= p_upper:
        # Entirely in token1
        amount0 = 0.0
        amount1 = L * (sqrtPb - sqrtPa)
    else:
        # In-range mix
        amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        amount1 = L * (sqrtP - sqrtPa)

    return amount0 * price + amount1


def lp_over_hodl_univ3(price_T: float, p_lower: float, p_upper: float, p0: float = 1.0) -> float:
    """
    Uniswap v3 LP performance vs 50/50 HODL at terminal price price_T.

    - Start with 50/50 HODL at price p0 (here p0 ~ 1).
    - LP provides concentrated liquidity in [p_lower, p_upper].
    - Returns LP_value_T / HODL_value_T.
    """
    # LP relative value (v3 position)
    v0 = uniswap_v3_position_value(p0, p_lower, p_upper, L=1.0)
    vT = uniswap_v3_position_value(price_T, p_lower, p_upper, L=1.0)
    lp_rel = vT / v0  # LP_T / LP_0

    # HODL relative value, starting 50/50 at p0
    R = price_T / p0
    hodl_rel = 0.5 * (1.0 + R)  # HODL_T / HODL_0 (HODL_0 = 1)

    return lp_rel / hodl_rel




def trade_outcome_constant_product(
    x_reserve: float,
    y_reserve: float,
    delta_x: float,
) -> Tuple[float, float, float]:
    """
    Simple Uniswap v2-style trade: user trades delta_x of token X into the pool.

    Invariant:
        x * y = k

    Parameters
    ----------
    x_reserve : float
        Initial reserve of token X (e.g. stablecoin).
    y_reserve : float
        Initial reserve of token Y (e.g. collateral / volatile asset).
    delta_x : float
        Amount of token X being swapped into the pool.

    Returns
    -------
    new_x : float
        New reserve of token X after the trade.
    new_y : float
        New reserve of token Y after the trade.
    exec_price : float
        Average execution price for this trade in units of Y per X.
    """
    if delta_x <= 0:
        raise ValueError("delta_x must be positive for this simple trade model.")

    k = x_reserve * y_reserve
    new_x = x_reserve + delta_x
    new_y = k / new_x

    # Total Y paid out to the trader
    delta_y = y_reserve - new_y
    exec_price = delta_y / delta_x
    return new_x, new_y, exec_price

def slippage_from_trade(
    x_reserve: float,
    y_reserve: float,
    delta_x: float,
) -> float:
    """
    Compute slippage of a trade as (exec_price - mid_price) / mid_price.

    Positive slippage means worse execution for the trader (price moved against them).
    """
    mid_price = y_reserve / x_reserve
    _, _, exec_price = trade_outcome_constant_product(x_reserve, y_reserve, delta_x)
    return (exec_price - mid_price) / mid_price

def slippage_vs_trade_fraction(
    x_reserve: float,
    y_reserve: float,
    fractions: np.ndarray,
) -> pd.DataFrame:
    """
    Compute slippage for a grid of trade sizes (fractions of x_reserve).

    Parameters
    ----------
    x_reserve : float
        Initial reserve of token X.
    y_reserve : float
        Initial reserve of token Y.
    fractions : np.ndarray
        Array of trade sizes as fractions of x_reserve, e.g. [0.01, 0.05, 0.1].

    Returns
    -------
    pd.DataFrame
        Columns: ['fraction', 'delta_x', 'slippage']
    """
    records = []
    for f in fractions:
        if f <= 0:
            continue
        delta_x = f * x_reserve
        slip = slippage_from_trade(x_reserve, y_reserve, delta_x)
        records.append(
            {
                "fraction": float(f),
                "delta_x": float(delta_x),
                "slippage": float(slip),
            }
        )
    return pd.DataFrame(records)

