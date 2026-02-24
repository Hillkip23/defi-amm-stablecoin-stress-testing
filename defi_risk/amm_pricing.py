import numpy as np
from typing import Tuple
import pandas as pd


def constant_product_price(x_reserve: float, y_reserve: float) -> float:
    """Return the AMM price y/x (price of token X in terms of token Y)."""
    if x_reserve <= 0 or y_reserve <= 0:
        raise ValueError("Reserves must be positive")
    return y_reserve / x_reserve


def pool_reserves_from_price(k: float, price: float) -> Tuple[float, float]:
    """
    Given invariant k = x*y and target price p = y/x, solve for (x, y).
    
    Returns:
        tuple: (x_reserve, y_reserve)
    """
    if price <= 0 or k <= 0:
        raise ValueError("Price and k must be positive")
    x = np.sqrt(k / price)
    y = np.sqrt(k * price)
    return x, y


def impermanent_loss(price_rel: float) -> float:
    """
    Impermanent loss (no fees) as a function of relative price change R = P_t / P_0.
    
    Formula: IL = 2*sqrt(R)/(1+R) - 1
    
    Returns negative values for losses (e.g., -0.057 for 5.7% loss).
    Works with scalars, numpy arrays, and pandas Series/DataFrames.
    """
    # Guard against division by zero
    denominator = 1.0 + price_rel
    if isinstance(denominator, (np.ndarray, pd.Series)):
        denominator = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    elif np.abs(denominator) < 1e-12:
        return np.nan
        
    return 2.0 * np.sqrt(price_rel) / denominator - 1.0


def lp_value_relative(price_rel: float) -> float:
    """LP position value relative to initial investment, assuming no fees.
    
    Returns: 1.0 + impermanent_loss(price_rel)
    """
    return 1.0 + impermanent_loss(price_rel)


def uniswap_v3_position_value(
    price: float, 
    p_lower: float, 
    p_upper: float, 
    L: float = 1.0
) -> float:
    """
    Value of a Uniswap v3 position (in token1 units) for given price and range.
    
    Uses standard Uniswap v3 liquidity formulas:
    - L = sqrt(xy) = liquidity parameter
    - Value calculated as token0_amount * price + token1_amount
    
    Args:
        price: Current price (token1 per token0)
        p_lower: Lower price bound of position
        p_upper: Upper price bound of position  
        L: Liquidity parameter (default 1.0 for relative value)
        
    Returns:
        Position value in token1 units
    """
    if p_lower <= 0 or p_upper <= 0:
        raise ValueError("Price bounds must be positive")
    if p_lower >= p_upper:
        raise ValueError("p_lower must be < p_upper")
    if price < 0:
        raise ValueError("Price cannot be negative")
    if L < 0:
        raise ValueError("Liquidity L cannot be negative")

    sqrtP = np.sqrt(price)
    sqrtPa = np.sqrt(p_lower)
    sqrtPb = np.sqrt(p_upper)

    if price <= p_lower:
        # Entirely in token0 (X), convert to token1 (Y) units at current price
        # amount0 = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount0 = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount1 = 0.0
        return amount0 * price
    elif price >= p_upper:
        # Entirely in token1 (Y)
        # amount1 = L * (sqrtPb - sqrtPa)
        amount0 = 0.0
        amount1 = L * (sqrtPb - sqrtPa)
        return amount1
    else:
        # In-range: mixed position
        # amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        # amount1 = L * (sqrtP - sqrtPa)
        amount0 = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        amount1 = L * (sqrtP - sqrtPa)
        return amount0 * price + amount1


def lp_over_hodl_univ3(
    price_T: float, 
    p_lower: float, 
    p_upper: float, 
    p0: float = 1.0
) -> float:
    """
    Uniswap v3 LP performance vs 50/50 HODL at terminal price price_T.
    
    Assumes:
    - Initial 50/50 portfolio at price p0
    - LP provides concentrated liquidity in [p_lower, p_upper]
    - Fees are NOT included in this calculation (terminal value only)
    
    Returns:
        Ratio of LP_value_T / HODL_value_T (>1 means LP outperformed)
    """
    if p0 <= 0:
        raise ValueError("Initial price p0 must be positive")
        
    # LP relative value (v3 position value / initial value)
    # Initial value at p0 with L=1 is the reference
    v0 = uniswap_v3_position_value(p0, p_lower, p_upper, L=1.0)
    vT = uniswap_v3_position_value(price_T, p_lower, p_upper, L=1.0)
    lp_rel = vT / v0  # LP_T / LP_0

    # HODL relative value, starting 50/50 at p0
    # HODL_0 = 0.5 + 0.5 = 1 (normalized)
    # HODL_T = 0.5 + 0.5*(price_T/p0) = 0.5*(1 + R)
    R = price_T / p0
    hodl_rel = 0.5 * (1.0 + R)  # HODL_T / HODL_0

    return lp_rel / hodl_rel


def trade_outcome_constant_product(
    x_reserve: float,
    y_reserve: float,
    delta_x: float,
) -> Tuple[float, float, float]:
    """
    Execute trade on constant product AMM (x * y = k).
    
    Trader adds delta_x of token X, receives delta_y of token Y.
    
    Args:
        x_reserve: Initial reserve of token X
        y_reserve: Initial reserve of token Y  
        delta_x: Amount of token X being swapped in (>0)
        
    Returns:
        tuple: (new_x_reserve, new_y_reserve, execution_price)
        where execution_price = delta_y / delta_x (Y per X)
    """
    if delta_x <= 0:
        raise ValueError("delta_x must be positive")
    if x_reserve <= 0 or y_reserve <= 0:
        raise ValueError("Reserves must be positive")
        
    k = x_reserve * y_reserve
    new_x = x_reserve + delta_x
    new_y = k / new_x
    
    if new_y <= 0:
        raise ValueError("Trade would drain the pool")

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
    
    Note: For typical AMM trades (buying X with Y), execution price will be 
    worse than mid_price, resulting in negative slippage.
    
    Returns:
        Slippage as decimal (e.g., -0.01 for 1% negative slippage)
    """
    if x_reserve <= 0 or y_reserve <= 0:
        raise ValueError("Reserves must be positive")
    if delta_x <= 0:
        raise ValueError("Trade size must be positive")
        
    mid_price = y_reserve / x_reserve
    _, _, exec_price = trade_outcome_constant_product(x_reserve, y_reserve, delta_x)
    
    if mid_price == 0:
        return np.nan
        
    return (exec_price - mid_price) / mid_price


def slippage_vs_trade_fraction(
    x_reserve: float,
    y_reserve: float,
    fractions: np.ndarray,
) -> pd.DataFrame:
    """
    Compute slippage for various trade sizes as fractions of x_reserve.
    
    Args:
        x_reserve: Initial reserve of token X
        y_reserve: Initial reserve of token Y
        fractions: Array of trade sizes as fractions of x_reserve (e.g., [0.01, 0.05])
        
    Returns:
        DataFrame with columns: ['fraction', 'delta_x', 'slippage']
    """
    records = []
    for f in fractions:
        if f <= 0:
            continue
        if f >= 1.0:
            # Skip trades that would remove entire reserve or more
            continue
        delta_x = f * x_reserve
        try:
            slip = slippage_from_trade(x_reserve, y_reserve, delta_x)
            records.append({
                "fraction": float(f),
                "delta_x": float(delta_x),
                "slippage": float(slip),
            })
        except ValueError:
            # Trade too large for pool
            break
            
    return pd.DataFrame(records)
