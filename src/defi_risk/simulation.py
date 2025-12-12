from typing import Optional

import numpy as np
import pandas as pd

from .amm_pricing import impermanent_loss


def simulate_gbm_price_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    mu: float,
    sigma: float,
    p0: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate GBM price paths P_t.

    Returns a DataFrame with shape (n_steps+1, n_paths), index = time grid.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    prices = np.zeros((n_steps + 1, n_paths))
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prices[t, :] = prices[t - 1, :] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def compute_lp_vs_hodl(
    prices: pd.DataFrame,
    fee_apr: float = 0.0,
) -> pd.DataFrame:
    """Given price paths, compute LP vs HODL performance.

    Assumptions:
      - initial 50/50 portfolio (1 unit of token A, P0 units of token B)
      - HODL value at time t: 1 + P_t/P0
      - LP relative performance given by IL formula, plus simple fee APR.
    """
    # Take first column as representative path
    p0 = prices.iloc[0, 0]
    rel_prices = prices / p0  # R = P_t / P0

    # Impermanent loss (matrix) and LP vs HODL factor
    il = impermanent_loss(rel_prices)
    lp_rel = 1.0 + il  # LP value / HODL (no fees)

    times = prices.index.values
    # Simple deterministic fee factor over the whole horizon
    fee_factor = 1.0 + fee_apr * (times / times[-1])
    lp_rel_with_fees = lp_rel.mul(fee_factor[:, None])

    # HODL value (normalized)
    hodl_value = 1.0 + rel_prices
    lp_value = lp_rel_with_fees * hodl_value

    df = pd.DataFrame(
        {
            "price": prices.iloc[:, 0],
            "rel_price": rel_prices.iloc[:, 0],
            "hodl_value": hodl_value.iloc[:, 0],
            "lp_value": lp_value.iloc[:, 0],
            "lp_over_hodl": lp_value.iloc[:, 0] / hodl_value.iloc[:, 0],
            "impermanent_loss": il.iloc[:, 0],
        },
        index=prices.index,
    )
    df.index.name = "time"
    return df



from .amm_pricing import impermanent_loss  # make sure this import is present at top
from typing import Optional

def monte_carlo_lp_summary(
    n_paths: int,
    n_steps: int,
    T: float,
    mu: float,
    sigma: float,
    p0: float = 1.0,
    fee_apr: float = 0.10,
    random_seed: Optional[int] = 123,
) -> pd.DataFrame:
    """
    Run a Monte Carlo experiment over many GBM price paths and summarize
    end-of-horizon LP vs HODL outcomes.

    Returns a DataFrame with one row per path and columns:
    - R: relative price change P_T / P_0
    - IL: impermanent loss at T
    - hodl_T: HODL value at T
    - lp_T: LP value at T
    - lp_over_hodl_T: LP_T / HODL_T
    """
    prices_paths = simulate_gbm_price_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        mu=mu,
        sigma=sigma,
        p0=p0,
        random_seed=random_seed,
    )

    P_T = prices_paths.iloc[-1, :]  # final prices across paths
    R = P_T / p0                    # relative price change

    IL = impermanent_loss(R)

    hodl_T = 1.0 + R                # HODL value
    lp_rel = 1.0 + IL               # LP / HODL (no fees)

    fee_factor = 1.0 + fee_apr      # simple 1-year APR approximation
    lp_T = lp_rel * hodl_T * fee_factor
    lp_over_hodl_T = lp_T / hodl_T

    return pd.DataFrame(
        {
            "R": R,
            "IL": IL,
            "hodl_T": hodl_T,
            "lp_T": lp_T,
            "lp_over_hodl_T": lp_over_hodl_T,
        }
    )


def compute_lp_vs_hodl_dynamic_fees(
    prices: pd.DataFrame,
    fee_rate: float = 0.003,    # e.g. 30 bps per trade
    volume_scale: float = 1.0,  # scales how much volume you assume per unit volatility
) -> pd.DataFrame:
    """
    Compute LP vs HODL with a simple dynamic fee model.

    Assumptions:
    - HODL: 1 unit of token A + P0 units of token B, same as before.
    - Fees are earned whenever the price moves (proxy for trading volume).
    - Per-step fee yield is proportional to |log-return| * volume_scale * fee_rate.
    - Fee yield compounds over time.

    This is more realistic than a flat APR, because fee income increases with volatility.
    """
    # Use first column as representative path
    price_series = prices.iloc[:, 0]
    p0 = price_series.iloc[0]
    rel_price = price_series / p0

    # HODL value
    hodl_value = 1.0 + rel_price

    # Impermanent loss over time
    il_series = impermanent_loss(rel_price)
    lp_rel_no_fees = 1.0 + il_series   # LP/HODL without fees

    # Log-returns as a proxy for trading intensity
    log_returns = np.log(price_series).diff().fillna(0.0).abs()

    # Step-wise fee yield; think of this as "fractional yield for this step"
    step_fee_yield = fee_rate * volume_scale * log_returns

    # Cumulative fee factor (compounded)
    fee_factor = (1.0 + step_fee_yield).cumprod()

    # LP/HODL including dynamic fees
    lp_rel_with_fees = lp_rel_no_fees * fee_factor

    lp_value = lp_rel_with_fees * hodl_value

    df = pd.DataFrame(
        {
            "price": price_series,
            "rel_price": rel_price,
            "hodl_value": hodl_value,
            "lp_value": lp_value,
            "lp_over_hodl": lp_value / hodl_value,
            "impermanent_loss": il_series,
            "fee_factor": fee_factor,
        },
        index=prices.index,
    )
    df.index.name = "time"
    return df
