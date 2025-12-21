# defi_risk/simulation.py

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
    """Simulate GBM price paths P_t with constant (mu, sigma).

    Returns a DataFrame with shape (n_steps+1, n_paths), index = time grid in years.
    """
    rng = np.random.default_rng(random_seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        prices[t, :] = prices[t - 1, :] * np.exp(drift + vol * z)

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------
# Regime-aware GBM
# --------------------------------------------------------------------------


def simulate_regime_gbm_price_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    mu_by_regime: dict,
    sigma_by_regime: dict,
    regime_path: np.ndarray,
    p0: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate GBM price paths with piecewise-constant (mu, sigma)
    depending on a regime index at each time step.
    """
    rng = np.random.default_rng(random_seed)

    regime_path = np.asarray(regime_path)
    if regime_path.shape[0] != n_steps:
        raise ValueError(
            f"regime_path must have length n_steps={n_steps}, got {regime_path.shape[0]}"
        )

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        reg = int(regime_path[t - 1])  # regime for step (t-1 → t)
        mu_t = float(mu_by_regime[reg])
        sigma_t = float(sigma_by_regime[reg])

        drift = (mu_t - 0.5 * sigma_t**2) * dt
        vol = sigma_t * np.sqrt(dt)

        z = rng.standard_normal(n_paths)
        prices[t, :] = prices[t - 1, :] * np.exp(drift + vol * z)

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def make_two_regime_path(
    n_steps: int,
    regime_low: int = 0,
    regime_high: int = 1,
    switch_step: Optional[int] = None,
) -> np.ndarray:
    """
    Helper to build a regime path:
      - start in low regime
      - optionally switch to high regime at switch_step
    """
    regime_path = np.full(n_steps, regime_low, dtype=int)
    if switch_step is not None and 0 <= switch_step < n_steps:
        regime_path[switch_step:] = regime_high
    return regime_path


# --------------------------------------------------------------------------
# LP / HODL computations
# --------------------------------------------------------------------------


def compute_lp_vs_hodl_path(
    prices_1path: pd.DataFrame,
    fee_apr: float = 0.0,
) -> pd.DataFrame:
    """
    Time-series LP vs HODL for ONE path (prices DataFrame with exactly 1 column).

    Assumptions:
      - initial 50/50 portfolio
      - HODL value: 1 + P_t/P0   (same normalization you were using)
      - LP/HODL (no fees): 1 + IL(R_t)
      - fees: linear APR accrual over time (in years): factor(t) = 1 + fee_apr * t
    """
    if prices_1path.shape[1] != 1:
        raise ValueError("compute_lp_vs_hodl_path expects a DataFrame with exactly 1 column.")

    p0 = float(prices_1path.iloc[0, 0])
    rel_prices = prices_1path / p0

    il = impermanent_loss(rel_prices)
    lp_rel = 1.0 + il

    times = prices_1path.index.values.astype(float)
    fee_factor = 1.0 + fee_apr * times
    lp_rel_with_fees = lp_rel.mul(fee_factor[:, None])

    hodl_value = 1.0 + rel_prices
    lp_value = lp_rel_with_fees * hodl_value

    df = pd.DataFrame(
        {
            "price": prices_1path.iloc[:, 0],
            "rel_price": rel_prices.iloc[:, 0],
            "hodl_value": hodl_value.iloc[:, 0],
            "lp_value": lp_value.iloc[:, 0],
            "lp_over_hodl": lp_value.iloc[:, 0] / hodl_value.iloc[:, 0],
            "impermanent_loss": il.iloc[:, 0],
        },
        index=prices_1path.index,
    )
    df.index.name = "time"
    return df


def compute_lp_vs_hodl_summary(
    prices: pd.DataFrame,
    fee_apr: float = 0.0,
) -> pd.DataFrame:
    """
    Vectorized terminal LP vs HODL summary for ALL paths.

    Input:
      prices: DataFrame shape (n_steps+1, n_paths)
    Output:
      DataFrame shape (n_paths, ...) with terminal metrics per path
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    p0 = float(prices.iloc[0, 0])
    rel_prices = prices / p0  # (T, N)

    il = impermanent_loss(rel_prices)
    lp_rel = 1.0 + il

    times = prices.index.values.astype(float)
    T_years = float(times[-1])

    fee_factor_T = 1.0 + fee_apr * T_years

    R_T = rel_prices.iloc[-1, :]
    il_T = il.iloc[-1, :]
    lp_over_hodl_T = lp_rel.iloc[-1, :] * fee_factor_T

    hodl_T = 1.0 + R_T
    lp_T = lp_over_hodl_T * hodl_T

    return pd.DataFrame(
        {
            "price": prices.iloc[-1, :].values,
            "rel_price": R_T.values,
            "hodl_value": hodl_T.values,
            "lp_value": lp_T.values,
            "lp_over_hodl": lp_over_hodl_T.values,
            "impermanent_loss": il_T.values,
        }
    )


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

    Returns one row per path.
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
    return compute_lp_vs_hodl_summary(prices_paths, fee_apr=fee_apr)


def compute_lp_vs_hodl_dynamic_fees(
    prices_1path: pd.DataFrame,
    fee_rate: float = 0.003,  # e.g. 30 bps per trade
    volume_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Compute LP vs HODL with a simple dynamic fee model for ONE path.

    Assumptions:
    - Fees earned per step ∝ |log-return| * volume_scale * fee_rate.
    - Fee yield compounds over time.
    """
    if prices_1path.shape[1] != 1:
        raise ValueError("compute_lp_vs_hodl_dynamic_fees expects a DataFrame with exactly 1 column.")

    price_series = prices_1path.iloc[:, 0]
    p0 = float(price_series.iloc[0])
    rel_price = price_series / p0

    hodl_value = 1.0 + rel_price

    il_series = impermanent_loss(rel_price)
    lp_rel_no_fees = 1.0 + il_series

    log_returns_abs = np.log(price_series).diff().fillna(0.0).abs()

    step_fee_yield = fee_rate * volume_scale * log_returns_abs
    fee_factor = (1.0 + step_fee_yield).cumprod()

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
        index=prices_1path.index,
    )
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------
# Backwards-compatible alias (keeps older app code / notebooks working)
# --------------------------------------------------------------------------
def compute_lp_vs_hodl(prices: pd.DataFrame, fee_apr: float = 0.0) -> pd.DataFrame:
    """
    Backwards-compatible wrapper.

    - If `prices` has 1 column: returns the full time-series (old behavior).
    - If `prices` has many columns: returns terminal summary per path (new behavior).
    """
    if prices.shape[1] == 1:
        return compute_lp_vs_hodl_path(prices, fee_apr=fee_apr)
    return compute_lp_vs_hodl_summary(prices, fee_apr=fee_apr)
