# src/peg_stress.py
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .peg_models import simulate_ou_peg_paths
from .amm_pricing import slippage_from_trade


# -----------------------------------------------------------
# 1) Depeg probabilities at horizon
# -----------------------------------------------------------
def depeg_probabilities(
    prices: pd.DataFrame,
    thresholds: Iterable[float] = (0.99, 0.95, 0.90),
) -> Dict[str, float]:
    """
    Compute P(p_T < θ) for a set of thresholds given peg path simulations.

    prices: DataFrame with shape (n_steps, n_paths); last row is p_T.
    thresholds: iterable of thresholds θ.
    """
    p_T = prices.iloc[-1, :]
    probs: Dict[str, float] = {}
    for thr in thresholds:
        key = f"p_T<{thr}"
        probs[key] = float((p_T < thr).mean())
    return probs


# -----------------------------------------------------------
# 1b) Depeg severity metrics over full paths
# -----------------------------------------------------------
def depeg_severity_metrics(
    prices: pd.DataFrame,
    threshold: float = 0.99,
) -> Dict[str, float]:
    """
    Compute severity metrics for depegs relative to a single threshold θ.

    Inputs
    ------
    prices : DataFrame (n_steps, n_paths)
        Simulated peg paths, one column per path.
    threshold : float
        Peg threshold θ (e.g. 0.99 or 1.00).

    Returns
    -------
    dict with keys:
        - expected_shortfall          : E[(θ - p_T)^+]
        - conditional_shortfall       : E[θ - p_T | p_T < θ]
        - time_under_peg              : fraction of time p_t < θ (averaged over paths)
        - worst_case_deviation        : min_t p_t (over all times, all paths)
        - depeg_probability           : P(p_T < θ)
    """
    # Convert to ndarray (T, N)
    if isinstance(prices, pd.DataFrame):
        arr = prices.values
    else:
        arr = np.asarray(prices)

    if arr.ndim != 2 or arr.size == 0:
        return {
            "expected_shortfall": 0.0,
            "conditional_shortfall": 0.0,
            "time_under_peg": 0.0,
            "worst_case_deviation": threshold,
            "depeg_probability": 0.0,
        }

    T, N = arr.shape

    # Terminal prices p_T
    p_T = arr[-1, :]  # (N,)
    depeg_mask_T = p_T < threshold
    depeg_prob = float(depeg_mask_T.mean()) if N > 0 else 0.0

    # Expected shortfall E[(θ - p_T)^+]
    shortfall_T = np.maximum(threshold - p_T, 0.0)
    expected_shortfall = float(shortfall_T.mean()) if N > 0 else 0.0

    # Conditional shortfall E[θ - p_T | p_T < θ]
    if depeg_prob > 0.0:
        conditional_shortfall = float(shortfall_T[depeg_mask_T].mean())
    else:
        conditional_shortfall = 0.0

    # Time under peg: 1/T ∫ 1_{p_t < θ} dt, approximated discretely
    below = arr < threshold  # (T, N)
    # mean over time then over paths = fraction of time under peg, averaged across paths
    time_under_peg = float(below.mean(axis=0).mean()) if N > 0 else 0.0

    # Worst-case deviation: min_t p_t over all times and paths
    worst_case_deviation = float(arr.min()) if N > 0 else threshold

    return {
        "expected_shortfall": expected_shortfall,
        "conditional_shortfall": conditional_shortfall,
        "time_under_peg": time_under_peg,
        "worst_case_deviation": worst_case_deviation,
        "depeg_probability": depeg_prob,
    }


# -----------------------------------------------------------
# 2) Run a single peg + liquidity scenario
# -----------------------------------------------------------
def run_peg_liquidity_scenario(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    p0: float,
    x_reserve: float,
    y_reserve: float,
    trade_fraction: float,
    thresholds: Iterable[float] = (0.99, 0.95, 0.90),
    peg: float = 1.0,
    random_seed: Optional[int] = 123,
) -> Dict:
    """
    Simulate OU peg paths, compute depeg probabilities at terminal time,
    and slippage for a single trade_fraction of the stablecoin reserve.
    """
    prices = simulate_ou_peg_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        kappa=kappa,
        sigma=sigma,
        p0=p0,
        peg=peg,
        random_seed=random_seed,
    )

    ou_probs = depeg_probabilities(prices, thresholds=thresholds)

    delta_x = trade_fraction * x_reserve
    slippage = slippage_from_trade(x_reserve, y_reserve, delta_x)

    return {
        "kappa": float(kappa),
        "sigma": float(sigma),
        "x_reserve": float(x_reserve),
        "y_reserve": float(y_reserve),
        "trade_fraction": float(trade_fraction),
        "ou_probs": ou_probs,
        "slippage": float(slippage),
    }
