#src/peg_stress.py
from typing import Dict, Iterable, Optional

import pandas as pd

from .peg_models import simulate_ou_peg_paths
from .amm_pricing import slippage_from_trade



# -----------------------------------------------------------
# 1) Define depeg_probabilities FIRST
# -----------------------------------------------------------
def depeg_probabilities(
    prices: pd.DataFrame,
    thresholds: Iterable[float] = (0.99, 0.95, 0.90),
) -> Dict[str, float]:
    p_T = prices.iloc[-1, :]
    probs: Dict[str, float] = {}
    for thr in thresholds:
        key = f"p_T<{thr}"
        probs[key] = float((p_T < thr).mean())
    return probs


# -----------------------------------------------------------
# 2) THEN define run_peg_liquidity_scenario
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
