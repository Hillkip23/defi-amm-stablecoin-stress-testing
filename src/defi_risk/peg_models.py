# src/peg_models.py

from typing import Optional
import numpy as np
import pandas as pd


def simulate_ou_peg_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate Ornstein–Uhlenbeck (OU) peg dynamics:

        dp_t = kappa * (peg - p_t) dt + sigma dW_t

    Parameters
    ----------
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    T : float
        Time horizon (e.g. in years).
    kappa : float
        Mean reversion speed towards `peg`.
    sigma : float
        Volatility of the peg process.
    p0 : float, default 1.0
        Initial price / peg level.
    peg : float, default 1.0
        Long-run mean (peg target).
    random_seed : Optional[int]
        Optional RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_steps + 1, n_paths)
        Index is the time grid from 0 to T.
        Each column is one simulated path.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]
        drift = kappa * (peg - prev) * dt
        diffusion = sigma * np.sqrt(dt) * z
        prices[t, :] = prev + drift + diffusion

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df

from defi_risk.peg_models import simulate_ou_peg_paths

paths = simulate_ou_peg_paths(
    n_paths=5,
    n_steps=365,
    T=1.0,
    kappa=5.0,
    sigma=0.02,
    p0=1.0,
    peg=1.0,
    random_seed=42,
)

print(paths.head())
print(paths.tail())


import matplotlib.pyplot as plt

paths.plot(legend=False)
plt.axhline(1.0, linestyle="--")
plt.title("OU Peg Simulation")
plt.show()
