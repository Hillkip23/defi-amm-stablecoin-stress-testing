# src/defi_risk/peg_models.py

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


# NEW: state-dependent OU peg dynamics -------------------------------------


def simulate_state_dependent_ou_peg_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa_base: float,
    sigma_base: float,
    stress_path: np.ndarray,
    kappa_sensitivity: float = 1.0,
    sigma_sensitivity: float = 1.0,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate OU peg dynamics where mean reversion (kappa) and volatility (sigma)
    depend on a stress indicator at each time step.

        dp_t = kappa_t * (peg - p_t) dt + sigma_t dW_t

    A simple specification is:

        kappa_t = kappa_base / (1 + kappa_sensitivity * stress_t)
        sigma_t = sigma_base * (1 + sigma_sensitivity * stress_t)

    so that higher stress weakens mean reversion and increases volatility.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    T : float
        Time horizon (e.g. in years).
    kappa_base : float
        Baseline mean reversion speed.
    sigma_base : float
        Baseline volatility of the peg process.
    stress_path : np.ndarray
        Array of length n_steps with non-negative stress levels
        (e.g. in [0, 1] or [0, +inf)).
    kappa_sensitivity : float
        How strongly stress weakens mean reversion.
    sigma_sensitivity : float
        How strongly stress amplifies volatility.
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
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    stress_path = np.asarray(stress_path, dtype=float)
    if stress_path.shape[0] != n_steps:
        raise ValueError(
            f"stress_path must have length n_steps={n_steps}, "
            f"got {stress_path.shape[0]}"
        )
    if np.any(stress_path < 0):
        raise ValueError("stress_path must be non-negative.")

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        stress_t = stress_path[t - 1]

        # State-dependent parameters
        kappa_t = kappa_base / (1.0 + kappa_sensitivity * stress_t)
        sigma_t = sigma_base * (1.0 + sigma_sensitivity * stress_t)

        drift = kappa_t * (peg - prev) * dt
        diffusion = sigma_t * np.sqrt(dt) * z
        prices[t, :] = prev + drift + diffusion

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def make_stress_path_two_regime(
    n_steps: int,
    stress_low: float = 0.0,
    stress_high: float = 1.0,
    switch_step: Optional[int] = None,
) -> np.ndarray:
    """
    Simple helper to build a stress path representing a calm-to-stressed scenario.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    stress_low : float
        Stress level in the calm regime (e.g. 0.0).
    stress_high : float
        Stress level in the stressed regime (e.g. 1.0).
    switch_step : Optional[int]
        Time step at which to switch from low to high stress.
        If None, stress remains at stress_low.

    Returns
    -------
    np.ndarray
        Array of length n_steps with stress levels.
    """
    stress = np.full(n_steps, stress_low, dtype=float)
    if switch_step is not None and 0 <= switch_step < n_steps:
        stress[switch_step:] = stress_high
    return stress


def simulate_ou_peg_paths_with_jumps(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate OU peg dynamics with occasional jump shocks:

        dp_t = kappa * (peg - p_t) dt + sigma dW_t + J_t

    where J_t is a jump term. We approximate a Poisson jump process by
    allowing at most one jump per time step:

        with probability lambda_jump * dt, add a jump ~ Normal(jump_mean, jump_std)
        otherwise, no jump.

    This is useful to model rare but meaningful peg shocks, e.g. news events,
    bank failures, or protocol exploits.

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
        Diffusion volatility of the peg process.
    lambda_jump : float
        Expected number of jumps per unit time (e.g. 0.5 for ~1 jump every 2 years).
    jump_mean : float
        Mean jump size (in price units). Negative values model downward shocks.
    jump_std : float
        Standard deviation of the jump size.
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
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    # Probability of a jump in each time step
    p_jump = lambda_jump * dt
    p_jump = min(max(p_jump, 0.0), 1.0)  # clip for safety

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        # Diffusion part (standard OU)
        drift = kappa * (peg - prev) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Jump part: at most one jump per time step per path
        jump_indicator = np.random.binomial(1, p_jump, size=n_paths)
        jumps = np.where(
            jump_indicator == 1,
            np.random.normal(loc=jump_mean, scale=jump_std, size=n_paths),
            0.0,
        )

        prices[t, :] = prev + drift + diffusion + jumps

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df
