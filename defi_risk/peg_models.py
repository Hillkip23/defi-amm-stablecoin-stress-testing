from typing import Optional, Literal, Dict
import numpy as np
import pandas as pd

PegModelName = Literal["basic_ou", "stress_ou", "ou_jumps"]

PEG_MODEL_LABELS: Dict[PegModelName, str] = {
    "basic_ou": "Basic OU",
    "stress_ou": "Stress-aware OU",
    "ou_jumps": "OU with jumps",
}


def _make_time_grid(n_steps: int, T: float) -> np.ndarray:
    return np.linspace(0.0, T, n_steps + 1)


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
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = _make_time_grid(n_steps, T)

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


def simulate_stress_aware_ou_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    alpha_kappa: float = 1.0,
    beta_sigma: float = 3.0,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = _make_time_grid(n_steps, T)

    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        dist = np.abs(prev - peg)

        kappa_eff = kappa / (1.0 + alpha_kappa * dist)
        sigma_eff = sigma * (1.0 + beta_sigma * dist)

        drift = kappa_eff * (peg - prev) * dt
        diffusion = sigma_eff * np.sqrt(dt) * z

        prices[t, :] = prev + drift + diffusion

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def simulate_ou_with_jumps(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    jump_intensity: float = 0.2,
    jump_mean: float = -0.05,
    jump_std: float = 0.03,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    times = _make_time_grid(n_steps, T)

    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        drift = kappa * (peg - prev) * dt
        diffusion = sigma * np.sqrt(dt) * z

        n_jumps = np.random.poisson(lam=jump_intensity * dt, size=n_paths)
        jump_shocks = np.where(
            n_jumps > 0,
            np.random.normal(loc=jump_mean, scale=jump_std, size=n_paths),
            0.0,
        )

        new_price = prev + drift + diffusion + jump_shocks
        prices[t, :] = np.clip(new_price, 0.1, None)

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def simulate_peg_paths(
    model: PegModelName,
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    p0: float = 1.0,
    peg: float = 1.0,
    random_seed: Optional[int] = None,
    alpha_kappa: float = 1.0,
    beta_sigma: float = 3.0,
    jump_intensity: float = 0.2,
    jump_mean: float = -0.05,
    jump_std: float = 0.03,
) -> pd.DataFrame:
    if model == "basic_ou":
        return simulate_ou_peg_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=sigma,
            p0=p0,
            peg=peg,
            random_seed=random_seed,
        )
    elif model == "stress_ou":
        return simulate_stress_aware_ou_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=sigma,
            alpha_kappa=alpha_kappa,
            beta_sigma=beta_sigma,
            p0=p0,
            peg=peg,
            random_seed=random_seed,
        )
    elif model == "ou_jumps":
        return simulate_ou_with_jumps(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=sigma,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            p0=p0,
            peg=peg,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown peg model: {model}")


__all__ = [
    "PegModelName",
    "PEG_MODEL_LABELS",
    "simulate_ou_peg_paths",
    "simulate_stress_aware_ou_paths",
    "simulate_ou_with_jumps",
    "simulate_peg_paths",
]
