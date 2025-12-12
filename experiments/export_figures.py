import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from defi_risk.peg_models import simulate_ou_peg_paths
from defi_risk.peg_stress import depeg_probabilities

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "peg_liquidity_grid.csv"


def export_depeg_vs_sigma(
    n_paths: int = 2000,
    n_steps: int = 252,
    T: float = 1.0,
    kappa: float = 5.0,
    p0: float = 1.0,
    peg: float = 1.0,
    sigma_min: float = 0.01,
    sigma_max: float = 0.05,
    n_sigma: int = 5,
    random_seed: int = 123,
):
    """
    Figure 1: Depeg probability vs peg volatility σ (OU model).
    Saves to figures/fig_depeg_vs_sigma.png
    """
    sigmas = np.linspace(sigma_min, sigma_max, n_sigma)
    thresholds = (0.99, 0.95, 0.90)
    rows = []

    for s in sigmas:
        prices = simulate_ou_peg_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=float(s),
            p0=p0,
            peg=peg,
            random_seed=random_seed,
        )
        probs = depeg_probabilities(prices, thresholds=thresholds)
        row = {"sigma": float(s)}
        row.update(probs)
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for thr in thresholds:
        col = f"p_T<{thr}"
        if col in df.columns:
            ax.plot(df["sigma"], df[col], marker="o", label=f"p_T < {thr}")

    ax.set_xlabel("Peg volatility σ")
    ax.set_ylabel("Depeg probability")
    ax.set_title(f"Depeg Probability vs Volatility (κ={kappa}, T={T} year)")
    ax.grid(True)
    ax.legend()

    out_path = FIG_DIR / "fig_depeg_vs_sigma.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def export_slippage_vs_trade_from_grid(
    sigma_val: float = 0.02,
    kappa_val: float = 5.0,
    reserves_val: float = 10_000_000.0,
):
    """
    Figure 2: Slippage vs trade size (% reserves) from precomputed grid.
    Uses data/peg_liquidity_grid.csv and saves figures/fig_slippage_vs_trade.png
    """
    df = pd.read_csv(CSV_PATH)

    subset = df[
        (df["sigma"] == sigma_val)
        & (df["kappa"] == kappa_val)
        & (df["reserves"] == reserves_val)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No rows in peg_liquidity_grid.csv for sigma={sigma_val}, "
            f"kappa={kappa_val}, reserves={reserves_val}"
        )

    subset["trade_pct"] = subset["trade_fraction"] * 100.0
    subset["slippage_pct"] = subset["slippage"] * 100.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(subset["trade_pct"], subset["slippage_pct"], marker="o")
    ax.set_xlabel("Trade size (% of reserves)")
    ax.set_ylabel("Slippage (%)")
    ax.set_title(
        f"AMM Slippage vs Trade Size\n"
        f"(σ={sigma_val}, κ={kappa_val}, reserves={reserves_val:,.0f})"
    )
    ax.grid(True)

    out_path = FIG_DIR / "fig_slippage_vs_trade.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def export_sigma_reserves_heatmap(
    trade_fraction_val: float = 0.10,
    kappa_val: float = 5.0,
    threshold_col: str = "p_T<0.95",
):
    """
    Figure 3: Heatmap of depeg probability across σ × reserves.
    Uses data/peg_liquidity_grid.csv and saves figures/fig_sigma_reserves_heatmap.png
    """
    df = pd.read_csv(CSV_PATH)

    if threshold_col not in df.columns:
        raise ValueError(
            f"Column {threshold_col} not found in peg_liquidity_grid.csv. "
            f"Available columns: {df.columns.tolist()}"
        )

    dfh = df[
        (df["kappa"] == kappa_val)
        & (df["trade_fraction"] == trade_fraction_val)
    ].copy()

    if dfh.empty:
        raise ValueError(
            f"No rows in peg_liquidity_grid.csv for kappa={kappa_val}, "
            f"trade_fraction={trade_fraction_val}"
        )

    sigmas = sorted(dfh["sigma"].unique())
    reserves = sorted(dfh["reserves"].unique())

    Z = np.zeros((len(reserves), len(sigmas)))

    for i, R in enumerate(reserves):
        for j, s in enumerate(sigmas):
            val = dfh[(dfh["reserves"] == R) & (dfh["sigma"] == s)][threshold_col].mean()
            Z[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        cmap="viridis",
        extent=[min(sigmas), max(sigmas), min(reserves), max(reserves)],
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Depeg probability ({threshold_col})")

    ax.set_xlabel("Peg volatility σ")
    ax.set_ylabel("Reserves")
    ax.set_title(
        f"Depeg Probability Heatmap\nκ={kappa_val}, trade={trade_fraction_val*100:.1f}% of reserves"
    )

    out_path = FIG_DIR / "fig_sigma_reserves_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    # 1) Depeg vs sigma (OU)
    export_depeg_vs_sigma()

    # 2) Slippage vs trade size using grid (tweak params as needed)
    export_slippage_vs_trade_from_grid(
        sigma_val=0.02,
        kappa_val=5.0,
        reserves_val=10_000_000.0,
    )

    # 3) Sigma × reserves heatmap
    export_sigma_reserves_heatmap(
        trade_fraction_val=0.10,
        kappa_val=5.0,
        threshold_col="p_T<0.95",
    )

    print("[DONE] All figures exported to 'figures/'")

