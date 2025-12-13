# experiments/export_paper_figures.py

"""
Generate paper figures from the stablecoin stress-testing framework.

Outputs:
  figures/depeg_vs_sigma.pdf           (Figure 2)
  figures/depeg_severity_heatmap.pdf   (Figure 3)
"""

import sys
from pathlib import Path
from typing import Optional  # <-- use Optional instead of "|" union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

sys.path.append(str(SRC_DIR))
FIG_DIR.mkdir(exist_ok=True)

from defi_risk.peg_models import simulate_ou_peg_paths
from defi_risk.peg_stress import depeg_probabilities


# ---------------------------------------------------------------------
# Figure 2: Depeg probability vs volatility sigma
# ---------------------------------------------------------------------
def make_depeg_vs_sigma_figure(
    n_paths: int = 2000,
    n_steps: int = 252,
    T: float = 1.0,
    kappa: float = 5.0,
    sigma_min: float = 0.01,
    sigma_max: float = 0.05,
    n_sigma: int = 7,
    thresholds=(0.99, 0.95, 0.90),
):
    """
    Replicates the logic of the 'Depeg vs volatility (σ)' tab in stablecoin_app.py
    and saves a clean Matplotlib figure for the paper.
    """
    sigma_grid = np.linspace(sigma_min, sigma_max, n_sigma)

    rows = []
    for s in sigma_grid:
        prices_s = simulate_ou_peg_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=float(s),
            p0=1.0,
            peg=1.0,
            random_seed=123,
        )
        probs_s = depeg_probabilities(prices_s, thresholds=thresholds)
        row = {"sigma": float(s)}
        row.update(probs_s)
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # Expect column names like "p_T<0.99", "p_T<0.95", "p_T<0.9"
    for thr in thresholds:
        col = f"p_T<{thr}"
        if col in df.columns:
            ax.plot(
                df["sigma"],
                df[col],
                marker="o",
                label=fr"$\mathbb{{P}}(p_T < {thr})$",
            )

    ax.set_xlabel(r"Peg volatility $\sigma$")
    ax.set_ylabel("Depeg probability")
    ax.set_title(
        r"Depeg probability vs volatility $\sigma$ "
        r"(OU peg, $\kappa={}$)".format(kappa)
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_pdf = FIG_DIR / "depeg_vs_sigma.pdf"
    out_png = FIG_DIR / "depeg_vs_sigma.png"
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"[OK] Saved Figure 2 (depeg vs sigma) to:\n  {out_pdf}\n  {out_png}")


# ---------------------------------------------------------------------
# Figure 3: Depeg probability heatmap (sigma × reserves)
# ---------------------------------------------------------------------
def make_depeg_heatmap_figure(
    grid_path: Optional[Path] = None,
    kappa_choice: float = 5.0,
    trade_fraction_choice: float = 0.10,  # 10% of reserves
    threshold_col: str = "p_T<0.95",
):
    """
    Uses the precomputed grid in data/peg_liquidity_grid.csv to produce
    a σ × reserves depeg heatmap, matching the logic of TAB 3 in stablecoin_app.py.
    """
    if grid_path is None:
        grid_path = DATA_DIR / "peg_liquidity_grid.csv"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid CSV not found at {grid_path}. "
            "Run `python -m experiments.run_peg_stress_grid` first."
        )

    grid_df = pd.read_csv(grid_path)

    if threshold_col not in grid_df.columns:
        raise ValueError(
            f"Threshold column '{threshold_col}' not found in grid. "
            f"Available: {[c for c in grid_df.columns if c.startswith('p_T<')]}"
        )

    # Filter by chosen kappa and trade fraction
    dfh = grid_df[
        (grid_df["kappa"] == kappa_choice)
        & (grid_df["trade_fraction"] == trade_fraction_choice)
    ].copy()

    if dfh.empty:
        raise ValueError(
            f"No rows found for kappa={kappa_choice}, "
            f"trade_fraction={trade_fraction_choice}. "
            "Check peg_liquidity_grid.csv or adjust parameters."
        )

    sigmas = sorted(dfh["sigma"].unique())
    reserves = sorted(dfh["reserves"].unique())

    Z = np.zeros((len(reserves), len(sigmas)))
    for i, R in enumerate(reserves):
        for j, s in enumerate(sigmas):
            mask = (dfh["reserves"] == R) & (dfh["sigma"] == s)
            val = dfh.loc[mask, threshold_col].mean()
            Z[i, j] = val

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[min(sigmas), max(sigmas), min(reserves), max(reserves)],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Depeg probability ({threshold_col})")

    ax.set_xlabel(r"Peg volatility $\sigma$")
    ax.set_ylabel("Pool reserves (units of stablecoin)")
    ax.set_title(
        r"Depeg heatmap: $\sigma \times$ reserves "
        fr"($\kappa={kappa_choice}$, trade={trade_fraction_choice*100:.0f}\%)"
    )

    fig.tight_layout()
    out_pdf = FIG_DIR / "depeg_severity_heatmap.pdf"
    out_png = FIG_DIR / "depeg_severity_heatmap.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"[OK] Saved Figure 3 (heatmap) to:\n  {out_pdf}\n  {out_png}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project root: {ROOT}")
    print(f"Using data dir: {DATA_DIR}")
    print(f"Using figures dir: {FIG_DIR}")

    # Figure 2
    make_depeg_vs_sigma_figure()

    # Figure 3
    make_depeg_heatmap_figure(
        kappa_choice=5.0,
        trade_fraction_choice=0.10,   # 10% of reserves
        threshold_col="p_T<0.95",
    )

    print("[DONE] All paper figures generated.")
