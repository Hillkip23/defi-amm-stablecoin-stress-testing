import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from defi_risk.peg_models import simulate_ou_peg_paths
from defi_risk.peg_stress import depeg_probabilities
from defi_risk.amm_pricing import slippage_vs_trade_fraction


# -------------------------------
# Cached helpers
# -------------------------------

@st.cache_data(show_spinner=False)
def simulate_ou_cached(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    p0: float,
    peg: float,
    seed: int,
) -> pd.DataFrame:
    return simulate_ou_peg_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        kappa=kappa,
        sigma=sigma,
        p0=p0,
        peg=peg,
        random_seed=seed,
    )


@st.cache_data(show_spinner=False)
def slippage_grid_cached(
    x_reserve: float,
    y_reserve: float,
    fractions: np.ndarray,
) -> pd.DataFrame:
    return slippage_vs_trade_fraction(x_reserve, y_reserve, fractions)


@st.cache_data(show_spinner=False)
def load_grid_csv() -> pd.DataFrame | None:
    """Load precomputed peg/liquidity grid if available."""
    csv_path = Path(__file__).parent / "data" / "peg_liquidity_grid.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


# -------------------------------
# Main app
# -------------------------------

def main():
    st.set_page_config(
        page_title="Stablecoin Peg & Liquidity Stress Lab",
        layout="wide",
    )

    st.title("🪙 Stablecoin Peg & Liquidity Stress Lab")

    st.markdown(
        """
        This app models **soft-pegged stablecoins** using an Ornstein–Uhlenbeck (OU) process
        and stress-tests **AMM liquidity** using a constant-product (Uniswap v2-style) pool.

        Use the tabs to explore:
        1. **Scenario Explorer** – single-parameter OU + AMM stress  
        2. **Depeg vs Volatility** – depeg probability curves across σ  
        3. **Heatmap** – depeg probability across σ × reserves (precomputed grid)
        """
    )

    # ---------------------------
    # Shared sidebar controls
    # ---------------------------
    st.sidebar.header("Global Simulation Controls")

    # Peg dynamics
    n_paths = st.sidebar.slider("Number of OU paths", 200, 5000, 1000, step=200)
    n_steps = st.sidebar.slider("Steps per path", 50, 730, 365, step=25)
    T = st.sidebar.selectbox("Horizon (years)", [0.25, 0.5, 1.0], index=2)

    kappa = st.sidebar.slider("Mean reversion speed κ", 0.5, 15.0, 5.0, step=0.5)
    sigma = st.sidebar.slider("Peg volatility σ", 0.001, 0.10, 0.02, step=0.001)
    p0 = st.sidebar.number_input("Initial price p₀", value=1.0, min_value=0.5, max_value=1.5, step=0.01)
    peg = st.sidebar.number_input("Peg target", value=1.0, min_value=0.5, max_value=1.5, step=0.01)

    # AMM params
    st.sidebar.markdown("---")
    st.sidebar.subheader("AMM Pool & Trade Stress")

    x_reserve = st.sidebar.number_input("Stablecoin reserves (X)", value=10_000_000.0, step=500_000.0)
    y_reserve = st.sidebar.number_input("Collateral reserves (Y)", value=10_000_000.0, step=500_000.0)

    max_trade_pct = st.sidebar.slider("Max trade size as % of reserves", 1.0, 50.0, 20.0, step=1.0)
    selected_trade_pct = st.sidebar.slider("Highlighted trade size (%)", 1.0, max_trade_pct, 10.0, step=1.0)

    seed = st.sidebar.number_input("Random seed", value=123, step=1)

    thresholds = (0.99, 0.95, 0.90)

    # Tabs for the three plots / views
    tab1, tab2, tab3 = st.tabs(
        ["Scenario explorer", "Depeg vs volatility (σ)", "Heatmap: σ × reserves"]
    )

    # ---------------------------------------------------
    # TAB 1 – Scenario explorer (OU paths + Slippage)
    # ---------------------------------------------------
    with tab1:
        st.subheader("1️⃣ Scenario Explorer")

        prices = simulate_ou_cached(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            kappa=kappa,
            sigma=sigma,
            p0=p0,
            peg=peg,
            seed=seed,
        )
        ou_probs = depeg_probabilities(prices, thresholds=thresholds)

        col1, col2 = st.columns(2)

        # Left: OU paths + distribution
        with col1:
            st.markdown(
                f"**Peg parameters:** κ = `{kappa}`, σ = `{sigma}`, T = `{T}` years, "
                f"paths = `{n_paths}`"
            )

            # Sample paths
            n_plot = min(50, n_paths)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(prices.iloc[:, :n_plot], alpha=0.4, linewidth=1)
            ax.axhline(peg, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.set_title("Sample Peg Paths (OU)")
            st.pyplot(fig, clear_figure=True)

            # Terminal distribution
            p_T = prices.iloc[-1, :]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(p_T, bins=40, alpha=0.8)
            ax2.axvline(peg, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel("Terminal price p_T")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of Terminal Peg")
            st.pyplot(fig2, clear_figure=True)

        # Right: Depeg probs + slippage curve
        with col2:
            st.markdown("**Depeg probabilities (OU only):**")

            prob_rows = []
            for thr in thresholds:
                key = f"p_T<{thr}"
                prob_rows.append(
                    {
                        "Threshold": f"p_T < {thr}",
                        "Probability": ou_probs.get(key, float("nan")) * 100.0,
                    }
                )
            prob_df = pd.DataFrame(prob_rows)
            st.table(prob_df.style.format({"Probability": "{:.2f}%"}))

            st.markdown(
                "These probabilities come **only** from the peg model. "
                "Liquidity stress adds further price impact on top."
            )

            st.markdown("---")
            st.subheader("Slippage vs Trade Size (AMM)")

            fractions = np.linspace(0.01, max_trade_pct / 100.0, 25)
            slip_df = slippage_grid_cached(x_reserve, y_reserve, fractions)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(slip_df["fraction"] * 100, slip_df["slippage"] * 100, marker="o")
            ax3.set_xlabel("Trade size (% of reserves)")
            ax3.set_ylabel("Slippage (%)")
            ax3.set_title("Constant-Product AMM Slippage Curve")
            ax3.grid(True)

            highlight = slip_df.iloc[
                (slip_df["fraction"] - selected_trade_pct / 100.0).abs().argmin()
            ]
            ax3.axvline(highlight["fraction"] * 100, linestyle="--", color="red")
            st.pyplot(fig3, clear_figure=True)

            st.markdown(
                f"**Highlighted trade:** ~`{selected_trade_pct:.1f}%` of reserves → "
                f"slippage ≈ `{highlight['slippage'] * 100:.2f}%`"
            )

    # ---------------------------------------------------
    # TAB 2 – Depeg probability vs volatility (σ)
    # ---------------------------------------------------
    with tab2:
        st.subheader("2️⃣ Depeg Probability vs Volatility σ")

        st.markdown(
            "We sweep over peg volatility σ while keeping κ, T and other parameters fixed, "
            "and plot depeg probabilities."
        )

        n_sigma = st.slider("Number of σ points", 3, 10, 5, step=1)
        sigma_min = st.number_input("Min σ", value=0.01, step=0.005, format="%.3f")
        sigma_max = st.number_input("Max σ", value=0.05, step=0.005, format="%.3f")

        sigma_grid = np.linspace(sigma_min, sigma_max, n_sigma)

        rows = []
        for s in sigma_grid:
            prices_s = simulate_ou_cached(
                n_paths=n_paths,
                n_steps=n_steps,
                T=T,
                kappa=kappa,
                sigma=float(s),
                p0=p0,
                peg=peg,
                seed=seed,
            )
            probs_s = depeg_probabilities(prices_s, thresholds=thresholds)
            row = {"sigma": float(s)}
            row.update(probs_s)
            rows.append(row)

        sigma_df = pd.DataFrame(rows)

        fig4, ax4 = plt.subplots(figsize=(7, 5))
        for thr in thresholds:
            col = f"p_T<{thr}"
            if col in sigma_df.columns:
                ax4.plot(sigma_df["sigma"], sigma_df[col], marker="o", label=col)
        ax4.set_xlabel("Volatility σ")
        ax4.set_ylabel("Depeg Probability")
        ax4.set_title(f"Depeg Probability vs σ (κ={kappa}, T={T}y)")
        ax4.grid(True)
        ax4.legend()
        st.pyplot(fig4, clear_figure=True)

        st.dataframe(sigma_df)

    # ---------------------------------------------------
    # TAB 3 – Heatmap σ × reserves → depeg probability
    # ---------------------------------------------------
    with tab3:
        st.subheader("3️⃣ Heatmap: Depeg Probability Across σ × Reserves")

        grid_df = load_grid_csv()
        if grid_df is None:
            st.warning(
                "No precomputed grid found at `data/peg_liquidity_grid.csv`.\n\n"
                "Run `python -m scripts.run_peg_stress_grid` from the project root "
                "to generate it, then reload this app."
            )
        else:
            st.markdown(
                "This heatmap uses the precomputed grid from `data/peg_liquidity_grid.csv`.\n"
                "It shows how depeg probability changes with peg volatility σ and AMM reserves."
            )

            kappa_choice = st.selectbox(
                "Select mean reversion κ",
                sorted(grid_df["kappa"].unique()),
                index=1 if 5.0 in grid_df["kappa"].unique() else 0,
            )
            trade_fraction_choice = st.selectbox(
                "Select trade size (fraction of reserves)",
                sorted(grid_df["trade_fraction"].unique()),
                format_func=lambda x: f"{x*100:.1f}%",
            )
            threshold_cols = [c for c in grid_df.columns if c.startswith("p_T<")]
            threshold_choice = st.selectbox(
                "Select depeg threshold", threshold_cols, index=1 if "p_T<0.95" in threshold_cols else 0
            )

            dfh = grid_df[
                (grid_df["kappa"] == kappa_choice)
                & (grid_df["trade_fraction"] == trade_fraction_choice)
            ]

            sigmas = sorted(dfh["sigma"].unique())
            reserves = sorted(dfh["reserves"].unique())
            Z = np.zeros((len(reserves), len(sigmas)))

            for i, R in enumerate(reserves):
                for j, s in enumerate(sigmas):
                    val = dfh[(dfh["reserves"] == R) & (dfh["sigma"] == s)][
                        threshold_choice
                    ].mean()
                    Z[i, j] = val

            fig5, ax5 = plt.subplots(figsize=(8, 6))
            im = ax5.imshow(
                Z,
                origin="lower",
                cmap="viridis",
                extent=[min(sigmas), max(sigmas), min(reserves), max(reserves)],
                aspect="auto",
            )
            cbar = fig5.colorbar(im, ax=ax5)
            cbar.set_label(f"Depeg Probability ({threshold_choice})")

            ax5.set_xlabel("Volatility σ")
            ax5.set_ylabel("Reserves")
            ax5.set_title(
                f"Depeg Probability Heatmap\nκ={kappa_choice}, trade={trade_fraction_choice*100:.1f}% of reserves"
            )
            st.pyplot(fig5, clear_figure=True)

            st.markdown("Underlying grid data:")
            st.dataframe(dfh.reset_index(drop=True))


if __name__ == "__main__":
    main()
