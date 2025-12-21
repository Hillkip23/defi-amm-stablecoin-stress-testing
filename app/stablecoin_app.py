# app/stablecoin_app.py

import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path so we can import `defi_risk`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# Imports (keep consistent)
# ============================
from defi_risk.simulation import (
    simulate_gbm_price_paths,
    compute_lp_vs_hodl_path,   # ⬅️ removed compute_lp_vs_hodl_summary here
)
from defi_risk.amm_pricing import (
    lp_over_hodl_univ3,
    slippage_vs_trade_fraction,
)
from defi_risk.peg_models import PEG_MODEL_LABELS, simulate_peg_paths
import defi_risk.peg_stress as peg_stress

from defi_risk.dune_client import (
    get_uniswap_eth_usdc_daily_prices,
    get_usdc_daily_prices,
)

# =====================================================
# Local LP summary helper (replaces imported version)
# =====================================================
def compute_lp_vs_hodl_summary(prices: pd.DataFrame, fee_apr: float) -> pd.DataFrame:
    """
    Simple LP vs HODL summary at horizon, path by path.
    """
    p_T = prices.iloc[-1, :].astype(float).values
    p0 = float(prices.iloc[0, 0])
    rel_move = p_T / p0 - 1.0

    hodl = p_T / p0
    il = -(rel_move ** 2)
    lp = (1.0 + fee_apr) * (1.0 + il)
    lp_over_hodl = lp / hodl

    return pd.DataFrame(
        {
            "price": p_T,
            "hodl": hodl,
            "lp": lp,
            "impermanent_loss": il,
            "lp_over_hodl": lp_over_hodl,
        }
    )


# =====================================================
# Helpers for calibration
# =====================================================

def load_price_series(source, max_years: int = 5) -> pd.Series:
    """
    Load a CSV (path or uploaded file) and return a cleaned daily close price
    series, trimmed to the last `max_years` years.
    """
    def _read(**kwargs):
        return pd.read_csv(source, **kwargs)

    df_raw = _read(comment="#")

    if df_raw.shape[1] < 2:
        if hasattr(source, "seek"):
            source.seek(0)
        df_raw = _read(skiprows=1)

    if df_raw.shape[1] < 2:
        raise ValueError("CSV must have at least two columns (date & price).")

    cols_lower = {c.lower(): c for c in df_raw.columns}
    date_col = cols_lower.get("date", df_raw.columns[0])

    price_col = None
    for key in ("close", "price", "adj close", "close_usd"):
        if key in cols_lower:
            price_col = cols_lower[key]
            break
    if price_col is None:
        price_col = df_raw.columns[1]

    df = df_raw[[date_col, price_col]].rename(
        columns={date_col: "date", price_col: "close"}
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date")

    if max_years is not None and not df.empty:
        cutoff = df["date"].max() - pd.Timedelta(days=365 * max_years)
        df = df[df["date"] >= cutoff]

    df = df.set_index("date")
    return df["close"]


def estimate_gbm_params(prices: pd.Series, trading_days: int = 252):
    """
    Estimate GBM drift and vol from a price series.
    Returns annualized mu, sigma and the log-returns.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    mu_daily = log_ret.mean()
    sigma_daily = log_ret.std(ddof=1)

    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    return mu_annual, sigma_annual, log_ret


def calibrate_ou_from_usdc_prices(prices: pd.Series) -> dict:
    """
    Estimate OU parameters (mu, sigma, kappa) from a USDC price series around 1.
    Treat deviations from 1 as an AR(1) and map to continuous-time OU.
    """
    x = prices.dropna().astype(float).values - 1.0
    if len(x) < 3:
        return {"mu": 1.0, "sigma": 0.01, "kappa": 1.0}

    dt = 1.0  # daily
    x_t = x[:-1]
    x_tp1 = x[1:]

    # Robust AR(1) via OLS with intercept: x_{t+1} = a + b x_t + eps
    X = np.vstack([np.ones_like(x_t), x_t]).T
    a, b = np.linalg.lstsq(X, x_tp1, rcond=None)[0]
    residuals = x_tp1 - (a + b * x_t)
    sigma_eps = np.std(residuals, ddof=2)

    kappa = -np.log(b) / dt if 0.0 < b < 1.0 else 1.0
    mu_dev = a / (1.0 - b) if abs(1.0 - b) > 1e-8 else 0.0
    mu = 1.0 + mu_dev

    denom = (1.0 - np.exp(-2.0 * kappa * dt))
    sigma = sigma_eps * np.sqrt(2.0 * kappa / denom) if denom > 1e-12 else float(sigma_eps)

    return {"mu": float(mu), "sigma": float(sigma), "kappa": float(kappa)}

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="DeFi AMM & Stablecoin Stress Lab",
    layout="wide",
)

# Initialize slider state before widgets
if "mu_slider" not in st.session_state:
    st.session_state.mu_slider = 0.0
if "sigma_slider" not in st.session_state:
    st.session_state.sigma_slider = 0.8

st.title("DeFi AMM & Stablecoin Stress Lab")

# ---- About / model overview ----
with st.expander("About this lab / model overview", expanded=True):
    st.markdown(
        """
This lab combines **DeFi AMM risk** and **stablecoin peg stress testing** in a single,
fully reproducible environment.

**What this app does**
- Simulates GBM price paths and evaluates LP vs HODL performance (including dynamic fees).
- Models soft-pegged stablecoins with OU, stress-aware OU, and OU-with-jumps dynamics.
- Quantifies not just *if* a stablecoin depegs, but *how severely* and *for how long*.
- Maps depeg and LP risk across volatility × liquidity surfaces.
- Links interactive simulations directly to the empirical calibrations used in the paper.

**Key questions it answers**
- When does LP provision remain attractive vs simply holding the asset?
- How much volatility and how little liquidity does it take to break a peg?
- How do dynamic v2-style fees compare to static concentrated v3 ranges for LPs?
"""
    )

# ---- Parameter documentation ----
with st.expander("Parameter documentation"):
    st.markdown(
        """
**μ (Drift)**: Annualized expected return of the risky asset (GBM).

**σ (Volatility)**: Annualized standard deviation of returns.

**κ (Mean reversion speed)**: How quickly the stablecoin price is pulled back to its peg in OU models.

**Fee APR**: Annualized trading fee yield earned by LPs (before IL and volatility).
"""
    )

# =====================================================
# LP quick presets (must come BEFORE sliders)
# =====================================================
st.markdown("#### Quick LP scenario presets")
lp_preset_col1, lp_preset_col2 = st.columns(2)
with lp_preset_col1:
    if st.button("Dynamic v2 vs static v3 (baseline)", key="lp_preset_baseline"):
        st.session_state.mu_slider = 0.0
        st.session_state.sigma_slider = 0.8

with lp_preset_col2:
    if st.button("High-volatility LP stress", key="lp_preset_stress"):
        st.session_state.mu_slider = 0.0
        st.session_state.sigma_slider = 1.2

# =====================================================
# Sidebar: core simulation parameters
# =====================================================
st.sidebar.header("Simulation Parameters")

default_mu = float(st.session_state.get("mu_slider", 0.0))
default_sigma = float(st.session_state.get("sigma_slider", 0.8))

n_paths = st.sidebar.slider("Number of Paths", 1, 5000, 1000)
n_steps = st.sidebar.slider("Steps per Path", 50, 500, 365)
T = st.sidebar.slider("Years (T)", 0.1, 5.0, 1.0)

mu = st.sidebar.slider(
    "Drift (mu)",
    -0.5,
    0.5,
    value=default_mu,
    key="mu_slider",
)

sigma = st.sidebar.slider(
    "Volatility (sigma)",
    0.1,
    2.0,
    value=default_sigma,
    key="sigma_slider",
)

fee_apr = st.sidebar.slider("Base Fee APR (constant model)", 0.0, 1.0, 0.1)
p0 = 1.0

st.sidebar.markdown("### Dynamic Fee Model")
fee_sensitivity = st.sidebar.slider(
    "Fee sensitivity to realized volatility", 0.0, 2.0, 0.5
)

st.sidebar.markdown("### Uniswap v3 Range (relative to P₀ = 1)")
p_lower = st.sidebar.slider("Lower bound", 0.2, 1.0, 0.8)
p_upper = st.sidebar.slider("Upper bound", 1.0, 5.0, 1.2)

# =====================================================
# Helper: run one Monte Carlo block and return (prices, summary_df)
# =====================================================
def run_mc_block(n_paths, n_steps, T, mu, sigma, fee_apr, p0=1.0, random_seed=42):
    prices = simulate_gbm_price_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        mu=mu,
        sigma=sigma,
        p0=p0,
        random_seed=random_seed,
    )
    summary_df = compute_lp_vs_hodl_summary(prices, fee_apr=fee_apr)
    return prices, summary_df

@st.cache_data(show_spinner=False)
def run_mc_block_cached(
    n_paths,
    n_steps,
    T,
    mu,
    sigma,
    fee_apr,
    p0=1.0,
    random_seed=42,
    _cache_version="v4",
):
    return run_mc_block(n_paths, n_steps, T, mu, sigma, fee_apr, p0, random_seed)

# =====================================================
# MAIN: run base simulation
# =====================================================
run_clicked = st.button("Run Simulation")

if run_clicked:
    with st.spinner("Running simulations..."):
        prices, summary_df = run_mc_block_cached(
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
            mu=mu,
            sigma=sigma,
            fee_apr=fee_apr,
            p0=p0,
            random_seed=42,
        )

    st.subheader("LP Performance Distribution (Terminal, all paths)")
    st.write(summary_df.describe())

    st.subheader("Summary of LP / HODL at Horizon")
    lp_stats = summary_df["lp_over_hodl"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    st.write(lp_stats)

    x = summary_df["lp_over_hodl"].values
    if len(x) >= 2:
        mean = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        st.caption(
            f"Mean LP/HODL ≈ {mean:.3f} (approx 95% CI: {mean-1.96*se:.3f} to {mean+1.96*se:.3f})"
        )

    st.subheader("Histogram of LP / HODL (static fee, v2)")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(summary_df["lp_over_hodl"], bins=50)
    ax_hist.set_xlabel("LP / HODL at horizon")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True)
    st.pyplot(fig_hist)

    # Dynamic fee approximation
    realized_vols = []
    dt = T / n_steps
    for i in range(n_paths):
        path_prices = prices.iloc[:, i]
        log_ret = np.log(path_prices / path_prices.shift(1)).dropna()
        vol_ann = log_ret.std(ddof=1) / np.sqrt(dt)
        realized_vols.append(vol_ann)

    summary_df["realized_vol"] = realized_vols
    dynamic_fee_apr = (
        fee_apr + fee_sensitivity * summary_df["realized_vol"]
    ).clip(lower=0.0, upper=1.0)
    summary_df["dynamic_fee_apr"] = dynamic_fee_apr
    summary_df["lp_over_hodl_dynamic"] = (1.0 + summary_df["impermanent_loss"]) * (
        1.0 + dynamic_fee_apr * T
    )

    if not summary_df.empty:
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download LP simulation results as CSV",
            data=csv,
            file_name="lp_simulation_results.csv",
            mime="text/csv",
        )

    st.subheader("Summary of LP / HODL with Dynamic Fees (Uniswap v2)")
    st.write(
        summary_df["lp_over_hodl_dynamic"].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
    )

    st.subheader("Histogram of LP / HODL with Dynamic Fees (v2)")
    fig_dyn, ax_dyn = plt.subplots()
    ax_dyn.hist(summary_df["lp_over_hodl_dynamic"], bins=50)
    ax_dyn.set_xlabel("LP / HODL at horizon (dynamic fees)")
    ax_dyn.set_ylabel("Frequency")
    ax_dyn.grid(True)
    st.pyplot(fig_dyn)

    # v3 comparison
    summary_df["lp_over_hodl_v3"] = summary_df["price"].apply(
        lambda p: lp_over_hodl_univ3(p, p_lower=p_lower, p_upper=p_upper)
    )

    st.subheader(
        f"Summary of LP / HODL (Uniswap v3, range [{p_lower:.2f}, {p_upper:.2f}])"
    )
    st.write(
        summary_df["lp_over_hodl_v3"].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
    )

    fig_v3, ax_v3 = plt.subplots()
    ax_v3.hist(summary_df["lp_over_hodl_v3"], bins=50)
    ax_v3.set_xlabel("LP / HODL at horizon (v3)")
    ax_v3.set_ylabel("Frequency")
    ax_v3.grid(True)
    st.pyplot(fig_v3)

    # Single-path visualizer
    st.subheader("Single Path Visualizer")

    path_idx = st.number_input(
        "Path index (0-based)",
        min_value=0,
        max_value=n_paths - 1,
        value=0,
        step=1,
    )

    df_path = compute_lp_vs_hodl_path(prices[[path_idx]], fee_apr=fee_apr)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Price Path**")
        fig_p, ax_p = plt.subplots()
        ax_p.plot(df_path.index.values, df_path["price"])
        ax_p.set_xlabel("Time (years)")
        ax_p.set_ylabel("Price")
        ax_p.grid(True)
        st.pyplot(fig_p)

    with col_p2:
        st.markdown("**LP vs HODL & IL over Time**")
        fig_v, ax_v = plt.subplots()
        ax_v.plot(df_path.index.values, df_path["lp_over_hodl"], label="LP / HODL")
        ax_v2 = ax_v.twinx()
        ax_v2.plot(
            df_path.index.values,
            df_path["impermanent_loss"],
            color="tab:red",
            label="IL",
        )
        ax_v.set_xlabel("Time (years)")
        ax_v.set_ylabel("LP / HODL")
        ax_v2.set_ylabel("Impermanent loss")
        ax_v.grid(True)
        st.pyplot(fig_v)

else:
    st.info("Adjust parameters on the left and click **Run Simulation**.")

# =====================================================
# Real-Data Calibration (GBM from Historical Prices)
# =====================================================

st.header("Real-Data Calibration (GBM from Historical Prices)")

asset_choice = st.selectbox(
    "Choose asset / data source",
    [
        "BTC (Binance USDT, local CSV)",
        "ETH (Binance USDT, local CSV)",
        "UNI (Binance USDT, local CSV)",
        "XRP (Binance USDT, local CSV)",
        "S&P 500 (^SPX, local CSV)",
        "ETH (on-chain, Dune prices.usd)",
        "Upload custom CSV...",
    ],
)

prices_series = None

file_map = {
    "BTC (Binance USDT, local CSV)": "data/btc_5y_daily.csv",
    "ETH (Binance USDT, local CSV)": "data/eth_5y_daily.csv",
    "UNI (Binance USDT, local CSV)": "data/uni_5y_daily.csv",
    "XRP (Binance USDT, local CSV)": "data/xrp_5y_daily.csv",
    "S&P 500 (^SPX, local CSV)": "data/sp500_5y_daily.csv",
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_uniswap_eth_usdc_daily_prices_cached():
    return get_uniswap_eth_usdc_daily_prices()

@st.cache_data(ttl=3600, show_spinner=False)
def get_usdc_daily_prices_cached():
    return get_usdc_daily_prices()

if asset_choice == "Upload custom CSV...":
    uploaded = st.file_uploader(
        "Upload CSV with columns like Date, Close (daily data)",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            prices_series = load_price_series(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

elif asset_choice == "ETH (on-chain, Dune prices.usd)":
    try:
        prices_series = get_uniswap_eth_usdc_daily_prices_cached()
    except Exception as e:
        st.error(f"Could not fetch on-chain prices from Dune: {e}")

else:
    path = file_map[asset_choice]
    if os.path.exists(path):
        try:
            prices_series = load_price_series(path)
        except Exception as e:
            st.error(f"Could not read {path}: {e}")
    else:
        st.warning(f"File not found: {path}. Put it in your data/ folder or update file_map.")

if prices_series is not None and not prices_series.empty:
    st.subheader("Cleaned Price Series (last ~5 years)")
    st.line_chart(prices_series)

    mu_hat, sigma_hat, log_ret = estimate_gbm_params(prices_series)

    st.subheader("Estimated GBM Parameters (annualized)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("μ (drift)", f"{mu_hat:.2%}")
    with col2:
        st.metric("σ (volatility)", f"{sigma_hat:.2%}")

    def apply_calibrated_params():
        st.session_state.mu_slider = float(mu_hat)
        st.session_state.sigma_slider = float(sigma_hat)
        st.session_state["calibration_applied"] = True

    st.button(
        "Use these parameters for simulation",
        on_click=apply_calibrated_params,
        key="apply_calibrated_params_btn",
    )

    if st.session_state.get("calibration_applied", False):
        st.success("Updated Drift μ and Volatility σ sliders from calibration.")

else:
    st.info("Select an asset and/or upload a CSV to run calibration.")

# =====================================================
# Stablecoin Peg & Liquidity Stress Lab (OU + AMM)
# =====================================================

st.header("🪙 Stablecoin Peg & Liquidity Stress Lab")

st.markdown(
    """
This module explores **stablecoin peg behavior** and **liquidity resilience** under
different levels of volatility, mean reversion, and pool depth.
"""
)

def load_grid_csv():
    csv_path = DATA_DIR / "peg_liquidity_grid.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)

tab1, tab2, tab3 = st.tabs(
    ["Scenario explorer", "Depeg vs volatility (σ)", "Heatmap: σ × reserves"]
)

# TAB 1 – Scenario explorer (OU + slippage)
with tab1:
    peg_model_key = st.selectbox(
        "Peg model",
        options=list(PEG_MODEL_LABELS.keys()),
        format_func=lambda k: PEG_MODEL_LABELS[k],
        key="peg_model_choice",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Peg Dynamics Parameters")

        n_paths_peg = st.slider(
            "Number of OU paths", 200, 5000, 1000, key="peg_n_paths"
        )
        n_steps_peg = st.slider(
            "Steps per path (peg)", 50, 365, 252, key="peg_n_steps"
        )
        T_peg = st.slider("Horizon (years)", 0.1, 2.0, 1.0, key="peg_T")
        kappa_peg = st.slider(
            "Mean reversion speed (κ)", 0.1, 10.0, 5.0, key="peg_kappa"
        )
        sigma_peg = st.slider(
            "Peg volatility (σ)", 0.001, 0.1, 0.02, key="peg_sigma"
        )
        p0_peg = st.slider("Initial price p₀", 0.90, 1.10, 1.00, key="peg_p0")
        peg_target = 1.0

        st.markdown("#### Stress Model Parameters")
        beta_sigma = st.slider(
            "Stress Amplification (β)",
            0.0, 10.0, 3.0,
            key="peg_beta_sigma",
        )

        st.markdown("#### Jump Process Parameters (for 'OU with jumps')")
        jump_intensity = st.slider(
            "Jump Intensity (λ)", 0.0, 1.0, 0.2, key="peg_jump_intensity"
        )
        jump_mean = st.slider(
            "Jump Mean", -0.1, 0.0, -0.05, key="peg_jump_mean"
        )
        jump_std = st.slider(
            "Jump Std Dev", 0.0, 0.1, 0.03, key="peg_jump_std"
        )

        seed_peg = 42

        st.markdown("#### Calibrate from USDC on-chain prices (Dune)")
        if st.button("Use USDC (Dune) for OU parameters", key="usdc_calib_btn"):
            try:
                usdc_prices = get_usdc_daily_prices_cached()
                params = calibrate_ou_from_usdc_prices(usdc_prices)
                st.session_state["peg_kappa"] = params["kappa"]
                st.session_state["peg_sigma"] = params["sigma"]
                st.session_state["peg_p0"] = params["mu"]
                st.success(
                    f"Calibrated from USDC: μ≈{params['mu']:.4f}, σ≈{params['sigma']:.4f}, κ≈{params['kappa']:.2f}"
                )
            except Exception as e:
                st.error(f"USDC calibration failed: {e}")

        severity_threshold = st.slider(
            "Severity Threshold (θ)",
            0.90, 1.00, 0.99,
            key="peg_severity_threshold",
        )

    with col_right:
        st.subheader("AMM Pool & Trade Stress")
        reserve_stable = st.number_input(
            "Stablecoin reserve in pool",
            100_000.0,
            100_000_000.0,
            10_000_000.0,
            step=100_000.0,
        )
        reserve_collateral = st.number_input(
            "Collateral reserve in pool",
            100_000.0,
            100_000_000.0,
            10_000_000.0,
            step=100_000.0,
        )
        max_trade_pct = st.slider(
            "Max trade size as % of reserves",
            1.0,
            50.0,
            20.0,
            key="peg_max_trade",
        )
        highlight_trade_pct = st.slider(
            "Highlighted trade size (%)",
            1.0,
            max_trade_pct,
            10.0,
            key="peg_highlight_trade",
        )

    run_peg = st.button("Run stablecoin scenario")

    if run_peg:
        prices_peg = simulate_peg_paths(
            model=peg_model_key,
            n_paths=n_paths_peg,
            n_steps=n_steps_peg,
            T=T_peg,
            kappa=float(st.session_state.get("peg_kappa", kappa_peg)),
            sigma=float(st.session_state.get("peg_sigma", sigma_peg)),
            p0=float(st.session_state.get("peg_p0", p0_peg)),
            peg=peg_target,
            random_seed=seed_peg,
            alpha_kappa=1.0,
            beta_sigma=beta_sigma,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
        )

        # 2. Standard depeg probabilities
        ou_probs = peg_stress.depeg_probabilities(
            prices_peg, thresholds=(0.99, 0.95, 0.90)
        )

        # 3. Full severity metrics at chosen θ
        severity_results = peg_stress.depeg_severity_metrics(
            prices_peg, threshold=severity_threshold
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Sample Peg Paths")
            n_plot = min(50, n_paths_peg)
            fig, ax = plt.subplots()
            ax.plot(prices_peg.iloc[:, :n_plot], alpha=0.4, linewidth=1)
            ax.axhline(peg_target, color="black", linestyle="--")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            st.subheader("Terminal Peg Distribution")
            p_T = prices_peg.iloc[-1, :]
            fig2, ax2 = plt.subplots()
            ax2.hist(p_T, bins=40)
            ax2.axvline(peg_target, color="black", linestyle="--")
            ax2.set_xlabel("p_T")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

        with c2:
            st.subheader("Depeg Severity Metrics (OU)")

            p = float(severity_results["depeg_probability"])
            n = int(prices_peg.shape[1])
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            ci_low = max(0.0, p - 1.96 * se)
            ci_high = min(1.0, p + 1.96 * se)

            severity_rows = [
                {
                    "Metric": "Expected Shortfall (E[(θ - p_T)^+])",
                    "Value": f"{severity_results['expected_shortfall']:.4f}",
                },
                {
                    "Metric": "Conditional Shortfall (E[θ - p_T | p_T < θ])",
                    "Value": f"{severity_results['conditional_shortfall']:.4f}",
                },
                {
                    "Metric": f"Time Under Peg (θ = {severity_threshold:.2f})",
                    "Value": f"{severity_results['time_under_peg'] * 100:.2f}%",
                },
                {
                    "Metric": "Worst Case Deviation (min p_t)",
                    "Value": f"{severity_results['worst_case_deviation']:.4f}",
                },
                {
                    "Metric": f"Depeg Probability (p_T < {severity_threshold:.2f})",
                    "Value": f"{p * 100:.2f}% (≈95% CI: {ci_low*100:.2f}–{ci_high*100:.2f})",
                },
            ]
            st.table(pd.DataFrame(severity_rows))

            st.subheader("Depeg Probabilities (Standard Thresholds)")
            rows = []
            for thr in (0.99, 0.95, 0.90):
                key = f"p_T<{thr}"
                rows.append(
                    {
                        "Threshold": f"p_T < {thr}",
                        "Probability (%)": ou_probs.get(key, 0.0) * 100,
                    }
                )
            st.table(pd.DataFrame(rows).style.format({"Probability (%)": "{:.2f}"}))

            st.subheader("Slippage vs Trade Size (AMM)")
            fractions = np.linspace(0.01, max_trade_pct / 100.0, 25)
            slip_df = slippage_vs_trade_fraction(
                reserve_stable, reserve_collateral, fractions
            )
            slip_df["trade_pct"] = slip_df["fraction"] * 100
            slip_df["slippage_pct"] = slip_df["slippage"] * 100

            fig3, ax3 = plt.subplots()
            ax3.plot(
                slip_df["trade_pct"], slip_df["slippage_pct"], marker="o"
            )
            ax3.set_xlabel("Trade size (% of reserves)")
            ax3.set_ylabel("Slippage (%)")
            ax3.grid(True)

            highlight = slip_df.iloc[
                (slip_df["trade_pct"] - highlight_trade_pct).abs().argmin()
            ]
            ax3.axvline(highlight["trade_pct"], linestyle="--", color="red")
            st.pyplot(fig3)

            st.markdown(
                f"Highlighted trade ≈ **{highlight['trade_pct']:.1f}%** of reserves → "
                f"slippage ≈ **{highlight['slippage_pct']:.2f}%**"
            )

# TAB 2 – Depeg probability vs σ
with tab2:
    st.subheader("Depeg Probability vs Volatility σ")

    n_paths_peg_curve = 2000
    n_steps_peg_curve = 252
    T_peg_curve = 1.0

    kappa_fixed = st.slider(
        "κ for this curve", 0.1, 10.0, 5.0, key="curve_kappa"
    )
    p0_fixed = 1.0
    peg_target = 1.0

    n_sigma = st.slider("Number of σ points", 3, 10, 5, key="curve_n_sigma")
    sigma_min = st.number_input(
        "Min σ", value=0.01, step=0.005, format="%.3f", key="curve_sigma_min"
    )
    sigma_max = st.number_input(
        "Max σ", value=0.05, step=0.005, format="%.3f", key="curve_sigma_max"
    )

    sigma_grid = np.linspace(sigma_min, sigma_max, n_sigma)
    rows = []
    for s in sigma_grid:
        prices_s = simulate_peg_paths(
            model="basic_ou",
            n_paths=n_paths_peg_curve,
            n_steps=n_steps_peg_curve,
            T=T_peg_curve,
            kappa=kappa_fixed,
            sigma=float(s),
            p0=p0_fixed,
            peg=peg_target,
            random_seed=123,
        )
        probs_s = peg_stress.depeg_probabilities(
            prices_s, thresholds=(0.99, 0.95, 0.90)
        )
        row = {"sigma": float(s)}
        row.update(probs_s)
        rows.append(row)

    sigma_df = pd.DataFrame(rows)

    fig4, ax4 = plt.subplots()
    for col in ["p_T<0.99", "p_T<0.95", "p_T<0.90"]:
        if col in sigma_df.columns:
            ax4.plot(sigma_df["sigma"], sigma_df[col], marker="o", label=col)
    ax4.set_xlabel("Volatility σ")
    ax4.set_ylabel("Depeg Probability")
    ax4.set_title(f"Depeg Probability vs σ (κ={kappa_fixed})")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

    st.dataframe(sigma_df)

# TAB 3 – Heatmap σ × reserves
with tab3:
    st.subheader("Depeg Probability Heatmap (σ × reserves)")

    grid_df = load_grid_csv()
    if grid_df is None:
        st.warning(
            f"No precomputed grid found at `{DATA_DIR / 'peg_liquidity_grid.csv'}`.\n\n"
            "Run `python experiments/run_peg_stress_grid.py` from the project root "
            "to generate it, then reload this app."
        )
    else:
        kappa_choice = st.selectbox(
            "Select κ", sorted(grid_df["kappa"].unique())
        )
        trade_fraction_choice = st.selectbox(
            "Trade size (fraction of reserves)",
            sorted(grid_df["trade_fraction"].unique()),
            format_func=lambda x: f"{x*100:.1f}%",
        )

        severity_cols = [
            "expected_shortfall",
            "conditional_shortfall",
            "time_under_peg",
            "worst_case_deviation",
        ]

        metric_options = [c for c in grid_df.columns if c.startswith("p_T<")] + [
            c for c in severity_cols if c in grid_df.columns
        ]

        metric_choice = st.selectbox(
            "Metric to display", metric_options, index=0
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
                val = dfh[
                    (dfh["reserves"] == R) & (dfh["sigma"] == s)
                ][metric_choice].mean()
                Z[i, j] = val

        fig5, ax5 = plt.subplots(figsize=(7, 5))
        im = ax5.imshow(
            Z,
            origin="lower",
            cmap="viridis",
            extent=[min(sigmas), max(sigmas), min(reserves), max(reserves)],
            aspect="auto",
        )
        cbar = fig5.colorbar(im, ax=ax5)
        cbar.set_label(f"Metric value ({metric_choice})")
        ax5.set_xlabel("Volatility σ")
        ax5.set_ylabel("Reserves")
        ax5.set_title(
            f"σ × Reserves Heatmap (κ={kappa_choice}, trade={trade_fraction_choice*100:.1f}%)"
        )
        st.pyplot(fig5)

        st.markdown("Underlying grid data:")
        st.dataframe(dfh.reset_index(drop=True))
