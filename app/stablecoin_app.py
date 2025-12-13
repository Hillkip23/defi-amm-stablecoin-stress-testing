

import os
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure src/ is on the Python path (works locally and on Streamlit Cloud)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

DATA_DIR = ROOT / "data"


from defi_risk.simulation import simulate_gbm_price_paths, compute_lp_vs_hodl
from defi_risk.amm_pricing import impermanent_loss, lp_over_hodl_univ3, slippage_vs_trade_fraction
from defi_risk.peg_models import simulate_ou_peg_paths
from defi_risk.peg_stress import depeg_probabilities


# =====================================================
# Helpers for calibration
# =====================================================

def load_price_series(source, max_years: int = 5) -> pd.Series:
    """
    Load a CSV (path or uploaded file) and return a cleaned daily close price
    series, trimmed to the last `max_years` years.

    Handles CryptoDataDownload-style CSVs that start with a URL line.
    """

    def _read(**kwargs):
        # Small helper so we can reuse the same source (path or UploadedFile)
        return pd.read_csv(source, **kwargs)

    # 1) First attempt: normal read, skipping '#' comments
    df_raw = _read(comment="#")

    # 2) If we only got a single column, try again skipping the first row
    #    (this fixes CryptoDataDownload files where the first row is a URL)
    if df_raw.shape[1] < 2:
        if hasattr(source, "seek"):
            # Reset pointer for UploadedFile / file-like objects
            source.seek(0)
        df_raw = _read(skiprows=1)

    # 3) Still broken? Then give a clear error.
    if df_raw.shape[1] < 2:
        raise ValueError("CSV must have at least two columns (date & price).")

    # ---- Normalisation of columns ----
    cols_lower = {c.lower(): c for c in df_raw.columns}

    # Date column: try 'date', otherwise first column
    date_col = cols_lower.get("date", df_raw.columns[0])

    # Price column: try common names, otherwise second column
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

    # Parse dates, drop bad rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    # Sort oldest → newest
    df = df.sort_values("date")

    # Keep last max_years
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
    sigma_daily = log_ret.std()

    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    return mu_annual, sigma_annual, log_ret


# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="AMM LP Quant Lab",
    layout="wide",
)

st.title("AMM LP Simulation Dashboard")


# =====================================================
# Sidebar: core simulation parameters
# =====================================================
st.sidebar.header("Simulation Parameters")

# Defaults (can be overwritten by calibration)
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

# Dynamic fee model
st.sidebar.markdown("### Dynamic Fee Model")
fee_sensitivity = st.sidebar.slider(
    "Fee sensitivity to realized volatility", 0.0, 2.0, 0.5
)

# Uniswap v3 range
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

    # Collect terminal statistics per path
    rows = []
    for i in range(n_paths):
        df_i = compute_lp_vs_hodl(prices[[i]], fee_apr=fee_apr)
        rows.append(df_i.iloc[-1])

    summary_df = pd.DataFrame(rows)
    return prices, summary_df


# =====================================================
# MAIN: run base simulation
# =====================================================
run_clicked = st.button("Run Simulation")

if run_clicked:
    st.write("Running simulations...")

    prices, summary_df = run_mc_block(
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        mu=mu,
        sigma=sigma,
        fee_apr=fee_apr,
        p0=p0,
        random_seed=42,
    )

    # -------------------------------------------------
    # A) Base distribution: LP vs HODL (v2, static fee)
    # -------------------------------------------------
    st.subheader("LP Performance Distribution (All Variables)")
    st.write(summary_df.describe())

    st.subheader("Summary of LP / HODL at Horizon")
    lp_stats = summary_df["lp_over_hodl"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    st.write(lp_stats)

    st.subheader("Histogram of LP / HODL")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(summary_df["lp_over_hodl"], bins=50)
    ax_hist.set_xlabel("LP / HODL at horizon")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True)
    st.pyplot(fig_hist)

    # -------------------------------------------------
    # C) Return decomposition: IL vs dynamic fees (v2)
    # -------------------------------------------------
    realized_vols = []
    for i in range(n_paths):
        path_prices = prices.iloc[:, i]
        log_ret = np.log(path_prices / path_prices.shift(1)).dropna()
        vol_ann = log_ret.std() * np.sqrt(n_steps / T)
        realized_vols.append(vol_ann)

    summary_df["realized_vol"] = realized_vols

    dynamic_fee_apr = fee_apr + fee_sensitivity * summary_df["realized_vol"]
    dynamic_fee_apr = dynamic_fee_apr.clip(lower=0.0, upper=1.0)
    summary_df["dynamic_fee_apr"] = dynamic_fee_apr

    # LP/HODL with NO fees: 1 + IL (IL negative)
    lp_no_fees = 1.0 + summary_df["impermanent_loss"]

    # Add fee contribution
    summary_df["lp_over_hodl_dynamic"] = lp_no_fees + dynamic_fee_apr * T

    st.subheader("Summary of LP / HODL with Dynamic Fees (Uniswap v2)")
    lp_dyn_stats = summary_df["lp_over_hodl_dynamic"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    st.write(lp_dyn_stats)

    st.subheader("Return Decomposition (Dynamic Fees vs HODL)")
    total_excess = summary_df["lp_over_hodl_dynamic"] - 1.0
    il_component = summary_df["impermanent_loss"]
    fee_component = dynamic_fee_apr * T

    decomp = pd.DataFrame(
        {
            "mean_component": [
                il_component.mean(),
                fee_component.mean(),
                total_excess.mean(),
            ]
        },
        index=["IL (negative)", "Fee income", "Total LP excess return"],
    )

    st.write(decomp)

    fig_decomp, ax_decomp = plt.subplots()
    ax_decomp.bar(decomp.index, decomp["mean_component"])
    ax_decomp.set_ylabel("Mean contribution")
    ax_decomp.set_xticklabels(decomp.index, rotation=20, ha="right")
    ax_decomp.grid(True, axis="y")
    st.pyplot(fig_decomp)

    st.subheader("Histogram of LP / HODL with Dynamic Fees (v2)")
    fig_dyn, ax_dyn = plt.subplots()
    ax_dyn.hist(summary_df["lp_over_hodl_dynamic"], bins=50)
    ax_dyn.set_xlabel("LP / HODL at horizon (dynamic fees)")
    ax_dyn.set_ylabel("Frequency")
    ax_dyn.grid(True)
    st.pyplot(fig_dyn)

    # -------------------------------------------------
    # B & E) Uniswap v3 concentrated LP vs HODL + range search
    # -------------------------------------------------
    summary_df["lp_over_hodl_v3"] = summary_df["price"].apply(
        lambda p: lp_over_hodl_univ3(p, p_lower=p_lower, p_upper=p_upper)
    )

    st.subheader(
        f"Summary of LP / HODL (Uniswap v3, range [{p_lower:.2f}, {p_upper:.2f}])"
    )
    lp_v3_stats = summary_df["lp_over_hodl_v3"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    st.write(lp_v3_stats)

    fig_v3, ax_v3 = plt.subplots()
    ax_v3.hist(summary_df["lp_over_hodl_v3"], bins=50)
    ax_v3.set_xlabel("LP / HODL at horizon (v3)")
    ax_v3.set_ylabel("Frequency")
    ax_v3.grid(True)
    st.pyplot(fig_v3)

    st.subheader("Uniswap v3: Simple Optimal Range Search")

    col_grid1, col_grid2 = st.columns(2)
    with col_grid1:
        grid_lower_min = st.number_input("Grid lower min", 0.2, 1.0, 0.6, step=0.05)
        grid_lower_max = st.number_input("Grid lower max", 0.2, 1.0, 0.9, step=0.05)
    with col_grid2:
        grid_upper_min = st.number_input("Grid upper min", 1.0, 5.0, 1.1, step=0.05)
        grid_upper_max = st.number_input("Grid upper max", 1.0, 5.0, 2.0, step=0.05)

    n_grid = st.slider("Grid resolution per axis", 3, 15, 7)

    if grid_lower_min >= grid_lower_max or grid_upper_min >= grid_upper_max:
        st.warning("Ensure lower min < lower max and upper min < upper max.")
    else:
        lowers = np.linspace(grid_lower_min, grid_lower_max, n_grid)
        uppers = np.linspace(grid_upper_min, grid_upper_max, n_grid)

        best_mean = -np.inf
        best_range = None

        prices_T = summary_df["price"].values
        results = []

        for L in lowers:
            for U in uppers:
                if L >= U:
                    continue
                vals = [lp_over_hodl_univ3(p, p_lower=L, p_upper=U) for p in prices_T]
                mean_val = np.mean(vals)
                results.append(
                    {"p_lower": L, "p_upper": U, "mean_lp_over_hodl": mean_val}
                )
                if mean_val > best_mean:
                    best_mean = mean_val
                    best_range = (L, U)

        opt_df = pd.DataFrame(results).sort_values(
            "mean_lp_over_hodl", ascending=False
        )

        st.write("Top candidate ranges by mean LP/HODL:")
        st.write(opt_df.head(10))

        if best_range is not None:
            st.success(
                f"Best range on this grid ≈ [{best_range[0]:.2f}, {best_range[1]:.2f}] "
                f"with mean LP/HODL ≈ {best_mean:.3f}"
            )

    # -------------------------------------------------
    # A) Stress scenarios: bull / bear / crab
    # -------------------------------------------------
    st.subheader("Stress Scenario Comparison (Uniswap v2, static fees)")

    scenarios = {
        "Base": dict(mu=mu, sigma=sigma, fee_apr=fee_apr),
        "Bull (up-trend, moderate vol)": dict(
            mu=mu + 0.2, sigma=sigma * 0.8, fee_apr=fee_apr
        ),
        "Bear (down-trend, high vol)": dict(
            mu=mu - 0.3, sigma=sigma * 1.5, fee_apr=fee_apr
        ),
        "Crab (flat, very high vol)": dict(mu=0.0, sigma=sigma * 2.0, fee_apr=fee_apr),
    }

    stress_rows = []
    for name, params in scenarios.items():
        _, stress_summary = run_mc_block(
            n_paths=min(1000, n_paths),
            n_steps=n_steps,
            T=T,
            mu=params["mu"],
            sigma=params["sigma"],
            fee_apr=params["fee_apr"],
            p0=p0,
            random_seed=123,
        )
        stats = stress_summary["lp_over_hodl"].describe(
            percentiles=[0.05, 0.5, 0.95]
        )
        stress_rows.append(
            {
                "scenario": name,
                "mu": params["mu"],
                "sigma": params["sigma"],
                "fee_apr": params["fee_apr"],
                "mean_lp_over_hodl": stats["mean"],
                "p5": stats["5%"],
                "median": stats["50%"],
                "p95": stats["95%"],
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    st.write(stress_df)

    fig_stress, ax_stress = plt.subplots()
    ax_stress.bar(stress_df["scenario"], stress_df["mean_lp_over_hodl"])
    ax_stress.set_ylabel("Mean LP / HODL at T")
    ax_stress.grid(True, axis="y")
    st.pyplot(fig_stress)

    # -------------------------------------------------
    # D) Single-path visualizer
    # -------------------------------------------------
    st.subheader("Single Path Visualizer")

    path_idx = st.number_input(
        "Path index (0-based)",
        min_value=0,
        max_value=n_paths - 1,
        value=0,
        step=1,
    )

    df_path = compute_lp_vs_hodl(prices[[path_idx]], fee_apr=fee_apr)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Price Path**")
        fig_p, ax_p = plt.subplots()
        ax_p.plot(df_path.index.values, df_path["price"])
        ax_p.set_xlabel("Time")
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
        ax_v.set_xlabel("Time")
        ax_v.set_ylabel("LP / HODL")
        ax_v2.set_ylabel("Impermanent loss")
        ax_v.grid(True)
        st.pyplot(fig_v)

else:
    st.info("Adjust parameters on the left and click **Run Simulation**.")


# =====================================================
# Real-Data Calibration (Feature E)
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
        "Upload custom CSV...",
    ],
)

prices_series = None

# Map choices to local CSV paths (adjust if your files live elsewhere)
file_map = {
    "BTC (Binance USDT, local CSV)": "data/btc_5y_daily.csv",
    "ETH (Binance USDT, local CSV)": "data/eth_5y_daily.csv",
    "UNI (Binance USDT, local CSV)": "data/uni_5y_daily.csv",
    "XRP (Binance USDT, local CSV)": "data/xrp_5y_daily.csv",
    "S&P 500 (^SPX, local CSV)": "data/sp500_5y_daily.csv",
}

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

    st.subheader("Distribution of Daily Log-Returns")
    fig_ret, ax_ret = plt.subplots()
    ax_ret.hist(log_ret, bins=50)
    ax_ret.set_xlabel("Daily log return")
    ax_ret.set_ylabel("Frequency")
    ax_ret.grid(True)
    st.pyplot(fig_ret)
    
    
        # -------------------------------------------
    # Autocorrelation & volatility clustering
    # -------------------------------------------
    st.subheader("Autocorrelation Diagnostics")

    # Choose how many lags to inspect
    max_lag = st.slider(
        "Max lag (days) for autocorrelation",
        min_value=1,
        max_value=60,
        value=20,
        step=1,
        key="acf_max_lag",
    )

    lags = list(range(1, max_lag + 1))

    # ACF of raw returns (should be near 0 for GBM)
    acf_ret = [log_ret.autocorr(lag=l) for l in lags]

    # ACF of squared returns (volatility clustering shows up here)
    log_ret_sq = log_ret ** 2
    acf_sq = [log_ret_sq.autocorr(lag=l) for l in lags]

    acf_df = pd.DataFrame(
        {
            "lag": lags,
            "acf_returns": acf_ret,
            "acf_squared_returns": acf_sq,
        }
    )
    st.write("First few autocorrelation values:")
    st.write(acf_df.head(10))

    # Plot ACF for returns
    fig_acf, ax_acf = plt.subplots()
    ax_acf.stem(lags, acf_ret, basefmt=" ")
    ax_acf.set_xlabel("Lag (days)")
    ax_acf.set_ylabel("Autocorrelation (returns)")
    ax_acf.set_title("ACF of Daily Returns")
    ax_acf.grid(True)
    st.pyplot(fig_acf)

    # Plot ACF for squared returns (volatility clustering)
    fig_acf2, ax_acf2 = plt.subplots()
    ax_acf2.stem(lags, acf_sq, basefmt=" ")
    ax_acf2.set_xlabel("Lag (days)")
    ax_acf2.set_ylabel("Autocorrelation (squared returns)")
    ax_acf2.set_title("ACF of Squared Returns (Volatility Clustering)")
    ax_acf2.grid(True)
    st.pyplot(fig_acf2)

    # -------------------------------------------
    # Rolling volatility (visual volatility clustering)
    # -------------------------------------------
    st.subheader("Rolling Volatility (annualized)")

    roll_window = st.selectbox(
        "Rolling window (days)",
        options=[7, 21, 63, 126],
        index=1,
        key="roll_vol_window",
    )

    rolling_vol = log_ret.rolling(roll_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    st.line_chart(rolling_vol.rename("Rolling σ (annualized)"))


    if st.button("Use these parameters for simulation"):
        st.session_state["mu_slider"] = float(mu_hat)
        st.session_state["sigma_slider"] = float(sigma_hat)
        st.success("Updated Drift μ and Volatility σ sliders from calibration.")
        st.experimental_rerun()
else:
    st.info("Select an asset and/or upload a CSV to run calibration.")

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import streamlit as st 
from defi_risk.stablecoin import ( simulate_mean_reverting_peg, slippage_curve, constant_product_slippage, )




# =====================================================
# 🪙 Stablecoin Peg & Liquidity Stress Lab (OU + AMM)
# =====================================================

st.header("🪙 Stablecoin Peg & Liquidity Stress Lab")

st.markdown(
    """
This module explores **stablecoin peg behavior** and **liquidity resilience** under
different levels of volatility, mean reversion, and pool depth.

Use the tabs below to explore:

1. **Scenario Explorer** – OU peg paths + slippage curve  
2. **Depeg vs Volatility** – depeg probability curves across σ  
3. **Heatmap** – depeg probability across σ × reserves (from precomputed grid)
"""
)


# Small helpers specific to this section
def simulate_ou_cached(n_paths, n_steps, T, kappa, sigma, p0, peg, seed):
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


def simulate_ou_stress_aware(
    n_paths: int,
    n_steps: int,
    T: float,
    kappa: float,
    sigma: float,
    p0: float,
    peg: float,
    seed: int,
    stress_mult: float = 3.0,
    stress_band: float = 0.01,
) -> pd.DataFrame:
    """
    OU with volatility that increases when the peg is stressed.

    - Near the peg (|p - peg| small) → volatility ~ sigma
    - As deviation approaches `stress_band` (e.g. 1%) → volatility ramps up
      towards `stress_mult * sigma`.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        deviation = np.abs(prev - peg)
        # 0 → 1 as deviation goes from 0 to stress_band
        stress_level = np.clip(deviation / stress_band, 0.0, 1.0)
        sigma_t = sigma * (1.0 + (stress_mult - 1.0) * stress_level)

        drift = kappa * (peg - prev) * dt
        diffusion = sigma_t * np.sqrt(dt) * z
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
    p0: float,
    peg: float,
    seed: int,
    jump_intensity: float = 1.0,   # ~1 jump per year
    jump_mean: float = -0.02,      # average 2% downward jump
    jump_std: float = 0.01,        # jump size variability
) -> pd.DataFrame:
    """
    OU with occasional jump shocks:

        dp_t = kappa(peg - p_t) dt + sigma dW_t + J_t

    where J_t is a jump term occurring with Poisson intensity `jump_intensity`
    and normally-distributed jump size.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    prices = np.zeros((n_steps + 1, n_paths), dtype=float)
    prices[0, :] = p0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        prev = prices[t - 1, :]

        drift = kappa * (peg - prev) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Bernoulli approximation to Poisson(λ dt)
        jump_prob = jump_intensity * dt
        jump_mask = np.random.rand(n_paths) < jump_prob
        jump_sizes = np.zeros(n_paths)
        if jump_mask.any():
            jump_sizes[jump_mask] = np.random.normal(
                loc=jump_mean,
                scale=jump_std,
                size=jump_mask.sum(),
            )

        prices[t, :] = prev + drift + diffusion + jump_sizes

    df = pd.DataFrame(prices, index=times)
    df.index.name = "time"
    return df


def load_grid_csv():
    """
    Load the precomputed σ × reserves depeg grid.

    Uses DATA_DIR so it works both locally and on Streamlit Cloud.
    """
    csv_path = DATA_DIR / "peg_liquidity_grid.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


tab1, tab2, tab3 = st.tabs(
    ["Scenario explorer", "Depeg vs volatility (σ)", "Heatmap: σ × reserves"]
)

# ---------------------------------------------
# TAB 1 – Scenario explorer (OU + slippage)
# ---------------------------------------------
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Peg Dynamics Parameters")

        peg_model = st.selectbox(
            "Peg model",
            ["Basic OU", "Stress-aware OU", "OU with jumps"],
            index=0,
            key="peg_model",
        )

        n_paths_peg = st.slider("Number of OU paths", 200, 5000, 1000, key="peg_n_paths")
        n_steps_peg = st.slider("Steps per path (peg)", 50, 365, 252, key="peg_n_steps")
        T_peg = st.slider("Horizon (years)", 0.1, 2.0, 1.0, key="peg_T")
        kappa_peg = st.slider("Mean reversion speed (κ)", 0.1, 10.0, 5.0, key="peg_kappa")
        sigma_peg = st.slider("Peg volatility (σ)", 0.001, 0.1, 0.02, key="peg_sigma")
        p0_peg = st.slider("Initial price p₀", 0.90, 1.10, 1.00, key="peg_p0")
        peg_target = 1.0
        seed_peg = 42

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
            "Max trade size as % of reserves", 1.0, 50.0, 20.0, key="peg_max_trade"
        )
        highlight_trade_pct = st.slider(
            "Highlighted trade size (%)", 1.0, max_trade_pct, 10.0, key="peg_highlight_trade"
        )

    run_peg = st.button("Run stablecoin scenario")

    if run_peg:
        if peg_model == "Basic OU":
            prices_peg = simulate_ou_cached(
                n_paths=n_paths_peg,
                n_steps=n_steps_peg,
                T=T_peg,
                kappa=kappa_peg,
                sigma=sigma_peg,
                p0=p0_peg,
                peg=peg_target,
                seed=seed_peg,
            )
        elif peg_model == "Stress-aware OU":
            prices_peg = simulate_ou_stress_aware(
                n_paths=n_paths_peg,
                n_steps=n_steps_peg,
                T=T_peg,
                kappa=kappa_peg,
                sigma=sigma_peg,
                p0=p0_peg,
                peg=peg_target,
                seed=seed_peg,
            )
        elif peg_model == "OU with jumps":
            prices_peg = simulate_ou_with_jumps(
                n_paths=n_paths_peg,
                n_steps=n_steps_peg,
                T=T_peg,
                kappa=kappa_peg,
                sigma=sigma_peg,
                p0=p0_peg,
                peg=peg_target,
                seed=seed_peg,
            )
        else:
            # Fallback (shouldn't happen)
            prices_peg = simulate_ou_cached(
                n_paths=n_paths_peg,
                n_steps=n_steps_peg,
                T=T_peg,
                kappa=kappa_peg,
                sigma=sigma_peg,
                p0=p0_peg,
                peg=peg_target,
                seed=seed_peg,
            )

        ou_probs = depeg_probabilities(prices_peg, thresholds=(0.99, 0.95, 0.90))

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
            st.subheader("Depeg Probabilities (OU)")
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
            ax3.plot(slip_df["trade_pct"], slip_df["slippage_pct"], marker="o")
            ax3.set_xlabel("Trade size (% of reserves)")
            ax3.set_ylabel("Slippage (%)")
            ax3.grid(True)

            # highlight chosen trade size
            highlight = slip_df.iloc[
                (slip_df["trade_pct"] - highlight_trade_pct).abs().argmin()
            ]
            ax3.axvline(highlight["trade_pct"], linestyle="--", color="red")
            st.pyplot(fig3)

            st.markdown(
                f"Highlighted trade ≈ **{highlight['trade_pct']:.1f}%** of reserves → "
                f"slippage ≈ **{highlight['slippage_pct']:.2f}%**"
            )

# ---------------------------------------------
# TAB 2 – Depeg probability vs σ
# ---------------------------------------------
with tab2:
    st.subheader("Depeg Probability vs Volatility σ")

    n_paths_peg = 2000
    n_steps_peg = 252
    T_peg = 1.0
    kappa_fixed = st.slider("κ for this curve", 0.1, 10.0, 5.0, key="curve_kappa")
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
        prices_s = simulate_ou_cached(
            n_paths=n_paths_peg,
            n_steps=n_steps_peg,
            T=T_peg,
            kappa=kappa_fixed,
            sigma=float(s),
            p0=p0_fixed,
            peg=peg_target,
            seed=123,
        )
        probs_s = depeg_probabilities(prices_s, thresholds=(0.99, 0.95, 0.90))
        row = {"sigma": float(s)}
        row.update(probs_s)
        rows.append(row)

    sigma_df = pd.DataFrame(rows)

    fig4, ax4 = plt.subplots()
    for col in ["p_T<0.99", "p_T<0.95", "p_T<0.9"]:
        if col in sigma_df.columns:
            ax4.plot(sigma_df["sigma"], sigma_df[col], marker="o", label=col)
    ax4.set_xlabel("Volatility σ")
    ax4.set_ylabel("Depeg Probability")
    ax4.set_title(f"Depeg Probability vs σ (κ={kappa_fixed})")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

    st.dataframe(sigma_df)

# ---------------------------------------------
# TAB 3 – Heatmap σ × reserves
# ---------------------------------------------
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
            "Select κ",
            sorted(grid_df["kappa"].unique()),
            index=1 if 5.0 in grid_df["kappa"].unique() else 0,
        )
        trade_fraction_choice = st.selectbox(
            "Trade size (fraction of reserves)",
            sorted(grid_df["trade_fraction"].unique()),
            format_func=lambda x: f"{x*100:.1f}%",
        )
        threshold_cols = [c for c in grid_df.columns if c.startswith("p_T<")]
        threshold_choice = st.selectbox(
            "Depeg threshold",
            threshold_cols,
            index=1 if "p_T<0.95" in threshold_cols else 0,
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
                ][threshold_choice].mean()
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
        cbar.set_label(f"Depeg Probability ({threshold_choice})")
        ax5.set_xlabel("Volatility σ")
        ax5.set_ylabel("Reserves")
        ax5.set_title(
            f"σ × Reserves Heatmap (κ={kappa_choice}, trade={trade_fraction_choice*100:.1f}%)"
        )
        st.pyplot(fig5)

        st.markdown("Underlying grid data:")
        st.dataframe(dfh.reset_index(drop=True))
