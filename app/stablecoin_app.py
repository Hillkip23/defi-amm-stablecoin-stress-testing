# app/stablecoin_app.py
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from defi_risk.simulation import simulate_gbm_price_paths, compute_lp_vs_hodl
from defi_risk.amm_pricing import (
    impermanent_loss,
    lp_over_hodl_univ3,
    slippage_vs_trade_fraction,
)
from defi_risk.peg_models import PEG_MODEL_LABELS, simulate_peg_paths
from defi_risk.peg_stress import depeg_probabilities
from defi_risk.stablecoin import (
    simulate_mean_reverting_peg,
    slippage_curve,
    constant_product_slippage,
)
from defi_risk.dune_client import (
    get_uniswap_eth_usdc_daily_prices,
    get_usdc_daily_prices,
)


def load_price_series(source, max_years: int = 5) -> pd.Series:
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
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    
    if max_years is not None and not df.empty:
        cutoff = df["date"].max() - pd.Timedelta(days=365 * max_years)
        df = df[df["date"] >= cutoff]
    
    return df.set_index("date")["close"]


def estimate_gbm_params(prices: pd.Series, trading_days: int = 252):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_daily = log_ret.mean()
    sigma_daily = log_ret.std()
    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)
    return mu_annual, sigma_annual, log_ret


def calibrate_ou_from_usdc_prices(prices: pd.Series) -> dict:
    """Calibrate OU from USDC prices with stationarity validation."""
    x = prices.dropna().astype(float).values - 1.0
    if len(x) < 2:
        return {"mu": 1.0, "sigma": 0.01, "kappa": 1.0, "stationary": True}

    dt = 1.0
    x_t = x[:-1]
    x_tp1 = x[1:]
    
    denominator = np.sum(x_t * x_t)
    if denominator < 1e-12:
        return {"mu": 1.0, "sigma": 0.01, "kappa": 1.0, "stationary": True}
        
    b = np.sum(x_t * x_tp1) / denominator
    a = np.mean(x_tp1 - b * x_t)
    residuals = x_tp1 - (a + b * x_t)
    sigma_eps = np.std(residuals, ddof=1)

    if not (0 < b < 1):
        return {
            "mu": float(1.0 + np.mean(x)), 
            "sigma": float(np.std(x)), 
            "kappa": 1.0,
            "stationary": False,
            "ar_coeff": float(b),
            "half_life_days": float('inf')
        }

    kappa = -np.log(b) / dt
    mu_dev = a / (1.0 - b) if abs(1.0 - b) > 1e-6 else 0.0
    mu = 1.0 + mu_dev
    
    if kappa > 0:
        sigma = sigma_eps * np.sqrt(2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * dt)))
    else:
        sigma = sigma_eps

    return {
        "mu": float(mu), 
        "sigma": float(sigma), 
        "kappa": float(kappa),
        "stationary": True,
        "half_life_days": float(np.log(2) / kappa) if kappa > 0 else float('inf')
    }


# Page config
st.set_page_config(page_title="DeFi AMM & Stablecoin Stress Lab", layout="wide")

# Initialize session state
for key, val in [("mu_slider", 0.0), ("sigma_slider", 0.8), ("calibration_applied", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

st.title("DeFi AMM & Stablecoin Stress Lab")
st.markdown("""
This dashboard provides stress testing for AMM LPs and stablecoin peg resilience using 
empirically calibrated stochastic models.
""")

# Sidebar
st.sidebar.header("Simulation Parameters")
n_paths = st.sidebar.slider("Number of Paths", 1, 5000, 1000)
n_steps = st.sidebar.slider("Steps per Path", 50, 500, 365)
T = st.sidebar.slider("Years (T)", 0.1, 5.0, 1.0)

mu = st.sidebar.slider("Drift (mu)", -0.5, 0.5, 
                       value=float(st.session_state.get("mu_slider", 0.0)), 
                       key="mu_slider")
sigma = st.sidebar.slider("Volatility (sigma)", 0.1, 2.0, 
                          value=float(st.session_state.get("sigma_slider", 0.8)), 
                          key="sigma_slider")

fee_apr = st.sidebar.slider("Base Fee APR", 0.0, 1.0, 0.1)
p0 = 1.0

st.sidebar.markdown("### Dynamic Fee Model")
fee_sensitivity = st.sidebar.slider("Fee sensitivity to realized volatility", 0.0, 2.0, 0.5)

st.sidebar.markdown("### Uniswap v3 Range")
p_lower = st.sidebar.slider("Lower bound", 0.2, 1.0, 0.8)
p_upper = st.sidebar.slider("Upper bound", 1.0, 5.0, 1.2)


@st.cache_data(ttl=300)
def run_mc_block(n_paths, n_steps, T, mu, sigma, fee_apr, p0=1.0, random_seed=42):
    prices = simulate_gbm_price_paths(
        n_paths=n_paths, n_steps=n_steps, T=T, mu=mu, sigma=sigma, p0=p0, random_seed=random_seed
    )
    rows = []
    for i in range(n_paths):
        df_i = compute_lp_vs_hodl(prices[[i]], fee_apr=fee_apr)
        rows.append(df_i.iloc[-1])
    return prices, pd.DataFrame(rows)


# Main simulation
run_clicked = st.button("Run Simulation")

if run_clicked:
    with st.spinner(f"Running {n_paths} Monte Carlo paths..."):
        prices, summary_df = run_mc_block(n_paths, n_steps, T, mu, sigma, fee_apr, p0)

    # LP Performance
    st.subheader("LP vs HODL Performance")
    st.write(summary_df["lp_over_hodl"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    fig, ax = plt.subplots()
    ax.hist(summary_df["lp_over_hodl"], bins=50, alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', label='Break-even')
    ax.set_xlabel("LP / HODL at horizon")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Dynamic fees calculation
    realized_vols = []
    for i in range(n_paths):
        path_prices = prices.iloc[:, i]
        log_ret = np.log(path_prices / path_prices.shift(1)).dropna()
        vol_ann = log_ret.std() * np.sqrt(n_steps / T)
        realized_vols.append(vol_ann)
    
    summary_df["realized_vol"] = realized_vols
    dynamic_fee_apr = (fee_apr + fee_sensitivity * summary_df["realized_vol"]).clip(0.0, 1.0)
    
    # Return decomposition (IL is negative, fees are positive)
    lp_no_fees = 1.0 + summary_df["impermanent_loss"]  # IL already negative
    summary_df["lp_over_hodl_dynamic"] = lp_no_fees + dynamic_fee_apr * T
    
    st.subheader("Return Decomposition (Dynamic Fees)")
    il_comp = summary_df["impermanent_loss"].mean()
    fee_comp = (dynamic_fee_apr * T).mean()
    total_comp = summary_df["lp_over_hodl_dynamic"].mean() - 1.0
    
    decomp = pd.DataFrame({
        "Component": [il_comp, fee_comp, total_comp]
    }, index=["IL (negative)", "Fee income", "Total excess return"])
    st.write(decomp)
    
    fig, ax = plt.subplots()
    colors = ['red' if x < 0 else 'green' for x in decomp["Component"]]
    ax.bar(decomp.index, decomp["Component"], color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Mean contribution")
    plt.xticks(rotation=25, ha='right')
    ax.grid(True, axis="y")
    st.pyplot(fig)

    # Uniswap v3 (simplified terminal price model)
    summary_df["lp_over_hodl_v3"] = summary_df["price"].apply(
        lambda p: lp_over_hodl_univ3(p, p_lower=p_lower, p_upper=p_upper)
    )
    
    st.subheader(f"Uniswap v3 Range [{p_lower:.2f}, {p_upper:.2f}]")
    st.write(summary_df["lp_over_hodl_v3"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    # V3 Grid Search
    st.subheader("Optimal Range Search")
    c1, c2 = st.columns(2)
    with c1:
        grid_lower_min = st.number_input("Lower min", 0.2, 1.0, 0.6, step=0.05)
        grid_lower_max = st.number_input("Lower max", 0.2, 1.0, 0.9, step=0.05)
    with c2:
        grid_upper_min = st.number_input("Upper min", 1.0, 5.0, 1.1, step=0.05)
        grid_upper_max = st.number_input("Upper max", 1.0, 5.0, 2.0, step=0.05)
    
    n_grid = st.slider("Grid resolution", 3, 15, 7)
    
    if grid_lower_min < grid_lower_max and grid_upper_min < grid_upper_max:
        lowers = np.linspace(grid_lower_min, grid_lower_max, n_grid)
        uppers = np.linspace(grid_upper_min, grid_upper_max, n_grid)
        
        progress = st.progress(0)
        results = []
        prices_T = summary_df["price"].values
        total = len(lowers) * len(uppers)
        count = 0
        
        best_mean, best_range = -np.inf, None
        
        for L in lowers:
            for U in uppers:
                if L >= U:
                    count += 1
                    continue
                vals = [lp_over_hodl_univ3(p, p_lower=L, p_upper=U) for p in prices_T]
                mean_val = np.mean(vals)
                results.append({"p_lower": L, "p_upper": U, "mean_lp_over_hodl": mean_val})
                if mean_val > best_mean:
                    best_mean, best_range = mean_val, (L, U)
                count += 1
                progress.progress(min(count/total, 1.0))
        
        opt_df = pd.DataFrame(results).sort_values("mean_lp_over_hodl", ascending=False)
        st.write("Top ranges:", opt_df.head(5))
        if best_range:
            st.success(f"Best: [{best_range[0]:.2f}, {best_range[1]:.2f}] with LP/HODL ≈ {best_mean:.3f}")

    # Stress scenarios
    st.subheader("Stress Scenarios")
    scenarios = {
        "Base": dict(mu=mu, sigma=sigma, fee_apr=fee_apr),
        "Bull": dict(mu=mu + 0.2, sigma=sigma * 0.8, fee_apr=fee_apr),
        "Bear": dict(mu=mu - 0.3, sigma=sigma * 1.5, fee_apr=fee_apr),
        "Crab": dict(mu=0.0, sigma=sigma * 2.0, fee_apr=fee_apr),
    }
    
    stress_rows = []
    for name, params in scenarios.items():
        _, stress_df = run_mc_block(min(1000, n_paths), n_steps, T, 
                                    params["mu"], params["sigma"], params["fee_apr"], 
                                    p0, random_seed=123)
        stats = stress_df["lp_over_hodl"].describe(percentiles=[0.05, 0.5, 0.95])
        stress_rows.append({
            "Scenario": name,
            "Mean": stats["mean"],
            "P5": stats["5%"],
            "Median": stats["50%"],
            "P95": stats["95%"],
        })
    
    st.write(pd.DataFrame(stress_rows))


# Calibration Section
st.header("GBM Calibration from Historical Data")

asset_choice = st.selectbox("Choose asset", [
    "BTC (CSV)", "ETH (CSV)", "UNI (CSV)", "XRP (CSV)", "S&P 500 (CSV)",
    "ETH (Dune on-chain)", "Upload custom CSV..."
])

file_map = {
    "BTC (CSV)": "data/btc_5y_daily.csv",
    "ETH (CSV)": "data/eth_5y_daily.csv",
    "UNI (CSV)": "data/uni_5y_daily.csv",
    "XRP (CSV)": "data/xrp_5y_daily.csv",
    "S&P 500 (CSV)": "data/sp500_5y_daily.csv",
}

prices_series = None

if asset_choice == "Upload custom CSV...":
    uploaded = st.file_uploader("Upload CSV (Date, Close)", type=["csv"])
    if uploaded:
        try:
            prices_series = load_price_series(uploaded)
        except Exception as e:
            st.error(f"Error: {e}")
elif asset_choice == "ETH (Dune on-chain)":
    try:
        prices_series = get_uniswap_eth_usdc_daily_prices()
    except Exception as e:
        st.error(f"Dune fetch failed: {e}")
else:
    path = file_map.get(asset_choice)
    if path and os.path.exists(path):
        prices_series = load_price_series(path)
    else:
        st.warning(f"File not found: {path}")

if prices_series is not None:
    st.line_chart(prices_series.rename("Price"))
    mu_hat, sigma_hat, log_ret = estimate_gbm_params(prices_series)
    
    c1, c2 = st.columns(2)
    c1.metric("Drift (μ)", f"{mu_hat:.2%}")
    c2.metric("Volatility (σ)", f"{sigma_hat:.2%}")
    
    # Autocorrelation
    max_lag = st.slider("ACF max lag", 1, 60, 20)
    lags = range(1, max_lag + 1)
    acf = [log_ret.autocorr(lag=l) for l in lags]
    acf_sq = [(log_ret**2).autocorr(lag=l) for l in lags]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.stem(lags, acf, basefmt=" ")
    ax1.set_title("Returns ACF")
    ax1.set_xlabel("Lag")
    ax2.stem(lags, acf_sq, basefmt=" ")
    ax2.set_title("Squared Returns ACF (Vol Clustering)")
    ax2.set_xlabel("Lag")
    st.pyplot(fig)
    
    if st.button("Use these parameters for simulation"):
        st.session_state.mu_slider = float(mu_hat)
        st.session_state.sigma_slider = float(sigma_hat)
        st.session_state.calibration_applied = True
        st.success("Parameters updated!")


# Stablecoin Stress Lab
st.header("🪙 Stablecoin Peg & Liquidity Stress Lab")

@st.cache_data(ttl=600)
def load_grid_csv():
    csv_path = DATA_DIR / "peg_liquidity_grid.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else None

tab1, tab2, tab3 = st.tabs(["Scenario Explorer", "Depeg vs σ", "σ × Reserves Heatmap"])

with tab1:
    model = st.selectbox("Peg model", list(PEG_MODEL_LABELS.keys()), 
                        format_func=lambda k: PEG_MODEL_LABELS[k])
    
    c1, c2 = st.columns(2)
    with c1:
        n_paths_peg = st.slider("Paths", 200, 5000, 1000, key="peg_paths")
        n_steps_peg = st.slider("Steps", 50, 365, 252, key="peg_steps")
        T_peg = st.slider("Horizon (years)", 0.1, 2.0, 1.0, key="peg_T")
        kappa_peg = st.slider("κ (mean reversion)", 0.1, 10.0, 5.0, key="peg_kappa")
        sigma_peg = st.slider("σ (volatility)", 0.001, 0.1, 0.02, key="peg_sigma")
        p0_peg = st.slider("Initial price", 0.90, 1.10, 1.00, key="peg_p0")
        
        if st.button("Calibrate from USDC (Dune)"):
            with st.spinner("Fetching..."):
                try:
                    usdc_prices = get_usdc_daily_prices()
                    params = calibrate_ou_from_usdc_prices(usdc_prices)
                    if params["stationary"]:
                        st.success(f"κ={params['kappa']:.2f}, σ={params['sigma']:.4f}, "
                                 f"half-life={params['half_life_days']:.1f}d")
                    else:
                        st.warning(f"Non-stationary (b={params['ar_coeff']:.3f})")
                except Exception as e:
                    st.error(f"Calibration failed: {e}")
    
    with c2:
        res_stable = st.number_input("Stable reserve", 100_
