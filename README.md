# DeFi AMM & Stablecoin Stress Testing Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/)

A unified, reproducible framework for stress testing Automated Market Makers (AMMs) and soft-pegged stablecoins using **exact stochastic calculus** and empirically calibrated models.

## 🌐 Live Demo

**[DeFi AMM & Stablecoin Stress Lab](https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/)**

Interactive dashboards for:
- **AMM LP Risk Analysis** – Monte Carlo simulation of LP vs HODL performance under volatility regimes
- **Stablecoin Peg Stress Testing** – Ornstein-Uhlenbeck models with exact solutions and compound Poisson jumps

All results are reproducible via the Streamlit interface or command-line export scripts.

---

## 📐 Mathematical Framework

### AMM & LP Simulation Engine

**Geometric Brownian Motion** for risky assets:
$$\mathrm{d}P_t = \mu P_t \,\mathrm{d}t + \sigma P_t \,\mathrm{d}W_t$$

- **Calibrated from on-chain data** (ETH, BTC via Dune) or custom CSV uploads
- **Impermanent Loss**: $IL(R) = \frac{2\sqrt{R}}{1+R} - 1$ where $R = P_T/P_0$
- **Dynamic Fees**: $fee_{APR}^{dynamic} = fee_{base} + \beta \cdot \sigma_{realized}$
- **Uniswap V3**: Concentrated liquidity ranges with grid-search optimization

### Stablecoin Peg Dynamics

**Mean-reverting processes with exact solutions**:

$$\mathrm{d}p_t = \kappa(p^* - p_t)\,\mathrm{d}t + \sigma_t \,\mathrm{d}W_t + \mathrm{d}J_t$$

| Model | Features | Use Case |
|-------|----------|----------|
| **Basic OU** | Constant $\sigma$, exact transition density | Normal market conditions |
| **Stress-Aware OU** | $\sigma_t = \sigma(1 + \beta\|p_t - p^*\|)$ | Volatility clustering during depegs |
| **OU with Jumps** | Compound Poisson $\mathrm{d}J_t$ (intensity $\lambda$, skewed jumps) | Tail risk / black swan events |

**Key Implementation Detail**: Unlike standard Euler-Maruyama approximations, we use the **exact OU transition density**:
$$p_t = p_{t-1}e^{-\kappa\Delta t} + p^*(1-e^{-\kappa\Delta t}) + \sigma\sqrt{\frac{1-e^{-2\kappa\Delta t}}{2\kappa}}Z_t$$

This eliminates discretization bias critical for accurate tail risk estimation.

### Empirical Calibration

**USDC On-Chain Calibration Pipeline**:
1. Fetch daily USDC/USD from Dune (`prices.usd` / `prices.day`)
2. Compute deviations: $x_t = p_t - 1$
3. Fit AR(1): $x_{t+1} = a + b x_t + \varepsilon_t$
4. Map to OU parameters:
   - $\kappa = -\ln(b)/\Delta t$ (mean reversion speed)
   - $\mu = 1 + a/(1-b)$ (long-run mean)
   - $\sigma = \sigma_\varepsilon\sqrt{2\kappa/(1-e^{-2\kappa\Delta t})}$ (volatility)
   - **Half-life**: $t_{1/2} = \ln(2)/\kappa$ (interpretable metric)

**Example Calibration** (March 2024):
- $\kappa = 0.56$ day$^{-1}$ → Half-life = **1.2 days**
- $\sigma = 0.0021$ (21 bps daily volatility)

---

## 🎯 Key Features

### 1. AMM Risk Analysis
- **LP vs HODL**: Terminal wealth distribution with fee accrual
- **Impermanent Loss Decomposition**: Separate IL (-) from fee income (+)
- **Regime Scenarios**: Base, Bull (low vol), Bear (high vol), Crab (flat/high vol)
- **V3 Range Optimization**: Grid search over price bands to maximize LP/HODL ratio

### 2. Stablecoin Stress Testing
- **Depeg Severity Metrics**:
  - Probability: $\mathbb{P}(p_T &lt; \theta)$
  - Expected Shortfall: $\mathbb{E}[(\theta - p_T)^+]$
  - Time-Under-Peg: $\frac{1}{T}\int_0^T \mathbf{1}_{\{p_t &lt; \theta\}}\mathrm{d}t$
  - Worst-case deviation: $\min_{t\in[0,T]} p_t$
- **Slippage Analysis**: Constant-product AMM slippage vs trade size as % of reserves
- **Stress Surfaces**: Heatmaps of depeg probability over $(\sigma, R)$ space

### 3. Production-Quality Numerics
✅ **Exact OU solution** (no Euler-Maruyama bias)  
✅ **Compound Poisson jumps** (properly sums multiple jumps per timestep)  
✅ **Stationarity validation** (AR(1) coefficient $b \in (0,1)$ check)  
✅ **Caching & progress bars** for large Monte Carlo runs  

---

## 📂 Repository Structure


Core modules (names may vary slightly with the actual repo)

- `app/stablecoin_app.py` – Streamlit entrypoint for the AMM & Stablecoin Stress Lab.  
- `defi_risk/simulation.py` – GBM path generation and LP vs HODL engine.  
- `defi_risk/amm_pricing.py` – constant‑product pricing and slippage utilities.  
- `defi_risk/peg_models.py` – OU, stress‑aware OU, and OU‑with‑jumps peg simulators.  
- `defi_risk/peg_stress.py` – depeg probabilities, severity metrics, and peg+liquidity scenarios.  
- `defi_risk/dune_client.py` – helper for pulling ETH and USDC data from Dune price tables. 
- `experiments/` – scripts that generate GBM calibrations, stress surfaces, and tables for the paper.  

All figures and tables in the paper are produced from this code via reproducible experiment scripts.


---

## 🚀 Quick Start

### Local Installation

```bash
git clone https://github.com/hillkip23/defi-amm-stablecoin-stress-testing.git
cd defi-amm-stablecoin-stress-testing

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the app
streamlit run app/stablecoin_app.py
