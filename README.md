# Stress Testing DeFi AMMs and Stablecoins

This repository accompanies the paper **“Stress Testing DeFi AMMs and Stablecoins Under Volatility and Liquidity Shocks”** (Cheruiyot, 2025) and the live **DeFi AMM & Stablecoin Stress Lab** Streamlit app. The project provides a unified, reproducible framework for stress testing automated market makers (AMMs) and soft‑pegged stablecoins using empirically calibrated stochastic models and interactive dashboards [file:248].

---

## Live Streamlit Lab

**DeFi AMM & Stablecoin Stress Lab**  
Unified dashboard for AMM liquidity‑provision risk and stablecoin peg resilience:

> https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/

The app exposes:

- **AMM & LP Risk** – GBM‑based simulations of LP vs HODL outcomes, impermanent loss, and fee income under different volatility regimes.  
- **Stablecoin Peg & Liquidity Stress Lab** – OU‑based peg models, depeg probabilities and severity metrics, and slippage as a function of pool depth and trade size [file:248].

All results in the paper are generated directly from this codebase via export scripts and are reproducible from the Streamlit interface [file:248].

---

## Research Overview

The paper studies how volatility shocks and liquidity withdrawals jointly affect:

- **LP profitability** in constant‑product AMMs (including dynamic v2‑style fees and static v3‑style concentrated ranges).  
- **Stablecoin peg stability**, measured not only by depeg probability but by how *badly* and *for how long* pegs fail [file:248].

Key elements:

- Risk assets follow a **regime‑aware GBM** calibrated to historical data (e.g., ETH using Dune’s `prices.usd` table; BTC; S&P 500) [file:248].  
- Stablecoin prices follow **mean‑reverting processes** (basic OU, stress‑aware OU, OU with jumps) calibrated to **USDC on‑chain data** via an AR(1) → OU mapping [file:248].  
- AMMs use **constant‑product pricing** with fee income, impermanent loss, and slippage fully modeled; concentrated‑liquidity ranges are analysed via grid search over price bands [file:248].  
- Stress surfaces over volatility and liquidity reveal sharp **phase transitions** in LP/HODL performance and peg stability that do not appear in average‑condition analyses [file:248].

---

## Main Features

### 1. AMM & LP Simulation Engine

- GBM process \( dP_t = \mu P_t dt + \sigma_t P_t dW_t \) with volatility calibrated from rolling realised volatility and multiple data sources (ETH, BTC, UNI, XRP, S&P 500, CSV uploads) [file:248].  
- Constant‑product AMM with slippage  
  \( xy = k, \ \text{slippage} = (P_{\text{exec}} - P_{\text{mid}})/P_{\text{mid}} \) [file:248].  
- LP performance vs HODL including:
  - Impermanent loss  
  - Static fee APR and volatility‑linked dynamic fees  
  - Concentrated‑liquidity ranges benchmarked against full‑range LPs [file:248].  
- Scenario presets (Base, Bull, Bear, Crab) to illustrate regime‑dependent LP outcomes [file:248].

### 2. Stablecoin Peg & Liquidity Stress Module

- Peg dynamics:  
  \( d p_t = \kappa(p^* - p_t)dt + \sigma_t dW_t + J_t \) with:
  - Basic OU (constant σ, no jumps)  
  - Stress‑aware OU with volatility amplification \( \sigma_t = \sigma (1 + \beta |p_t - p^*|) \)  
  - OU with downward jumps (Poisson intensity λ, negatively skewed jump sizes) [file:248].  
- **USDC on‑chain calibration**:
  1. Fetch daily USDC/USD from Dune `prices.usd`.  
  2. Compute deviations \( x_t = p_t - 1 \).  
  3. Fit AR(1) \( x_{t+1} = a + b x_t + \varepsilon_t \).  
  4. Map to OU: \( \kappa = -\ln b \), \( \mu = 1 + a/(1-b) \), \( \sigma \) from residual variance [file:248].  
  Dashboard button “Use USDC (Dune) for OU parameters” pre‑loads \( \mu \approx 1.0006, \sigma \approx 0.0021, \kappa \approx 0.55 \) (half‑life ≈ 1.3 days) [file:248].

- **Depeg severity metrics** computed over simulated paths:
  - Depeg probability \( \mathbb{P}(p_T < \theta) \)  
  - Expected shortfall \( \mathbb{E}[(\theta - p_T)^+] \)  
  - Conditional shortfall \( \mathbb{E}[\theta - p_T \mid p_T < \theta] \)  
  - Time‑under‑peg \( \frac{1}{T}\int_0^T 1_{\{p_t < \theta\}} dt \)  
  - Worst‑case deviation \( \min_{t \in [0,T]} p_t \) [file:248].

- Stress surfaces and heatmaps over volatility σ and pool reserves R highlight nonlinear transitions where mild depegs become frequent and severe once σ crosses critical bands [file:248].

---

## Repository Structure

Core modules (names may vary slightly with the actual repo):

- `app/stablecoin_app.py` – Streamlit entrypoint for the AMM & Stablecoin Stress Lab.  
- `defi_risk/simulation.py` – GBM path generation and LP vs HODL engine.  
- `defi_risk/amm_pricing.py` – constant‑product pricing and slippage utilities.  
- `defi_risk/peg_models.py` – OU, stress‑aware OU, and OU‑with‑jumps peg simulators.  
- `defi_risk/peg_stress.py` – depeg probabilities, severity metrics, and peg+liquidity scenarios.  
- `defi_risk/dune_client.py` – helper for pulling ETH and USDC data from Dune `prices.usd`.  
- `experiments/` – scripts that generate GBM calibrations, stress surfaces, and tables for the paper [file:248].

All figures and tables in the paper are produced from this code via reproducible experiment scripts [file:248].

---

## Getting Started

### Local installation

git clone https://github.com/your-handle/defi-amm-stablecoin-stress-testing.git
cd defi-amm-stablecoin-stress-testing
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt



### Using the Streamlit Lab

1. Open the live app (or your local instance).  
2. In the **Risk Asset / AMM** tab, choose an asset (e.g., ETH Dune), select a regime (Base/Bull/Bear/Crab), and run simulations to visualise LP vs HODL distributions.  
3. In the **Stablecoin Peg & Liquidity Stress Lab**, either:
   - Use the **USDC baseline (on‑chain)** or “Use USDC (Dune) for OU parameters” preset, or  
   - Manually set \( \kappa, \sigma, \mu \), reserves, and trade size.  
4. Inspect sample peg paths, depeg probabilities, severity metrics, and slippage curves; switch to the volatility and heatmap tabs to explore phase transitions across σ and liquidity [file:248].

---

## Citation

If you use this code or dashboard, please cite:

> Hillary Cheruiyot (2025), *Stress Testing DeFi AMMs and Stablecoins Under Volatility and Liquidity Shocks*.






# DeFi AMM & Stablecoin Stress Testing

A Python framework and interactive dashboards for **stress testing automated market makers (AMMs)** and **soft-pegged stablecoins** under volatility and liquidity shocks.

This project focuses on **risk, not yield**: how AMM liquidity providers and stablecoin pegs behave when markets move fast, liquidity thins out, or both.

---

## 🔗 Live Dashboards

### AMM LP Simulation Dashboard  
Explore LP performance vs HODL under volatility, fee models, and Uniswap v2/v3 mechanics.

👉 https://defi-amm-stablecoin-stress-testing-ykubq3cknumnygyjttmmxc.streamlit.app/

### Stablecoin Peg & Liquidity Stress Lab  
Explore stablecoin peg dynamics, depeg probabilities, and liquidity-driven slippage risk.

👉 https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/


---

## Why this matters (non-quant version)

Stablecoins and AMMs are often treated as “boring plumbing,” until they suddenly fail.  
When volatility spikes or liquidity evaporates, small design assumptions can turn into large losses.

This framework helps answer questions like:

- How likely is a stablecoin to break its peg under stress?
- How much liquidity is actually needed to keep prices stable?
- When do LP fees stop compensating for impermanent loss?

Rather than backtesting profits, the goal is **stress testing**: understanding *where systems break* before they do in production.

---

## What this project does

### AMM Risk & LP Analysis
- Simulates asset prices using Monte Carlo GBM
- Compares LP vs HODL outcomes across volatility regimes
- Models impermanent loss and fee income
- Supports Uniswap v2 and concentrated liquidity (v3-style) ranges
- Stress-tests LP performance under bull, bear, and high-volatility scenarios

### Stablecoin Peg Stress Testing
- Models peg dynamics using mean-reverting (OU) processes
- Estimates depeg probabilities at different thresholds (e.g. 1%, 5%, 10%)
- Links peg stability to AMM liquidity depth
- Visualizes σ × reserves heatmaps from precomputed stress grids
- Explores slippage growth as trade size increases

---

## What this framework uniquely answers

Most DeFi tools focus on **expected returns**.  
This framework focuses on **tail risk**.

It explicitly connects:
- **Volatility → peg deviation**
- **Liquidity depth → slippage and depeg risk**
- **AMM mechanics → systemic stability**

This allows analysis of historical stress events (e.g. USDC 2023, Curve pool stress, UST-style dynamics) without relying on protocol-specific assumptions.

---

## Repository structure (high level)

src/defi_risk/ Core models (AMM math, GBM, OU, stress metrics)
experiments/ Reproducible stress-test scripts
app/ Streamlit dashboards
data/ Market data + precomputed stress grids
figures/ Generated plots
report/ Research notes / paper (optional)



---

## Running locally

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

Run the dashboards:
streamlit run app/streamlit_app.py
streamlit run app/stablecoin_app.py


Scope & assumptions

This framework is intentionally simplified:

Prices follow reduced-form stochastic processes (GBM, OU)

Stablecoin pegs are modeled without explicit on-chain arbitrage

AMM liquidity is treated as exogenous

No governance, oracle failure, or reflexive mint/burn loops

This is a stress-testing lab, not a production trading simulator.

Intended use

Scenario analysis

Risk intuition building

Protocol design exploration

Education and research

Not intended for live trading or financial advice.
