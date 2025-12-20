# Stress Testing DeFi AMMs and Stablecoins

This repository accompanies the paper **“Stress Testing DeFi AMMs and Stablecoins Under Volatility and Liquidity Shocks”** (Cheruiyot, 2025) and the live **DeFi AMM & Stablecoin Stress Lab** Streamlit app. The project provides a unified, reproducible framework for stress testing automated market makers (AMMs) and soft‑pegged stablecoins using empirically calibrated stochastic models and interactive dashboards.

## Live Streamlit Lab

**DeFi AMM & Stablecoin Stress Lab**  
Unified dashboard for AMM liquidity‑provision risk and stablecoin peg resilience:

> https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/

The app exposes:

- **AMM & LP Risk** – GBM‑based simulations of LP vs HODL outcomes, impermanent loss, and fee income under different volatility regimes.  
- **Stablecoin Peg & Liquidity Stress Lab** – OU‑based peg models, depeg probabilities and severity metrics, and slippage as a function of pool depth and trade size.

All results in the paper are generated directly from this codebase via export scripts and are reproducible from the Streamlit interface.


## Research Overview

The paper studies how volatility shocks and liquidity withdrawals jointly affect:

- **LP profitability** in constant‑product AMMs (including dynamic v2‑style fees and static v3‑style concentrated ranges).  
- **Stablecoin peg stability**, measured not only by depeg probability but by how *badly* and *for how long* pegs fail.

Key elements:

- Risk assets follow a **regime‑aware GBM** calibrated to historical data (e.g., ETH using Dune’s `prices.usd` table; BTC; S&P 500).  
- Stablecoin prices follow **mean‑reverting processes** (basic OU, stress‑aware OU, OU with jumps) calibrated to **USDC on‑chain data** via an AR(1) → OU mapping.  
- AMMs use **constant‑product pricing** with fee income, impermanent loss, and slippage fully modeled; concentrated‑liquidity ranges are analysed via grid search over price bands.  
- Stress surfaces over volatility and liquidity reveal sharp **phase transitions** in LP/HODL performance and peg stability that do not appear in average‑condition analyses.


## Main Features

### 1. AMM & LP Simulation Engine

<<<<<<< HEAD
- GBM process \( dP_t = \mu P_t dt + \sigma_t P_t dW_t \) with volatility calibrated from rolling realised volatility and multiple data sources (ETH, BTC, UNI, XRP, S&P 500, CSV uploads).  
- Constant‑product AMM with slippage  
  \( xy = k, \ \text{slippage} = (P_{\text{exec}} - P_{\text{mid}})/P_{\text{mid}} \).  
=======
- GBM process \( dP_t = \mu P_t dt + \sigma_t P_t dW_t \) with volatility calibrated from rolling realised volatility and multiple data sources (ETH, BTC, UNI, XRP, S&P 500, CSV uploads)... dPt=μPtdt+σtPtdWt
 
- Constant‑product AMM with slippage  
  xy = k, \ \text{slippage} = (P_{\text{exec}} - P_{\text{mid}})/P_{\text{mid}} \).
  constant = xy=k
  slippage= (y/xy).(−Δx/(x(x+Δx))= −Δx/(x+Δx)
>>>>>>> 3b5632b65f8c54211051ec1d906e4731a1b355e1
- LP performance vs HODL including:
  - Impermanent loss  
  - Static fee APR and volatility‑linked dynamic fees  
  - Concentrated‑liquidity ranges benchmarked against full‑range LPs.  
- Scenario presets (Base, Bull, Bear, Crab) to illustrate regime‑dependent LP outcomes.

### 2. Stablecoin Peg & Liquidity Stress Module

- Peg dynamics:  
  \( d p_t = \kappa(p^* - p_t)dt + \sigma_t dW_t + J_t \) with:
  - Basic OU (constant σ, no jumps)  
  - Stress‑aware OU with volatility amplification \( \sigma_t = \sigma (1 + \beta |p_t - p^*|) \)  
  - OU with downward jumps (Poisson intensity λ, negatively skewed jump sizes).  
- **USDC on‑chain calibration**:
  1. Fetch daily USDC/USD from Dune `prices.usd`.  
  2. Compute deviations \( x_t = p_t - 1 \).  
  3. Fit AR(1) \( x_{t+1} = a + b x_t + \varepsilon_t \).  
  4. Map to OU: \( \kappa = -\ln b \), \( \mu = 1 + a/(1-b) \), \( \sigma \) from residual variance.  
  Dashboard button “Use USDC (Dune) for OU parameters” pre‑loads \( \mu \approx 1.0006, \sigma \approx 0.0021, \kappa \approx 0.55 \) (half‑life ≈ 1.3 days).

- **Depeg severity metrics** computed over simulated paths:
  - Depeg probability \( \mathbb{P}(p_T < \theta) \)  
  - Expected shortfall \( \mathbb{E}[(\theta - p_T)^+] \)  
  - Conditional shortfall \( \mathbb{E}[\theta - p_T \mid p_T < \theta] \)  
  - Time‑under‑peg \( \frac{1}{T}\int_0^T 1_{\{p_t < \theta\}} dt \)  
  - Worst‑case deviation \( \min_{t \in [0,T]} p_t \).

- Stress surfaces and heatmaps over volatility σ and pool reserves R highlight nonlinear transitions where mild depegs become frequent and severe once σ crosses critical bands.

## Repository Structure

Core modules (names may vary slightly with the actual repo):

- `app/stablecoin_app.py` – Streamlit entrypoint for the AMM & Stablecoin Stress Lab.  
- `defi_risk/simulation.py` – GBM path generation and LP vs HODL engine.  
- `defi_risk/amm_pricing.py` – constant‑product pricing and slippage utilities.  
- `defi_risk/peg_models.py` – OU, stress‑aware OU, and OU‑with‑jumps peg simulators.  
- `defi_risk/peg_stress.py` – depeg probabilities, severity metrics, and peg+liquidity scenarios.  
- `defi_risk/dune_client.py` – helper for pulling ETH and USDC data from Dune `prices.usd`.  
- `experiments/` – scripts that generate GBM calibrations, stress surfaces, and tables for the paper.

All figures and tables in the paper are produced from this code via reproducible experiment scripts.


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
4. Inspect sample peg paths, depeg probabilities, severity metrics, and slippage curves; switch to the volatility and heatmap tabs to explore phase transitions across σ and liquidity.



<<<<<<< HEAD

=======
>>>>>>> 3b5632b65f8c54211051ec1d906e4731a1b355e1
