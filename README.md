# Stress Testing DeFi AMMs and Stablecoins

The project provides a unified, reproducible framework for stress testing automated market makers (AMMs) and soft‑pegged stablecoins using empirically calibrated stochastic models and interactive dashboards.[web:18][web:19]

## Live Streamlit Lab

**DeFi AMM & Stablecoin Stress Lab**  
Unified dashboard for AMM liquidity‑provision risk and stablecoin peg resilience:[web:18]

> https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/

The app exposes:

- **AMM & LP Risk** – GBM‑based simulations of LP vs HODL outcomes, impermanent loss, and fee income under different volatility regimes.[web:19]  
- **Stablecoin Peg & Liquidity Stress Lab** – OU‑based peg models, depeg probabilities and severity metrics, and slippage as a function of pool depth and trade size.[web:45][web:48]

All results in the paper are generated directly from this codebase via export scripts and are reproducible from the Streamlit interface.[web:21]

## Research Overview

The paper studies how volatility shocks and liquidity withdrawals jointly affect:[web:21]

- **LP profitability** in constant‑product AMMs (including dynamic v2‑style fees and static v3‑style concentrated ranges).[web:19][web:50]  
- **Stablecoin peg stability**, measured not only by depeg probability but by how badly and for how long pegs fail.[web:24][web:51]

Key elements:

- Risk assets follow a **regime‑aware GBM** calibrated to historical data (e.g., ETH using Dune’s `prices.usd` / `prices.day` tables; BTC; S&P 500).[web:18][web:41]  
- Stablecoin prices follow **mean‑reverting processes** (basic OU, stress‑aware OU, OU with jumps) calibrated to **USDC on‑chain data** via an AR(1) → OU mapping.[web:18][web:45]  
- AMMs use **constant‑product pricing** with fee income, impermanent loss, and slippage fully modeled; concentrated‑liquidity ranges are analysed via grid search over price bands.[web:19][web:47]  
- Stress surfaces over volatility and liquidity reveal sharp **phase transitions** in LP/HODL performance and peg stability that do not appear in average‑condition analyses.[web:21][web:51]

## Main Features

### 1. AMM & LP Simulation Engine

- GBM process \(\mathrm{d}P_t = \mu P_t \mathrm{d}t + \sigma_t P_t \mathrm{d}W_t\) with volatility calibrated from rolling realised volatility and multiple data sources (ETH, BTC, UNI, XRP, S&P 500, CSV uploads).[web:18]  
- Constant‑product AMM with slippage  
  \[
  xy = k
  \]  
  and slippage defined as  
  \[
  \text{slippage} = \frac{P_{\text{exec}} - P_{\text{mid}}}{P_{\text{mid}}}.
  \]
- LP performance vs HODL including:
  - Impermanent loss.  
  - Static fee APR and volatility‑linked dynamic fees.  
  - Concentrated‑liquidity ranges benchmarked against full‑range LPs.  
- Scenario presets (Base, Bull, Bear, Crab) to illustrate regime‑dependent LP outcomes.[web:19]

### 2. Stablecoin Peg & Liquidity Stress Module

- Peg dynamics:  
  \[
  \mathrm{d}p_t = \kappa(p^* - p_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t + J_t
  \]
  with:
  - Basic OU (constant \(\sigma\), no jumps).  
  - Stress‑aware OU with volatility amplification \(\sigma_t = \sigma (1 + \beta |p_t - p^*|)\).  
  - OU with downward jumps (Poisson intensity \(\lambda\), negatively skewed jump sizes).[web:45][web:48]

- **USDC on‑chain calibration**:[web:18][web:41]  
  1. Fetch daily USDC/USD from Dune price tables (legacy `prices.usd` or `prices.day`).  
  2. Compute deviations \(x_t = p_t - 1\).  
  3. Fit AR(1) \(x_{t+1} = a + b x_t + \varepsilon_t\).  
  4. Map to OU: \(\kappa = -\ln b\), \(\mu = 1 + a/(1-b)\), \(\sigma\) from residual variance.  

  The dashboard button “Use USDC (Dune) for OU parameters” pre‑loads a baseline parameter set estimated from USDC data.[web:18]

- **Depeg severity metrics** computed over simulated paths:[web:21][web:24]  
  - Depeg probability \(\mathbb{P}(p_T < \theta)\).  
  - Expected shortfall \(\mathbb{E}[(\theta - p_T)^+]\).  
  - Conditional shortfall \(\mathbb{E}[\theta - p_T \mid p_T < \theta]\).  
  - Time‑under‑peg \(\frac{1}{T}\int_0^T 1_{\{p_t < \theta\}} \mathrm{d}t\).  
  - Worst‑case deviation \(\min_{t \in [0,T]} p_t\).  

- Stress surfaces and heatmaps over volatility \(\sigma\) and pool reserves \(R\) highlight nonlinear transitions where mild depegs become frequent and severe once \(\sigma\) crosses critical bands.[web:21][web:51]

## Repository Structure

Core modules (names may vary slightly with the actual repo):[web:19]

- `app/stablecoin_app.py` – Streamlit entrypoint for the AMM & Stablecoin Stress Lab.  
- `defi_risk/simulation.py` – GBM path generation and LP vs HODL engine.  
- `defi_risk/amm_pricing.py` – constant‑product pricing and slippage utilities.  
- `defi_risk/peg_models.py` – OU, stress‑aware OU, and OU‑with‑jumps peg simulators.  
- `defi_risk/peg_stress.py` – depeg probabilities, severity metrics, and peg+liquidity scenarios.  
- `defi_risk/dune_client.py` – helper for pulling ETH and USDC data from Dune price tables.[web:18][web:42]  
- `experiments/` – scripts that generate GBM calibrations, stress surfaces, and tables for the paper.  

All figures and tables in the paper are produced from this code via reproducible experiment scripts.[web:21]

## Getting Started

### Local installation

```bash
git clone https://github.com/hillkip23/defi-amm-stablecoin-stress-testing.git
cd defi-amm-stablecoin-stress-testing
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
