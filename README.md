# DeFi AMM & Stablecoin Stress Testing

🔗 **Live dashboards**

- **AMM & Stablecoin Stress Testing Dashboard**  
  https://defi-amm-stablecoin-stress-testing-ykubq3cknumnygyjttmmxc.streamlit.app/

- **Stablecoin-Focused Stress Testing Dashboard**  
  https://hillkip23-defi-amm-stablecoin-stress-t-appstablecoin-app-yfedl4.streamlit.app/

---

## Overview

A Python framework and interactive Streamlit dashboard for **stress testing
automated market makers (AMMs)** and **soft-pegged stablecoins**.

The project uses Monte Carlo simulation and simple stochastic models to analyze
how volatility and liquidity stress affect:

- AMM slippage and liquidity depth  
- Stablecoin peg stability  
- Depeg risk under adverse market conditions  

---

## What this project does

This repository provides tools to:

- Simulate **stablecoin peg dynamics** using mean-reverting processes  
- Estimate **depeg probabilities** under different volatility regimes  
- Measure **AMM slippage** as a function of trade size and pool liquidity  
- Run repeatable **stress-test scenarios**  
- Explore results interactively through a **Streamlit dashboard**  

The focus is on **risk analysis and stress testing**, rather than full
on-chain execution or trading simulation.

---

## Repository structure

```text
├── src/defi_risk/        # Core simulation and pricing logic
├── experiments/          # Reproducible stress-test scripts
├── app/                  # Streamlit dashboards
├── data/                 # Input data and generated grids
├── figures/              # Generated figures
└── requirements.txt


Installation

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt

Running analyses

From the repository root:

# Depeg probability and volatility–liquidity grid
PYTHONPATH=src python experiments/run_peg_stress_grid.py

# Example stablecoin peg paths
PYTHONPATH=src python experiments/export_peg_paths.py

# AMM slippage vs trade size
PYTHONPATH=src python experiments/export_figures.py

Running the dashboard locally
PYTHONPATH=src streamlit run app/streamlit_app.py


The dashboard allows interactive exploration of:

Stablecoin peg behavior

Depeg probability surfaces

AMM slippage versus trade size and liquidity

Liquidity stress scenarios

Scope and assumptions

Risk asset prices follow simplified stochastic dynamics

Stablecoin pegs are modeled using reduced-form mean reversion

AMM pricing uses constant-product mechanics

Liquidity provision and withdrawal are treated as exogenous

This framework is intended for scenario analysis, stress testing,
and parameter sensitivity exploration.