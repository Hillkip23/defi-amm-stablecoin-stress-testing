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
