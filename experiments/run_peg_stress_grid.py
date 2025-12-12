# scripts/run_peg_stress_grid.py

import os
from pathlib import Path

import numpy as np
import pandas as pd


from defi_risk.peg_stress import run_peg_liquidity_scenario



def main():
    # --- 1. Define grids of parameters ---

    sigmas = [0.01, 0.02, 0.03, 0.05]          # peg volatility
    kappas = [1.0, 5.0, 10.0]                  # mean reversion speed
    reserves = [10_000_000, 2_000_000, 500_000]  # stablecoin + collateral reserves
    trade_fractions = [0.01, 0.05, 0.10, 0.20] # trade size as % of reserves

    n_paths = 5000
    n_steps = 365
    T = 1.0
    p0 = 1.0

    thresholds = (0.99, 0.95, 0.90)

    rows = []

    # --- 2. Run grid of scenarios ---

    for sigma in sigmas:
        for kappa in kappas:
            for R in reserves:
                for tf in trade_fractions:
                    res = run_peg_liquidity_scenario(
                        n_paths=n_paths,
                        n_steps=n_steps,
                        T=T,
                        kappa=kappa,
                        sigma=sigma,
                        p0=p0,
                        x_reserve=R,
                        y_reserve=R,  # assume 1:1 pool initially
                        trade_fraction=tf,
                        thresholds=thresholds,
                        peg=1.0,
                        random_seed=123,
                    )

                    row = {
                        "sigma": sigma,
                        "kappa": kappa,
                        "reserves": R,
                        "trade_fraction": tf,
                        "slippage": res["slippage"],
                    }
                    # unpack ou_probs dict into columns
                    for key, val in res["ou_probs"].items():
                        row[key] = val

                    rows.append(row)

    df = pd.DataFrame(rows)

    # --- 3. Save results ---

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "peg_liquidity_grid.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()

