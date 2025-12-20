"""
Lightweight Dune API client for pulling on-chain DeFi data
into the DeFi AMM & Stablecoin Stress Lab.

Current focus:
- Daily mid-price series for the Uniswap v3 ETH-USDC 0.05% pool on Ethereum.

You need a Dune API key set as the environment variable DUNE_API_KEY.
See: https://docs.dune.com/api-reference/apis/quickstart-analyze
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
import requests


DUNE_API_KEY = os.getenv("DUNE_API_KEY")
DUNE_BASE_URL = "https://api.dune.com/api/v1"



class DuneClientError(Exception):
    """Custom error for Dune client issues."""


def _get_headers() -> dict:
    if not DUNE_API_KEY:
        raise DuneClientError(
            "DUNE_API_KEY environment variable is not set. "
            "Create an API key in Dune and export it before running the app."
        )
    return {"X-DUNE-API-KEY": DUNE_API_KEY}


def execute_query_and_wait(query_id: int, params: Optional[dict] = None, poll_interval: float = 2.0) -> pd.DataFrame:
    """
    Execute a saved Dune query and wait for results.

    Parameters
    ----------
    query_id : int
        The numeric ID of a saved Dune query.
    params : dict, optional
        Parameter values for the query (if it uses :named parameters).
    poll_interval : float
        Seconds to wait between status checks.

    Returns
    -------
    pd.DataFrame
        Result rows as a pandas DataFrame.
    """
    headers = _get_headers()

    # 1) Start execution
    start_url = f"{DUNE_BASE_URL}/query/{query_id}/execute"
    payload = {"parameters": params or {}}
    r = requests.post(start_url, headers=headers, json=payload)
    if r.status_code != 200:
        raise DuneClientError(f"Failed to start query {query_id}: {r.status_code} {r.text}")

    execution_id = r.json().get("execution_id")
    if not execution_id:
        raise DuneClientError(f"No execution_id returned for query {query_id}: {r.text}")

    # 2) Poll status
    status_url = f"{DUNE_BASE_URL}/execution/{execution_id}/status"
    results_url = f"{DUNE_BASE_URL}/execution/{execution_id}/results"

    while True:
        s = requests.get(status_url, headers=headers)
        if s.status_code != 200:
            raise DuneClientError(f"Error polling status for {execution_id}: {s.status_code} {s.text}")

        state = s.json().get("state")
        if state in ("QUERY_STATE_COMPLETED", "SUCCESS"):
            break
        if state in ("QUERY_STATE_FAILED", "ERROR"):
            raise DuneClientError(f"Dune query {query_id} failed: {s.text}")

        time.sleep(poll_interval)

    # 3) Fetch results
    res = requests.get(results_url, headers=headers)
    if res.status_code != 200:
        raise DuneClientError(f"Error fetching results for {execution_id}: {res.status_code} {res.text}")

    rows = res.json().get("result", {}).get("rows", [])
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# === High-level helper for your app =========================================


def get_uniswap_eth_usdc_daily_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Return a daily close/mid-price series for Uniswap v3 ETH/USDC 0.05% pool.

    This wraps a saved Dune query that should:
    - Filter trades or pool observations for the canonical ETH-USDC 0.05% pool.
    - Aggregate to one row per day with columns: date, price_usd.

    Parameters
    ----------
    start_date : str, optional
        Filter start date, e.g. '2021-01-01' (if your query supports it).
    end_date : str, optional
        Filter end date, e.g. '2025-01-01'.

    Returns
    -------
    pd.Series
        pandas Series indexed by pandas.Timestamp with name 'close'.
    """
    # TODO: replace with your real Dune query ID
    QUERY_ID = 6364165

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    df = execute_query_and_wait(QUERY_ID, params=params)

    if df.empty:
        raise DuneClientError("Dune ETH-USDC daily price query returned no rows.")

    # Expect columns: 'date', 'price_usd'; adjust if your query names differ
    if "date" not in df.columns or "price_usd" not in df.columns:
        raise DuneClientError(f"Unexpected columns from Dune: {df.columns.tolist()}")

    s = (
        df
        .assign(date=pd.to_datetime(df["date"], utc=True))
        .set_index("date")["price_usd"]
        .sort_index()
        .rename("close")
    )
    return s


def get_usdc_daily_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Daily USDC/USD price series from Dune prices.usd."""
    QUERY_ID = 6364523  # replace with your actual id, e.g. 6364xxx

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    df = execute_query_and_wait(QUERY_ID, params=params)

    if df.empty:
        raise DuneClientError("Dune USDC daily price query returned no rows.")

    if "date" not in df.columns or "price_usd" not in df.columns:
        raise DuneClientError(f"Unexpected columns from Dune: {df.columns.tolist()}")

    s = (
        df
        .assign(date=pd.to_datetime(df["date"], utc=True))
        .set_index("date")["price_usd"]
        .sort_index()
        .rename("close")
    )
    return s

