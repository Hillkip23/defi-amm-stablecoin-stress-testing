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
    """
    Return headers for Dune API.

    Raises
    ------
    DuneClientError
        If DUNE_API_KEY is not set.
    """
    if not DUNE_API_KEY:
        raise DuneClientError(
            "DUNE_API_KEY environment variable is not set. "
            "Either export a Dune API key or use local/CSV data instead."
        )
    # Header name is case-insensitive; this matches Dune docs.
    return {"x-dune-api-key": DUNE_API_KEY}


def _raise_for_status(query_id: int, resp: requests.Response, context: str) -> None:
    """
    Centralised HTTP error handling with friendly messages.
    """
    if resp.status_code == 401:
        # Fastest path: tell the user to keep using CSV if they do not want to debug the key.
        raise DuneClientError(
            f"Dune returned 401 (invalid API key) while {context} for query {query_id}. "
            "Double-check DUNE_API_KEY, or skip Dune and rely on local/CSV data in the app."
        )
    if resp.status_code != 200:
        raise DuneClientError(
            f"HTTP {resp.status_code} while {context} for query {query_id}: {resp.text}"
        )


def execute_query_and_wait(
    query_id: int,
    params: Optional[dict] = None,
    poll_interval: float = 2.0,
) -> pd.DataFrame:
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
    _raise_for_status(query_id, r, "starting execution")

    execution_id = r.json().get("execution_id")
    if not execution_id:
        raise DuneClientError(
            f"No execution_id returned for query {query_id}: {r.text}"
        )

    # 2) Poll status
    status_url = f"{DUNE_BASE_URL}/execution/{execution_id}/status"
    results_url = f"{DUNE_BASE_URL}/execution/{execution_id}/results"

    while True:
        s = requests.get(status_url, headers=headers)
        _raise_for_status(query_id, s, "polling status")

        state = s.json().get("state")
        if state in ("QUERY_STATE_COMPLETED", "SUCCESS"):
            break
        if state in ("QUERY_STATE_FAILED", "ERROR"):
            raise DuneClientError(
                f"Dune query {query_id} failed with state={state}: {s.text}"
            )

        time.sleep(poll_interval)

    # 3) Fetch results
    res = requests.get(results_url, headers=headers)
    _raise_for_status(query_id, res, "fetching results")

    rows = res.json().get("result", {}).get("rows", [])
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# === High-level helpers for your app =========================================


def get_uniswap_eth_usdc_daily_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Return a daily close/mid-price series for Uniswap v3 ETH/USDC 0.05% pool.

    This wraps a saved Dune query that should:
    - Filter trades or pool observations for the canonical ETH-USDC 0.05% pool.
    - Aggregate to one row per day with columns: date, price_usd.
    """
    QUERY_ID = 6364165  # TODO: replace with your real Dune query ID

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    df = execute_query_and_wait(QUERY_ID, params=params)

    if df.empty:
        raise DuneClientError(
            "Dune ETH-USDC daily price query returned no rows. "
            "If this persists, export the query as CSV from Dune and load it via the app."
        )

    if "date" not in df.columns or "price_usd" not in df.columns:
        raise DuneClientError(f"Unexpected columns from Dune: {df.columns.tolist()}")

    s = (
        df.assign(date=pd.to_datetime(df["date"], utc=True))
        .set_index("date")["price_usd"]
        .sort_index()
        .rename("close")
    )
    return s


def get_usdc_daily_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Daily USDC/USD price series from Dune prices.usd.
    """
    QUERY_ID = 6364523  # your saved USDC prices.usd query id

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    df = execute_query_and_wait(QUERY_ID, params=params)

    if df.empty:
        raise DuneClientError(
            "Dune USDC daily price query returned no rows. "
            "You can instead download prices.usd as CSV from Dune and upload it in the app."
        )

    if "date" not in df.columns or "price_usd" not in df.columns:
        raise DuneClientError(f"Unexpected columns from Dune: {df.columns.tolist()}")

    s = (
        df.assign(date=pd.to_datetime(df["date"], utc=True))
        .set_index("date")["price_usd"]
        .sort_index()
        .rename("close")
    )
    return s
