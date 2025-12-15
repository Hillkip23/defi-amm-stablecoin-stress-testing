import matplotlib.pyplot as plt
import pandas as pd


def plot_price(df: pd.DataFrame):
    plt.figure()
    plt.plot(df.index, df["price"])
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Simulated Price Path")
    plt.grid(True)


def plot_values(df: pd.DataFrame):
    plt.figure()
    plt.plot(df.index, df["hodl_value"], label="HODL")
    plt.plot(df.index, df["lp_value"], label="LP")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value (normalized)")
    plt.title("LP vs HODL Value")
    plt.legend()
    plt.grid(True)


def plot_lp_over_hodl(df: pd.DataFrame):
    plt.figure()
    plt.plot(df.index, df["lp_over_hodl"])
    plt.xlabel("Time")
    plt.ylabel("LP / HODL")
    plt.title("LP Performance vs HODL")
    plt.grid(True)


def plot_il(df: pd.DataFrame):
    plt.figure()
    plt.plot(df.index, df["impermanent_loss"])
    plt.xlabel("Time")
    plt.ylabel("Impermanent Loss (fraction)")
    plt.title("Impermanent Loss Over Time")
    plt.grid(True)
