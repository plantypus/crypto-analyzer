"""
Module backtest.py
------------------
Backtest simple d’une stratégie long-only basée sur RSI et SMA.
"""

import pandas as pd
from core.indicators import sma, rsi

def backtest_long_only(df: pd.DataFrame, rsi_buy=30, rsi_sell=70, sma_short=20, sma_long=200) -> dict:
    """
    Backtest long-only basique : achat si conditions RSI/SMA, vente sinon.
    """
    df = df.copy()
    df["sma_short"] = sma(df["close"], sma_short)
    df["sma_long"] = sma(df["close"], sma_long)
    df["rsi"] = rsi(df["close"])

    position = 0
    equity = 1000  # portefeuille initial
    for i in range(len(df)):
        row = df.iloc[i]
        if position == 0:
            if row["rsi"] < rsi_buy and row["sma_short"] > row["sma_long"]:
                position = equity / row["close"]  # acheter
                equity = 0
        else:
            if row["rsi"] > rsi_sell:
                equity = position * row["close"]  # vendre
                position = 0
    if position > 0:
        equity = position * df["close"].iloc[-1]
    
    return {"equity": equity, "return": (equity - 1000) / 1000}
