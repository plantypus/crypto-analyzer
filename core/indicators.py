"""
Module indicators.py
--------------------
Calcul des indicateurs techniques (SMA, Bollinger, RSI, ATR).
"""

import pandas as pd

def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Moyenne mobile simple."""
    return series.rolling(window).mean()

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2):
    """Bandes de Bollinger (haut, bas)."""
    sma_ = sma(series, window)
    std = series.rolling(window).std()
    return sma_, sma_ + num_std * std, sma_ - num_std * std

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()
