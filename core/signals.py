"""
Module signals.py
-----------------
Définit les règles simples pour générer des signaux d'achat/vente.
"""

import pandas as pd

def buy_signal(price: float, sma200: float, rsi_val: float, bb_low: float) -> bool:
    """
    Retourne True si les conditions d’achat sont réunies.
    Critères :
    - Prix > SMA200 (tendance long terme haussière)
    - RSI < 30 (survente)
    - Prix <= Bollinger bas
    """
    return (price > sma200) and (rsi_val < 30) and (price <= bb_low)

def sell_signal(price: float, rsi_val: float, bb_high: float) -> bool:
    """
    Retourne True si conditions de vente.
    Critères :
    - RSI > 70 (surachat)
    - OU prix >= Bollinger haut
    """
    return (rsi_val > 70) or (price >= bb_high)
