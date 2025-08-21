"""
Module data.py
--------------
Responsable de la récupération des prix de cryptomonnaies depuis l’API CoinGecko.
"""

import os
import requests
import pandas as pd
from datetime import datetime

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_market_chart(coin_id: str, vs: str = "eur", days: int = 90) -> pd.DataFrame:
    """
    Récupère l’historique des prix via CoinGecko.
    
    Parameters:
        coin_id (str): identifiant de la crypto (ex: 'bitcoin').
        vs (str): devise de référence ('eur' ou 'usd').
        days (int): nombre de jours d’historique (1–365, max).
    
    Returns:
        pd.DataFrame: colonnes = ['date', 'close'].
    """
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days}

    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    return df
