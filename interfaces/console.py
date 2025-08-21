"""
Interface console
-----------------
Analyse rapide en CLI.
"""

import argparse
from core.data import fetch_market_chart
from core.indicators import sma, bollinger_bands, rsi
from core.signals import buy_signal, sell_signal
from core.levels import propose_levels

def main():
    parser = argparse.ArgumentParser(description="Analyse crypto en CLI")
    parser.add_argument("--coin", type=str, required=True, help="Identifiant de la crypto (ex: bitcoin)")
    parser.add_argument("--vs", type=str, default="eur", help="Devise (eur/usd)")
    parser.add_argument("--days", type=int, default=90, help="Nombre de jours d’historique")
    args = parser.parse_args()

    df = fetch_market_chart(args.coin, vs=args.vs, days=args.days)
    df["sma200"] = sma(df["close"], 200)
    df["bb_mid"], df["bb_high"], df["bb_low"] = bollinger_bands(df["close"])
    df["rsi"] = rsi(df["close"])

    latest = df.iloc[-1]
    signal_buy = buy_signal(latest["close"], latest["sma200"], latest["rsi"], latest["bb_low"])
    signal_sell = sell_signal(latest["close"], latest["rsi"], latest["bb_high"])
    levels = propose_levels(latest["close"], atr=200)  # exemple fixe

    print(f"Dernier prix: {latest['close']:.2f} {args.vs.upper()}")
    print(f"RSI: {latest['rsi']:.2f}")
    print(f"Signal Achat: {'Oui' if signal_buy else 'Non'}")
    print(f"Signal Vente: {'Oui' if signal_sell else 'Non'}")
    print(f"Niveaux proposés: {levels}")

if __name__ == "__main__":
    main()
