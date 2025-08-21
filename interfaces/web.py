# interfaces/web.py
# -*- coding: utf-8 -*-
"""
Interface Streamlit (v2.0 – modulaire)
--------------------------------------
UI interactive s'appuyant sur le moteur 'core':
- Récupération des données (core.data)
- Indicateurs (core.indicators)
- Signaux (core.signals)
- Niveaux conseillés (core.levels)
- Backtest (core.backtest)

Lancer :
    python crypto_analyser.py web
ou :
    streamlit run interfaces/web.py
"""

from __future__ import annotations
import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Moteur (core)
from crypto_analyzer.core.data import fetch_market_chart
from crypto_analyzer.core.indicators import sma, bollinger_bands, rsi
from crypto_analyzer.core.signals import buy_signal, sell_signal
from crypto_analyzer.core.levels import propose_levels
from crypto_analyzer.core.backtest import backtest_long_only


# -------------------------------------------------------------------
# Aide/Glossaire (markdown enrichi)
# -------------------------------------------------------------------
HELP_MD = """
# 📘 Guide des paramètres et indicateurs

## 1. Moyennes mobiles (SMA)
**SMA (Simple Moving Average)** = *moyenne arithmétique des prix de clôture sur une période donnée*.  

- **SMA20** → tendance **court terme**  
- **SMA50** → tendance **moyen terme**  
- **SMA200** → tendance **long terme**  

💡 **Exemple** :  
➡️ Si le prix du Bitcoin est **au-dessus de la SMA200**, la tendance est **haussière**.  
➡️ En dessous = tendance **baissière**.

---

## 2. Bandes de Bollinger (BB)
Les **Bandes de Bollinger** encadrent le prix autour d’une moyenne mobile :  
- **BB Haut** = SMA20 + *σ* × écart-type  
- **BB Bas** = SMA20 – *σ* × écart-type  

👉 Elles reflètent la **volatilité** :  
- Bandes larges → marché **volatile**  
- Bandes resserrées → marché **calme**, souvent avant un mouvement fort  

💡 **Exemple** :  
➡️ Prix touche la **bande basse** → signal d’**achat potentiel** (prix survendu).  
➡️ Prix touche la **bande haute** → signal de **vente** (prix suracheté).

---

## 3. Volatilité
La **volatilité** = amplitude des variations de prix.  

- Faible = marché calme  
- Forte = marché risqué mais opportunités possibles  

💡 **Exemple** :  
➡️ Un actif à **2%/jour** = stable  
➡️ Une crypto à **15%/jour** = variations très fortes

---

## 4. Termes techniques de trading
- **Ordre limite** : ordre exécuté seulement si le prix atteint ton niveau fixé  
  *Exemple : acheter du BTC à 50 000€ si le prix descend jusque-là*  

- **Target (objectif)** : niveau où tu prends tes profits  
  *Exemple : achat à 50 000€, target à 60 000€ → +20%*  

- **Stop (stop-loss)** : niveau où tu coupes tes pertes  
  *Exemple : achat à 50 000€, stop à 47 000€ → perte max ~6%*  

---

## 5. RSI (Relative Strength Index)
Le **RSI** mesure la force des hausses vs baisses (0 → 100).  

- RSI < 30 = **survente** → possible achat  
- RSI > 70 = **surachat** → possible vente  

💡 **Exemple** :  
➡️ RSI = 25 + prix sur Bollinger bas = **signal d’achat fort**.

---

## 6. ATR (Average True Range)
L’**ATR** mesure l’amplitude moyenne des mouvements.  

- ATR élevé → fortes fluctuations → stop plus large nécessaire  
- ATR faible → stop serré possible  

💡 **Exemple** :  
➡️ ATR = 1200€ → un stop à 2000€ est réaliste  
➡️ Stop à 200€ = trop serré → risque d’être touché inutilement

---

# ⚖️ Décider d’acheter ou vendre

Une décision solide combine plusieurs signaux :  

1. **Tendance générale (SMA200)** : au-dessus = haussier, en-dessous = baissier  
2. **RSI** : < 30 → achat / > 70 → vente  
3. **Bandes de Bollinger** : bas = opportunité d’achat / haut = alerte vente  
4. **Target & Stop** : définis AVANT d’entrer en position  
5. **ATR & Volatilité** : ajuster les stops à la dynamique du marché  

---

## 🔑 Exemple concret

- BTC = 50 000€  
- Prix **au-dessus SMA200** → tendance long terme haussière  
- RSI = 28 + prix sur Bollinger bas → **signal d’achat**  
- Plan : acheter à 50 000€, **target = 55 000€ (+10%)**, **stop = 48 000€ (–4%)**  
- ATR (proxy) ≈ 1200€ → stop de 2000€ (~4%) cohérent  

👉 **Résumé** :  
- **Acheter** = prix bas + RSI bas + Bollinger bas + tendance long terme haussière  
- **Vendre** = prix haut + RSI haut + Bollinger haut + rupture de support  
- **Toujours fixer target & stop** pour gérer le risque
"""


# -------------------------------------------------------------------
# Utilitaires UI
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def list_popular_coins(vs: str = "eur", per_page: int = 50) -> pd.DataFrame:
    """
    Retourne une liste de cryptos classées par capitalisation (proxy popularité).
    Cache 1h pour limiter le rate-limit. Fallback local si erreur.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    try:
        resp = requests.get(url, params={
            "vs_currency": vs, "order": "market_cap_desc",
            "per_page": per_page, "page": 1
        }, timeout=20)
        if resp.status_code == 429:
            raise RuntimeError("Rate limit 429")
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        return df[["id", "symbol", "name", "market_cap_rank"]].sort_values("market_cap_rank")
    except Exception:
        fallback = [
            {"id": "bitcoin",     "symbol": "btc",  "name": "Bitcoin",    "market_cap_rank": 1},
            {"id": "ethereum",    "symbol": "eth",  "name": "Ethereum",   "market_cap_rank": 2},
            {"id": "tether",      "symbol": "usdt", "name": "Tether",     "market_cap_rank": 3},
            {"id": "binancecoin", "symbol": "bnb",  "name": "BNB",        "market_cap_rank": 4},
            {"id": "ripple",      "symbol": "xrp",  "name": "XRP",        "market_cap_rank": 5},
            {"id": "cardano",     "symbol": "ada",  "name": "Cardano",    "market_cap_rank": 6},
            {"id": "solana",      "symbol": "sol",  "name": "Solana",     "market_cap_rank": 7},
            {"id": "dogecoin",    "symbol": "doge", "name": "Dogecoin",   "market_cap_rank": 8},
            {"id": "tron",        "symbol": "trx",  "name": "TRON",       "market_cap_rank": 9},
            {"id": "polkadot",    "symbol": "dot",  "name": "Polkadot",   "market_cap_rank": 10},
        ]
        return pd.DataFrame(fallback)


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_coin_id(query: str) -> str:
    """
    Résout un symbole/nom/id vers l'id CoinGecko (simple, avec cache).
    - Tente /search, sinon /coins/list.
    """
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Entrée vide.")

    # 1) /search
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": q}, timeout=20
        )
        if r.status_code == 200:
            js = r.json()
            for c in js.get("coins", []):
                if c.get("symbol", "").lower() == q or c.get("id", "").lower() == q:
                    return c["id"]
            if js.get("coins"):
                return js["coins"][0]["id"]
    except Exception:
        pass

    # 2) /coins/list
    r = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=25)
    r.raise_for_status()
    arr = r.json()
    for c in arr:
        if c.get("symbol", "").lower() == q:
            return c["id"]
    for c in arr:
        if c.get("id", "").lower() == q:
            return c["id"]
    for c in arr:
        if c.get("name", "").lower() == q:
            return c["id"]
    raise ValueError(f"Impossible de résoudre '{query}'.")


def make_plotly_figure(
    df: pd.DataFrame,
    title: str,
    vs: str = "EUR",
    show_sma20: bool = True,
    show_sma50: bool = True,
    show_sma200: bool = True,
    show_bbands: bool = True,
    buy_level: float | None = None,
    target_level: float | None = None,
    stop_level: float | None = None,
) -> go.Figure:
    """
    Construit un graphique Plotly interactif :
    - Valeur au survol + % depuis le début
    - Cases à cocher SMA/Bollinger
    - Format X auto (jj-mm-aaaa ou HH:MM si < 24h)
    """
    start_price = float(df["close"].iloc[0])
    pct_from_start = (df["close"] / start_price - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], mode="lines", name="Prix",
        hovertemplate=(
            "Date: %{x}<br>"
            + f"Prix: %{{y:.2f}} {vs}<br>"
            + "% depuis début: %{customdata:.2f}%<extra></extra>"
        ),
        customdata=pct_from_start,
    ))

    if show_bbands and {"bb_high", "bb_low"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], name="Bollinger haut (20, 2σ)", mode="lines", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], name="Bollinger bas (20, 2σ)", mode="lines", line=dict(dash="dot")))

    if show_sma20 and "sma20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], name="SMA20", mode="lines"))
    if show_sma50 and "sma50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA50", mode="lines"))
    if show_sma200 and "sma200" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="SMA200", mode="lines"))

    for lvl, label in ((buy_level, "Achat"), (target_level, "Objectif"), (stop_level, "Stop")):
        if lvl is not None and lvl > 0:
            fig.add_hline(y=lvl, line_dash="dash", annotation_text=f"{label} ({lvl:.2f} {vs})", annotation_position="top left")

    span_seconds = (df.index[-1] - df.index[0]).total_seconds()
    if span_seconds < 24 * 3600:
        tickformat = "%H:%M"
        title_extra = f" — aujourd'hui ({df.index[-1].date()})"
    else:
        tickformat = "%d-%m-%Y"
        title_extra = ""

    fig.update_xaxes(tickformat=tickformat)
    fig.update_layout(
        title=title + title_extra,
        xaxis_title="Date", yaxis_title=f"Prix ({vs})",
        hovermode="x unified"
    )
    return fig


# -------------------------------------------------------------------
# Application
# -------------------------------------------------------------------
def run_app():
    st.set_page_config(page_title="Crypto Analyser", layout="wide")
    st.title("Crypto Analyser — Modulaire (v2.0)")

    tabs = st.tabs(["Analyse", "Backtest (≤ 1 an)", "📘 Aide & Glossaire"])

    # === Onglet Analyse ===
    with tabs[0]:
        with st.sidebar:
            st.header("Paramètres d'analyse")
            try:
                df_pop = list_popular_coins("eur", per_page=50)
                options = [f"{row.symbol.upper()} — {row.name} ({row.id})" for _, row in df_pop.iterrows()]
            except Exception as e:
                st.warning(f"Impossible de charger la liste des cryptos populaires : {e}")
                df_pop, options = pd.DataFrame(), []

            coin_input = st.text_input("Cryptomonnaie", value="btc",
                                       help="Symbole (BTC) ou nom (bitcoin). Tu peux aussi choisir ci-dessous.")
            if options:
                selection = st.selectbox("…ou choisir (par capitalisation)", options)
                if selection:
                    coin_input = selection.split("(")[-1].rstrip(")")

            vs = st.selectbox("Devise", ["EUR", "USD"], index=0)
            days = st.number_input("Fenêtre (jours)", min_value=1, max_value=365, value=90, step=1)

            st.markdown("**Courbes à afficher**")
            show_sma20 = st.checkbox("SMA20", value=True)
            show_sma50 = st.checkbox("SMA50", value=True)
            show_sma200 = st.checkbox("SMA200", value=True)
            show_bbands = st.checkbox("Bandes de Bollinger", value=True)

            buy_lvl = st.number_input("Votre prix d'achat (optionnel)", min_value=0.0, value=0.0)
            target_lvl = st.number_input("Objectif (optionnel)", min_value=0.0, value=0.0)
            stop_lvl = st.number_input("Stop (optionnel)", min_value=0.0, value=0.0)

            lancer = st.button("Analyser", use_container_width=True)

        if lancer:
            try:
                coin_id = resolve_coin_id(coin_input)
                df = fetch_market_chart(coin_id, vs=vs.lower(), days=int(days))

                # Indicateurs
                df["sma20"] = sma(df["close"], 20)
                df["sma50"] = sma(df["close"], 50)
                df["sma200"] = sma(df["close"], 200)
                mid, high, low = bollinger_bands(df["close"], window=20, num_std=2)
                df["bb_mid"], df["bb_high"], df["bb_low"] = mid, high, low
                df["rsi14"] = rsi(df["close"], 14)
                # ATR proxy (close-only)
                df["atr14_proxy"] = df["close"].diff().abs().rolling(14).mean()

                last = df.iloc[-1]
                sig_buy = buy_signal(last["close"], last.get("sma200", last["close"]), last["rsi14"], last["bb_low"])
                sig_sell = sell_signal(last["close"], last["rsi14"], last["bb_high"])

                # Niveaux proposés
                atr_val = float(last["atr14_proxy"]) if pd.notna(last["atr14_proxy"]) else 0.0
                levels = propose_levels(float(last["close"]), atr_val)

                # Cartes
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dernier prix", f"{last['close']:.2f} {vs}")
                # Variation 24h (si dispo)
                if len(df) > 25:
                    c2.metric("Var. 24h", f"{(df['close'].iloc[-1]/df['close'].iloc[-25]-1)*100:.2f}%")
                # Volatilité approx
                if len(df) > 5:
                    vol = df["close"].pct_change().dropna().std() * (len(df) ** 0.5) * 100
                    c3.metric("Volatilité (approx)", f"{vol:.2f}%")
                # Drawdown
                dd = df["close"] / df["close"].cummax() - 1
                c4.metric("Pire baisse (DD)", f"{dd.min()*100:.2f}%")

                st.subheader("Signaux & niveaux proposés")
                st.write({
                    "Signal Achat": "Oui" if sig_buy else "Non",
                    "Signal Vente": "Oui" if sig_sell else "Non",
                    "Prix limite conseillé": f"{levels['limite']:.2f} {vs}",
                    "Stop": f"{levels['stop']:.2f} {vs}",
                    "Target": f"{levels['target']:.2f} {vs}",
                })

                # Graphique interactif
                fig = make_plotly_figure(
                    df,
                    title=f"{coin_id} | {int(days)} jours | {vs}",
                    vs=vs,
                    show_sma20=show_sma20,
                    show_sma50=show_sma50,
                    show_sma200=show_sma200,
                    show_bbands=show_bbands,
                    buy_level=(buy_lvl or None),
                    target_level=(target_lvl or None),
                    stop_level=(stop_lvl or None),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur : {e}")

    # === Onglet Backtest (≤ 1 an) ===
    with tabs[1]:
        st.subheader("Backtest long-only (≤ 1 an)")
        colA, colB, colC = st.columns(3)
        with colA:
            coin_bt = st.text_input("Crypto (symbole/nom/id)", value="btc")
            vs_bt = st.selectbox("Devise", ["EUR", "USD"], index=0)
            bt_days = st.selectbox("Fenêtre", ["180", "365"], index=1)
        with colB:
            rsi_buy = st.slider("RSI achat", min_value=10, max_value=40, value=30, step=1)
            rsi_sell = st.slider("RSI vente", min_value=60, max_value=90, value=70, step=1)
        with colC:
            sma_short = st.selectbox("SMA courte", [20, 50, 100], index=1)
            sma_long = st.selectbox("SMA longue", [100, 150, 200], index=2)

        if st.button("Lancer le backtest", use_container_width=True):
            try:
                coin_id = resolve_coin_id(coin_bt)
                df_bt = fetch_market_chart(coin_id, vs=vs_bt.lower(), days=int(bt_days))
                res = backtest_long_only(df_bt, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
                                         sma_short=int(sma_short), sma_long=int(sma_long))
                st.success("Backtest terminé")
                st.write({
                    "Capital final (base 1000)": f"{res['equity']:.2f}",
                    "Rendement": f"{res['return']*100:.2f}%",
                })

                # Courbe d'équité (Plotly)
                eq = pd.Series([1000, res["equity"]], index=[df_bt.index[0], df_bt.index[-1]])
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines+markers", name="Équité"))
                fig_bt.update_layout(title=f"Équité — {coin_id} | {vs_bt}",
                                     xaxis_title="Date", yaxis_title="Capital")
                st.plotly_chart(fig_bt, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur backtest : {e}")

    # === Onglet Aide & Glossaire ===
    with tabs[2]:
        st.markdown(HELP_MD, unsafe_allow_html=True)


# Pour lancer directement : streamlit run interfaces/web.py
if __name__ == "__main__":
    run_app()
