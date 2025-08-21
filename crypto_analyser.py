#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_analyser.py — Analyse crypto : CLI + Interface Streamlit (en un seul fichier)
===================================================================================

Version : **1.2.2** (ajout onglet 📘 Aide & Glossaire)

Objectif
--------
Outil pédagogique et modulaire pour :
- récupérer les prix d'une cryptomonnaie (API publique CoinGecko),
- afficher un graphique clair avec indicateurs,
- fournir des signaux simples (achat/vente),
- proposer un prix **limite** d'achat, un **stop** et un **target**,
- fonctionner à la fois en **ligne de commande** ET en **interface web Streamlit**.

Conception (inspirée des "10 règles NASA")
------------------------------------------
1) **Simplicité** : fonctions courtes, une responsabilité principale chacune.
2) **Interfaces claires** : entrées/sorties explicites (docstrings), pas d'effets de bord cachés.
3) **Validation** : vérifs d'arguments et exceptions explicites.
4) **Lisibilité** : noms en français, commentaires concrets.
5) **Modularité** : couches séparées (données / indicateurs / signaux / présentation).
6) **Testabilité** : fonctions pures (réutilisables dans d'autres applis ou tests).
7) **Évitement des globales** : tout passe par paramètres ou retours.
8) **Tolérance aux pannes** : reprises simples (retries HTTP, messages d'erreur utiles).
9) **Traçabilité** : impressions structurées côté CLI.
10) **Évolutivité** : prêt à être intégré dans une app (API/Streamlit/mobile) sans réécrire le cœur.

Prérequis :
    pip install requests pandas matplotlib python-dateutil streamlit plotly

Utilisation :
    # Mode console (CLI)
    python crypto_analyser.py --coin btc --days 180 --vs EUR --buy 52000 --target 65000 --stop 48000

    # Mode interface web (formulaire + bulles d'aide)
    streamlit run crypto_analyser.py

Remarques :
- CoinGecko n'exige pas de clé API pour les endpoints utilisés ici.
- Les couleurs des tracés restent sobres pour maximiser la lisibilité.

Changelog
---------
- **v1.2.2** : ajout de l’onglet **📘 Aide & Glossaire** (markdown enrichi) intégré à l’UI Streamlit.
- **v1.2.1** : gestion du rate-limit CoinGecko (backoff 429, cache, fallback) + per_page réduit.
- **v1.2** :
  - Graphique **interactif** (Plotly) en mode Streamlit avec **valeur au survol** et **% depuis le début**.
  - **Nombre de jours personnalisé** (entrée 1–365) au lieu de listes fixes.
  - **Cases à cocher** pour afficher/masquer SMA20/50/200 et Bandes de Bollinger.
  - Format de date de l’axe X : `jj-mm-aaaa`, ou **HH:MM** si la fenêtre < 24h (titre ajoute "aujourd'hui").
  - Libellés clairs **Oui/Non** pour les signaux.
- **v1.1** : `fetch_market_chart` peut basculer sur `/market_chart/range` si `days > 365` (utile mais non requis pour v1.2).
- **v1.0** : première version fonctionnelle (CLI + Streamlit, indicateurs, signaux, backtest simple).** : première version fonctionnelle (CLI + Streamlit, indicateurs, signaux, backtest simple).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from dateutil import tz
import matplotlib.pyplot as plt

__version__ = "1.2.2"

# =============================
# 1) Couche Données / CoinGecko
# =============================
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _http_get(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = 20) -> dict:
    """Effectue une requête GET avec retries simples.

    Args:
        url: URL absolue.
        params: paramètres de requête.
        retries: nombre de tentatives.
        timeout: délai max en secondes.
    Returns:
        dict JSON décodé.
    Raises:
        HTTPError si dernier essai échoue.
    """
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.2 * (attempt + 1))
    r.raise_for_status()


def resolve_coin_id(query: str) -> str:
    """Résout un symbole/nom (ex: "btc", "bitcoin") vers l'ID CoinGecko (ex: "bitcoin").

    Stratégie :
    1) /search (plus rapide et filtré)
    2) /coins/list (fallback complet)
    """
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Paramètre 'query' vide.")

    # 1) Essai via /search
    try:
        data = _http_get(f"{COINGECKO_BASE}/search", params={"query": q})
        for c in data.get("coins", []):
            if c.get("symbol", "").lower() == q or c.get("id", "").lower() == q:
                return c["id"]
        if data.get("coins"):
            return data["coins"][0]["id"]
    except Exception:
        pass

    # 2) Fallback via /coins/list
    coins = _http_get(f"{COINGECKO_BASE}/coins/list")
    for c in coins:
        if c.get("symbol", "").lower() == q:
            return c["id"]
    for c in coins:
        if c.get("id", "").lower() == q:
            return c["id"]
    for c in coins:
        if c.get("name", "").lower() == q:
            return c["id"]
    raise ValueError(f"Impossible de résoudre le coin '{query}'.")


def list_popular_coins(vs: str = "EUR", per_page: int = 100) -> pd.DataFrame:
    """Retourne un DataFrame des cryptos classées par capitalisation (popularité proxy).

    Utilise /coins/markets order=market_cap_desc. Colonnes clés: id, symbol, name, market_cap_rank.
    """
    vs = (vs or "EUR").lower()
    data = _http_get(
        f"{COINGECKO_BASE}/coins/markets",
        params={"vs_currency": vs, "order": "market_cap_desc", "per_page": per_page, "page": 1},
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=["id", "symbol", "name", "market_cap_rank"])  
    df = pd.DataFrame(data)
    return df[["id", "symbol", "name", "market_cap_rank"]].sort_values("market_cap_rank")


# =============================
# 2) Modèle & Récupération série
# =============================
@dataclass
class PriceFrame:
    """Conteneur des prix.

    Attributes:
        df: DataFrame indexé en datetime local avec colonne 'prix'.
        vs: Devise de cotation (ex. EUR, USD).
        coin_id: Identifiant CoinGecko.
    """
    df: pd.DataFrame
    vs: str
    coin_id: str


def fetch_market_chart(coin_id: str, vs: str = "EUR", days: str = "90", interval: Optional[str] = None) -> PriceFrame:
    """Récupère l'historique de prix chez CoinGecko.

    - Pour `days` > 365 (ex. 730 pour ~2 ans), utilise `/market_chart/range` pour éviter 401 Unauthorized.
    - Sinon utilise `/market_chart` classique.
    """
    if not coin_id:
        raise ValueError("'coin_id' requis")

    vs_lc = (vs or "EUR").lower()

    use_range = False
    if isinstance(days, str) and days.isdigit():
        use_range = int(days) > 365
    elif isinstance(days, (int, float)):
        use_range = int(days) > 365

    if use_range:
        # Fenêtre en jours -> timestamps Unix (secondes)
        import time as _time
        now = int(_time.time())
        from_ts = now - int(days) * 24 * 3600
        params = {"vs_currency": vs_lc, "from": from_ts, "to": now}
        data = _http_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range", params=params)
    else:
        params = {"vs_currency": vs_lc, "days": str(days)}
        if interval:
            params["interval"] = interval
        data = _http_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", params=params)

    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("Pas de données retournées par l'API.")
    df = pd.DataFrame(prices, columns=["ts", "prix"])  # ts en ms UTC
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(tz.tzlocal())
    df = df.set_index("ts").sort_index()
    return PriceFrame(df=df, vs=vs.upper(), coin_id=coin_id)


# =============================
# 3) Indicateurs & Statistiques
# =============================


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple avec tolérance aux petites séries."""
    return series.rolling(window, min_periods=max(1, window // 3)).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute SMA20/50/200, Bandes de Bollinger (20, 2σ), ATR(14), RSI(14).

    Retourne un **nouveau** DataFrame pour éviter les effets de bord.
    """
    d = df.copy()
    p = d["prix"]
    # SMA
    d["SMA20"] = rolling_mean(p, 20)
    d["SMA50"] = rolling_mean(p, 50)
    d["SMA200"] = rolling_mean(p, 200)
    # Bollinger
    std20 = p.rolling(20).std()
    d["BB_Haut"] = d["SMA20"] + 2 * std20
    d["BB_Bas"] = d["SMA20"] - 2 * std20
    # ATR (approx via variation absolue faute d'OHLC)
    tr = p.diff().abs()
    d["ATR14"] = tr.rolling(14).mean()
    # RSI(14) (version simple : SMA des gains/pertes)
    delta = p.diff()
    gains = delta.clip(lower=0)
    pertes = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = pertes.rolling(14).mean().replace(0, 1e-12)
    rs = avg_gain / avg_loss
    d["RSI14"] = 100 - (100 / (1 + rs))
    return d


def compute_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Calcule quelques statistiques pédagogiques sur la série 'prix'."""
    p = df["prix"]
    ret = p.pct_change().dropna()
    dd = (p / p.cummax() - 1.0)
    return {
        "Dernier prix": float(p.iloc[-1]),
        "Variation 24h (%)": float((p.iloc[-1] / p.iloc[-min(25, len(p))] - 1) * 100) if len(p) > 25 else None,
        "Volatilité estimée (%)": float(ret.std() * 100 * (len(ret) ** 0.5)) if len(ret) > 5 else None,
        "Pire baisse (drawdown) (%)": float(dd.min() * 100),
    }


def pnl_from_buy(df: pd.DataFrame, buy_price: float, qty: Optional[float] = None) -> Tuple[pd.DataFrame, float, Optional[float]]:
    """Ajoute la perf vs prix d'achat et calcule le P&L courant.

    Returns:
        (df_out, perf_en_%_courante, pnl_absolu_ou_None)
    """
    out = df.copy()
    out["Perf_vs_Achat (%)"] = (out["prix"] / buy_price - 1.0) * 100.0
    last_perf = float(out["Perf_vs_Achat (%)"].iloc[-1])
    last_pnl = None
    if qty is not None:
        last_pnl = float((out["prix"].iloc[-1] - buy_price) * qty)
    return out, last_perf, last_pnl


# =============================
# 4) Signaux & Niveaux conseillés
# =============================


def buy_signal_row(r: pd.Series) -> bool:
    """Signal d'achat simple : tendance OK + survente contrôlée.
    - Tendance : SMA50 > SMA200
    - Survente : RSI14 < 30 ET prix < Bande de Bollinger basse
    """
    if any(c not in r for c in ("SMA50", "SMA200", "RSI14", "BB_Bas", "prix")):
        return False
    tendance_ok = r["SMA50"] > r["SMA200"]
    setup = (r["RSI14"] < 30) and (r["prix"] < r["BB_Bas"])
    return bool(tendance_ok and setup)


def sell_signal_row(r: pd.Series) -> bool:
    """Signal de vente simple : surachat ou débordement de bande haute.
    - RSI14 > 70 OU prix > Bande de Bollinger haute
    """
    if any(c not in r for c in ("RSI14", "BB_Haut", "prix")):
        return False
    return bool((r["RSI14"] > 70) or (r["prix"] > r["BB_Haut"]))


def propose_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Propose des niveaux : limite (achat), stop, target basés sur l'ATR.

    Heuristique :
    - limite ≈ max(min des 10 derniers jours, prix - 0.5*ATR)
    - stop   ≈ prix - 1.0*ATR
    - target ≈ prix + 2.0*ATR
    """
    r = df.iloc[-1]
    atr = float(r.get("ATR14", 0) or 0)
    prix = float(r["prix"])  # prix courant
    limite = max(df["prix"].tail(10).min(), prix - 0.5 * atr) if atr > 0 else prix * 0.99
    stop = prix - 1.0 * atr if atr > 0 else prix * 0.98
    target = prix + 2.0 * atr if atr > 0 else prix * 1.02
    return {"limite": float(limite), "stop": float(stop), "target": float(target)}


# =============================
# 5) Présentation (matplotlib + Plotly)
# =============================


def plot_price(
    df: pd.DataFrame,
    title: str,
    buy: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    vs: str = "EUR",
    show_indicators: bool = True,
    annotate_signals: bool = True,
    show: bool = True,
):
    """Affiche le graphique principal (matplotlib) pour le mode CLI."""
    fig = plt.figure(figsize=(11.5, 6))
    ax = df["prix"].plot(label="Prix")

    if show_indicators:
        if {"BB_Haut", "BB_Bas"}.issubset(df.columns):
            df["BB_Haut"].plot(ax=ax, linestyle=":", label="Bollinger haut (20, 2σ)")
            df["BB_Bas"].plot(ax=ax, linestyle=":", label="Bollinger bas (20, 2σ)")
        for c in ("SMA20", "SMA50", "SMA200"):
            if c in df.columns:
                df[c].plot(ax=ax, label=c)

    # Zones d'aide achat/target/stop (légères)
    pmin, pmax = float(df["prix"].min()), float(df["prix"].max())
    if buy is not None:
        ax.axhline(buy, linestyle="--", label=f"Achat ({buy} {vs})")
        ax.fill_between(df.index, pmin, buy, alpha=0.05)
    if target is not None:
        ax.axhline(target, linestyle="--", label=f"Objectif ({target} {vs})")
        ax.fill_between(df.index, target, pmax, alpha=0.05)
    if stop is not None:
        ax.axhline(stop, linestyle="--", label=f"Stop ({stop} {vs})")
        ax.fill_between(df.index, pmin, stop, alpha=0.05)

    # Annotation des signaux sur le dernier point
    if annotate_signals and {"RSI14", "BB_Bas", "BB_Haut", "SMA50", "SMA200"}.issubset(df.columns):
        r = df.iloc[-1]
        x, y = df.index[-1], float(r["prix"])
        if buy_signal_row(r):
            ax.scatter([x], [y], s=60, marker="^", label="Signal Achat")
            ax.annotate("Achat", xy=(x, y), xytext=(0, 18), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-"), ha="center")
        if sell_signal_row(r):
            ax.scatter([x], [y], s=60, marker="v", label="Signal Vente")
            ax.annotate("Vente", xy=(x, y), xytext=(0, -22), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-"), ha="center")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Prix ({vs})")
    ax.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def make_plotly_figure(
    df: pd.DataFrame,
    title: str,
    vs: str = "EUR",
    show_sma20: bool = True,
    show_sma50: bool = True,
    show_sma200: bool = True,
    show_bbands: bool = True,
    buy: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
):
    """Construit un graphique **interactif** Plotly pour Streamlit.

    - Affiche la valeur au survol + **% depuis le début**.
    - Cases à cocher pour masquer/afficher SMA et Bandes de Bollinger.
    - Axe X formaté en `jj-mm-aaaa` ou `HH:MM` si fenêtre < 24h (titre ajoute "aujourd'hui").
    """
    import plotly.graph_objects as go

    # % depuis début
    start_price = float(df["prix"].iloc[0])
    pct_from_start = (df["prix"] / start_price - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["prix"],
        mode="lines",
        name="Prix",
        hovertemplate=(
            "Date: %{x}<br>"
            + f"Prix: %{{y:.2f}} {vs}<br>"
            + "% depuis début: %{customdata:.2f}%<extra></extra>"
        ),
        customdata=pct_from_start,
    ))

    if show_bbands and {"BB_Haut", "BB_Bas"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Haut"], name="Bollinger haut (20, 2σ)", mode="lines", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Bas"], name="Bollinger bas (20, 2σ)", mode="lines", line=dict(dash="dot")))

    if show_sma20 and "SMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"))
    if show_sma50 and "SMA50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", mode="lines"))
    if show_sma200 and "SMA200" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", mode="lines"))

    # Lignes horizontales
    for lvl, label in ((buy, "Achat"), (target, "Objectif"), (stop, "Stop")):
        if lvl is not None and lvl > 0:
            fig.add_hline(y=lvl, line_dash="dash", annotation_text=f"{label} ({lvl} {vs})", annotation_position="top left")

    # Axe X format
    span_seconds = (df.index[-1] - df.index[0]).total_seconds()
    if span_seconds < 24 * 3600:
        tickformat = "%H:%M"
        title_extra = f" — aujourd'hui ({df.index[-1].date()})"
    else:
        tickformat = "%d-%m-%Y"
        title_extra = ""
    fig.update_xaxes(tickformat=tickformat)

    fig.update_layout(title=title + title_extra, xaxis_title="Date", yaxis_title=f"Prix ({vs})", hovermode="x unified")
    return fig


def plot_equity_curve(eq: pd.Series, title: str = "Courbe d'équité", show: bool = True):
    """Trace la courbe d'équité (matplotlib). Retourne la figure pour intégration Streamlit.
    Args:
        show: si True, plt.show(); sinon retourne la figure.
    """
    fig = plt.figure(figsize=(11.5, 4.5))
    eq.plot()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Capital (base=1)')
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# =============================
# 6) Backtest (1 à 2 ans) — validation des paramètres
# =============================

def generate_signals(df: pd.DataFrame, rsi_buy: float = 30, rsi_sell: float = 70,
                     sma_short: int = 50, sma_long: int = 200, bb_mult: float = 2.0) -> pd.DataFrame:
    """Génère colonnes booléennes 'sig_buy' et 'sig_sell' selon règles simples.

    Règles (pédagogiques) :
      - Achat si SMA_short > SMA_long ET RSI < rsi_buy ET prix < (SMA20 - bb_mult*std20)
      - Vente si RSI > rsi_sell OU prix > (SMA20 + bb_mult*std20)

    Returns: nouveau DataFrame avec colonnes sig_buy/sig_sell (bool).
    """
    d = compute_indicators(df)
    std20 = d['prix'].rolling(20).std()
    bb_low = d['SMA20'] - bb_mult * std20
    bb_high = d['SMA20'] + bb_mult * std20

    d['sig_buy'] = (d[f'SMA{sma_short}'] > d[f'SMA{sma_long}']) & (d['RSI14'] < rsi_buy) & (d['prix'] < bb_low)
    d['sig_sell'] = (d['RSI14'] > rsi_sell) | (d['prix'] > bb_high)
    return d


def backtest_long_only(df: pd.DataFrame, rsi_buy: float = 30, rsi_sell: float = 70,
                       sma_short: int = 50, sma_long: int = 200, bb_mult: float = 2.0,
                       fee_bps: float = 10.0, slip_bps: float = 5.0) -> Dict[str, object]:
    """Backtest très simple, long-only, une position à la fois.

    Hypothèses :
      - Entrée si sig_buy True ; sortie si sig_sell True.
      - Exécution au **bar suivant** (évite le look-ahead) au prix de clôture suivant.
      - Frais+slippage appliqués à l'entrée et à la sortie (en points de base).
      - Pas de stop/target intraday (close-only).

    Returns:
      dict avec métriques et DataFrame 'res' (incluant courbe d'équité et trades).
    """
    d = generate_signals(df, rsi_buy, rsi_sell, sma_short, sma_long, bb_mult).copy()
    p = d['prix']
    next_p = p.shift(-1)  # exécution au bar suivant

    in_pos = False
    entry_px = None
    equity = []  # rendement cumulé
    trade_rets = []
    trade_list = []
    cumret = 1.0

    # frais exprimés en proportion
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0

    for i in range(len(d)):
        if i == len(d) - 1:
            # pas d'exécution possible au-delà
            equity.append(cumret)
            break
        price = p.iloc[i]
        price_next = next_p.iloc[i]

        if not in_pos:
            if d['sig_buy'].iloc[i]:
                # entrer demain au prix_next augmenté des coûts
                entry_px = price_next * (1 + fee + slip)
                in_pos = True
                trade_list.append({'type': 'BUY', 'date': p.index[i+1], 'price': float(entry_px)})
            equity.append(cumret)
        else:
            # en position : rendement journalier = price_next/price courant (proxy)
            daily_ret = price_next / price
            cumret *= daily_ret
            # vérifier signal de sortie
            if d['sig_sell'].iloc[i]:
                exit_px_gross = price_next
                exit_px = exit_px_gross * (1 - fee - slip)
                trade_ret = exit_px / entry_px
                trade_rets.append(trade_ret)
                trade_list.append({'type': 'SELL', 'date': p.index[i+1], 'price': float(exit_px), 'ret': float(trade_ret-1)})
                in_pos = False
                entry_px = None
            equity.append(cumret)

    # métriques
    eq = pd.Series(equity, index=p.index[:len(equity)])
    rets = eq.pct_change().fillna(0)
    # max drawdown
    rollmax = eq.cummax()
    dd = eq / rollmax - 1
    maxdd = float(dd.min())
    total_ret = float(eq.iloc[-1] - 1)
    n_days = (p.index[-1] - p.index[0]).days or 1
    cagr = (1 + total_ret) ** (365.25 / n_days) - 1 if n_days > 365 else total_ret
    sharpe_like = (rets.mean() / (rets.std() + 1e-12)) * (365.25 ** 0.5) if rets.std() > 0 else 0.0

    wins = [r for r in trade_rets if r > 1]
    losses = [r for r in trade_rets if r <= 1]
    win_rate = len(wins) / len(trade_rets) if trade_rets else 0.0
    avg_win = (sum(wins)/len(wins)-1) if wins else 0.0
    avg_loss = (sum(losses)/len(losses)-1) if losses else 0.0

    return {
        'metrics': {
            'Rendement total': total_ret,
            'CAGR (approx)': cagr,
            'Max Drawdown': maxdd,
            'Sharpe-like': float(sharpe_like),
            'Nb trades': len(trade_rets),
            'Taux de réussite': float(win_rate),
            'Gain moyen (trades gagnants)': float(avg_win),
            'Perte moyenne (trades perdants)': float(avg_loss),
        },
        'res': d.join(pd.DataFrame({'equity': eq})),
        'trades': trade_list,
        'params': {
            'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
            'sma_short': sma_short, 'sma_long': sma_long, 'bb_mult': bb_mult,
            'fee_bps': fee_bps, 'slip_bps': slip_bps,
        }
    }


# =============================
# 7) CLI principale
# =============================

def run_cli(argv: Optional[List[str]] = None) -> int:
    """Point d'entrée Console (ligne de commande)."""
    p = argparse.ArgumentParser(description="Analyse crypto : Graphique + Indicateurs + Signaux + P&L + Backtest")
    p.add_argument("--coin", required=True, help="Symbole/nom (ex: btc, bitcoin, eth, ethereum)")
    p.add_argument("--vs", default="EUR", help="Devise (EUR/USD)")
    p.add_argument("--days", default="90", help="Fenêtre: 1,7,30,90,180,365,max")
    p.add_argument("--interval", default=None, help="Optionnel: hourly/daily")
    p.add_argument("--buy", type=float, default=None, help="Votre prix d'achat (optionnel)")
    p.add_argument("--qty", type=float, default=None, help="Quantité détenue (pour P&L)")
    p.add_argument("--target", type=float, default=None, help="Objectif (optionnel)")
    p.add_argument("--stop", type=float, default=None, help="Stop (optionnel)")
    p.add_argument("--no-plot", action="store_true", help="N'affiche pas le graphique")
    # Backtest params (par défaut 365j pour éviter limitations publiques)
    p.add_argument("--bt", action="store_true", help="Exécuter un backtest (fenêtre recommandée ≤ 365j)")
    p.add_argument("--bt_days", default="365", help="Fenêtre historique pour backtest (jours ou 'max')")
    p.add_argument("--rsi_buy", type=float, default=30.0, help="Seuil RSI pour achat")
    p.add_argument("--rsi_sell", type=float, default=70.0, help="Seuil RSI pour vente")
    p.add_argument("--sma_short", type=int, default=50, help="Période SMA courte")
    p.add_argument("--sma_long", type=int, default=200, help="Période SMA longue")
    p.add_argument("--bb_mult", type=float, default=2.0, help="Multiplicateur bandes de Bollinger")
    p.add_argument("--fee_bps", type=float, default=10.0, help="Frais en points de base par trade")
    p.add_argument("--slip_bps", type=float, default=5.0, help="Slippage en points de base par exécution")

    args = p.parse_args(argv)

    try:
        coin_id = resolve_coin_id(args.coin)
    except Exception as e:
        print(f"[Erreur] Résolution du coin: {e}")
        return 2

    # Mode backtest
    if args.bt:
        try:
            pf = fetch_market_chart(coin_id, vs=args.vs, days=args.bt_days, interval=args.interval)
        except Exception as e:
            print(f"[Erreur] Récupération des données: {e}")
            return 3
        bt = backtest_long_only(
            pf.df, rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell,
            sma_short=args.sma_short, sma_long=args.sma_long, bb_mult=args.bb_mult,
            fee_bps=args.fee_bps, slip_bps=args.slip_bps,
        )
        print("===== Backtest (long-only) =====")
        for k, v in bt['metrics'].items():
            if isinstance(v, float):
                if k in ("Rendement total", "CAGR (approx)", "Max Drawdown"):
                    print(f"{k}: {v*100:.2f}%")
                else:
                    print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        # Tracé équité
        if not args.no_plot:
            plot_equity_curve(bt['res']['equity'], title=f"Équité — {args.coin} | {args.vs}")
        return 0

    # Mode analyse standard
    try:
        pf = fetch_market_chart(coin_id, vs=args.vs, days=args.days, interval=args.interval)
    except Exception as e:
        print(f"[Erreur] Récupération des données: {e}")
        return 3

    df = compute_indicators(pf.df)

    # Stats
    stats = compute_stats(df)
    print("===== Infos =====")
    print(f"Coin: {pf.coin_id} | Devise: {pf.vs} | Points: {len(df)}")
    for k, v in stats.items():
        if v is not None:
            txt = f"{v:.2f}%" if "(%)" in k else f"{v}"
            print(f"{k}: {txt}")

    # P&L
    if args.buy is not None:
        df, last_perf, last_pnl = pnl_from_buy(df, args.buy, args.qty)
        print("----- Performance vs achat -----")
        print(f"Perf actuelle: {last_perf:.2f}%")
        if last_pnl is not None:
            print(f"P&L actuel: {last_pnl:.2f} {pf.vs}")

    # Signaux + propositions
    r = df.iloc[-1]
    is_buy = buy_signal_row(r)
    is_sell = sell_signal_row(r)
    levels = propose_levels(df)
    print("===== Signaux =====")
    print(f"Signal Achat: {'Oui' if is_buy else 'Non'} | Signal Vente: {'Oui' if is_sell else 'Non'}")
    print(f"Limite: {levels['limite']:.2f} {pf.vs} | Stop: {levels['stop']:.2f} {pf.vs} | Target: {levels['target']:.2f} {pf.vs}")

    if not args.no_plot:
        title = f"{pf.coin_id} | {args.days} jours | {pf.vs}"
        plot_price(df, title, buy=args.buy, target=args.target, stop=args.stop, vs=pf.vs)

    return 0


# =============================
# 8) Interface Web Streamlit (interactif)
# =============================

# Contenu pédagogique (markdown enrichi) pour l’onglet Aide
help_md = """
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
- ATR = 1200€ → stop de 2000€ (~4%) cohérent  

👉 **Résumé** :  
- **Acheter** = prix bas + RSI bas + Bollinger bas + tendance long terme haussière  
- **Vendre** = prix haut + RSI haut + Bollinger haut + rupture de support  
- **Toujours fixer target & stop** pour gérer le risque
"""

def run_streamlit_app():
    """Lance l'interface Streamlit (formulaire + graph + bulles d'aide + backtest)."""
    import streamlit as st

    st.set_page_config(page_title="Analyse Crypto", layout="wide")
    st.title("Analyse Crypto – Indicateurs & Signaux pédagogiques")

    tabs = st.tabs(["Analyse", "Backtest (≤ 1 an)", "📘 Aide & Glossaire"])

    # === Onglet Analyse ===
    with tabs[0]:
        with st.sidebar:
            st.header("Paramètres d'analyse")
            try:
                df_pop = list_popular_coins(vs="EUR", per_page=100)
                options = [f"{row.symbol.upper()} — {row.name} ({row.id})" for _, row in df_pop.iterrows()]
            except Exception as e:
                st.warning(f"Impossible de charger la liste des cryptos populaires : {e}")
                df_pop, options = pd.DataFrame(), []

            coin_input = st.text_input(
                "Cryptomonnaie",
                value="btc",
                help="Saisissez un symbole (BTC) ou un nom (bitcoin). Vous pouvez aussi choisir ci-dessous.",
            )
            if options:
                selection = st.selectbox(
                    "Ou choisissez (classé par capitalisation)", options,
                    help="Classement par capitalisation (proxy de popularité).",
                )
                if selection:
                    coin_input = selection.split("(")[-1].rstrip(")")

            vs = st.selectbox("Devise", ["EUR", "USD"], index=0, help="Devise d'affichage.")
            # Jours personnalisés (1 à 365 pour compatibilité API publique)
            days = st.number_input("Fenêtre (jours)", min_value=1, max_value=365, value=90, step=1,
                                   help="Nombre de jours d'analyse (personnalisé).")

            st.markdown("**Courbes à afficher**")
            show_sma20 = st.checkbox("SMA20", value=True)
            show_sma50 = st.checkbox("SMA50", value=True)
            show_sma200 = st.checkbox("SMA200", value=True)
            show_bbands = st.checkbox("Bandes de Bollinger", value=True)

            buy = st.number_input("Votre prix d'achat (optionnel)", min_value=0.0, value=0.0)
            target = st.number_input("Objectif (optionnel)", min_value=0.0, value=0.0)
            stop = st.number_input("Stop (optionnel)", min_value=0.0, value=0.0)
            lancer = st.button("Analyser", key="btn_analyse")

            sidebar_opts = dict(show_sma20=show_sma20, show_sma50=show_sma50,
                                show_sma200=show_sma200, show_bbands=show_bbands)

        if lancer:
            try:
                coin_id = resolve_coin_id(coin_input)
                pf = fetch_market_chart(coin_id, vs=vs, days=str(int(days)))
                df = compute_indicators(pf.df)
                stats = compute_stats(df)
                r = df.iloc[-1]
                is_buy = buy_signal_row(r)
                is_sell = sell_signal_row(r)
                levels = propose_levels(df)

                # Cartes
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dernier prix", f"{stats['Dernier prix']:.2f} {pf.vs}")
                if stats.get("Variation 24h (%)") is not None:
                    c2.metric("Var. 24h", f"{stats['Variation 24h (%)']:.2f}%")
                if stats.get("Volatilité estimée (%)") is not None:
                    c3.metric("Volatilité (approx)", f"{stats['Volatilité estimée (%)']:.2f}%")
                c4.metric("Pire baisse (DD)", f"{stats['Pire baisse (drawdown) (%)']:.2f}%")

                st.subheader("Signaux & niveaux proposés")
                st.write({
                    "Signal Achat": "Oui" if is_buy else "Non",
                    "Signal Vente": "Oui" if is_sell else "Non",
                    "Prix limite conseillé": f"{levels['limite']:.2f} {pf.vs}",
                    "Stop": f"{levels['stop']:.2f} {pf.vs}",
                    "Target": f"{levels['target']:.2f} {pf.vs}",
                })

                # Graphique interactif
                fig = make_plotly_figure(
                    df,
                    title=f"{pf.coin_id} | {int(days)} jours | {pf.vs}",
                    vs=pf.vs,
                    buy=(buy or None), target=(target or None), stop=(stop or None),
                    **sidebar_opts,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur : {e}")

    # === Onglet Backtest (≤ 1 an pour API publique) ===
    with tabs[1]:
        st.subheader("Backtest long-only (≤ 1 an)")
        colA, colB, colC = st.columns(3)
        with colA:
            coin_bt = st.text_input("Crypto (id/symbole)", value="btc",
                                    help="Symbole ou nom. Ex: btc, bitcoin.")
            vs_bt = st.selectbox("Devise", ["EUR", "USD"], index=0)
            bt_days = st.selectbox("Fenêtre historique", ["180", "365"], index=1,
                                   help="Limité à 1 an sur API publique.")
        with colB:
            rsi_buy = st.slider("RSI achat", min_value=10, max_value=40, value=30, step=1)
            rsi_sell = st.slider("RSI vente", min_value=60, max_value=90, value=70, step=1)
            bb_mult = st.slider("Bollinger σ", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        with colC:
            sma_short = st.selectbox("SMA courte", [20, 50, 100], index=1)
            sma_long = st.selectbox("SMA longue", [100, 150, 200], index=2)
            fee_bps = st.number_input("Frais (bps)", min_value=0.0, value=10.0, help="1 bps = 0,01%")
            slip_bps = st.number_input("Slippage (bps)", min_value=0.0, value=5.0, help="Glissement à l'exécution")

        if st.button("Lancer le backtest", key="btn_bt"):
            try:
                coin_id = resolve_coin_id(coin_bt)
                pf_bt = fetch_market_chart(coin_id, vs=vs_bt, days=bt_days)
                bt = backtest_long_only(
                    pf_bt.df,
                    rsi_buy=float(rsi_buy), rsi_sell=float(rsi_sell),
                    sma_short=int(sma_short), sma_long=int(sma_long), bb_mult=float(bb_mult),
                    fee_bps=float(fee_bps), slip_bps=float(slip_bps),
                )
                m = bt['metrics']
                st.success("Backtest terminé")
                st.write({
                    "Rendement total": f"{m['Rendement total']*100:.2f}%",
                    "CAGR (approx)": f"{m['CAGR (approx)']*100:.2f}%",
                    "Max Drawdown": f"{m['Max Drawdown']*100:.2f}%",
                    "Sharpe-like": f"{m['Sharpe-like']:.2f}",
                    "Nb trades": int(m['Nb trades']),
                    "Taux de réussite": f"{m['Taux de réussite']*100:.1f}%",
                    "Gain moyen (trades gagnants)": f"{m['Gain moyen (trades gagnants)']*100:.2f}%",
                    "Perte moyenne (trades perdants)": f"{m['Perte moyenne (trades perdants)']*100:.2f}%",
                })
                # Courbe d'équité
                fig2 = plot_equity_curve(bt['res']['equity'], title=f"Équité — {coin_bt} | {vs_bt}", show=False)
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Erreur backtest : {e}")

    # === Onglet Aide & Glossaire ===
    with tabs[2]:
        st.markdown(help_md, unsafe_allow_html=True)


# =============================
# 9) Point d'entrée (CLI ou Streamlit)
# =============================
if __name__ == "__main__":
    import sys
    # S'il y a des arguments -> on lance la CLI.
    if len(sys.argv) > 1:
        raise SystemExit(run_cli(sys.argv[1:]))
    # Sinon, on tente de lancer l'interface Streamlit.
    else:
        try:
            run_streamlit_app()
        except ModuleNotFoundError:
            print("Streamlit n'est pas installé.")
            print("Installez-le avec: pip install streamlit")
            print("Ou lancez en mode console, ex.:")
            print("python crypto_analyser.py --coin btc --days 90 --vs EUR")
