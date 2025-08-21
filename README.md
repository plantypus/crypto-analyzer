# Crypto Analyser

Outil pédagogique et interactif pour analyser les cryptomonnaies.  
Permet de :
- Récupérer les prix via l’API publique de **CoinGecko**.
- Visualiser des **graphes interactifs** avec indicateurs techniques.
- Générer des **signaux simples d’achat/vente**.
- Proposer des niveaux d’**ordre limite, stop-loss et target**.
- Faire un **backtest simple** (1 an) pour valider les paramètres.
- Fonctionner en **console (CLI)** ou en **interface web Streamlit**.

---

## 🚀 Fonctionnalités
- **Indicateurs intégrés :**
  - SMA20, SMA50, SMA200
  - Bandes de Bollinger (20, 2σ)
  - RSI (14)
  - ATR (14)
- **Stats clés :**
  - Dernier prix
  - Variation 24h
  - Volatilité estimée
  - Drawdown max
- **Signaux pédagogiques :**
  - Achat si *SMA50 > SMA200* **et** *RSI < 30* **et** *Prix < Bollinger basse*
  - Vente si *RSI > 70* **ou** *Prix > Bollinger haute*
- **Graphique interactif :**
  - Valeur affichée au survol
  - % de variation depuis le départ
  - Format date adapté (jour/heure)
  - Cases à cocher pour afficher/masquer indicateurs
- **Glossaire intégré** : explication de chaque indicateur et termes de trading.

---

## 📦 Installation

Prérequis Python 3.9+ :
requests
pandas
matplotlib
python-dateutil
streamlit
plotly

```bash
pip install -r requirements.txt


