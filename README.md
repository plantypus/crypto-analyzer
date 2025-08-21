# Crypto Analyzer

Outil d'analyse technique pour cryptomonnaies.

## Installation

1. Cloner le dépôt et se placer dedans.
2. Créer un environnement virtuel (optionnel mais conseillé) :

   ```bash
   python -m venv venv
   source venv/bin/activate  # sous Windows : venv\Scripts\activate
   ```

3. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Console (CLI)

```bash
python crypto_analyzer.py cli --coin bitcoin --days 90 --vs eur
```

### Interface Web (Streamlit)

```bash
python crypto_analyzer.py web --port 8501
# ou
streamlit run web.py
```

### API (FastAPI)

```bash
python crypto_analyzer.py api --port 8000
```

## Structure du projet

- `data.py` : récupération des données de marché via CoinGecko.
- `indicators.py` : calcul des indicateurs techniques (SMA, Bollinger, RSI, ATR).
- `signals.py` : règles simples de signaux d'achat/vente.
- `levels.py` : proposition de niveaux de trading (target, stop, limite).
- `backtest.py` : backtest basique d'une stratégie long-only.
- `console.py` : interface en ligne de commande.
- `web.py` : interface utilisateur Streamlit.
- `crypto_analyzer.py` : point d'entrée pour lancer CLI, interface web ou API.
## Stratégie Git

- `main` : branche stable contenant les versions prêtes pour la production.
- `develop` : branche d'intégration où les fonctionnalités sont assemblées avant d'être fusionnées dans `main`.
- `feature/*` : branches dérivées de `develop` pour développer de nouvelles fonctionnalités ou corrections.
