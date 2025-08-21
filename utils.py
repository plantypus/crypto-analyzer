"""
Module utils.py
---------------
Fonctions utilitaires : formatage, logs, etc.
"""

def format_price(price: float, vs: str = "EUR") -> str:
    """Formatage prix avec devise."""
    return f"{price:,.2f} {vs.upper()}"

def pct_change(old: float, new: float) -> float:
    """Variation en % entre deux valeurs."""
    return ((new - old) / old) * 100
