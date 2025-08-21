"""
Module levels.py
----------------
Propose des niveaux de stop, target et limite en fonction du prix et de la volatilité.
"""

def propose_levels(price: float, atr: float) -> dict:
    """
    Calcule niveaux de trading.
    
    Target = prix + 2 * ATR
    Stop   = prix - 1 * ATR
    Limite = prix - 0.5 * ATR
    """
    return {
        "target": price + 2 * atr,
        "stop": price - 1 * atr,
        "limite": price - 0.5 * atr,
    }
