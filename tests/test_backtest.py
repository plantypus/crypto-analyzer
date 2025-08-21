import importlib
import sys
import types

import pandas as pd
import pytest

import indicators

# Create a dummy 'core.indicators' module so that backtest can import it
core_module = types.ModuleType("core")
core_module.indicators = indicators
sys.modules["core"] = core_module
sys.modules["core.indicators"] = indicators

import backtest


def test_backtest_long_only(monkeypatch):
    df = pd.DataFrame({"close": [10, 11, 12]})

    def fake_sma(series, window):
        return pd.Series([2, 2, 2], index=series.index)

    def fake_rsi(series):
        return pd.Series([20, 80, 50], index=series.index)

    monkeypatch.setattr(backtest, "sma", fake_sma)
    monkeypatch.setattr(backtest, "rsi", fake_rsi)

    result = backtest.backtest_long_only(df, rsi_buy=30, rsi_sell=70)
    assert result["equity"] == pytest.approx(1100)
    assert result["return"] == pytest.approx(0.1)
