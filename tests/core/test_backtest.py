import pandas as pd
import pytest

from crypto_analyzer.core import backtest


def test_backtest_long_only(monkeypatch):
    df = pd.DataFrame({"close": [10, 11, 12]})

    def fake_sma(series, window):
        if window == 20:
            return pd.Series([3, 3, 3], index=series.index)
        return pd.Series([2, 2, 2], index=series.index)

    def fake_rsi(series):
        return pd.Series([20, 80, 50], index=series.index)

    monkeypatch.setattr(backtest, "sma", fake_sma)
    monkeypatch.setattr(backtest, "rsi", fake_rsi)

    result = backtest.backtest_long_only(df, rsi_buy=30, rsi_sell=70)
    assert result["equity"] == pytest.approx(1100)
    assert result["return"] == pytest.approx(0.1)
