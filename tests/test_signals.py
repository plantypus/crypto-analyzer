import pytest

from signals import buy_signal, sell_signal


@pytest.mark.parametrize(
    "price,sma200,rsi,bb_low,expected",
    [
        (105, 100, 20, 110, True),
        (95, 100, 20, 110, False),
        (105, 100, 35, 110, False),
        (105, 100, 20, 100, False),
    ],
)
def test_buy_signal(price, sma200, rsi, bb_low, expected):
    assert buy_signal(price, sma200, rsi, bb_low) == expected


@pytest.mark.parametrize(
    "price,rsi,bb_high,expected",
    [
        (110, 80, 120, True),  # RSI > 70
        (110, 60, 100, True),  # Price >= Bollinger high
        (110, 60, 120, False),
    ],
)
def test_sell_signal(price, rsi, bb_high, expected):
    assert sell_signal(price, rsi, bb_high) == expected
