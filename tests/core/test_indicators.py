import numpy as np
import pandas as pd

from crypto_analyzer.core import indicators


def test_sma_basic():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = indicators.sma(series, window=3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected)
