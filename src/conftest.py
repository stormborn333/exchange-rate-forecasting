"""
File for generating data for unit tests using pytest
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def example_data():
    """
    Fixture that generates example DataFrame with random 'Close' prices.

    This fixture creates a Pandas DataFrame with a date range from January 1, 2020,
    to January 1, 2024, with daily frequency. The 'Close' column of the DataFrame
    contains randomly generated prices.

    Returns:
        pd.DataFrame: DataFrame with index named 'Date' and 'Close' prices.

    Example usage:
        df = example_data()
    """
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    df = pd.DataFrame(index=dates)
    df["Close"] = np.random.rand(len(dates))
    df.index.name = "Date"
    return df


@pytest.fixture
def example_dict():
    """
    Fixture that provides an example dictionary for testing purposes.

    Returns:
        dict: A dictionary {'a': 1, 'b': 2, 'c': 3}.

    Example usage:
        data = example_dict()
        assert 'a' in data
        assert data['b'] == 2
    """
    return {"a": 1, "b": 2, "c": 3}
