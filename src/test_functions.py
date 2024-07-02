"""
Testing functions from functions.py file
"""

import pytest
import pandas as pd
from functions import (
    get_yfinance_data,
    get_key_by_value,
    prepare_data_box_plot,
    prepare_data_decomp_trend,
)


def test_get_yfinance_data():
    """
    Test function for the get_yfinance_data function.

    This function tests the behavior of get_yfinance_data by verifying:
    - It returns a Pandas DataFrame.
    - The returned DataFrame has a 'Close' column.
    - The length of the returned DataFrame is greater than 0.

    Example usage:
        Run this test function to ensure get_yfinance_data behaves as expected.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    ticker = "ES=F"
    s_date = "2020-01-01"
    e_date = "2021-01-01"

    yf_data = get_yfinance_data(ticker, s_date, e_date)

    assert isinstance(yf_data, pd.DataFrame)
    assert "Close" in yf_data.columns
    assert len(yf_data) > 0


@pytest.mark.parametrize("value, expected_key", [(2, "b"), (4, None)])
def test_get_key_by_value(example_dict, value, expected_key):
    """
    Test function for the get_key_by_value function.

    This function tests the behavior of get_key_by_value by verifying:
    - It correctly retrieves the key associated with the given value from example_dict.

    Args:
        example_dict (dict): Example dictionary used for testing.
        value: Value to search for in example_dict.
        expected_key: Expected key associated with the value in example_dict.

    Example usage:
        Run this test function to ensure get_key_by_value behaves as expected.

    Raises:
        AssertionError: If the retrieved key does not match the expected_key.
    """
    key = get_key_by_value(example_dict, value)
    assert key == expected_key


def test_prepare_data_box_plot(example_data):
    """
    Test function for the prepare_data_box_plot function.

    This function tests the behavior of prepare_data_box_plot by verifying:
    - It adds 'Month' and 'Year' columns to the DataFrame.
    - The length of the prepared DataFrame is greater than 0.

    Args:
        example_data (pd.DataFrame): Example DataFrame used for testing.

    Example usage:
        Run this test function to ensure prepare_data_box_plot behaves as expected.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    start_date = "2022-01-01"
    end_date = "2024-01-01"

    prepared_data = prepare_data_box_plot(example_data, start_date, end_date)

    assert "Month" in prepared_data.columns
    assert "Year" in prepared_data.columns
    assert len(prepared_data) > 0


def test_prepare_data_decomp_trend(example_data):
    """
    Test function for the prepare_data_decomp_trend function.

    This function tests the behavior of prepare_data_decomp_trend by verifying:
    - It adds a 'trend' column to the 'data_decomp' DataFrame.
    - The length of 'y_pred' matches the length of 'data_decomp'.

    Args:
        example_data (pd.DataFrame): Example DataFrame used for testing.

    Example usage:
        Run this test function to ensure prepare_data_decomp_trend behaves as expected.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    _, data_decomp, y_pred = prepare_data_decomp_trend(example_data)

    assert "trend" in data_decomp.columns
    assert len(y_pred) == len(data_decomp)
