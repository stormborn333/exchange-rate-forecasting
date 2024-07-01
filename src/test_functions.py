import pytest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from functions import (
    get_yfinance_data,
    get_key_by_value,
    prepare_data_box_plot,
    prepare_data_decomp_trend,
)

def test_get_yfinance_data():
    ticker = "ES=F"
    s_date = "2020-01-01"
    e_date = "2021-01-01"

    yf_data = get_yfinance_data(ticker, s_date, e_date)

    assert isinstance(yf_data, pd.DataFrame)
    assert "Close" in yf_data.columns
    assert len(yf_data) > 0

@pytest.mark.parametrize("value, expected_key", [(2, "b"), (4, None)])
def test_get_key_by_value(example_dict, value, expected_key):
    key = get_key_by_value(example_dict, value)
    assert key == expected_key

def test_prepare_data_box_plot(example_data):
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    prepared_data = prepare_data_box_plot(example_data, start_date, end_date)
    
    assert "Month" in prepared_data.columns
    assert "Year" in prepared_data.columns
    assert len(prepared_data) > 0

def test_prepare_data_decomp_trend(example_data):
    result, data_decomp, y_pred = prepare_data_decomp_trend(example_data)
    
    assert "trend" in data_decomp.columns
    assert len(y_pred) == len(data_decomp)