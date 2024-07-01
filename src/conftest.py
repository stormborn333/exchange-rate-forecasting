import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def example_data():
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    df = pd.DataFrame(index=dates)
    df["Close"] = np.random.rand(len(dates))
    df.index.name = "Date"
    return df

@pytest.fixture
def example_dict():
    return {"a": 1, "b": 2, "c": 3}