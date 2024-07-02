"""
File containing functions reused in a multiple times in different Python files and notebooks
"""

import datetime
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates


def get_yfinance_data(ticker, start_date, end_date):
    """
    Retrieve historical stock data from Yahoo Finance.

    This function uses the yfinance library to download historical stock data for a
    specified ticker and date range.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical stock data for the specified
                      ticker and date range. The DataFrame includes columns such as
                      'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def get_key_by_value(d, value):
    """
    Retrieve the key associated with a given value in a dictionary.

    This function searches through a dictionary and returns the key that corresponds
    to the specified value. It is useful for looking up keys based on values in JSON
    files or other dictionaries.

    Parameters:
    d (dict): The dictionary to search through.
    value (any): The value to search for.

    Returns:
    any: The key associated with the specified value if found, otherwise None.
    """
    for key, val in d.items():
        if val == value:
            return key
    return None


def prepare_data_box_plot(data, start_date, end_date):
    """
    Prepare data for a box plot by adding 'Month' and 'Year' columns and filtering the date range.

    This function adds 'Month' and 'Year' columns to the data based on the index, which is assumed
    to be a date. It then calculates the number of business days between the start and end dates
    and filters the data to include only the most recent business days within this range.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing historical stock data.
    start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: The modified DataFrame with 'Month' and 'Year' columns added and filtered
                      to include only the most recent business days within the specified date range.
    """
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    days = (
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
        - datetime.datetime.strptime(start_date, "%Y-%m-%d")
    ).days
    days_ses = int(days / 7 * 5)
    data = data[-days_ses:]

    return data


def prepare_data_decomp_trend(data):
    """
    Decompose the trend component of the stock closing price data and perform linear regression.

    This function performs seasonal decomposition of the closing price data to extract the trend
    component. It then fits a linear regression model to the trend data and makes predictions.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing historical stock data with a 'Close'
    column.

    Returns:
    tuple: A tuple containing:
        - result (statsmodels.tsa.seasonal.DecomposeResult): The result of the seasonal
        decomposition.
        - data_decomp (pandas.DataFrame): A DataFrame containing the decomposed trend data with date
        and date number.
        - y_pred (numpy.ndarray): The predicted trend values from the linear regression model.
    """
    result = seasonal_decompose(data.Close[:], model="None", period=360)
    result_trend = result.trend.dropna()
    data_decomp = result_trend.reset_index()
    data_decomp["Date_num"] = mdates.date2num(data_decomp.Date)
    model = LinearRegression()
    x = data_decomp[["Date_num"]]
    y = data_decomp["trend"]
    model.fit(x, y)
    y_pred = model.predict(x)

    return result, data_decomp, y_pred
