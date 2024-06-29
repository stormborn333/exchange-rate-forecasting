"""
File containing functions reused in a multiple times in different Python files and notebooks
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

def get_yfinance_data(ticker, start_date, end_date):

    data = yf.download(ticker, start = start_date, end = end_date)
    return data

def get_key_by_value(d, value):

    for key, val in d.items():
        if val == value:
            return key
    return None

def prepare_data_box_plot(data, start_date, end_date):

    data['Month'] = data.index.month
    data['Year'] = data.index.year
    days = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
    days_ses = int(days / 7 * 5)
    data = data[-days_ses:]

    return data

def prepare_data_decomp_trend(data):

    result = seasonal_decompose(data.Close[:], model='None', period=360)
    result_trend = result.trend.dropna()
    data_decomp = result_trend.reset_index()
    data_decomp['Date_num'] = mdates.date2num(data_decomp.Date)
    model = LinearRegression()
    X = data_decomp[['Date_num']]
    y = data_decomp['trend']
    model.fit(X, y)
    y_pred = model.predict(X)

    return result, data_decomp, y_pred

