"""
File containing functions reused in a multiple times in different Python files and notebooks
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime

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

