"""
File containing functions reused in a multiple times in different Python files and notebooks
"""

import pandas as pd
import numpy as np
import yfinance as yf

def get_yfinance_data(ticker, start_date, end_date):

    data = yf.download(ticker, start = start_date, end = end_date)
    return data

def get_key_by_value(d, value):

    for key, val in d.items():
        if val == value:
            return key
    return None
    

