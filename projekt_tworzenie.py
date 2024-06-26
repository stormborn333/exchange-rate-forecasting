#!/usr/bin/env python
# coding: utf-8

# In[1]:


# System wspomagania decyzji inwestorskich
# Paweł Kowalski


# In[2]:


# sprawdzenieczasu działania programu
import time as tm
start_time = tm.time()


# In[3]:


# import bibliotek
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib 
import sklearn  
import statsmodels 
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from plik_tickers import tickers 


# In[4]:


# Sprawdzanie wersji Phytona
import sys 
print(f"Python: {sys.version}")


# In[5]:


# Sprawdzanie wersji bibliotek
print(f"yfinance: {yf.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"tensorflow: {tf.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"statsmodels: {statsmodels.__version__}")


# In[6]:


# ustawienie ziarna i czyszczenie
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


# In[7]:


# wybór tickers'a , tu musi być np. lista rozwijana
for symbol, name in tickers.items():
    print(f"Symbol: {symbol} - Name: {name}")


# In[8]:


# pobranie danych ze strony Yahoo (po testach wpisać np. rok  2000 !
start = "2004-01-01"
end = date.today().strftime("%Y-%m-%d")
ticker = "BZ=F" # możliwość wyboru np 10
data = yf.download(ticker, start=start, end=end)
print(data.shape)


# In[9]:


# Kontrola jakości danych
print(data.info())
print(data.describe())
print('head', data.head(5))
print('tail', data.tail(5))
print(type(data))