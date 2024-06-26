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


# In[10]:


# wybór danych z kolumny 'Close'
series = data.Close.to_numpy()
#print(series)


# In[11]:


# Tworzenie wykresu z danych
plt.plot(data.index, data["Close"], linestyle='-', color='darkblue')  # Wykres linii
plt.xlabel('Czas')
plt.ylabel('Wartość [USD]')
plt.title(f'Wartość zamknięcia dla {tickers[ticker]}')
plt.grid(color='lightgray')
plt.tight_layout()
plt.show()


# In[12]:


# def. wykres
def plot_series(time, series_scaled, format="-", start = 0, end = None):
    plt.plot(time[start:end], series_scaled[start:end], format)
    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.grid(True)


# In[13]:


# przygotowanie danych do wykresu
data_1 = data
data_1['Month'] = data_1.index.month
data_1['Year'] = data_1.index.year

days = 1825 # ostatnie dni
days_ses = int(days/7*5) # dni sesyjne
data_3 = data_1[-days_ses:]


# In[14]:


# Tworzenie wykresu pudełkowego
fig, ax1 = plt.subplots(ncols=1, figsize=(12, 8))
sns.boxplot(data=data_3, x='Month', y='Close', ax=ax1, hue='Month', palette='muted',width=7)
plt.ylabel('Wartość zamknięcia [USD]')
plt.xlabel('Miesiące')
plt.title(f'Wykres pudełkowy dla {tickers[ticker]} z podziałem na miesiące, ostatnie {days} dni')  # Tytuł wykresu
plt.grid(color='lightgray')
ax1.get_legend().remove()
plt.tight_layout()
plt.show()


# In[15]:


# przygotowanie danych do dekompozycji + regresja
result = seasonal_decompose(data.Close[:], model='None', period=720)
result_trend = result.trend.dropna()
data_decomp = result_trend.reset_index()
data_decomp['Date_num'] = mdates.date2num(data_decomp.Date)
model = LinearRegression()
X = data_decomp[['Date_num']]
y = data_decomp['trend']
model.fit(X, y)
y_pred = model.predict(X)


# In[16]:


# przygotowanie danych do dekompozycji + regresja
result_observed = result.observed.dropna()
data_obs = result_observed.reset_index()
data_obs['Date_num'] = mdates.date2num(data_obs.Date)
model = LinearRegression()
X1 = data_obs[['Date_num']]
y1 = data_obs['Close']
model.fit(X1, y1)
y_pred_1 = model.predict(X1)


# In[17]:


#wykres dekompozycji + regresja
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1 ]})
ax0.set_title(f'Wykres dla {tickers[ticker]}')
ax0.plot(result.observed)
ax0.plot(data_obs['Date'], y_pred_1, color='red')
ax1.set_title('Trend')
ax1.plot(result.trend, color='purple')
plt.grid(color='gray')
plt.show()

