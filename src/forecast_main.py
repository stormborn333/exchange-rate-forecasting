"""
File containing main function responsible for making forecast based on input parameters provided by the user in the Dash application
"""

# import bibliotek
import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from functions import get_yfinance_data, get_key_by_value
import json

# ustawienie ziarna i czyszczenie
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

EPOCHS = 50
WINDOW_SIZE = 25
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 300
    
def make_forecast(start_date, end_date, ticker_name, n_steps):

    #tickers data
    with open("data-utils/data-raw/tickers.json", 'r') as f:
        data_tickers = json.load(f)
    ticker = get_key_by_value(data_tickers, ticker_name)

    #yahoo finance data
    data = get_yfinance_data(ticker, start_date, end_date)
    series = data.Close.to_numpy() #y of the model

    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    # Przygotowanie danych do uczenia
    time = np.arange(len(series_scaled), dtype="float32")
    split_time = int(len(series_scaled) * 0.8)
    x_train = series_scaled[:split_time]
    x_valid = series_scaled[split_time:]
    time_valid = time[split_time:]

    dataset = windowed_dataset(x_train, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)
    valid_dataset = windowed_dataset(x_valid, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)
    
    # Definicja modelu
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(100, input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(100),
        tf.keras.layers.Dense(1),
    ])
    
    # zatrzymanie przy braku postępów
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
#     # kompilowanie i trenowanie modelu
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.7)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
    history = model.fit(dataset, epochs=EPOCHS , verbose=1, validation_data=valid_dataset, callbacks=[stop_early])
    
    # przewidywanie wartości na podstawie modelu wytrenowanego
    forecast = model_forecast(model, series_scaled[split_time - WINDOW_SIZE: -1], WINDOW_SIZE)[:, 0]

    # odwrócenie skalowania danych
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
    x_valid = scaler.inverse_transform(x_valid.reshape(-1, 1))
    
    # obliczenie błędu
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mae_metric.update_state(x_valid, forecast)
    mae = mae_metric.result().numpy()
    mean_value = np.mean(x_valid)
    percentage_error = np.abs(mae / mean_value) * 100
    percentage_error_rounded = round(percentage_error, 2)
    
    # wyznaczenie predykcji na kolejne dni, a następnie odwrócenie skalowania danych
    future_forecast = []
    last_window = series_scaled[-WINDOW_SIZE:]
    
    for _ in range(n_steps):
        prediction = model.predict(last_window[np.newaxis])
        future_forecast.append(prediction[0, 0])
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction[0, 0]
    
    future_forecast = np.array(future_forecast)
    future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))
    
    # Zwrócenie błądu i predykcji
    return percentage_error_rounded, future_forecast, forecast.flatten(), x_valid.flatten(), time_valid
   
def windowed_dataset(series_scaled, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE):
        series_scaled = tf.expand_dims(series_scaled, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices(series_scaled)
        dataset = dataset.window(WINDOW_SIZE + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(WINDOW_SIZE + 1))
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(BATCH_SIZE).prefetch(1)
        return dataset

def model_forecast(model, series_scaled, WINDOW_SIZE):
    series = tf.expand_dims(series_scaled, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series_scaled)
    ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
    
