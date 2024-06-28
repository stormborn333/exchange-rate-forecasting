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
from plik_tickers import tickers  # Załóżmy, że to plik zawiera słownik tickers
import json

# ustawienie ziarna i czyszczenie
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

EPOCHS = 50
WINDOW_SIZE = 25
WATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 300
    
def analyze_ticker(start_date, end_date, ticker_name, n_steps):

    #tickers data
    with open("data-utils/data-raw/tickers.json", 'r') as f:
        data_tickers = json.load(f)
    ticker = get_key_by_value(data, ticker_name)

    #yahoo finance data
    data = yf.download(ticker, start=start, end=end)
    series = data.Close.to_numpy() #y of the model
    
#     # przygotowanie danych do wykresu
#     data_1 = data.copy()
#     data_1['Month'] = data_1.index.month
#     data_1['Year'] = data_1.index.year
#     days = (datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days
#     days_ses = int(days / 7 * 5)
#     data_2 = data_1[-days_ses:]
    
#     # Tworzenie wykresu pudełkowego
#     fig, ax1 = plt.subplots(ncols=1, figsize=(12, 8))
#     sns.boxplot(data=data_2, x='Month', y='Close', ax=ax1, hue='Month', palette='muted',width=7)
#     plt.ylabel('Wartość zamknięcia [USD]')
#     plt.xlabel('Miesiące')
#     plt.title(f'Wykres pudełkowy dla {tickers[ticker]}, ostatnie {days} dni')  # Tytuł wykresu
#     plt.grid(color='lightgray')
#     ax1.get_legend().remove()
#     plt.tight_layout()
#     plt.savefig("pudelkowy.jpg")
#     plt.close()
    
#     # Przygotowanie danych do dekompozycji + regresja
#     result = seasonal_decompose(data.Close[:], model='None', period=360)
#     result_trend = result.trend.dropna()
#     data_decomp = result_trend.reset_index()
#     data_decomp['Date_num'] = mdates.date2num(data_decomp.Date)
#     model = LinearRegression()
#     X = data_decomp[['Date_num']]
#     y = data_decomp['trend']
#     model.fit(X, y)
#     y_pred = model.predict(X)
    
#     # wykres dekompozycji + regresja
#     fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
#     ax0.set_title(f'Wykres dla {tickers[ticker]}')
#     ax0.plot(result.observed)
#     ax0.plot(data_decomp['Date'], y_pred, color='red', label='Regresja liniowa')
#     ax1.set_title('Trend')
#     ax1.plot(result.trend, color='purple')
#     ax0.set_ylabel('Wartość [USD]')
#     ax1.set_ylabel('Wartość [USD]')
#     ax0.set_xlabel('Lata')
#     ax1.set_xlabel('Lata')
#     plt.grid(color='gray')
#     ax0.legend()
#     plt.savefig("dekompozycja.jpg")
#     plt.close()
    
#     # skalowanie danych
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
#     # Przygotowanie danych do uczenia
#     time = np.arange(len(series_scaled), dtype="float32")
#     split_time = int(len(series_scaled) * 0.8)
#     x_train = series_scaled[:split_time]
#     x_valid = series_scaled[split_time:]
#     time_valid = time[split_time:]
    
#     # ustawienie parametrów uczenia
#     EPOCHS = 50
#     WINDOW_SIZE = 25
#     BATCH_SIZE = 64
#     SHUFFLE_BUFFER_SIZE = 300
    
#     # Funkcja do tworzenia zbioru danych
#     def windowed_dataset(series_scaled, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE):
#         series_scaled = tf.expand_dims(series_scaled, axis=-1)
#         dataset = tf.data.Dataset.from_tensor_slices(series_scaled)
#         dataset = dataset.window(WINDOW_SIZE + 1, shift=1, drop_remainder=True)
#         dataset = dataset.flat_map(lambda window: window.batch(WINDOW_SIZE + 1))
#         dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).map(lambda window: (window[:-1], window[-1]))
#         dataset = dataset.batch(BATCH_SIZE).prefetch(1)
#         return dataset
    
#     dataset = windowed_dataset(x_train, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)
#     valid_dataset = windowed_dataset(x_valid, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)
    
#     # Definicja modelu
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.SimpleRNN(100, input_shape=[None, 1], return_sequences=True),
#         tf.keras.layers.SimpleRNN(100),
#         tf.keras.layers.Dense(1),
#     ])
    
#     # zatrzymanie przy braku postępów
#     stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
#     # kompilowanie i trenowanie modelu
#     optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.7)
#     model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
#     history = model.fit(dataset, epochs=EPOCHS , verbose=1, validation_data=valid_dataset, callbacks=[stop_early])
    
#     # funkcja przewidywania wartości na podstawie modelu wytrenowanego
#     def model_forecast(model, series_scaled, WINDOW_SIZE):
#         series = tf.expand_dims(series_scaled, axis=-1)
#         ds = tf.data.Dataset.from_tensor_slices(series_scaled)
#         ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
#         ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
#         ds = ds.batch(32).prefetch(1)
#         forecast = model.predict(ds)
#         return forecast
    
#     # przewidywanie wartości na podstawie modelu wytrenowanego
#     forecast = model_forecast(model, series_scaled[split_time - WINDOW_SIZE: -1], WINDOW_SIZE)[:, 0]
    
#     # odwrócenie skalowania danych
#     forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
#     x_valid = scaler.inverse_transform(x_valid.reshape(-1, 1))
    
#     # obliczenie błędu
#     mae_metric = tf.keras.metrics.MeanAbsoluteError()
#     mae_metric.update_state(x_valid, forecast)
#     mae = mae_metric.result().numpy()
#     mean_value = np.mean(x_valid)
#     percentage_error = np.abs(mae / mean_value) * 100
#     percentage_error_rounded = round(percentage_error, 2)
    
#     # wyznaczenie predykcji na kolejne dni, a następnie odwrócenie skalowania danych
#     future_forecast = []
#     last_window = series_scaled[-WINDOW_SIZE:]
    
#     for _ in range(num_days):
#         prediction = model.predict(last_window[np.newaxis])
#         future_forecast.append(prediction[0, 0])
#         last_window = np.roll(last_window, -1)
#         last_window[-1] = prediction[0, 0]
    
#     future_forecast = np.array(future_forecast)
#     future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))
    
#     # wykres predykcji
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_valid[-200:], x_valid[-200:], label='dane treningowe')
#     plt.plot(time_valid[-200:], forecast[-200:], label='dane walidacyjne')
#     time_future = np.arange(len(series), len(series) + num_days)
#     plt.title(f'Wartość zamknięcia dla {tickers[ticker]}')
#     plt.ylabel('Wartość [USD]')
#     plt.xlabel('Dni')
#     plt.plot(time_future, future_forecast, label='prognoza')
#     plt.legend()
#     plt.savefig("prognoza.jpg")
#     plt.close()

#     # Zwrócenie błądu i predykcji
#     return percentage_error_rounded, future_forecast
   
# # ------------------------------------------------------------------------------------ 

def get_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None



# # Przykładowe dane - test
# start_date = "2014-01-01" # minimalnie 10 lat aby to miało sens, max od 2008
# end_date = "2024-01-01"
# selected_ticker = "BZ=F"
# days_to_forecast = 10

# error_percentage, predicted_values = analyze_ticker(start_date, end_date, selected_ticker, days_to_forecast)

# print(f"\nBłąd procentowy: {error_percentage}% \n")
# print(f"Wartości predykcji:\n{predicted_values}\n")

# # Wyświetlenie wygenerowanych plików JPG
# from IPython.display import Image, display

# filenames = ["zamkniecie.jpg", "pudelkowy.jpg", "dekompozycja.jpg", "prognoza.jpg"]

# for filename in filenames:
#     display(Image(filename=filename))