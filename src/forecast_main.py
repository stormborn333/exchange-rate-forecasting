"""
File containing main function responsible for making forecast
 based on input parameters provided by the user in the Dash application
"""

# library import
import json
import numpy as np
from tensorflow import (  # pylint: disable=no-name-in-module)
    keras,
    random,
    expand_dims,
    data,
)
from sklearn.preprocessing import MinMaxScaler
from functions import get_yfinance_data, get_key_by_value

# grain setting and cleaning
keras.backend.clear_session()
random.set_seed(42)
np.random.seed(42)

# declared necessary variables
EPOCHS = 50
WINDOW_SIZE = 25
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 300


# function that trains the model and prediction values
# pylint: disable=too-many-locals
def make_forecast(start_date: str, end_date: str, ticker_name: str, n_steps: int):
    """
    Generates a forecast for a given financial ticker over a
    specified date range using a SimpleRNN model.

    Parameters:
    start_date(str): The start date for the historical data fetch in 'YYYY-MM-DD' format.
    end_date(str): The end date for the historical data fetch in 'YYYY-MM-DD' format.
    ticker_name(str): The name of the ticker for which the forecast is to be made.
    n_steps(int): The number of future steps to forecast beyond the end date.

    Returns:
    tuple
        A tuple containing the following elements:
        - percentage_error_rounded (float): The rounded percentage error of the forecast.
        - future_forecast (np.ndarray): The forecasted values for the future steps.
        - forecast (np.ndarray): The forecasted values on the validation set.
        - x_valid (np.ndarray): The actual validation data.
        - time_valid (np.ndarray): The time steps corresponding to the validation data.
    """

    # tickers data
    with open("data-utils/data-raw/tickers.json", "r", encoding="utf-8") as f:
        data_tickers = json.load(f)
    ticker = get_key_by_value(data_tickers, ticker_name)

    # yahoo finance data
    data_yf = get_yfinance_data(ticker, start_date, end_date)
    series = data_yf.Close.to_numpy()  # y of the model
    # data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Preparing data for training
    time = np.arange(len(series_scaled), dtype="float32")
    split_time = int(len(series_scaled) * 0.8)
    x_train = series_scaled[:split_time]
    x_valid = series_scaled[split_time:]
    time_valid = time[split_time:]

    dataset = windowed_dataset(x_train)
    valid_dataset = windowed_dataset(x_valid)

    # Model definition
    model = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(100, input_shape=[None, 1], return_sequences=True),
            keras.layers.SimpleRNN(100),
            keras.layers.Dense(1),
        ]
    )

    # stopping when no progress is madel
    stop_early = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    # building and training the model
    optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.7)
    model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    model.fit(
        dataset,
        epochs=EPOCHS,
        verbose=1,
        validation_data=valid_dataset,
        callbacks=[stop_early],
    )

    # predicting values based on a trained model
    forecast = model_forecast(model, series_scaled[split_time - WINDOW_SIZE : -1])[:, 0]

    # invert data scaling
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
    x_valid = scaler.inverse_transform(x_valid.reshape(-1, 1))

    # error calculation
    mae_metric = keras.metrics.MeanAbsoluteError()
    mae_metric.update_state(x_valid, forecast)
    mae = mae_metric.result().numpy()
    mean_value = np.mean(x_valid)
    percentage_error = np.abs(mae / mean_value) * 100
    percentage_error_rounded = round(percentage_error, 2)

    # making predictions for subsequent days and then inverting the data scaling
    future_forecast = []
    last_window = series_scaled[-WINDOW_SIZE:]

    for _ in range(n_steps):
        prediction = model.predict(last_window[np.newaxis])
        future_forecast.append(prediction[0, 0])
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction[0, 0]
    # transforming future forecasts from a normalized form to the original data scale
    future_forecast = np.array(future_forecast)
    future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))
    # Error and prediction returns
    return (
        percentage_error_rounded,
        future_forecast,
        forecast.flatten(),
        x_valid.flatten(),
        time_valid,
    )


# function that prepares data for the training model
def windowed_dataset(series_scaled: np.ndarray):
    """
    Prepares a TensorFlow dataset for time series forecasting by creating
    sliding windows of the input series.

    Parameters:
    series_scaled(np.ndarray): The scaled time series data.

    Returns:
    dataset(data.Dataset): A TensorFlow dataset containing tuples of
    (input_window, target_value) ready for training a time series model.
    """
    series_scaled = expand_dims(series_scaled, axis=-1)
    dataset = data.Dataset.from_tensor_slices(series_scaled)
    dataset = dataset.window(WINDOW_SIZE + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(WINDOW_SIZE + 1))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).map(
        lambda window: (window[:-1], window[-1])
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(1)
    return dataset


# function for forecasting using a machine learning model
def model_forecast(model: keras.Model, series_scaled: np.ndarray):
    """
    Generates forecasts using a trained model on a scaled time series dataset.

    Parameters:

    model(keras.Model): The trained TensorFlow model used for making predictions.
    series_scaled(np.ndarray): The scaled time series data to forecast.

    Returns:
    forecast(np.ndarray): The forecasted values generated by the model.
    """
    ds = data.Dataset.from_tensor_slices(series_scaled)
    ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
