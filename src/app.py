"""
Main file of the project responsible for final dashboard created in Dash
"""

import json
from datetime import datetime
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
from functions import (
    get_yfinance_data,
    get_key_by_value,
    prepare_data_box_plot,
    prepare_data_decomp_trend,
)
from forecast_main import make_forecast

# getting tickers data
with open("data-utils/data-raw/tickers.json", "r", encoding="utf-8") as f:
    data_tickers = json.load(f)
ticker_names = list(data_tickers.values())[11:]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    [
        html.H1("Investment Decision Support System"),
        # slicer for selecting ticker
        html.Label("Select Financing Instrument:"),
        dcc.Dropdown(
            id="base-ticker-dropdown",
            options=[
                {"label": ticker_name, "value": ticker_name}
                for ticker_name in ticker_names
            ],
            value="EUR",
        ),
        # slicer for selecting forecast steps
        html.Label("Number of Days to Predict Ahead:"),
        dcc.Input(
            id="predict-days-input",
            type="number",
            value=5,  # Default value
            min=1,  # Min value
            max=10,  # Max value
        ),
        # calendar slicer for selecting date range
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id="date-picker-range",
            start_date="2008-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            display_format="YYYY-MM-DD",
            min_date_allowed="2008-01-01",
            max_date_allowed=datetime.now().strftime("%Y-%m-%d"),
        ),
        # buttons
        html.Button("Generate results", id="generate-results-button", n_clicks=0),
        html.Button("Reset", id="reset-button", n_clicks=0),
        # line graphs
        dcc.Graph(id="stock-price"),
        dcc.Graph(id="box-plot", style={"display": "none"}),
        dcc.Graph(id="lr-plot", style={"display": "none"}),
        dcc.Graph(id="decomp-trend-plot", style={"display": "none"}),
        dcc.Graph(id="forecast-plot", style={"display": "none"}),
        # result ables
        dash_table.DataTable(
            id="predicted-table",
            style_table={"overflowX": "scroll"},
            sort_action="native",
            sort_mode="single",
            sort_by=[],
        ),
    ]
)


# dash app callbacks
@app.callback(
    [
        Output("stock-price", "figure"),
        Output("stock-price", "style"),
        Output("box-plot", "figure"),
        Output("box-plot", "style"),
        Output("lr-plot", "figure"),
        Output("lr-plot", "style"),
        Output("decomp-trend-plot", "figure"),
        Output("decomp-trend-plot", "style"),
        Output("forecast-plot", "figure"),
        Output("forecast-plot", "style"),
        Output("predicted-table", "columns"),
        Output("predicted-table", "data"),
        Output("predict-days-input", "value"),
        Output("base-ticker-dropdown", "value"),
        Output("date-picker-range", "start_date"),
        Output("date-picker-range", "end_date"),
    ],
    [Input("generate-results-button", "n_clicks"), Input("reset-button", "n_clicks")],
    [
        State("predict-days-input", "value"),
        State("base-ticker-dropdown", "value"),
        State("date-picker-range", "start_date"),
        State("date-picker-range", "end_date"),
    ],
)

# function for creating visualizations
def update_graph( # pylint: disable=too-many-locals
    _sclicks, _rclicks, predict_days: int, selected_ticker: str, start_date: str, end_date: str
):
    """
    Update the stock price graphs and forecast based on user interaction.

    This function is triggered by user actions such as clicking the generate
    results button or the reset button. Depending on the button clicked, it
    updates various graphs and forecast data, or resets them to default values.

    Parameters:
    _sclicks (int): Number of times the generate results button has been clicked.
    _rclicks (int): Number of times the reset button has been clicked.
    predict_days (int): Number of days to predict into the future.
    selected_ticker (str): The selected stock ticker.
    start_date (str): The start date for the data range.
    end_date (str): The end date for the data range.

    Returns:
    tuple: Contains updated or reset values for:
        - Line plot for stock close prices.
        - Display status for line plot.
        - Box plot for stock price distribution by months.
        - Display status for box plot.
        - Linear regression trend plot.
        - Display status for linear regression plot.
        - Decomposed trend plot.
        - Display status for decomposed trend plot.
        - Forecast plot.
        - Display status for forecast plot.
        - Table columns for forecast results.
        - Table data for forecast results.
        - Predict days.
        - Selected ticker.
        - Start date.
        - End date.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        return (
            dash.no_update,
            {"display": "none"},
            dash.no_update,
            {"display": "none"},
            dash.no_update,
            {"display": "none"},
            dash.no_update,
            {"display": "none"},
            dash.no_update,
            {"display": "none"},
            [],
            [],
            5,
            "S&P 500",
            "2008-01-01",
            datetime.now().strftime("%Y-%m-%d")) # default values after reset

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "generate-results-button":

        # geting ticker and yfinance data
        ticker = get_key_by_value(data_tickers, selected_ticker)
        data = get_yfinance_data(ticker, start_date, end_date)
        print(data)

        # line plot for Close values
        fig_line = px.line(
            data, x=data.index, y="Close", title=f"Stock Price for {selected_ticker}"
        )
        fig_line.update_xaxes(title="Date")
        fig_line.update_yaxes(title="Price")

        # box plot
        data_box = prepare_data_box_plot(data, start_date, end_date)
        fig_box = px.box(
            data_box,
            y="Close",
            x="Month",
            title=f"Stock Price Distribution for {selected_ticker} by Months",
        )
        fig_box.update_xaxes(title="Month")
        fig_box.update_yaxes(title="Price")

        # linear regression
        result_decomp, data_decomp, lr_pred = prepare_data_decomp_trend(data)
        n = len(data_decomp)
        fig_lr = px.line(
            x=data_decomp["Date"],
            y=[result_decomp.observed[-n:], lr_pred],
            title=f"Linear trend for {selected_ticker}",
            color_discrete_sequence=["blue", "red"],
        )
        fig_lr.update_xaxes(title="Date")
        fig_lr.update_yaxes(title="Value")
        fig_lr.data[0].name = "Observed"
        fig_lr.data[1].name = "Linear Trend"
        fig_lr.update_layout(legend_title_text="")

        # decomposed trend
        fig_trend = px.line(
            result_decomp.trend,
            title=f"Decomposed trend for {selected_ticker}",
            color_discrete_sequence=["purple"],
        )
        fig_trend.update_xaxes(title="Date")
        fig_trend.update_yaxes(title="Value")
        fig_trend.data[0].name = "Decomposed Trend"
        fig_trend.update_layout(legend_title_text="")

        # making forecast using NN
        _, future_forecast, forecast, x_valid, time_valid = make_forecast(
            start_date, end_date, selected_ticker, predict_days
        )
        n_valid = len(time_valid)  # number of predictions from val set
        data_valid = data[-n_valid:]  # val data to be used for visualization
        future_dates = pd.date_range(
            start=data_valid.index[-1], periods=len(future_forecast) + 1, freq="D"
        )[
            1:
        ]  # generating future dates based on provided param
        dates_val = data_valid.index  # val dates
        combined_dates = np.concatenate(
            [dates_val, future_dates]
        )  # combining val and future dates
        # transforming values to be in the same shape
        x_valid_tranformed = list(x_valid) + [np.nan] * len(future_forecast)
        forecast_transformed = list(forecast) + [np.nan] * len(future_forecast)
        future_forecast_transformed = [np.nan] * len(forecast) + list(future_forecast)
        # df contaning combined data
        df_forecast = pd.DataFrame(
            {
                "dates": combined_dates,
                "X_valid": x_valid_tranformed,
                "forecast": forecast_transformed,
                "future_forecast": future_forecast_transformed,
            }
        )
        df_forecast = df_forecast [-100:]
        df_forecast = df_forecast.melt(
            id_vars=["dates"],
            value_vars=["X_valid", "forecast", "future_forecast"],
            var_name="type",
            value_name="value",
        )
        df_forecast["value"] = df_forecast["value"].astype(float)
        fig_forecast = px.line(
            df_forecast,
            x="dates",
            y="value",
            color="type",
            title=f"Forecast for {selected_ticker}",
        )
        fig_forecast.update_xaxes(title="Date")
        fig_forecast.update_yaxes(title="Value")
        fig_forecast.data[0].name = "Predicted Value [Validation Data]"
        fig_forecast.data[1].name = "Actual Value [Validation Data]"
        fig_forecast.data[2].name = f"Forecast for the next {predict_days} days"
        fig_forecast.update_layout(legend_title_text="")

        # result table
        df_results = pd.DataFrame(
            {
                "Date": future_dates.strftime('%Y-%m-%d'),
                "Prediction": future_forecast.flatten(),
            }
        )
        columns = [{"name": i, "id": i} for i in df_results.columns]
        data_records = df_results.to_dict("records")

        return (
            fig_line,
            {"display": "block"},
            fig_box,
            {"display": "block"},
            fig_lr,
            {"display": "block"},
            fig_trend,
            {"display": "block"},
            fig_forecast,
            {"display": "block"},
            columns,
            data_records,
            predict_days,
            selected_ticker,
            start_date,
            end_date,
        )  # output values

    if button_id == "reset-button":
        # reset the graph and input value
        return (
            {},
            {"display": "none"},
            {},
            {"display": "none"},
            {},
            {"display": "none"},
            {},
            {"display": "none"},
            {},
            {"display": "none"},
            [],
            [],
            5,
            "S&P 500",
            "2008-01-01",
             datetime.now().strftime("%Y-%m-%d"))

    return (
        dash.no_update,
        {"display": "none"},
        dash.no_update,
        {"display": "none"},
        dash.no_update,
        {"display": "none"},
        dash.no_update,
        {"display": "none"},
        dash.no_update,
        {"display": "none"},
        [],
        [],
        5,
        "S&P 500",
        "2008-01-01",
         datetime.now().strftime("%Y-%m-%d"))


# local version of Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

# running Dash app using Docker
# server = app.server
# if __name__ == "__main__":
#     app.run_server(host="0.0.0.0", port=8080)
