"""
Main file of the project responsible for final dashboard created in Dash
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import pickle
from datetime import datetime, timedelta
from er_forecast import make_lr_forecast
from functions import get_yfinance_data, get_key_by_value
import json
import datetime

#getting tickers data
with open("data-utils/data-raw/tickers.json", 'r') as f:
        data_tickers = json.load(f)
ticker_names = list(data_tickers.values())

#example data
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Investment Decision Support System"),

    html.Label('Select Financing Instrument:'),
    dcc.Dropdown(
        id='base-ticker-dropdown',
        options=[{'label': ticker_name, 'value': ticker_name} for ticker_name in ticker_names],
        value='EUR'
    ),

    html.Label('Number of Days to Predict Ahead:'),
    dcc.Input(
        id='predict-days-input',
        type='number',
        value=5,  # Default value
        min=1,
        max=10
    ),

    html.Label('Select Date Range:'),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date="2008-01-01",
        end_date="2024-05-31",
        display_format='YYYY-MM-DD',
        min_date_allowed = "2008-01-01",
        max_date_allowed = "2024-05-31"
    ),

    html.Button('Generate results', id='generate-results-button', n_clicks=0),
    html.Button('Reset', id='reset-button', n_clicks=0),

    dcc.Graph(
        id='stock-price'
    )

])

@app.callback(
    [Output('stock-price', 'figure'),
     Output('stock-price', 'style'),
     Output('predict-days-input', 'value'),
     Output('base-ticker-dropdown', 'value'),
    Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date')],
    [Input('generate-results-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('predict-days-input', 'value'),
     State('base-ticker-dropdown', 'value'),
      State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)

def update_graph(show_clicks, reset_clicks, predict_days, selected_ticker, start_date, end_date):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, {'display': 'none'}, 5, 'S&P 500', "2008-01-01", "2024-05-31"  # Default value for predict-days-input

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'generate-results-button':

        ticker = get_key_by_value(data_tickers, selected_ticker)
        data = get_yfinance_data(ticker, start_date, end_date)

        fig = px.line(data, x=data.index, y='Close', title=f'Stock Price for {selected_ticker}')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Price')

        return fig, {'display': 'block'}, predict_days, selected_ticker, start_date, end_date

    elif button_id == 'reset-button':
        # Reset the graph and input value
        return {}, {'display': 'none'}, 5, 'S&P 500', "2008-01-01", "2024-05-31"  # Reset value for predict-days-input

    return dash.no_update, {'display': 'none'}, 5, 'S&P 500', "2008-01-01", "2024-05-31"

#Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# server = app.server

# if __name__ == "__main__":
#     app.run_server(host="0.0.0.0", port=8080)

