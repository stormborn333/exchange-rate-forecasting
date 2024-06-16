"""
Main file of the project responsible for final dashboard created in Dash
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import os
import pickle
from datetime import datetime, timedelta
from er_forecast import make_lr_forecast

#getting data used for training lr model
with open(
    "data-utils/data-raw/lr_data.csv",
    encoding="utf8",
    errors="ignore",
) as f:
    df_lr: pd.DataFrame = pd.read_csv(f)
X = df_lr['X']
y = df_lr['y']

#transforming X into dates
d_today = datetime.today()
d_100_ago = d_today - timedelta(days=100)
d_range = pd.date_range(start=d_100_ago, end=d_today, periods=100)
X_dates = [d_range[i-1] for i in X]

#supported currencies
available_currencies = ['EUR', 'PLN']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Exchange Rates Forecasting"),

    html.Label('Select Base Currency:'),
    dcc.Dropdown(
        id='base-currency-dropdown',
        options=[{'label': currency, 'value': currency} for currency in available_currencies],
        value='EUR'
    ),

    html.Label('Select Target Currency:'),
    dcc.Dropdown(
        id='target-currency-dropdown',
        options=[{'label': currency, 'value': currency} for currency in available_currencies],
        value='PLN'
    ),

    html.Label('Number of Days to Predict Ahead:'),
    dcc.Input(
        id='predict-days-input',
        type='number',
        value=7  # Default value
    ),

    dcc.Graph(
        id='line-plot'
    ),

    html.H2("Predicted Exchange Rates"),
    dash_table.DataTable(
        id='predicted-table',
        columns=[
            {"name": "Date", "id": "Date"},
            {"name": "Predicted Value", "id": "Predicted Value"}
        ],
        data=[],
        style_table={'overflowX': 'scroll'},
        sort_action="native",
        sort_mode="single", 
        sort_by=[]
    )
])

@app.callback(
    Output('line-plot', 'figure'),
    Output('predicted-table', 'data'),
    [
        Input('base-currency-dropdown', 'value'),
        Input('target-currency-dropdown', 'value'),
        Input('predict-days-input', 'value')
    ]
)
def update_plot(base_currency, target_currency, predict_days):

    #making predictions
    max_X = np.max(X)
    y_pred = make_lr_forecast(max_X, predict_days)
    X_dates_future = pd.date_range(start=X_dates[-1] + timedelta(days=1), 
                                   periods=int(predict_days))

    data = [
        {'x': pd.to_datetime(X_dates).date, 'y': y, 'type': 'scatter', 'mode': 'lines', 'name': 'Historical values'},
        {'x': pd.to_datetime(X_dates_future).date, 'y': y_pred, 'type': 'scatter', 'mode': 'lines', 'name': 'Forecast'}
    ]
    
    # Layout of the plot
    layout = {
        'title': f'Exchange Rate: {base_currency} to {target_currency} [{predict_days} Days Forecast]',
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': 'Exchange Rate'}
    }

    # Data for the table
    predicted_data = [
        {'Date': date, 'Predicted Value': value} for date, value in zip(pd.to_datetime(X_dates_future).date.tolist(), y_pred.tolist())
    ]

    return {'data': data, 'layout': layout}, predicted_data


# Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)

