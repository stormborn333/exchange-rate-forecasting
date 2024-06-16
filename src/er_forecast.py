"""
File containing functions responsible for making forecast based on provided model and number of future steps
"""

import pandas as pd
import numpy as np
import pickle

def make_lr_forecast(X_max: int, n_steps: int):

    #loading model from pickle file
    with open("models/lr_model.pkl", 'rb') as f:
        model = pickle.load(f)

    #making prediction based on provided parameter
    n_steps = int(n_steps)
    X_pred = [X_max + n for n in range(1,n_steps + 1)]
    y_pred = model.predict(np.array(X_pred).reshape(-1, 1))

    return y_pred