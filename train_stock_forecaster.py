import numpy as np
import pandas as pd
from build_stocks_model import build_model
from load_stocks_data import load_stocks_forecasting_data

def build_and_train_stocks_forecaster(data_fpath: str, n_prev_days: int):
    '''
        Build and train the initial model used to forecast stocks.
    '''
    data_x, data_y = load_stocks_forecasting_data(fpath=data_fpath, n_prev_days=n_prev_days)

    model = build_model(n_prev_days=n_prev_days, n_features=6)

    model.summary()

    model.fit(data_x, data_y, epochs=20)

    return model
