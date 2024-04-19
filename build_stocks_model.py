import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, LeakyReLU

def build_model(n_prev_days, n_features, loss='mean_absolute_error'):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(n_prev_days, n_features), return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model
