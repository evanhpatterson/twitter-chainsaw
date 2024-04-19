import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_stocks_forecasting_data(fpath: str, n_prev_days: int):

    # load the dataframe
    df = pd.read_csv(fpath)

    # convert to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # get the deltas from day to day
    df = df.set_index("Date").diff()
    df = df.drop(df.index[0])

    # preprocess by normalizing the values
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # start collecting data
    data_x = []
    data_y = []

    # iterate through; the input should be the last few days
    for i in range(n_prev_days, data.shape[0]):
        data_x.append(data[i-n_prev_days:i]) # the past n days
        data_y.append(data[i]) # the current day
    
    # convert lists to arrays
    data_x = np.array(data_x, dtype=np.float32)
    data_y = np.array(data_y, dtype=np.float32)

    # shuffle the data
    perm = np.random.permutation(len(data_x))

    data_x = data_x[perm]
    data_y = data_y[perm]

    return data_x, data_y


if __name__=="__main__":
    fpath = "data/NVDA.csv"
    data_x, data_y = load_stocks_forecasting_data(fpath=fpath, n_prev_days=30)
