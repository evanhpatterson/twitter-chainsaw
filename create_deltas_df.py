'''
    Run this file to create a dataframe that stores the change in stock price each day.
'''
import numpy as np
import pandas as pd
import os

if __name__=="__main__":
    fpath = 'data/NVDA.csv'

    fpath_no_ext, ext = os.path.splitext(fpath)

    # load the dataframe
    df = pd.read_csv(fpath)

    # convert to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # get the deltas from day to day
    df = df.set_index("Date").diff()

    df.dropna().to_csv(f"{fpath_no_ext}-delta{ext}")
