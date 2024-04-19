from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    delta = end - start
    
    return [start + timedelta(days=i) for i in range(delta.days + 1)]

def gen_stock_data(start_date, end_date):
    data_dict = dict()
    dates = generate_date_range(start_date=start_date, end_date=end_date)
    data_dict["Date"] = dates
    num_rows = len(dates)
    data_dict["Open"] = np.random.randint(10, 100, size=num_rows)
    data_dict["High"] = np.random.randint(10, 100, size=num_rows)
    data_dict["Low"] = np.random.randint(10, 100, size=num_rows)
    data_dict["Close"] = np.random.randint(10, 100, size=num_rows)
    data_dict["Adj Close"] = np.random.randint(10, 100, size=num_rows)
    data_dict["Volume"] = np.random.randint(1_000, 10_000, size=num_rows)

    return pd.DataFrame(data=data_dict)
