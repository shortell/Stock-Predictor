import numpy as np
import torch
from data_loader import get_stock_window
import pandas as pd

def create_training_data(stock_df, trends_df_list):
    X, y = [], []
    for i in range(len(trends_df_list) - 1):
        trends_df = trends_df_list[i]
        next_trends_df = trends_df_list[i + 1]
        trends_df.index = pd.to_datetime(trends_df.index)
        first_date, last_date = trends_df.index.min(), trends_df.index.max()
        next_day = last_date + pd.Timedelta(days=1)
        
        if next_day in stock_df.index and next_day in next_trends_df.index:
            stock_window = get_stock_window(stock_df, first_date, last_date)
            input_data = pd.concat([stock_window, trends_df], axis=1)
            X.append(input_data.values)
            
            stock_next_day_values = stock_df.loc[next_day][['Close', 'Volume']].values
            google_trends_next_day_value = next_trends_df.loc[next_day].values
            target_output = np.append(stock_next_day_values, google_trends_next_day_value)
            y.append(target_output)

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return X_tensor.permute(0, 1, 2), y_tensor

def split_data(X, y, train_ratio=0.8):
    split_idx = int(train_ratio * len(X))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
