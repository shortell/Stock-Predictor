import os
import pandas as pd

def load_trends_files(trends_dir):
    trend_files = sorted([os.path.join(trends_dir, f)
                          for f in os.listdir(trends_dir) if f.endswith('.csv')])
    return [pd.read_csv(file, parse_dates=['date']).set_index('date') for file in trend_files]

def load_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file, parse_dates=['Date']).set_index('Date')
    return stock_data[['Close', 'Volume']]

def get_first_and_last_date(trends_df):
    first_date = trends_df.index[0].strftime('%Y-%m-%d')
    last_date = trends_df.index[-1].strftime('%Y-%m-%d')
    return first_date, last_date

def get_stock_window(stock_df, start_date, end_date):
    return stock_df[start_date:end_date]
