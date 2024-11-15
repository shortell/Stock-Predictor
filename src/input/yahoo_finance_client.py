import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def get_intraday_stock_data(ticker: str, start_date: str, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch stock data for a given ticker at a specific interval.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AMZN").
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format (default is None for latest data).
    - interval (str): Interval between data points ('1m', '5m', '15m', '30m', '1h', '1d').

    Returns:
    - pd.DataFrame: DataFrame with stock data (Date, Open, High, Low, Close, and Volume).
    """
    try:
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            print(f"No data found for {ticker} at {interval} interval.")
            return pd.DataFrame()

        # Remove timezone to simplify saving
        data.index = data.index.tz_localize(None)

        # Create a daily frequency DataFrame with all dates in the range
        all_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(all_dates)

        # Forward fill the Close column to handle missing days
        data['Close'] = data['Close'].ffill()

        # Set Open, High, and Low to be the same as the previous Close for missing days
        data['Open'] = data['Open'].combine_first(data['Close'])
        data['High'] = data['High'].combine_first(data['Close'])
        data['Low'] = data['Low'].combine_first(data['Close'])

        # Set Volume to 0 for missing days
        data['Volume'] = data['Volume'].fillna(0)

        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def save_to_csv(data, ticker, start_date, end_date, output_dir='src/data/yahoo_finance'):
    """
    Save the stock data (Date, Open, High, Low, Close, Volume) to a single CSV file.

    Parameters:
    - data (pd.DataFrame): DataFrame with stock data.
    - ticker (str): The stock ticker symbol.
    - start_date (str): Start date of the data.
    - end_date (str): End date of the data.
    - output_dir (str): Directory to store the CSV file (default is 'src/data/yahoo_finance').

    Returns:
    None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename
    filename = f"{output_dir}/{ticker}_{start_date}_to_{end_date}.csv"

    # Save the data to CSV, including the Date index as a column
    data.to_csv(filename, index_label='Date')
    print(f"Saved data to {filename}")

# Example usage
if __name__ == "__main__":
    ticker = "BBY"
    start_date = "2020-12-28"
    end_date = "2024-10-29"
    interval = '1d'  # You can change this to '1m', '5m', etc., for intraday data

    # Fetch the stock data
    data = get_intraday_stock_data(ticker, start_date, end_date, interval)

    # Save the entire data to one CSV file
    if not data.empty:
        save_to_csv(data, ticker, start_date, end_date)
