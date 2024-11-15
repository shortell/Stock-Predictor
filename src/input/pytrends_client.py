from pytrends.request import TrendReq 
# import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import os
from datetime import datetime, timedelta

# Initialize pytrends object
pytrends = TrendReq(hl='en-US', timeout=(4, 10))

def build_payload(kw_list, timeframe, cat=0, geo='US', gprop=''):
    """
    Build the payload for Google Trends API request.

    Parameters:
    - kw_list (list): List of keywords to be included in the request.
    - cat (int): Category to narrow down the search (default is 0 for 'All categories').
    - timeframe (str): Time range for the data (default is 'now 1-d' for the past 1 day).
    - geo (str): Geographical location for the search (default is 'US' for the United States).
    - gprop (str): Google property to filter the search results (default is an empty string).

    Returns:
    None
    """
    try:
        pytrends.build_payload(
            kw_list,
            cat=cat,
            timeframe=timeframe,
            geo=geo,
            gprop=gprop
        )
        return True
    except Exception as e:
        print(f"Error building payload: {e}")
        return False
    
def fetch_interest_over_time():
    """
    Retrieve the interest over time data using pytrends library.

    Returns:
    DataFrame: A pandas DataFrame containing the interest over time data.
    """
    try:
        return pytrends.interest_over_time()
    except Exception as e:
        print(f"Error fetching interest over time data: {e}")
        return None

def convert_to_est(df, datetime_column='date'):
    """
    Convert the datetime column in the DataFrame to US Eastern Standard Time (EST).

    Parameters:
    - df (DataFrame): The pandas DataFrame containing a datetime column.
    - datetime_column (str): The name of the datetime column to convert (default is 'date').

    Returns:
    DataFrame: A new DataFrame with the datetime column converted to EST.
    """
    try:
        # Set timezone to UTC first, as Google Trends timestamps are in UTC
        df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True)

        # Convert to US/Eastern time zone
        df[datetime_column] = df[datetime_column].dt.tz_convert('US/Eastern')

        return df
    except Exception as e:
        print(f"Error converting to EST: {e}")
        return df
    
def save_to_csv(data, start, end, search_term, base_dir='data/google_trends'):
    """
    Save a DataFrame to a CSV file named after the given timeframe,
    inside a directory named after the search term.

    Parameters:
    - data (DataFrame): The DataFrame to save.
    - start (str): The start date of the timeframe.
    - end (str): The end date of the timeframe.
    - search_term (str): The search term being collected.
    - base_dir (str): Base directory to store all Google Trends data (default is 'data/google_trends').

    Returns:
    None
    """
    # Remove the 'isPartial' column if it exists
    if 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])
    # Create the search term directory inside the base directory
    search_term_dir = os.path.join(base_dir, search_term.replace(" ", "_"))
    os.makedirs(search_term_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Construct the filename based on the timeframe
    filename = f"{search_term_dir}/{start}_to_{end}.csv"
    data.to_csv(filename, index=False)
    print(f"Saved data to {filename}")

# def fetch_trends_data_with_retry(search_term, start_date, window_size, base_dir='data/google_trends'):
#     """
#     Collects Google Trends data in sliding 30-day windows, moving forward by one day,
#     and saves each window's data to a CSV inside a search term-specific directory.

#     Parameters:
#     - search_term (str): The keyword to search for.
#     - start_date (str): The starting date in 'YYYY-MM-DD' format.
#     - window_size (int): Number of days for each sliding window (e.g., 30).
#     - base_dir (str): Base directory to store all Google Trends data (default is 'data/google_trends').

#     Returns:
#     None
#     """
#     # Parse the initial date
#     current_date = datetime.strptime(start_date, '%Y-%m-%d')

#     while True:
#         # Set the window range
#         start = current_date.strftime('%Y-%m-%d')
#         end = (current_date + timedelta(days=window_size - 1)).strftime('%Y-%m-%d')
#         timeframe = f'{start} {end}'
#         print(f"Fetching data for timeframe: {timeframe}")

#         # Use helper function to build payload
#         if build_payload([search_term], timeframe):
#             data = fetch_interest_over_time()
#             if data is not None and not data.empty:
#                 data = data.reset_index()  # Convert the datetime index to a column
#                 save_to_csv(data, start, end, search_term, base_dir)
#             else:
#                 print(f"No data for {timeframe}. Retrying...")
#         else:
#             print(f"Failed to build payload for {timeframe}. Retrying...")

#         # Infinite retry mechanism with random backoff on failures
#         if data is None or data.empty:
#             wait_time = random.randint(30, 90)  # Random wait between 30 to 90 seconds
#             print(f"Waiting {wait_time} seconds before retrying...")
#             time.sleep(wait_time)
#             continue  # Retry the same timeframe

#         # Stop if we've reached today's date
#         if current_date >= datetime.today() - timedelta(days=window_size):
#             break

#         # Move the window forward by one day
#         current_date += timedelta(days=1)
def fetch_trends_data_with_retry(search_terms, start_date, window_size, base_dir='src/data/google_trends'):
    """
    Collects Google Trends data in sliding 30-day windows, moving forward by one day,
    and saves each window's data to a CSV file for each search term within a combined directory.

    Parameters:
    - search_terms (list of str): A list of keywords to search for.
    - start_date (str): The starting date in 'YYYY-MM-DD' format.
    - window_size (int): Number of days for each sliding window (e.g., 30).
    - base_dir (str): Base directory to store all Google Trends data (default is 'data/google_trends').

    Returns:
    None
    """
    # Ensure search_terms is a list
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    # Parse the initial date
    current_date = datetime.strptime(start_date, '%Y-%m-%d')

    while True:
        # Set the window range
        start = current_date.strftime('%Y-%m-%d')
        end = (current_date + timedelta(days=window_size - 1)).strftime('%Y-%m-%d')
        timeframe = f'{start} {end}'
        print(f"Fetching data for terms {search_terms} for timeframe: {timeframe}")

        # Use helper function to build payload for all terms at once
        if build_payload(search_terms, timeframe):
            data = fetch_interest_over_time()
            if data is not None and not data.empty:
                data = data.reset_index()  # Convert the datetime index to a column
                save_to_csv(data, start, end, '_'.join(search_terms), base_dir)
            else:
                print(f"No data for timeframe {timeframe}. Retrying...")

        else:
            print(f"Failed to build payload for timeframe {timeframe}. Retrying...")

        # Infinite retry mechanism with random backoff on failures
        if data is None or data.empty:
            wait_time = random.randint(30, 90)  # Random wait between 30 to 90 seconds
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            continue  # Retry the same timeframe

        # Stop if we've reached today's date
        if current_date >= datetime.today() - timedelta(days=window_size):
            break

        # Move the window forward by one day
        current_date += timedelta(days=1)


search_terms = ['best buy stock', 'bby']
start_date = '2021-01-01'
window_size = 30

fetch_trends_data_with_retry(search_terms, start_date, window_size)



