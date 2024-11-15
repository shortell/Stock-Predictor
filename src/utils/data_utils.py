import json
import yfinance as yf
import pandas as pd
from database.tables.companies import add_company

def json_str_to_dict(json_string):
    try:
        # Convert the JSON string to a Python dictionary
        dictionary = json.loads(json_string)
        return dictionary
    except json.JSONDecodeError as e:
        # Handle the error if the JSON string is not properly formatted
        print(json_string)
        print(f"Error decoding JSON: {e}")
        return None
    
def print_stock_mentions(mentions):
    """
    Formats and prints the stock mentions data in a readable table format.

    Parameters:
        - mentions (list of tuples): Each tuple contains (company_ticker, mention_count, percentage_of_total).
    """
    # Print header
    print(f"{'Ticker':<15} {'Mentions':<10} {'Percentage':<15}")
    print("=" * 40)  # Separator line
    
    # Print each mention in a formatted way
    for ticker, count, percentage in mentions:
        # Safely handle None values
        count_str = str(count) if count is not None else 'N/A'
        
        # Check if percentage is a valid number before formatting
        if percentage is None:
            percentage_str = 'N/A'
        else:
            try:
                percentage_str = f"{percentage:.2f}%"
            except ValueError:
                percentage_str = 'N/A'  # Handle unexpected errors

        print(f"{ticker:<15} {count_str:<10} {percentage_str:<15}")
    
    print("=" * 40)  # Final separator line

