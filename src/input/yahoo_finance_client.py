import yfinance
import pandas as pd


def fetch_nasdaq_listings(num_companies):
    # Get a list of all Nasdaq-listed companies
    nasdaq_tickers = yfinance.Ticker("NDAQ").info['tickers']

    # Select a random sample of n companies
    tickers = nasdaq_tickers.sample(num_companies)

    # Fetch data for the selected tickers
    data = yfinance.download(tickers, period="max")

    return data, tickers


# Number of companies to fetch
num_companies = 1000

# Fetch data
data, tickers = fetch_nasdaq_listings(num_companies)

# Extract company names
company_names = data['shortName']

for ticker, company_name in zip(tickers, company_names):
    print(f"{ticker}: {company_name}")

# Create DataFrame
df = pd.DataFrame({'ticker': tickers, 'company_name': company_names})

# Save to CSV
df.to_csv("stock_data.csv", index=False)
