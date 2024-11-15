import time
import random


from input.reddit_client import initialize_reddit, get_subreddit_by_name, get_posts_from_subreddit, get_comments_from_post
from input.pytrends_client import build_payload, fetch_interest_over_time
from input.yahoo_finance_client import get_intraday_stock_data, normalize_stock_prices, plot_normalized_prices
from database.tables.redditors import add_redditors_bulk
from database.tables.posts import add_posts_bulk, get_posts_by_subreddit_in_last_n_days
from database.tables.comments import add_comments_bulk
from database.db_utils import create_schema, drop_schema
from database.tables.subreddits import add_subreddit
from database.tables.post_sentiments import bulk_add_post_sentiments, get_posts_without_sentiments, get_stock_mentions_avg_sentiment_descending, get_stock_mentions_count_descending
from database.tables.companies import bulk_add_companies, get_companies
from sentiment_analysis import analyze_post_sentiment, separate_company_sentiment
from utils.data_utils import json_str_to_dict, print_stock_mentions
# from llm_apis.openai_client import openai_create_embedding
from yahoo_fin.stock_info import tickers_sp500, tickers_nasdaq, tickers_dow


import matplotlib.pyplot as plt


def collect_posts_metadata(reddit, subreddit_name, time_filter='day', limit=None):
    subreddit = get_subreddit_by_name(reddit, subreddit_name)
    if subreddit is None:
        return None
    add_subreddit(subreddit.id, subreddit_name, False)
    post_metadata, post_text = get_posts_from_subreddit(
        subreddit, time_filter, limit)
    # redditors must be inserted first because posts reference redditors
    add_redditors_bulk(post_metadata)
    add_posts_bulk(post_metadata, subreddit.id)
    return post_text


def collect_posts_sentiment(posts_text):
    unprocessed_posts = get_posts_without_sentiments(list(posts_text.keys()))
    print(unprocessed_posts)
    print(f"Number of posts without sentiments: {len(unprocessed_posts)}")
    new_records = []
    unique_companies = set()
    for post_id in unprocessed_posts:
        print(f"ANALYZING: {post_id}")
        text = posts_text[post_id]
        # response = analyze_post_sentiment(text)
        # print("UNCLEANED TEXT STARTS HERE")
        # print(text)
        # print("UNCLEANED TEXT ENDS HERE")
        response = separate_company_sentiment(text)
        print("RESPONSE STARTS HERE")
        print(response)
        print("RESPONSE ENDS HERE")
    #     try:
    #         records = json_str_to_dict(response)
    #         if len(records) == 0:
    #             print(f"No sentiment found for post: {post_id}")
    #             record = (post_id, None, None)
    #             new_records.append(record)
    #         else:
    #             for record in records:
    #                 ticker = record['ticker']
    #                 sentiment = record['sentiment']
    #                 unique_companies.add(ticker)
    #                 record = (post_id, ticker, sentiment)
    #                 print(record)
    #                 new_records.append(record)
    #     except:
    #         print(f"Skipping post due to API error: {post_id}")
    #         continue
    # bulk_add_companies(unique_companies)
    # bulk_add_post_sentiments(new_records)


def collect_comments_metadata(reddit, posts, limit=None):
    """
    Collects comments from a list of posts.

    Args:
        reddit (praw.Reddit): An instance of the Reddit API wrapper.
        posts (list): A list of post IDs to collect comments from.
        limit (int, optional): The maximum number of comments to collect. Defaults to None.

        Returns:
        dict: A dictionary containing the comment text.
    """
    total_num_comments = 0
    for post_id in posts:
        print(f"Collecting comments for post: {post_id}")
        post_obj = reddit.submission(id=post_id)
        comment_metadata, comments_text = get_comments_from_post(
            post_obj, limit)
        total_num_comments += len(comments_text)
        add_redditors_bulk(comment_metadata)

        add_comments_bulk(comment_metadata, post_id)
    return comments_text


def collect_comment_sentiment(comments_text):
    pass


def collect_stock_and_google_trends_data(ticker, start_date, end_date, search_term, timeframe='now 7-d'):
    build_payload([search_term], timeframe=timeframe)
    google_trends_data = fetch_interest_over_time()

    stock_data = get_intraday_stock_data(ticker, start_date, end_date)
    if not stock_data.empty:
        normalized_stock_data = normalize_stock_prices(stock_data)
         # Plotting both datasets on the same graph
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot Google Trends data
        ax1.set_xlabel('Datetime')
        ax1.set_ylabel('Google Trends Interest', color='tab:blue')
        ax1.plot(google_trends_data.index, google_trends_data[search_term], color='tab:blue', label='Google Trends Interest')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis to plot the normalized stock prices
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Stock Price', color='tab:green')
        ax2.plot(normalized_stock_data['Datetime'], normalized_stock_data['Normalized Price'], color='tab:green', label='Normalized Stock Price')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Add a title and legend
        plt.title(f'{ticker} Stock Price vs Google Trends Interest for "{search_term}"')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        # Display the plot
        plt.show()

collect_stock_and_google_trends_data(
    ticker='AAPL',
    start_date='2024-10-15',
    end_date='2024-10-22',
    search_term='apple stock',
    timeframe='now 7-d'
)





# drop_schema()
# create_schema()

# reddit = initialize_reddit()

# post_text = collect_posts_metadata(reddit, 'stocks', 'day')
# print(post_text)
# execution_time = end_time - start_time
# print(f"Metadata collection time: {execution_time} seconds")
# print(f"Number of posts collected: {len(post_text)}")


# for key in post_text:
#     text = post_text[key]
#     response = openai_create_embedding(text)
#     print(len(response))

# start_time = time.time()
# collect_posts_sentiment(post_text)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Sentiment analysis time: {execution_time} seconds")

# start_time = time.time()
# comment_text = collect_comments_metadata(reddit, list(post_text.keys()), 75)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Comment collection time: {execution_time} seconds")
# print(f"Number of comments collected: {comment_text}")

# stock_mentions = get_stock_mentions_count_descending()

# for record in stock_mentions:
#     ticker, count, percentage, avg_sentiment = record
#     rounded_percentage = round(
#         percentage, 4) if percentage is not None else None
#     print(f"Ticker: {ticker}, Count: {count}, Percentage: {
#           rounded_percentage}, AVG Sentiment: {avg_sentiment}")

# stock_mentions = get_stock_mentions_avg_sentiment_descending()

# for record in stock_mentions:
#     ticker, count, percentage, avg_sentiment = record
#     # if avg_sentiment == 0.0:
#     #     continue
#     if count < 2:
#         continue
#     rounded_percentage = round(percentage, 4) if percentage is not None else None
#     print(f"Ticker: {ticker}, Count: {count}, Percentage: {rounded_percentage}, AVG Sentiment: {avg_sentiment}")


def clean_ticker(ticker):
    """Cleans ticker symbols by stripping whitespace and converting to uppercase."""
    return ticker.strip().upper()

def get_all_tickers():
    tickers = set()
    try:
        sp500_tickers = {clean_ticker(ticker) for ticker in tickers_sp500()}
        tickers.update(sp500_tickers)
        nasdaq_tickers = {clean_ticker(ticker) for ticker in tickers_nasdaq()}
        tickers.update(nasdaq_tickers)
        dow_tickers = {clean_ticker(ticker) for ticker in tickers_dow()}
        tickers.update(dow_tickers)
    except Exception as e:
        print(f"Failed to retrieve tickers: {e}")
    return tickers


def add_all_companies():
    """Adds all companies to the database."""
    tickers = get_all_tickers()
    bulk_add_companies(tickers)

# add_all_companies()

# start_time = time.time()

# all_companies = get_companies()
# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")
# # print(all_companies)
# i = 1  
# row = []
# for company in all_companies:
#     row.append(company[0])

#     if i % 50 == 0:
#         print(row)
#         row = []

#     i += 1

# print(len(all_companies))





