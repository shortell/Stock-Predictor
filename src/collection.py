import time
import random


from input.reddit_client import initialize_reddit, get_subreddit_by_name, get_posts_from_subreddit, get_comments_from_post
from database.tables.redditors import add_redditors_bulk
from database.tables.posts import add_posts_bulk, get_posts_by_subreddit_in_last_n_days
from database.tables.comments import add_comments_bulk
from database.db_utils import create_schema, drop_schema
from database.tables.subreddits import add_subreddit
from database.tables.post_stock_sentiments import bulk_add_post_stock_sentiment, get_posts_without_sentiments
from database.tables.companies import bulk_add_companies
from sentiment_analysis import analyze_post_sentiment
from utils.data_utils import json_str_to_dict
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError


def collect_posts_metadata(reddit, subreddit_name, time_filter='day', limit=None):
    subreddit = get_subreddit_by_name(reddit, subreddit_name)
    if subreddit is None:
        return None
    add_subreddit(subreddit.id, subreddit_name, False)
    post_metadata, post_text = get_posts_from_subreddit(
        subreddit, time_filter, limit)
    add_redditors_bulk(post_metadata) # redditors must be inserted first because posts reference redditors
    add_posts_bulk(post_metadata, subreddit.id)
    return post_text



def collect_comments(reddit, posts, limit=None):
    """
    Collects comments from a list of posts.
    
    Args:
        reddit (praw.Reddit): An instance of the Reddit API wrapper.
        posts (list): A list of post IDs to collect comments from.
        limit (int, optional): The maximum number of comments to collect. Defaults to None.
        
        Returns:
        dict: A dictionary containing the comment text.
    """
    for post_id in posts:
        post_obj = reddit.submission(id=post_id)
        comment_metadata, comment_text = get_comments_from_post(post_obj, limit)
        add_redditors_bulk(comment_metadata)
        add_comments_bulk(comment_metadata, post_id)
    return comment_text


def analyze_and_collect_post_sentiments(subreddit_name, time_filter='day', limit=None):
    reddit = initialize_reddit()
    posts_text = collect_posts_metadata(reddit, subreddit_name, time_filter, limit)
    unprocessed_posts = get_posts_without_sentiments(list(posts_text.keys()))
    print(f"Number of posts without sentiments: {len(unprocessed_posts)}")
    new_records = []
    for post_id in unprocessed_posts:
        text = posts_text[post_id]
        response = analyze_post_sentiment(text)
        if response is None:
            print(f"Skipping post due to API error: {post_id}")
            continue
        elif len(response) == 0:
            print(f"No sentiment found for post: {post_id}")
            record = (post_id, None, None)
            new_records.append(record)
        else:
            records = json_str_to_dict(response)
            for record in records:
                ticker = record['ticker']
                sentiment = record['sentiment']
                new_records.append((post_id, ticker, sentiment))
    bulk_add_post_stock_sentiment(new_records)




    # for post_id, post_text in post_text.items():
    #     response = analyze_post_sentiment_with_retry(post_text)
    #     if response is None:
    #         print(f"Skipping post due to API error: {post_id}")
    #         continue
    #     if len(response) == 0:
    #         print(f"No sentiment found for post: {post_id}")
    #         continue



# drop_schema()
# create_schema()
# reddit = initialize_reddit()
# start_time = time.time()
# post_text = collect_posts_metadata(reddit, 'wallstreetbets', 'day')
# end_time = time.time()
# execution_time = end_time - start_time

# print(f"Execution time: {execution_time} seconds")
# i = 0
# for post_id, text in post_text.items():
#     print(f"Post #{i}")
#     print(f"{post_id}:\n{text}\n\n")
#     i += 1

# records = []
# unique_companies = set()
# start_time = time.time()

# for post_id, post_text in post_text.items():
#     response = analyze_post_sentiment_with_retry(post_text)
#     if response is None:
#         print(f"Skipping post due to API error: {post_text}")
#         continue
    
#     sentiments = json_str_to_dict(response)
#     if not sentiments:
#         records.append((post_id, None, None))
#         print(post_text)
#         continue
    
#     for record in sentiments:
#         print(record)
#         ticker = record['ticker']
#         sentiment = record['sentiment']
#         unique_companies.add(ticker)
#         records.append((post_id, ticker, sentiment))
        
     # rate limit is 1000 requests per minute, so 1 request every 0.06 seconds

# end_time = time.time()
# print(f"Sentiment Score computation time: {end_time - start_time} seconds")

# # Bulk add and final steps (same as before)

# print(f"Number of posts analyzed: {len(post_text)}")
# print(f"Number of unique companies: {len(unique_companies)}")
# print(f"Unique companies: {unique_companies}")


