import time
import random


from input.reddit_client import initialize_reddit, get_subreddit_by_name, get_posts_from_subreddit, get_comments_from_post
from database.tables.redditors import add_redditors_bulk
from database.tables.posts import add_posts_bulk, get_posts_by_subreddit_in_last_n_days
from database.tables.comments import add_comments_bulk
from database.db_utils import create_schema, drop_schema
from database.tables.subreddits import add_subreddit
from database.tables.post_sentiments import bulk_add_post_sentiments, get_posts_without_sentiments
from database.tables.companies import bulk_add_companies
from sentiment_analysis import analyze_post_sentiment
from utils.data_utils import json_str_to_dict


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
        text = posts_text[post_id]
        response = analyze_post_sentiment(text)
        try:
            records = json_str_to_dict(response)
            if len(records) == 0:
                print(f"No sentiment found for post: {post_id}")
                record = (post_id, None, None)
                new_records.append(record)
            else:
                for record in records:
                    ticker = record['ticker']
                    sentiment = record['sentiment']
                    unique_companies.add(ticker)
                    record = (post_id, ticker, sentiment)
                    print(record)
                    new_records.append(record)
        except:
            print(f"Skipping post due to API error: {post_id}")
            continue

        # if response is None:
        #     print(f"Skipping post due to API error: {post_id}")
        #     continue
        # elif len(json_str_to_dict(response)) == 0:
        #     print(f"No sentiment found for post: {post_id}")
        #     record = (post_id, None, None)
        #     # print(record)
        #     new_records.append(record)
        # else:
        #     records = json_str_to_dict(response)
        #     # print(len(records))
        #     for record in records:
        #         ticker = record['ticker']
        #         sentiment = record['sentiment']
        #         unique_companies.add(ticker)
        #         record = (post_id, ticker, sentiment)
        #         print(record)
        #         new_records.append(record)
                # print(new_records)
    bulk_add_companies(unique_companies)
    bulk_add_post_sentiments(new_records)


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
        comment_metadata, comment_text = get_comments_from_post(
            post_obj, limit)
        total_num_comments += len(comment_text)
        add_redditors_bulk(comment_metadata)
        add_comments_bulk(comment_metadata, post_id)
    return total_num_comments


def collect_comment_sentiment(comments_text):
    pass


drop_schema()
create_schema()

reddit = initialize_reddit()

start_time = time.time()
post_text = collect_posts_metadata(reddit, 'stockmarket', 'week')
# print(post_text)
end_time = time.time()
execution_time = end_time - start_time
print(f"Metadata collection time: {execution_time} seconds")
print(f"Number of posts collected: {len(post_text)}")

start_time = time.time()
collect_posts_sentiment(post_text)
end_time = time.time()
execution_time = end_time - start_time
print(f"Sentiment analysis time: {execution_time} seconds")

start_time = time.time()
comment_text = collect_comments_metadata(reddit, list(post_text.keys()), 75)
end_time = time.time()
execution_time = end_time - start_time
print(f"Comment collection time: {execution_time} seconds")
print(f"Number of comments collected: {comment_text}")
