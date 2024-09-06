import os
import yaml
import datetime as dt
import time
import praw


def initialize_reddit():
    """
    Initialize the Reddit instance using environment variables.

    Returns:
    praw.Reddit or None: An instance of the Reddit API wrapper if successful, otherwise None.
    """
    try:
        config = {}
        yml_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))), 'config', 'api_auth.yml')
        with open(yml_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        reddit = praw.Reddit(
            client_id=config['REDDIT_CLIENT_ID'],
            client_secret=config['REDDIT_CLIENT_SECRET'],
            user_agent="Accessing Reddit threads",
            check_for_async=False,
        )

        return reddit

    except Exception as e:
        print(f"Error initializing Reddit: {e}")
        return None


def get_subreddit_by_name(reddit, subreddit_name):
    """
    Gets a subreddit by its name using the PRAW API.

    Args:
        reddit (praw.Reddit): An instance of the Reddit API wrapper.
        subreddit_name (str): The name of the subreddit to retrieve.

    Returns:
        praw.models.Subreddit or None: The subreddit object if found, otherwise None.
    """

    try:
        subreddit = reddit.subreddit(subreddit_name)
        return subreddit
    except Exception as e:
        print(f"Error getting subreddit by name: {e}")
        return None


def get_posts_from_subreddit(subreddit, time_filter='day', limit=None):
    """
    Get posts from a specific subreddit for a given date.

    Parameters:
        subreddit (praw.Subreddit): An instance of the Subreddit API wrapper.
        date (datetime.datetime): The date object for the desired posts.

    Returns:
        list: A list of posts for the specified subreddit and date.
    """
    post_metadata = []
    post_text = {}

    search_results = subreddit.top(time_filter=time_filter, limit=limit)

    for post in search_results:
        # Filter stickied and no selftext posts
        if post.stickied or not post.is_self:
            continue
        post_id = post.id
        user_name = str(post.author)
        num_upvotes = post.score
        upvote_ratio = post.upvote_ratio
        time_created = dt.datetime.utcfromtimestamp(post.created_utc)
        post_str = f"{post.title}\n\n{post.selftext}"
        post_text.update({post_id: post_str})

        record = {
            'post_id': post_id,
            'user_name': user_name,
            'num_upvotes': num_upvotes,
            'upvote_ratio': upvote_ratio,
            'time_created': time_created
        }
        post_metadata.append(record)

    return post_metadata, post_text


def get_comments_from_post(post, limit=None):
    """
    Get comments from a specific post.

    Parameters:
        post_id (str): The ID of the post to retrieve comments from.

    Returns:
        list: A list of comments for the specified post.
    """
    comment_metadata = []
    comment_text = {}

    post.comments.replace_more(limit=limit)
    for comment in post.comments.list():
        comment_id = comment.id
        user_name = str(comment.author)
        num_upvotes = comment.score
        time_created = dt.datetime.utcfromtimestamp(comment.created_utc)
        record = (comment_id, user_name, num_upvotes, time_created)
        record = {
            'comment_id': comment_id,
            'user_name': user_name,
            'num_upvotes': num_upvotes,
            'time_created': time_created
        }
        comment_metadata.append(record)
        comment_text.update({comment_id: comment.body})

    return comment_metadata, comment_text


def get_comments_from_post(reddit, post_id, limit=None):
    """
    Get comments from a specific post.

    Parameters:
        reddit (praw.Reddit): An instance of the Reddit API wrapper.
        post_id (str): The ID of the post to retrieve comments from.
        limit (int): The maximum number of comments to retrieve.

        Returns:
        list: A list of comments for the specified post.
    """
    post = reddit.submission(id=post_id)
    comment_metadata = []
    comment_text = {}

    post.comments.replace_more(limit=limit)
    for comment in post.comments.list():
        comment_id = comment.id
        user_name = str(comment.author)
        num_upvotes = comment.score
        time_created = dt.datetime.utcfromtimestamp(comment.created_utc)
        record = {
            'comment_id': comment_id,
            'user_name': user_name,
            'num_upvotes': num_upvotes,
            'time_created': time_created
        }
        comment_metadata.append(record)
        comment_text.update({comment_id: comment.body})

    return comment_metadata, comment_text
