import os
from dotenv import load_dotenv

import praw


def initialize_reddit():
    """
    Initialize the Reddit instance using environment variables.

    Returns:
    praw.Reddit or None: An instance of the Reddit API wrapper if successful, otherwise None.
    """
    try:
        load_dotenv()

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="Accessing Reddit threads",
            check_for_async=False,
        )
        return reddit
    except Exception as _:
        return None
