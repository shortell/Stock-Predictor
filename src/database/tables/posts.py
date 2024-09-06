from ..postgres_utils import exec_get_all, connect


def add_posts_bulk(posts, subreddit_id):
  """
  Adds or updates a list of posts to the database in a bulk operation.

  Parameters:
      - posts (list of dicts): A list of dictionaries where each key corresponds to a column name.
      - subreddit_id (str): The ID of the subreddit the posts belong to.

  Returns:
      - bool: True if all posts were successfully added or updated, False otherwise.
  """

  query = """
  INSERT INTO rsi.posts (id, subreddit_id, redditor, num_upvotes, upvote_ratio, time_created)
  VALUES (%s, %s, %s, %s, %s, %s)
  ON CONFLICT (id) DO UPDATE SET
      num_upvotes = EXCLUDED.num_upvotes,
      upvote_ratio = EXCLUDED.upvote_ratio;
  """

  values = [(post['post_id'], subreddit_id, post['user_name'], post['num_upvotes'],
             post['upvote_ratio'], post['time_created']) for post in posts]

  with connect() as conn:  # Use a context manager for automatic connection closing
      try:
          cur = conn.cursor()
          # No need for mogrify, directly pass values list to executemany
          cur.executemany(query, values)
          conn.commit()
          return True
      except Exception as e:
          print(f"Error adding posts to database: {e}")
          return False


def get_posts_by_subreddit_in_last_n_days(subreddit_id, n_days):
    """
    Retrieves posts from a subreddit in the last n days.
    Parameters:
    - subreddit_id (str): The ID of the subreddit to retrieve posts from.
    - n_days (int): The number of days to retrieve posts from.
    Returns:
    - list: A list of dictionaries representing the posts.
    """
    query = """
    SELECT id, redditor, num_upvotes, upvote_ratio, time_created FROM rsi.posts
    WHERE subreddit_id = %s
    AND time_created >= NOW() - INTERVAL '%s days';
    """
    try:
        result = exec_get_all(query, (subreddit_id, n_days))
        return result
    except Exception as e:
        print(f"Error retrieving posts from database: {e}")
        return []
