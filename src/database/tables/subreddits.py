from ..postgres_utils import exec_commit


def add_subreddit(id, name, is_active=False):
    """
    Adds a subreddit to the database.
    Parameters:
    - subreddit_name (str): The name of the subreddit to be added.
    Returns:
    - bool: True if the subreddit was successfully added, False otherwise.
    """
    query = """
    INSERT INTO rsi.subreddits (id, name, is_active)
    VALUES (%s, %s, %s);
    """
    try:
        exec_commit(query, (id, name, is_active))
        return True
    except Exception as e:
        print(f"Error adding subreddit to database: {e}")
        return False
