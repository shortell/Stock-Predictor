from ..postgres_utils import exec_get_all, connect


def add_comments_bulk(comments, post_id):
    """
    Adds a list of comments to the database in a bulk operation.
    Parameters:
    - comments (list of dicts): A list of dictionaries where each key corresponds to a column name.
    - post_id (str): The ID of the post the comments belong to.
    Returns:
    - bool: True if all comments were successfully added, False otherwise.
    """
    query = """
    INSERT INTO psi.comments (id, post_id, redditor, num_upvotes, time_created)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
      num_upvotes = EXCLUDED.num_upvotes;
    """
    values = [(comment['comment_id'], post_id, comment['user_name'],
               comment['num_upvotes'], comment['time_created']) for comment in comments]
    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.executemany(query, values)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error adding comments to database: {e}")
        return False


def get_comments_by_post(post_id):
    """
    Retrieves comments from a post.
    Parameters:
    - post_id (str): The ID of the post to retrieve comments from.
    Returns:
    - list: A list of dictionaries representing the comments.
    """
    query = """
    SELECT comment_id, user_id, num_upvotes, time_created FROM psi.comments
    WHERE post_id = %s;
    """
    try:
        result = exec_get_all(query, (post_id,))
        return result
    except Exception as e:
        print(f"Error retrieving comments from database: {e}")
        return []
