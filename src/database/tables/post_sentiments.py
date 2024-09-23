from ..postgres_utils import connect

def bulk_add_post_sentiments(sentiments):
    """
    Adds a list of stock sentiments to the database in a bulk operation.

    Parameters:
        - sentiments (list of tuples): A list of tuples containing post ID, a ticker, and a sentiment value.

    Returns:
        - bool: True if all sentiments were successfully added, False otherwise.
    """

    query = """
    INSERT INTO psi.post_sentiments (post_id, company_ticker, sentiment_score)
    VALUES (%s, %s, %s)
    ON CONFLICT (post_id, company_ticker) DO NOTHING;
    """

    with connect() as conn:  # Use a context manager for automatic connection closing
        try:
            cur = conn.cursor()
            cur.executemany(query, sentiments)  # Use executemany for bulk inserts
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding post stock sentiments to database: {e}")
            return False

        
def get_posts_without_sentiments(post_ids):
    """
    Given a list of post IDs, returns all post IDs that do not have a record in post_stock_sentiments.

    Parameters:
        - post_ids (list of str): A list of post IDs to check.

    Returns:
        - list of str: Post IDs that do not have a record in post_stock_sentiments.
    """
    query = """
    SELECT id FROM psi.posts
    WHERE id = ANY(%s)
    AND id NOT IN (
        SELECT post_id FROM psi.post_sentiments);
    """
    
    with connect() as conn:
        try:
            cur = conn.cursor()
            cur.execute(query, (post_ids,))
            result = cur.fetchall()
            return [row[0] for row in result]
        except Exception as e:
            print(f"Error fetching posts without sentiments: {e}")
            return []