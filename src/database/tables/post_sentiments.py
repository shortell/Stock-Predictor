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
    INSERT INTO rsst.post_sentiments (post_id, company_ticker, sentiment_score)
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
    SELECT id FROM rsst.posts
    WHERE id = ANY(%s)
    AND id NOT IN (
        SELECT post_id FROM rsst.post_sentiments);
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
        
def get_stock_mentions_count_descending():
    """
    Retrieves each stock ticker mentioned in the post_sentiments table,
    the count of mentions, the percentage of total mentions,
    and the average sentiment for each ticker,
    ordered by the count of mentions in descending order.

    Returns:
        - list of tuples: Each tuple contains (company_ticker, mention_count, percentage_of_total, avg_sentiment).
    """
    query = """
    SELECT company_ticker, 
           COUNT(*) AS mention_count, 
           COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS percentage_of_total,
           AVG(sentiment_score) AS avg_sentiment
    FROM rsst.post_sentiments
    GROUP BY company_ticker
    ORDER BY mention_count DESC;  -- Order by mention count in descending order
    """
    
    with connect() as conn:
        try:
            cur = conn.cursor()
            cur.execute(query)
            result = cur.fetchall()
            return [(row[0], row[1], row[2], row[3]) for row in result]  # Return avg_sentiment as well
        except Exception as e:
            print(f"Error fetching stock mentions: {e}")
            return []
        
def get_stock_mentions_avg_sentiment_descending():
    """
    Retrieves each stock ticker mentioned in the post_sentiments table,
    the count of mentions, the percentage of total mentions,
    and the average sentiment for each ticker,
    ordered by the average sentiment score in descending order.

    Returns:
        - list of tuples: Each tuple contains (company_ticker, mention_count, percentage_of_total, avg_sentiment).
    """
    query = """
    SELECT company_ticker, 
           COUNT(*) AS mention_count, 
           COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS percentage_of_total,
           AVG(sentiment_score) AS avg_sentiment
    FROM rsst.post_sentiments
    GROUP BY company_ticker
    ORDER BY avg_sentiment DESC;  -- Order by average sentiment in descending order
    """
    
    with connect() as conn:
        try:
            cur = conn.cursor()
            cur.execute(query)
            result = cur.fetchall()
            return [(row[0], row[1], row[2], row[3]) for row in result]  # Return all required fields
        except Exception as e:
            print(f"Error fetching stock mentions: {e}")
            return []