from ..postgres_utils import exec_get_all, connect, exec_commit

def add_labeled_posts(post_sentiment_id): 
    query = """
    INSERT INTO rsst.labeled_posts (post_sentiment_id)
    VALUES (%s)
    ON CONFLICT (post_sentiment_id) DO UPDATE SET
      post_sentiment_id = EXCLUDED.post_sentiment_id;
    """
    try:
        exec_commit(query, (post_sentiment_id,))
        return True
    except Exception as e:
        print(f"Error adding labeled post to database: {e}")
        return False
    
def get_labeled_posts():
    """
    uses a join to get all labeled posts

    """
    query = """
    SELECT p.id, p.sentiment, p.ticker, p.time_created, p.title, p.text, p.url, p.num_comments, p.num_upvotes, p.num_downvotes, p.num_awards
    FROM rsst.labeled_posts lp
    JOIN rsst.post_sentiments p
    ON lp.post_sentiment_id = p.id;
    """
    try:
        result = exec_get_all(query)
        return result
    except Exception as e:
        print(f"Error retrieving labeled posts from database: {e}")
        return []
    
def delete_labeled_post(post_sentiment_id):
    query = """
    DELETE FROM rsst.labeled_posts
    WHERE post_sentiment_id = %s;
    """
    try:
        exec_commit(query, (post_sentiment_id,))
        return True
    except Exception as e:
        print(f"Error deleting labeled post from database: {e}")
        return False