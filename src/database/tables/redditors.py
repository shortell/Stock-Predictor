from ..postgres_utils import connect


def add_redditors_bulk(content_metadata):
    """
    Adds a list of Reddit users to the database in a bulk operation.

    Parameters:
        - content_metadata (list of dicts): A list of dictionaries containing content information.

    Returns:
        - bool: True if all Reddit users were successfully added, False otherwise.
    """

    user_names = [(record['user_name'],) for record in content_metadata]

    query = """
  INSERT INTO rsst.redditors (user_name)
  VALUES (%s)
  ON CONFLICT (user_name) DO NOTHING;
  """

    with connect() as conn:  # Use a context manager for automatic connection closing
        try:
            cur = conn.cursor()
            # No need for mogrify, directly pass values list to executemany
            cur.executemany(query, user_names)
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding redditors to database: {e}")
            return False
