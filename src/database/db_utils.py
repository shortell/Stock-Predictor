from .postgres_utils import exec_sql_file, exec_commit


def create_schema():
    """
    Creates the schema for the database.
    """
    try:
        exec_sql_file('queries/schema.sql')
        return True
    except Exception as e:
        print(f"Error creating schema: {e}")
        return False
    
def drop_schema():
    """
    Drops the schema for the database.
    """
    try:
        exec_commit('DROP SCHEMA IF EXISTS rsi CASCADE;')
        return True
    except Exception as e:
        print(f"Error dropping schema: {e}")
        return False