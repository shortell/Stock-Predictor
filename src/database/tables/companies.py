from ..postgres_utils import exec_commit, exec_get_one, connect


def add_company(ticker, company_name):
    """
    Adds a company to the database.
    Parameters:
    - ticker (str): The ticker symbol of the company to be added.
    - company_name (str): The name of the company to be added.
    Returns:
    - bool: True if the company was successfully added, False otherwise.
    """
    query = """
    INSERT INTO psi.companies (ticker, name)
    VALUES (%s, %s);
    """
    try:
        exec_commit(query, (company_name, ticker))
        return True
    except Exception as e:
        print(f"Error adding company to database: {e}")
        return False
    

def bulk_add_companies(companies: set):
    values = [(company,) for company in companies]
    query = """
    INSERT INTO psi.companies (ticker)
    VALUES (%s)
    ON CONFLICT (ticker) DO NOTHING;
    """
    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.executemany(query, values)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error bulk adding companies to database: {e}")
        return False
    
def update_company_name(ticker, company_name):
    """
    Updates the name of a company in the database.
    Parameters:
    - ticker (str): The ticker symbol of the company to be updated.
    - company_name (str): The new name of the company.
    Returns:
    - bool: True if the company name was successfully updated, False otherwise.
    """
    query = """
    UPDATE psi.companies
    SET name = %s
    WHERE ticker = %s;
    """
    try:
        exec_commit(query, (company_name, ticker))
        return True
    except Exception as e:
        print(f"Error updating company name in database: {e}")
        return False


def delete_company(ticker):
    """
    Deletes a company from the database.
    Parameters:
    - ticker (str): The ticker symbol of the company to be deleted.
    Returns:
    - bool: True if the company was successfully deleted, False otherwise.
    """
    query = """
    DELETE FROM psi.companies
    WHERE ticker = %s;
    """
    try:
        exec_commit(query, (ticker,))
        return True
    except Exception as e:
        print(f"Error deleting company from database: {e}")
        return False


def get_company(id):
    """
    Retrieves a company from the database by its ID.
    Parameters:
    - id (int): The ID of the company to retrieve.
    Returns:
    - dict: A dictionary representing the company.
    """
    query = """
    SELECT * FROM rss.companies
    WHERE id = %s;
    """
    try:
        result = exec_get_one(query, (id,))
        return result
    except Exception as e:
        print(f"Error retrieving company from database: {e}")
        return None
