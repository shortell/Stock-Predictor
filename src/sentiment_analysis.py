import re
import unicodedata
import json

from pydantic import BaseModel
from llm_apis.gemini_client import generate_JSON
from llm_apis.openai_client import create_structured_JSON_openai, create_completion_JSON_openai





def clean_reddit_post(text: str) -> str:
    # Remove URLs and image URLs
    text = re.sub(
        r'http\S+|www\.\S+|\S+\.(jpg|jpeg|png|gif|webp)\S*', '', text)

    # Remove emojis
    def remove_emoji(text):
        return ''.join(
            char for char in text
            if not unicodedata.category(char).startswith('So')
        )
    text = remove_emoji(text)

    # Remove newlines, tabs, and extra spaces
    # text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newlines and tabs with spaces
    # text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space

    # Strip leading and trailing whitespace
    return text.strip()


# def separate_company_sentiment(text: str) -> dict:
#     """
#     Extracts the company name and sentiment from the input text.
#     Parameters:
#     - text (str): The input text.
#     Returns:
#     - dict: A dictionary containing the company ticker and the part of the text talking about that company.
#     """
#     cleaned_text = clean_reddit_post(text)
#     text_length = len(cleaned_text)
#     system_message = ""  # come up with a prompt that will help the model understand the task of separating
#     user_message = cleaned_text
#     max_tokens = 0  # figure out way to determine max tokens to be appropriate for any text size
#     response = create_structured_JSON_openai(system_message, user_message, max_tokens, StockDiscussion)

def separate_company_sentiment(text: str):
    """
    Extracts stock tickers mentioned in the text along with relevant discussion parts.
    
    Parameters:
    - text (str): The input Reddit post or comment.

    Returns:
    - dict: A dictionary where keys are stock tickers and values are the parts of the text discussing them.
    """
    cleaned_text = clean_reddit_post(text)
    print("CLEANED TEXT STARTS HERE")
    print(cleaned_text)
    print("CLEANED TEXT ENDS HERE")
    

    # Define the system message to instruct the LLM clearly.
    system_message = (
    """
    Extract stock tickers and relevant discussions from the following Reddit post or comment. 
    Identify all stock tickers and the parts of the text that mention or discuss them. Try to capture everything the user says about each stock. As well as context for accurate sentiment analysis. 
    If a company is referred to by name without a ticker, replace it with its known ticker. 
    In cases where multiple companies or tickers appear in the same sentence, associate each part of the text with the correct ticker. 
    If no valid ticker exists for a mentioned company, ignore it. 
    Output a JSON dictionary where keys are stock tickers, and values are the related text excerpts.
    Each ticker should be unique, containing all the relevant text discussing that stock.

    Example output: {
      'AAPL': 'Apple reported better-than-expected earnings.',
      'TSLA': 'Tesla's stock is soaring after the announcement.'
    }
    """
)

    # Create the API request using the openai_client
    response = create_completion_JSON_openai(
        system_message=system_message,
        user_message=cleaned_text,
        max_tokens=4096,
    )

    return response


def analyze_post_sentiment(post_text: str):
    first_shot = f"""Read the following Reddit post and identify all stocks mentioned. The stock may be referred to by the name of the company or by their stock ticker. Ignore URLs and image links. If a company is not publicly traded ignore it:
    {post_text}\n\n"""
    second_shot = """For each unique stock identified, return a list of dictionaries where each dictionary includes:
    1. The stock ticker.
    2. A sentiment score (between -1 and 1) based on the overall sentiment of the mention.

    Use this JSON schema:
    sentiment = {"ticker": str, "sentiment": float}
    Return a `list[sentiment]`:
    """
    prompt = first_shot + second_shot

    response = generate_JSON(prompt)
    return response
