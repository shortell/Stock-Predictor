from ai_models.gemini_client import generate_JSON


# def analyze_post_sentiment(post_text: str):
#     prompt = f"""
#     Analyze the following text for anything related to companies (e.g., company names, products, services, CEOs, etc.).:
#     \"\"\"
#     {post_text}
#     \"\"\"
#     """
    
#     json_prompt = """
#     For each company identified get its stock ticker and analyze the sentiment of the post on a scale from -1.00 (most negative) to 1.00 (most positive).
#     Thoroughly analyze the sentiment of the post to provide the most accurate sentiment score.
#     Use this JSON schema:
#     sentiment = {"ticker": str, "sentiment": float}
#     Return a `list[sentiment]`
#     """
#     response = generate_JSON(prompt + json_prompt)
#     return response

def analyze_post_sentiment(post_text: str):
    first_shot = f"""Read the following Reddit post and identify all companies mentioned. The company names could be mentioned directly, by their stock ticker, or through a related entity such as a product, service, or employee. Ignore URLs and image links. If a company is not publicly traded ignore it:
    {post_text}\n\n"""
    second_shot = """For each unique company identified, return a list of dictionaries where each dictionary includes:
    1. The company's stock ticker.
    2. A sentiment score (between -1 and 1) based on the overall sentiment of the mention.

    Use this JSON schema:
    sentiment = {"ticker": str, "sentiment": float}
    Return a `list[sentiment]`:
    """
    prompt = first_shot + second_shot
    
    response = generate_JSON(prompt)
    return response

