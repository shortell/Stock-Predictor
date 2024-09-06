from ai_models.gemini_client import generate_JSON


def analyze_post_sentiment(post_text: str):
    prompt = f"""
    Analyze the following text for anything related to companies (e.g., company names, products, services, CEOs, etc.).:
    \"\"\"
    {post_text}
    \"\"\"
    """
    
    json_prompt = """
    For each company identified get its stock ticker and analyze the sentiment of the post on a scale from -1 (most negative) to 1 (most positive).
    Thoroughly analyze the sentiment of the post to provide the most accurate sentiment score.
    Use this JSON schema:
    sentiment = {"ticker": str, "sentiment": float}
    Return a `list[sentiment]`
    """
    response = generate_JSON(prompt + json_prompt)
    return response
