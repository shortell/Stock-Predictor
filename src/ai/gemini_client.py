import os
from dotenv import load_dotenv


import google.generativeai as genai


def generate_JSON(prompt, gemini_model='gemini-1.5-flash'):
    # Load environment variables from .env file
    load_dotenv()

    # Configure the API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    model = genai.GenerativeModel(gemini_model,
                                  generation_config={"response_mime_type": "application/json"})

    response = model.generate_content(prompt)
    return response.text
