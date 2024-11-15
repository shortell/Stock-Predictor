import os
import yaml
import google.generativeai as gemini
import time
import random

RATE_LIMITS = {
    'gemini-1.5-flash': 0.06,  # 60 seconds / 1000 requests     
    'gemini-1.5-pro': 0.17,    # 60 seconds / 360 requests     
}

config = {}
yml_path = os.path.join(os.path.dirname(__file__), '../../config/api_auth.yml')
with open(yml_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Configure the API key
gemini.configure(api_key=config["GEMINI_API_KEY"])

def generate_JSON(prompt, gemini_model='gemini-1.5-flash', max_retries=3):
    model = gemini.GenerativeModel(gemini_model,
                                  generation_config={"response_mime_type": "application/json"})

    retries = 0
    rate_limit = RATE_LIMITS[gemini_model]
    
    while retries < max_retries:
        try:
            response = model.generate_content(prompt, generation_config=gemini.GenerationConfig(
                temperature=0.0, # could be a parameter, but we want consistent results
            ))
            time.sleep(rate_limit)  # Enforce rate limiting after each successful request
            return response.text
        except Exception as e:
            retries += 1
            backoff_time = min(rate_limit * (2 ** retries) + random.uniform(0, 1), 60)  # Exponential backoff with jitter
            print(f"Error occurred: {e}. Retrying {retries}/{max_retries} in {backoff_time:.2f} seconds...")
            time.sleep(backoff_time)  # Wait before retrying

    print("Max retries reached. Could not fetch data.")
    return None  # Return None if all retries fail


