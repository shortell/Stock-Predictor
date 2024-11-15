import os
import yaml
from pydantic import BaseModel

from openai import OpenAI

# Load the configuration file for API keys
config = {}
yml_path = os.path.join(os.path.dirname(__file__), '../../config/api_auth.yml')

try:
    with open(yml_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
except FileNotFoundError as e:
    raise RuntimeError(f"Configuration file not found at {yml_path}") from e
except yaml.YAMLError as e:
    raise RuntimeError("Error parsing the YAML configuration file.") from e

# Initialize the OpenAI client with API key
try:
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
except KeyError as e:
    raise RuntimeError("API key not found in the configuration file.") from e

COMPLETION_MODEL = 'gpt-4o-mini-2024-07-18'
EMBEDDING_MODEL = 'text-embedding-3-small'

def create_completion_openai(system_message, user_message, max_tokens, temperature=0.0, model=COMPLETION_MODEL):
    """
    Creates a chat completion using OpenAI's API.

    :param system_message: System-level instructions or context.
    :param user_message: User's input message for the model to process.
    :param max_tokens: Maximum number of tokens for the completion.
    :param temperature: Controls the randomness of the output (0.0 is deterministic).
    :param model: The model to use for completion.
    :return: The generated completion as a string.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during completion creation: {e}")
        return ""

def create_completion_JSON_openai(system_message, user_message, max_tokens, temperature=0.0, model=COMPLETION_MODEL):
    """
    Creates a JSON-structured chat completion using OpenAI's API.

    :param system_message: System-level instructions or context.
    :param user_message: User's input message for the model to process.
    :param max_tokens: Maximum number of tokens for the completion.
    :param temperature: Controls the randomness of the output (0.0 is deterministic).
    :param model: The model to use for completion.
    :return: The JSON-structured response as a string.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during JSON completion creation: {e}")
        return {}
    
class StockDiscussion(BaseModel):
    """Pydantic model to structure stock discussion data."""
    stock_mentions: dict[str, str]

def create_structured_JSON_openai(system_message, user_message, max_tokens, temperature=0.0, model=COMPLETION_MODEL):
    """
    Creates a structured JSON completion using OpenAI's API.

    :param system_message: System-level instructions or context.
    :param user_message: User's input message for the model to process.
    :param max_tokens: Maximum number of tokens for the completion.
    :param response_format: Desired response format (e.g., JSON object).
    :param temperature: Controls the randomness of the output (0.0 is deterministic).
    :param model: The model to use for completion.
    :return: The parsed structured JSON response.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=StockDiscussion
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error during structured JSON completion creation: {e}")
        return {}

def create_embedding_openai(text, model=EMBEDDING_MODEL):
    """
    Creates an embedding for a given text using OpenAI's text-embedding-3-small model.

    :param text: The text to create an embedding for.
    :return: A list of floats representing the embedding. The length of the list is 1536.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error during embedding creation: {e}")
        return []
