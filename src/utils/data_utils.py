import json

def json_str_to_dict(json_string):
    try:
        # Convert the JSON string to a Python dictionary
        dictionary = json.loads(json_string)
        return dictionary
    except json.JSONDecodeError as e:
        # Handle the error if the JSON string is not properly formatted
        print(json_string)
        print(f"Error decoding JSON: {e}")
        return None