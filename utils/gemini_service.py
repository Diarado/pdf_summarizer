"""
This module handles API key loading, model configuration, and provides
a function to generate content based on a given prompt.
"""
import os
import google.generativeai as genai
from pathlib import Path

API_KEY_PATH = Path(__file__).parent.parent / "api_keys" / "gemini.txt"

def load_and_configure_api():
    """
    Loads the API key from the specified file and configures the Gemini client.
    """
    try:
        if not API_KEY_PATH.is_file():
            print(f"Error: API key file not found at {API_KEY_PATH}")
            return False
        
        api_key = API_KEY_PATH.read_text().strip()

        if not api_key:
            print(f"Error: API key file at {API_KEY_PATH} is empty.")
            return False

        genai.configure(api_key=api_key)
        return True

    except Exception as e:
        print(f"An error occurred during API configuration: {e}")
        return False

def get_gemini_response(prompt: str, model_name: str = 'gemini-2.0-flash') -> str:
    """
    Generates a response from the Gemini model for a given prompt.

    Args:
        prompt (str): The text prompt to send to the model.
        model_name (str): The name of the model to use.

    Returns:
        str: response, or an error message.
    """
    try:
        if not load_and_configure_api():
            return "Error: API configuration failed."

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"An error occurred while generating the response: {e}"