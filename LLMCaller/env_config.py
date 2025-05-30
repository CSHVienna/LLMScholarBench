import os
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
load_dotenv(dotenv_path='../.env')

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENALEX_EMAIL = os.getenv('OPENALEX_EMAIL')

def get_groq_api_key():
    """Get Groq API key from environment variables"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    return GROQ_API_KEY

def get_openalex_email():
    """Get OpenAlex email from environment variables (optional)"""
    return OPENALEX_EMAIL 