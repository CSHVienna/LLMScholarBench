import os
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
load_dotenv(dotenv_path='../.env')

# Get email from environment variable (optional)
OPENALEX_EMAIL = os.getenv('OPENALEX_EMAIL')