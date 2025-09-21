import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from configparser import ConfigParser
from config.loader import get_global_config

class GeminiAPI:
    def __init__(self, config, usage_tracker_path='usage_tracker.json'):
        self.config = config
        self.usage_tracker_path = usage_tracker_path
        self.credentials_path = self._get_credentials_path()
        self.project_id, self.location, self.service_account_file = self._load_google_config()
        self.credentials = self._load_credentials()

    def _get_credentials_path(self):
        """Get credentials path from global configuration"""
        global_config = get_global_config()
        credentials_dir = global_config.get('credentials_dir')
        if not credentials_dir:
            raise ValueError("credentials_dir not found in global configuration. Please set it in llm_setup.json")
        return credentials_dir

    def _load_google_config(self):
        """Load Google configuration from config.ini"""
        config = ConfigParser()
        keys_directory = os.path.join(self.credentials_path, '.keys')
        config.read(os.path.join(keys_directory, 'config.ini'))

        project_id = config.get('google', 'project_id')
        location = config.get('google', 'location')
        service_account_filename = config.get('google', 'service_account_file')

        if not os.path.isabs(service_account_filename):
            service_account_file = os.path.join(keys_directory, service_account_filename)
        else:
            service_account_file = service_account_filename

        return project_id, location, service_account_file

    def _load_credentials(self):
        """Load Google service account credentials"""
        return service_account.Credentials.from_service_account_file(self.service_account_file)

    async def generate_response(self, prompt):
        """Generate response using Gemini API - returns FULL response like OpenRouter"""

        # Check if this is a grounded request
        is_grounded = self.config.get('grounded', False)

        if is_grounded:
            return await self._generate_grounded_response(prompt)
        else:
            return await self._generate_normal_response(prompt)

    async def _generate_normal_response(self, prompt):
        """Generate normal Vertex AI response (no grounding)"""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location, credentials=self.credentials)

            # Create model
            model = GenerativeModel(self.config['model'])
            generation_config = GenerationConfig(
                temperature=self.config.get('temperature', 0.9)
                # NO max_output_tokens - let it use defaults
            )

            # Generate response
            response = model.generate_content(prompt, generation_config=generation_config)

            # Return the FULL response object - this is what gets saved
            return response

        except Exception as e:
            print(f"Error with Gemini normal API: {e}")
            raise e

    async def _generate_grounded_response(self, prompt):
        """Generate Google Search grounded response"""
        try:
            # Create scoped credentials for API calls
            scoped_credentials = self.credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

            # Build API URL
            api_url = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
                f"{self.project_id}/locations/{self.location}/publishers/google/models/{self.config['model']}:generateContent"
            )

            # Build payload
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"googleSearch": {}}],
                "generationConfig": {
                    "temperature": self.config.get('temperature', 1.0)
                    # NO maxOutputTokens - let it use defaults
                }
            }

            # Make API call
            session = AuthorizedSession(scoped_credentials)
            response = session.post(
                api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()

            # Return the FULL JSON response - this is what gets saved
            return response.json()

        except Exception as e:
            print(f"Error with Gemini grounded API: {e}")
            raise e