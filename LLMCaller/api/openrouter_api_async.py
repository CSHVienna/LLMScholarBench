import os
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from config.loader import get_global_config

class OpenRouterAPIAsync:
    """
    Simplified async OpenRouter API client for paid models.
    No rate limiting - runs fully asynchronously.
    """

    def __init__(self, config, usage_tracker_path='usage_tracker.json'):
        self.config = config
        self.usage_tracker_path = usage_tracker_path
        self.api_key = self._load_api_key()
        self.client = self._create_client()

    def _load_api_key(self) -> str:
        """Load API key from centralized credentials directory"""
        # First try environment variable (if already set)
        key = os.getenv('OPENROUTER_API_KEY')
        if key:
            return key

        # Load from config if provided, otherwise use global config
        credentials_dir = self.config.get('credentials_dir')
        if not credentials_dir:
            global_config = get_global_config()
            credentials_dir = global_config.get('credentials_dir')

        if not credentials_dir:
            raise ValueError("credentials_dir not found in global configuration")

        env_file = os.path.join(credentials_dir, '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            key = os.getenv('OPENROUTER_API_KEY')
            if key:
                return key

        raise ValueError(f"No OpenRouter API key found. Please set OPENROUTER_API_KEY in {env_file}")

    def _create_client(self):
        """Create AsyncOpenAI client configured for OpenRouter"""
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    def get_usage_summary(self) -> dict:
        """Get usage summary (simplified for async - no rate tracking)"""
        return {
            'mode': 'async',
            'rate_limiting': 'disabled'
        }

    async def generate_response(self, prompt):
        """Generate response using OpenRouter API - pure async, no rate limiting"""
        try:
            # Prepare extra headers
            extra_headers = {
                "HTTP-Referer": "https://github.com/your-username/LLMScholar-Audits",
                "X-Title": "LLMScholar Audits"
            }

            # Prepare extra_body for sub-provider enforcement (if specified)
            extra_body = {}
            if 'sub_provider' in self.config and self.config['sub_provider']:
                extra_body["provider"] = {
                    "order": [self.config['sub_provider']],
                    "allow_fallbacks": False
                }

            # Prepare API call parameters
            api_params = {
                "extra_headers": extra_headers,
                "messages": [
                    {"role": "system", "content": self.config['system_message']},
                    {"role": "user", "content": prompt}
                ],
                "model": self.config['model'],
                "temperature": self.config['temperature'],
                "stop": self.config['stop'],
                "stream": self.config['stream']
            }

            # Add max_tokens if specified
            if 'max_tokens' in self.config:
                api_params['max_tokens'] = self.config['max_tokens']

            # Add extra_body if needed
            if extra_body:
                api_params['extra_body'] = extra_body

            # Make async API call (no rate limiting!)
            chat_completion = await self.client.chat.completions.create(**api_params)

            return chat_completion

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Error with OpenRouter API (async): {e}")
            raise e
