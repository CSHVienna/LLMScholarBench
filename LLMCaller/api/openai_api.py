import os
import json
from datetime import datetime
from typing import Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

class OpenAIAPI:
    def __init__(self, config, usage_tracker_path='usage_tracker.json'):
        load_dotenv()
        self.config = config
        self.usage_tracker_path = usage_tracker_path
        self.api_key = self._load_api_key()
        self.client = self._create_client()
        
    def _load_api_key(self) -> str:
        """Load OpenAI API key from environment variables"""
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
        return key
    
    def _create_client(self):
        """Create AsyncOpenAI client"""
        return AsyncOpenAI(api_key=self.api_key)
    
    async def generate_response(self, prompt, max_attempts=None):
        """Call OpenAI API - no rate limiting needed"""
        if max_attempts is None:
            max_attempts = self.config.get('max_attempts', 3)
        
        # Build messages array like OpenRouterAPI
        messages = [
            {"role": "system", "content": self.config['system_message']},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config['model'],
                    messages=messages,
                    temperature=self.config.get('temperature', 0),
                    stream=self.config.get('stream', False)
                )
                
                # Track usage
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                self._track_usage(usage_info)
                
                # Return the response object like OpenRouterAPI
                return response
                
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt == max_attempts - 1:
                    raise e
                
                # Brief retry delay
                import asyncio
                await asyncio.sleep(1)
    
    def _track_usage(self, usage_info):
        """Track API usage"""
        try:
            # Try to load existing usage data
            if os.path.exists(self.usage_tracker_path):
                with open(self.usage_tracker_path, 'r') as f:
                    usage_data = json.load(f)
            else:
                usage_data = {}
            
            # Initialize model entry if not exists
            model_name = self.config['model']
            if model_name not in usage_data:
                usage_data[model_name] = {
                    'total_calls': 0,
                    'total_tokens': 0,
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'first_call': datetime.now().isoformat(),
                    'last_call': None,
                    'provider': 'openai'
                }
            
            # Update usage
            usage_data[model_name]['total_calls'] += 1
            usage_data[model_name]['total_tokens'] += usage_info['total_tokens']
            usage_data[model_name]['total_prompt_tokens'] += usage_info['prompt_tokens']
            usage_data[model_name]['total_completion_tokens'] += usage_info['completion_tokens']
            usage_data[model_name]['last_call'] = datetime.now().isoformat()
            
            # Save updated usage data
            with open(self.usage_tracker_path, 'w') as f:
                json.dump(usage_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not track usage: {e}")
    
    async def wait_for_rate_limit_reset(self) -> bool:
        """
        No-op for OpenAI since we don't have rate limits like OpenRouter.
        Always returns True (ready to proceed).
        """
        return True
    
    async def check_rate_limit_status(self) -> tuple[bool, None]:
        """
        No-op for OpenAI since we don't have rate limits like OpenRouter.
        Always returns (True, None) meaning ready to proceed.
        """
        return True, None
    
    def get_usage_summary(self) -> dict:
        """Get usage summary for OpenAI API"""
        return {
            'provider': 'openai',
            'model': self.config['model'],
            'note': 'No rate limits applied for OpenAI models'
        }