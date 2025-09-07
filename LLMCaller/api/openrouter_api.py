import os
import time
import asyncio
from datetime import datetime
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

class OpenRouterAPI:
    # Class-level rate limiting shared across all instances
    _global_minute_calls = []
    _global_rate_limit_lock = asyncio.Lock()
    def __init__(self, config, usage_tracker_path='usage_tracker.json'):
        load_dotenv()
        self.config = config
        self.usage_tracker_path = usage_tracker_path
        self.api_key = self._load_api_key()
        self.client = self._create_client()
        
    def _load_api_key(self) -> str:
        """Load single API key from environment variables"""
        key = os.getenv('OPENROUTER_API_KEY')
        if not key:
            raise ValueError("No OpenRouter API key found. Please set OPENROUTER_API_KEY in your .env file")
        return key
    
    def _create_client(self):
        """Create AsyncOpenAI client configured for OpenRouter"""
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
    
    async def _wait_for_rate_limit(self):
        """Wait if we've hit the rate limit using global shared rate limiting"""
        async with OpenRouterAPI._global_rate_limit_lock:
            now = time.time()
            # Remove calls older than 1 minute from global tracker
            OpenRouterAPI._global_minute_calls = [call_time for call_time in OpenRouterAPI._global_minute_calls if now - call_time < 60]
            
            # Use conservative limit well below the 20 calls per minute shown in headers
            max_calls_per_minute = 12  # Very conservative limit to avoid any rate limiting issues
            
            if len(OpenRouterAPI._global_minute_calls) >= max_calls_per_minute:
                wait_time = 60 - (now - OpenRouterAPI._global_minute_calls[0]) + 1
                print(f"Global rate limit reached ({max_calls_per_minute}/min). Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                # Clear old calls after waiting
                now = time.time()
                OpenRouterAPI._global_minute_calls = [call_time for call_time in OpenRouterAPI._global_minute_calls if now - call_time < 60]
    
    def get_usage_summary(self) -> dict:
        """Get a simple usage summary using global rate tracking"""
        return {
            'calls_in_current_minute': len([
                call_time for call_time in OpenRouterAPI._global_minute_calls 
                if time.time() - call_time < 60
            ]),
            'rate_limit': '12 calls per minute (global)'
        }

    async def generate_response(self, prompt):
        """Generate response using OpenRouter API with rate limiting"""
        # Check per-minute rate limit
        await self._wait_for_rate_limit()
        
        try:
            # Prepare extra headers
            extra_headers = {
                "HTTP-Referer": "https://github.com/your-username/LLMScholar-Audits",
                "X-Title": "LLMScholar Audits"
            }
            
            # Add provider preference if specified
            if 'provider' in self.config and self.config['provider']:
                extra_headers["X-OpenRouter-Provider"] = self.config['provider']
            
            chat_completion = await self.client.chat.completions.create(
                extra_headers=extra_headers,
                messages=[
                    {"role": "system", "content": self.config['system_message']},
                    {"role": "user", "content": prompt}
                ],
                model=self.config['model'],
                temperature=self.config['temperature'],
                stop=self.config['stop'],
                stream=self.config['stream']
            )
            
            # Update global rate limit tracking
            async with OpenRouterAPI._global_rate_limit_lock:
                OpenRouterAPI._global_minute_calls.append(time.time())
            
            return chat_completion
            
        except Exception as e:
            print(f"Error with OpenRouter API: {e}")
            raise e

def test_api_key():
    """Test function to verify OpenRouter API key works"""
    import requests
    import json
    
    try:
        # Create API client
        config = {
            'model': 'deepseek/deepseek-chat-v3.1:free',
            'temperature': 0,
            'stop': None,
            'stream': False,
            'system_message': 'You are a helpful assistant.'
        }
        
        client = OpenRouterAPI(config)
        print(f"✅ Successfully loaded API key")
        
        # Test the API key
        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={
                "Authorization": f"Bearer {client.api_key}"
            }
        )
        
        if response.status_code == 200:
            key_info = response.json()
            print(f"✅ API key is valid")
            print(f"   Usage: ${key_info.get('usage', 0)}")
            print(f"   Limit: ${key_info.get('limit', 0)}")
            print(f"   Is Free Tier: {key_info.get('is_free_tier', False)}")
        else:
            print(f"❌ API key validation failed: {response.status_code} - {response.text}")
        
        # Test a simple chat completion
        print(f"\n🧪 Testing chat completion...")
        try:
            import asyncio
            response = asyncio.run(client.generate_response("Say 'Hello, world!' in exactly those words."))
            print(f"✅ Chat completion successful!")
            print(f"   Response: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"❌ Chat completion failed: {e}")
            
        # Show usage summary
        print(f"\n📊 Usage Summary:")
        usage_summary = client.get_usage_summary()
        print(json.dumps(usage_summary, indent=2))
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_api_key()