import os
import time
import asyncio
import json
from datetime import datetime
from typing import Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from config.loader import get_global_config

class OpenRouterAPI:
    # Class-level rate limiting shared across all instances
    _global_minute_calls = []
    _global_rate_limit_lock = None
    
    @classmethod
    def _get_lock(cls):
        """Get or create the global rate limit lock"""
        if cls._global_rate_limit_lock is None:
            cls._global_rate_limit_lock = asyncio.Lock()
        return cls._global_rate_limit_lock
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

        # Load from centralized credentials
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
    
    async def _wait_for_rate_limit(self):
        """Wait if we've hit the rate limit using global shared rate limiting"""
        async with self._get_lock():
            now = time.time()
            # Remove calls older than 1 minute from global tracker
            OpenRouterAPI._global_minute_calls = [call_time for call_time in OpenRouterAPI._global_minute_calls if now - call_time < 60]
            
            # Use rate limit with small buffer (16 - 1 = 15)
            max_calls_per_minute = 15  # Conservative limit under 16/minute for free models
            
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
            'rate_limit': '15 calls per minute (global)'
        }
    
    async def check_rate_limit_status(self) -> Tuple[bool, Optional[float]]:
        """
        Pre-flight rate limit check - make a minimal test call to see if we're rate limited
        
        Returns:
            (can_proceed: bool, wait_time_seconds: Optional[float])
            - (True, None): No rate limit, can proceed
            - (False, wait_seconds): Rate limited, should wait wait_seconds before proceeding
        """
        try:
            # Make a minimal test call
            test_prompt = "Hi"  # Shortest possible prompt
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": test_prompt}],
                model=self.config['model'],
                temperature=0,
                max_tokens=1,  # Minimize token usage
                stream=False
            )
            
            # Success - no rate limit active
            print("✅ Pre-flight check: No rate limit detected, proceeding...")
            return True, None
            
        except Exception as e:
            error_str = str(e)
            
            # Check if this is a 429 rate limit error
            if "429" in error_str and "Rate limit exceeded" in error_str:
                print(f"⏳ Pre-flight check: Rate limit detected - {error_str}")
                
                # Try to extract reset timestamp from error
                try:
                    # Look for X-RateLimit-Reset in the error message
                    if "X-RateLimit-Reset" in error_str:
                        # Extract the timestamp - it's in the metadata headers
                        # Error format: {...'X-RateLimit-Reset': '1757285940000'...}
                        start = error_str.find("'X-RateLimit-Reset': '") + len("'X-RateLimit-Reset': '")
                        end = error_str.find("'", start)
                        reset_timestamp_ms = int(error_str[start:end])
                        
                        # Convert to seconds and calculate wait time
                        reset_timestamp_seconds = reset_timestamp_ms / 1000
                        current_time = time.time()
                        wait_time = max(0, reset_timestamp_seconds - current_time)
                        
                        print(f"📊 Rate limit will reset in {wait_time:.1f} seconds")
                        return False, wait_time
                    
                except Exception as parse_error:
                    print(f"⚠️  Could not parse reset time: {parse_error}")
                
                # Fallback: use default wait time
                print("⏳ Using default 60-second wait time")
                return False, 60.0
            
            else:
                # Some other error - re-raise it
                print(f"❌ Pre-flight check failed with non-rate-limit error: {e}")
                raise e

    async def wait_for_rate_limit_reset(self) -> bool:
        """
        Perform pre-flight check and wait if rate limited
        
        Returns:
            True: Ready to proceed
            False: Should abort (some non-rate-limit error occurred)
        """
        try:
            can_proceed, wait_time = await self.check_rate_limit_status()
            
            if can_proceed:
                return True
            
            # We're rate limited - wait for reset
            if wait_time and wait_time > 0:
                print(f"⏳ Rate limit detected. Waiting {wait_time:.1f} seconds for reset...")
                
                # Show progress during long waits
                if wait_time > 10:
                    # Show countdown for long waits
                    remaining = wait_time
                    while remaining > 0:
                        mins, secs = divmod(remaining, 60)
                        print(f"⏰ Rate limit reset in {int(mins):02d}:{int(secs):02d}")
                        sleep_time = min(30, remaining)  # Update every 30s or remaining time
                        await asyncio.sleep(sleep_time)
                        remaining -= sleep_time
                else:
                    await asyncio.sleep(wait_time)
                
                print("✅ Rate limit should be reset now, proceeding...")
                return True
            
            return True  # Fallback to proceed
            
        except Exception as e:
            print(f"❌ Pre-flight rate limit check failed: {e}")
            return False  # Signal to abort

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
            async with self._get_lock():
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