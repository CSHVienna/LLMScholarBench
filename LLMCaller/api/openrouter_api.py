import os
import json
import hashlib
from datetime import datetime, date
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

class OpenRouterAPI:
    def __init__(self, config, usage_tracker_path='usage_tracker.json'):
        load_dotenv()
        self.config = config
        self.usage_tracker_path = usage_tracker_path
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.usage_data = self._load_usage_data()
        self.client = self._create_client()
        
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f'OPENROUTER_API_KEY_{i}')
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        if not keys:
            # Fallback to single key
            key = os.getenv('OPENROUTER_API_KEY')
            if key:
                keys.append(key)
        
        if not keys:
            raise ValueError("No OpenRouter API keys found in environment variables. Please set OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, etc. in your .env file")
        
        return keys
    
    def _create_client(self):
        """Create OpenAI client configured for OpenRouter"""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_keys[self.current_key_index]
        )
    
    def _load_usage_data(self) -> dict:
        """Load usage tracking data"""
        if os.path.exists(self.usage_tracker_path):
            with open(self.usage_tracker_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_usage_data(self):
        """Save usage tracking data"""
        with open(self.usage_tracker_path, 'w') as f:
            json.dump(self.usage_data, f, indent=2, default=str)
    
    def _get_key_hash(self, key: str) -> str:
        """Get a hash of the API key for tracking (privacy)"""
        return hashlib.sha256(key.encode()).hexdigest()[:10]
    
    def _update_usage(self, key: str):
        """Update usage count for a key"""
        key_hash = self._get_key_hash(key)
        today = date.today().isoformat()
        
        if key_hash not in self.usage_data:
            self.usage_data[key_hash] = {}
        
        if today not in self.usage_data[key_hash]:
            self.usage_data[key_hash][today] = {'count': 0, 'active': True}
        
        self.usage_data[key_hash][today]['count'] += 1
        self._save_usage_data()
    
    def _should_rotate_key(self) -> bool:
        """Check if current key has reached daily limit"""
        current_key = self.api_keys[self.current_key_index]
        key_hash = self._get_key_hash(current_key)
        today = date.today().isoformat()
        
        if key_hash in self.usage_data and today in self.usage_data[key_hash]:
            return self.usage_data[key_hash][today]['count'] >= 1000
        return False
    
    def _rotate_to_next_key(self) -> bool:
        """Rotate to next available key under daily limit"""
        original_index = self.current_key_index
        
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if not self._should_rotate_key():
                self.client = self._create_client()
                print(f"Rotated to API key {self.current_key_index + 1}")
                return True
        
        self.current_key_index = original_index
        return False
    
    def get_usage_summary(self) -> dict:
        """Get a summary of usage across all keys"""
        summary = {
            'total_keys': len(self.api_keys),
            'current_key': self.current_key_index + 1,
            'daily_usage': {}
        }
        
        today = date.today().isoformat()
        for key_hash, usage_data in self.usage_data.items():
            if today in usage_data:
                summary['daily_usage'][f'key_{key_hash}'] = usage_data[today]['count']
        
        return summary

    def generate_response(self, prompt):
        """Generate response using OpenRouter API with automatic key rotation"""
        # Check if we need to rotate keys
        if self._should_rotate_key():
            if not self._rotate_to_next_key():
                raise Exception("All API keys have reached daily limit (1000 calls)")
        
        try:
            chat_completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/your-username/LLMScholar-Audits",
                    "X-Title": "LLMScholar Audits"
                },
                messages=[
                    {"role": "system", "content": self.config['system_message']},
                    {"role": "user", "content": prompt}
                ],
                model=self.config['model'],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                top_p=self.config['top_p'],
                stop=self.config['stop'],
                stream=self.config['stream']
            )
            
            # Update usage tracking
            current_key = self.api_keys[self.current_key_index]
            self._update_usage(current_key)
            
            return chat_completion
            
        except Exception as e:
            print(f"Error with API key {self.current_key_index + 1}: {e}")
            # Try to rotate to next key
            if self._rotate_to_next_key():
                return self.generate_response(prompt)  # Retry with new key
            else:
                raise e

def test_api_keys():
    """Test function to verify OpenRouter API keys work"""
    import requests
    import json
    
    try:
        # Create API client
        config = {
            'model': 'deepseek/deepseek-chat-v3.1:free',
            'temperature': 0,
            'max_tokens': 100,
            'top_p': 0.1,
            'stop': None,
            'stream': False,
            'system_message': 'You are a helpful assistant.'
        }
        
        client = OpenRouterAPI(config)
        print(f"✅ Successfully loaded {len(client.api_keys)} API key(s)")
        
        # Test each API key
        for i, key in enumerate(client.api_keys):
            print(f"\n🔑 Testing API key {i+1}...")
            
            response = requests.get(
                url="https://openrouter.ai/api/v1/key",
                headers={
                    "Authorization": f"Bearer {key}"
                }
            )
            
            if response.status_code == 200:
                key_info = response.json()
                print(f"✅ Key {i+1} is valid")
                print(f"   Usage: ${key_info.get('usage', 0)}")
                print(f"   Limit: ${key_info.get('limit', 0)}")
                print(f"   Is Free Tier: {key_info.get('is_free_tier', False)}")
            else:
                print(f"❌ Key {i+1} failed: {response.status_code} - {response.text}")
        
        # Test a simple chat completion
        print(f"\n🧪 Testing chat completion...")
        try:
            response = client.generate_response("Say 'Hello, world!' in exactly those words.")
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
    test_api_keys()