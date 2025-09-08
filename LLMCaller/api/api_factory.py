from .openrouter_api import OpenRouterAPI
from .openai_api import OpenAIAPI

def create_api_client(config, usage_tracker_path='usage_tracker.json'):
    """Factory function to create the appropriate API client based on provider"""
    provider = config.get('provider', 'openrouter')  # default to openrouter for backward compatibility
    
    if provider == 'openai':
        return OpenAIAPI(config, usage_tracker_path)
    elif provider == 'openrouter':
        return OpenRouterAPI(config, usage_tracker_path)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'openrouter'")