from .openrouter_api import OpenRouterAPI
from .gemini_api import GeminiAPI

def create_api_client(config, usage_tracker_path='usage_tracker.json', rate_limit=None):
    """Factory function to create the appropriate API client based on provider"""
    provider = config.get('provider', 'openrouter')  # default to openrouter for backward compatibility

    if provider == 'openrouter':
        if rate_limit is not None:
            return OpenRouterAPI(config, usage_tracker_path, rate_limit=rate_limit)
        return OpenRouterAPI(config, usage_tracker_path)
    elif provider == 'gemini':
        return GeminiAPI(config, usage_tracker_path)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'openrouter', 'gemini'")