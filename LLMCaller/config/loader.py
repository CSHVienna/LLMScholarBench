import json
import os
import csv
from functools import lru_cache

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_category_variables():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'category_variables.json')
    return load_config(config_path)

def load_llm_setup(model_name):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'llm_setup.json')
    all_configs = load_config(config_path)
    
    if model_name not in all_configs:
        raise ValueError(f"Model '{model_name}' not found in configuration.")
    
    model_config = all_configs[model_name].copy()
    
    # Resolve system message reference
    if 'system_message_ref' in model_config:
        system_messages = load_system_messages()
        message_ref = model_config.pop('system_message_ref')
        if message_ref in system_messages:
            model_config['system_message'] = system_messages[message_ref]
        else:
            raise ValueError(f"System message reference '{message_ref}' not found in system_messages.json")
    
    # Load provider info from cached CSV only if not explicitly set
    if 'provider' not in model_config:
        provider_map = _load_provider_info()
        if model_config['model'] in provider_map:
            model_config['provider'] = provider_map[model_config['model']]
        else:
            # Default to openrouter for backward compatibility
            model_config['provider'] = 'openrouter'
    
    return model_config

def load_system_messages():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'system_messages.json')
    return load_config(config_path)

@lru_cache(maxsize=1)
def _load_provider_info():
    """Cache provider info from CSV to avoid re-reading on every call"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(config_dir, 'new_models_open_router.csv')
    provider_map = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row['model name'].strip() if 'model name' in row else row[' size'].strip()
                provider = row['provider'].strip() if 'provider' in row else ''
                if provider:
                    provider_map[model_name] = provider
    except Exception as e:
        print(f"Warning: Could not load provider info from CSV: {e}")
    return provider_map

def get_available_models(provider_filter=None):
    """Get list of available model names, optionally filtered by provider"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'llm_setup.json')
    all_configs = load_config(config_path)
    models = [key for key in all_configs.keys() if key != 'global']

    if provider_filter:
        filtered = []
        for model in models:
            config = all_configs[model]
            if config.get('provider') == provider_filter:
                filtered.append(model)
        return filtered

    return models

def get_global_config():
    """Get global configuration settings with environment variable overrides"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'llm_setup.json')
    all_configs = load_config(config_path)
    global_config = all_configs.get('global', {})

    # Override with environment variables if they exist
    if os.getenv('LLMCALLER_CREDENTIALS'):
        global_config['credentials_dir'] = os.getenv('LLMCALLER_CREDENTIALS')

    if os.getenv('LLMCALLER_OUTPUT'):
        global_config['output_dir'] = os.getenv('LLMCALLER_OUTPUT')

    return global_config

def load_twin_scientists_config():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'twin_scientists_config.json')
    return load_config(config_path)