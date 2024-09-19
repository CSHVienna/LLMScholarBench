import json
import os

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
    
    return all_configs[model_name]

def load_twin_scientists_config():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'twin_scientists_config.json')
    return load_config(config_path)