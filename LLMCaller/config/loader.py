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
    
    model_config = all_configs[model_name].copy()
    
    # Load provider info from CSV
    csv_path = os.path.join(config_dir, 'new_models_open_router.csv')
    try:
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name_csv = row['model name'].strip() if 'model name' in row else row[' size'].strip()
                if model_name_csv == model_config['model']:
                    provider = row['provider'].strip() if 'provider' in row else ''
                    if provider:
                        model_config['provider'] = provider
                    break
    except Exception as e:
        print(f"Warning: Could not load provider info from CSV: {e}")
    
    return model_config

def load_twin_scientists_config():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, 'twin_scientists_config.json')
    return load_config(config_path)