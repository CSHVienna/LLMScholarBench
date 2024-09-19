import os
import json
from datetime import datetime

def save_attempt(result, run_dir):
    category = result['category']
    variable = result['variable']
    attempt = result['attempt']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    category_var_dir = os.path.join(run_dir, f"{category}_{variable}")
    os.makedirs(category_var_dir, exist_ok=True)

    result_filename = f"attempt{attempt}_{timestamp}.json"
    result_path = os.path.join(category_var_dir, result_filename)
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result_path