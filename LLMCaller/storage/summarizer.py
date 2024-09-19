import os
import json
from datetime import datetime

def update_summary(result, run_dir):
    summary_path = os.path.join(run_dir, "experiment_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    category = result['category']
    variable = result['variable']
    
    if category not in summary:
        summary[category] = {}
    
    if variable not in summary[category]:
        summary[category][variable] = {"attempts": []}
    
    attempt_summary = {
        "attempt": result['attempt'],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "is_valid": result.get('validation_result', {}).get('is_valid', False)
    }

    if 'error' in result:
        attempt_summary["error"] = result['error']
    
    summary[category][variable]["attempts"].append(attempt_summary)
    summary[category][variable]["latest_attempt"] = result['attempt']
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path