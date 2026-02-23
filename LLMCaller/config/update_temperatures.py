#!/usr/bin/env python3
"""Update temperature values in llm_setup.json from best_temperatures CSV."""

import json
import csv
import sys
from pathlib import Path

def update_temperatures(csv_path, json_path):
    """Update temperatures in llm_setup.json from CSV file."""
    # Read CSV and build model->temperature mapping
    temp_mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp_mapping[row['model']] = float(row['temperature'])

    # Load llm_setup.json
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Update temperatures
    updated_count = 0
    for model_name, model_config in config['models'].items():
        if model_name in temp_mapping:
            old_temp = model_config['temperature']
            new_temp = temp_mapping[model_name]
            model_config['temperature'] = new_temp
            print(f"{model_name}: {old_temp} -> {new_temp}")
            updated_count += 1

    # Save updated config
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nUpdated {updated_count} models")

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/data/datasets/LLMScholar-Audits/Auditor/results/temperature-analysis/best_temperatures_by_factuality.csv'
    json_path = Path(__file__).parent / 'llm_setup.json'

    update_temperatures(csv_path, json_path)
