import json
import os
import shutil
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_enhanced_data(oa_file, aps_file):
    oa_data = load_json(oa_file)
    aps_data = load_json(aps_file)
    
    # Merge OA and APS data
    for hash_key in oa_data:
        if hash_key in aps_data:
            oa_data[hash_key].update(aps_data[hash_key])
    
    return oa_data

def update_organized_results(source_dir, target_dir, enhanced_data):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.startswith("validation_result_") and file.endswith(".json"):
                source_file = os.path.join(root, file)
                relative_path = os.path.relpath(source_file, source_dir)
                target_file = os.path.join(target_dir, relative_path)
                
                # Ensure target directory exists
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                
                # Load the original file
                original_data = load_json(source_file)
                
                # Get the hash from the filename
                hash_key = file.split('_')[2].split('.')[0]
                
                if hash_key in enhanced_data:
                    # Update the data with enhanced information
                    original_data.update(enhanced_data[hash_key])
                
                # Save the updated file
                save_json(original_data, target_file)

def main():
    source_dir = "././experiments_validation_results/organized_results"
    target_dir = "././experiments_validation_results/updated_organized_results"
    oa_file = "././experiments_validation_results/enhanced_unique_answers_oa_check.json"
    aps_file = "././experiments_validation_results/enhanced_unique_answers_aps_check.json"
    
    print("Loading enhanced data...")
    enhanced_data = load_enhanced_data(oa_file, aps_file)
    
    print("Updating organized results...")
    update_organized_results(source_dir, target_dir, enhanced_data)
    
    print(f"Updated results saved in {target_dir}")

if __name__ == "__main__":
    main()
