import os
import json
from pathlib import Path

def load_author_demographics(file_path):
    with open(file_path, 'r') as f:
        return {json.loads(line)['openalex_id']: json.loads(line) for line in f}

def update_enhanced_authors(file_path, author_demographics):

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\nError loading {file_path}: {e}\n")
        return
    
    if 'enhanced_authors' in data and isinstance(data['enhanced_authors'], list):
        for author_info in data['enhanced_authors']:
            if 'key' in author_info and author_info['key'] in author_demographics:
                author_demo = author_demographics[author_info['key']]
                author_info['gender'] = author_demo.get('gender', 'not in aps')
                author_info['ethnicity'] = author_demo.get('ethnicity', 'not in aps')
            else:
                author_info['gender'] = 'not in aps'
                author_info['ethnicity'] = 'not in aps'
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {file_path}")
    else:
        print(f"No 'enhanced_authors' list found in {file_path}")

def process_directory(directory_path, author_demographics):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json') and file.startswith('validation_result'):
                file_path = os.path.join(root, file)
                update_enhanced_authors(file_path, author_demographics)

def main():
    author_demographics_file = "././organised_data/authors_demographics.json"
    results_dir = "././experiments_validation_results/updated_organized_results"

    author_demographics = load_author_demographics(author_demographics_file)
    print(f"Loaded {len(author_demographics)} author demographics")
    process_directory(results_dir, author_demographics)

if __name__ == "__main__":
    main()