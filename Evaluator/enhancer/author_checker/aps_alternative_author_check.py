import os
import json

# Function to load JSON data (standard JSON)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to load JSON data (one JSON per line)
def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Function to save JSON data (standard JSON)
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to update JSON files in the experiment directory
def update_experiment_jsons(root_dir, authors_data):
    openalex_ids = {author['openalex_id'] for author in authors_data}

    for config in os.listdir(root_dir):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path):
            continue

        for run in os.listdir(config_path):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            for use_case in os.listdir(run_path):
                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                # Iterate over files to find and update the enhanced authors JSON
                for file in os.listdir(use_case_path):
                    if file.startswith("validation_result_") and file.endswith(".json"):
                        file_path = os.path.join(use_case_path, file)
                        data = load_json(file_path)

                        # Update "has_published_in_aps" for each candidate within enhanced authors
                        for author in data.get('enhanced_authors', []):
                            for candidate in author.get('candidates', []):
                                if candidate['id'] in openalex_ids:
                                    candidate['has_published_in_aps'] = True
                                else:
                                    candidate['has_published_in_aps'] = False

                        # Save the updated JSON back to the file
                        save_json(data, file_path)
                        print(f"Updated file: {file_path}")

if __name__ == "__main__":
    input_dir = "././experiments_validation_results/updated_organized_results"
    authors_file_path = "././organised_data/authors.json"

    # Load authors data from NDJSON file
    authors_data = load_ndjson(authors_file_path)

    # Update the JSON files in the experiment directory
    update_experiment_jsons(input_dir, authors_data)
