import os
import json

# Function to load JSON data (standard JSON)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to save JSON data
def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to load a JSON array from a file (standard JSON array format)
def load_json_array(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)  # Load entire content as a JSON array

# Function to update the data with Nobel laureate information
def update_nobel_data(root_dir, nobel_data):
    # Create a lookup dictionary for fast Nobel laureate checks
    nobel_openalex_ids = {entry['openalex_id']: entry for entry in nobel_data}

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

                for file in os.listdir(use_case_path):
                    if file.startswith("validation_result_") and file.endswith(".json"):
                        file_path = os.path.join(use_case_path, file)
                        data = load_json(file_path)

                        enhanced_authors = data.get('enhanced_authors', [])
                        
                        for author in enhanced_authors:
                            author_key = author.get('key')

                            # Add Nobel laureate information if the author key is valid
                            if author_key and author_key in nobel_openalex_ids:
                                nobel_info = nobel_openalex_ids[author_key]
                                author['is_nobel_laureate'] = True
                                author['nobel_year'] = nobel_info.get('year')
                                author['nobel_category'] = nobel_info.get('category')
                            else:
                                author['is_nobel_laureate'] = False
                                author['nobel_year'] = None
                                author['nobel_category'] = None

                        # Save the updated data back to the JSON file
                        save_json(file_path, data)
                        print(f"Updated file: {file_path}")

def main():
    input_dir = "././experiments_validation_results/updated_organized_results"
    nobel_file_path = "././organised_data/nobel_prize_with_openalex.json"

    # Load Nobel laureate data from JSON file
    nobel_data = load_json_array(nobel_file_path)

    print("Updating data with Nobel laureate information...")
    update_nobel_data(input_dir, nobel_data)

    print(f"All files in {input_dir} updated successfully.")

if __name__ == "__main__":
    main()
