import os
import json
import logging

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_files(root_dir):
    for config in os.listdir(root_dir):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path):
            continue

        logging.info(f"Processing configuration '{config}'...")
        for run in os.listdir(config_path):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            logging.info(f"  Processing run '{run}'...")
            for use_case in os.listdir(run_path):
                if not use_case.startswith('field'):
                    continue  # Only process use cases starting with 'field'

                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                logging.info(f"    Processing use case '{use_case}'...")
                for file_name in os.listdir(use_case_path):
                    if not file_name.startswith('validation_result_') or not file_name.endswith('.json'):
                        continue

                    file_path = os.path.join(use_case_path, file_name)
                    data = load_json(file_path)

                    # Check if 'enhanced_authors' and 'enhanced_publications' exist
                    enhanced_authors = data.get('enhanced_authors', [])
                    enhanced_publications = data.get('enhanced_publications', [])

                    if not enhanced_authors or not enhanced_publications:
                        logging.info(f"      Skipping file '{file_name}' due to missing data.")
                        continue

                    # Check that lengths of 'enhanced_authors' and 'enhanced_publications' are the same
                    if len(enhanced_authors) != len(enhanced_publications):
                        logging.warning(f"      Length mismatch between 'enhanced_authors' and 'enhanced_publications' in file '{file_name}'. Skipping file.")
                        continue

                    # Remove existing 'correct_authorship' or 'correct_attribution' fields from enhanced_publications
                    for publication in enhanced_publications:
                        publication.pop('correct_authorship', None)
                        publication.pop('correct_attribution', None)

                    # For each author, check if their 'key' is among the 'author_ids' of the corresponding publication
                    for i, (author, publication) in enumerate(zip(enhanced_authors, enhanced_publications)):
                        author_key = author.get('key')
                        author_ids = publication.get('author_ids', [])

                        if not author_key:
                            # If the 'key' is missing, set 'correct_authorship' to False
                            logging.debug(f"        Author at index {i} missing 'key'. Setting 'correct_authorship' to False in publication.")
                            publication['correct_authorship'] = False
                            continue

                        if author_key in author_ids:
                            publication['correct_authorship'] = True
                        else:
                            publication['correct_authorship'] = False

                    # Save the updated data back to the file
                    save_json(data, file_path)
                    logging.info(f"      Updated data saved to '{file_name}'")

    logging.info("Processing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_directory = "././experiments_validation_results/updated_organized_results"
    process_files(root_directory)
