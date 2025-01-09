import os
import json
import logging

def load_publications_database(db_file_path):
    """
    Load the publications database from a newline-separated JSON file.
    Returns a dictionary mapping normalized DOIs to publication data.
    """
    publications = {}
    with open(db_file_path, 'r') as f:
        for line in f:
            try:
                pub = json.loads(line)
                doi = pub.get('doi', '').strip().lower()
                if doi:
                    publications[doi] = pub
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line: {e}")
    return publications

def normalize_doi(doi):
    """
    Normalize a DOI by removing URL prefixes and converting to lowercase.
    """
    if doi:
        doi = doi.strip().lower()
        prefixes = ['http://dx.doi.org/', 'https://dx.doi.org/', 'http://doi.org/', 'https://doi.org/', 'doi:']
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break
    return doi

def process_files(root_dir, publications_db):
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

                    # Check if 'enhanced_publications' exist
                    enhanced_publications = data.get('enhanced_publications', [])

                    if not enhanced_publications:
                        logging.info(f"      No 'enhanced_publications' in file '{file_name}', skipping.")
                        continue

                    # Process each publication
                    for pub in enhanced_publications:
                        doi = pub.get('DOI')
                        normalized_doi = normalize_doi(doi)
                        if not normalized_doi:
                            logging.warning(f"        Publication with missing or invalid DOI: {doi}")
                            pub['is_aps_publication'] = False
                            continue

                        if normalized_doi in publications_db:
                            pub['is_aps_publication'] = True
                            pub['new_id_publication'] = publications_db[normalized_doi].get('new_id_publication')
                            pub['aps_id_publication'] = publications_db[normalized_doi].get('aps_id_publication')
                        else:
                            pub['is_aps_publication'] = False

                    # Save the updated data back to the file
                    save_json(data, file_path)
                    logging.info(f"      Updated data saved to '{file_name}'")

    logging.info("Processing complete.")

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_directory = "././experiments_validation_results/updated_organized_results"
    publications_db_file = "././organised_data/publications.json"

    logging.info("Loading publications database...")
    publications_db = load_publications_database(publications_db_file)
    logging.info(f"Loaded {len(publications_db)} publications into the database.")

    process_files(root_directory, publications_db)
