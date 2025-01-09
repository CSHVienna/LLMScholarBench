import os
import json
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def check_doi_in_openalex(doi, doi_cache):
    if doi in doi_cache:
        return doi_cache[doi]

    base_url = "https://api.openalex.org/works"
    params = {
        "filter": f"doi:{doi}",
        "mailto": "your-email@example.com"  # Replace with your email
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['meta']['count'] > 0:
            work = data['results'][0]
            oa_work_id_full = work['id']  # Full OpenAlex work ID (URL)
            oa_work_id = oa_work_id_full.split('/')[-1]  # Stripped ID

            publication_year = work.get('publication_year', None)
            author_ids = []
            authorships = work.get('authorships', [])
            for authorship in authorships:
                author = authorship.get('author', {})
                author_id_full = author.get('id', None)
                if author_id_full:
                    author_id = author_id_full.split('/')[-1]  # Stripped author ID
                    author_ids.append(author_id)
            result = {
                'is_in_openalex': True,
                'oa_work_id': oa_work_id,
                'publication_year': publication_year,
                'author_ids': author_ids
            }
        else:
            result = {
                'is_in_openalex': False,
                'oa_work_id': None,
                'publication_year': None,
                'author_ids': []
            }

        doi_cache[doi] = result
        return result

    except requests.RequestException as e:
        logging.error(f"Error querying OpenAlex API for DOI '{doi}': {e}")
        result = {
            'is_in_openalex': False,
            'oa_work_id': None,
            'publication_year': None,
            'author_ids': []
        }
        doi_cache[doi] = result
        return result

def process_files(root_dir):

    doi_cache = {}

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

                    validation_result = data.get('validation_result', {})
                    is_valid = validation_result.get('is_valid', False)
                    if not is_valid:
                        logging.info(f"      Skipping invalid data in '{file_name}'")
                        continue

                    extracted_data = validation_result.get('extracted_data', [])
                    if not extracted_data:
                        logging.info(f"      No extracted data in '{file_name}'")
                        continue

                    enhanced_publications = []
                    for item in extracted_data:
                        doi = item.get('DOI')
                        if not doi:
                            logging.warning(f"        No DOI found in item: {item}")
                            continue

                        # Enhance DOI data
                        doi_info = check_doi_in_openalex(doi, doi_cache)
                        enhanced_item = {**item, **doi_info}
                        enhanced_publications.append(enhanced_item)

                    # Append enhanced publications to data
                    data['enhanced_publications'] = enhanced_publications

                    # Update author authorships based on DOI information
                    enhanced_authors = data.get('enhanced_authors', [])
                    for author in enhanced_authors:
                        if not author.get('is_in_openalex'):
                            continue  # Skip authors not in OpenAlex

                        author_key = author.get('key')
                        if not author_key:
                            continue

                        # Check if the author's key is among the author IDs in any of the DOIs
                        for pub in enhanced_publications:
                            if author_key in pub.get('author_ids', []):
                                author['has_authorship_in_publications'] = True
                                break
                        else:
                            author['has_authorship_in_publications'] = False

                    # Save the updated data back to the file
                    save_json(data, file_path)
                    logging.info(f"      Enhanced data saved to '{file_name}'")

    logging.info("Processing complete.")

if __name__ == "__main__":
    root_directory = "././experiments_validation_results/updated_organized_results"
    process_files(root_directory)
