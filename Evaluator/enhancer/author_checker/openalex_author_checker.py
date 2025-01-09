import os
import json
import requests
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mapping_configuration(file_path):
    logging.info(f"Loading mapping configuration from '{file_path}'...")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_enhanced_data(data, file_path):
    logging.info(f"Saving enhanced data to '{file_path}'...")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def check_author(name, cache):
    if name in cache:
        return cache[name]

    base_url = "https://api.openalex.org/authors"
    params = {
        "filter": f"display_name.search:\"{name}\"",
        "sort": "works_count:desc",
        "per-page": 6
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        
        if results:
            primary_author = results[0]
            is_in_openalex = True
            key = primary_author['id'].split('/')[-1]  # Remove 'https://openalex.org/' part
            display_name = primary_author['display_name']
            
            candidates = [
                {
                    'id': candidate['id'].split('/')[-1],
                    'display_name': candidate['display_name']
                }
                for candidate in results[1:6]
            ]
        else:
            is_in_openalex = False
            key = None
            display_name = None
            candidates = []
        
        result = {
            'is_in_openalex': is_in_openalex,
            'key': key,
            'display_name': display_name,
            'candidates': candidates
        }
        
        cache[name] = result
        return result
        
    except requests.RequestException as e:
        logging.error(f"Error querying OpenAlex API for '{name}': {e}")
        result = {
            'is_in_openalex': False,
            'key': None,
            'display_name': None,
            'candidates': []
        }
        cache[name] = result
        return result

def process_unique_answers(mapping_config):
    cache = {}
    enhanced_data = {}

    logging.info("Processing unique answers...")
    for content_hash, data in tqdm(mapping_config['unique_answers'].items(), desc="Processing Unique Answers"):
        validation_result = data['validation_result']
        extracted_data = validation_result.get('extracted_data', [])
        
        if extracted_data is None:
            logging.warning(f"No extracted_data for content_hash: {content_hash}")
            extracted_data = []
        
        enhanced_authors = []
        for author in tqdm(extracted_data, desc="Checking Authors", leave=False):
            name = author.get('Name')
            if name:
                #logging.info(f"Checking author '{name}'...")
                author_info = check_author(name, cache)
                enhanced_author = {**author, **author_info}
                enhanced_authors.append(enhanced_author)
            else:
                logging.warning(f"Author without Name in content_hash: {content_hash}")
        
        enhanced_data[content_hash] = {
            'reference_file': data['reference_file'],
            'enhanced_authors': enhanced_authors
        }
    
    logging.info("Completed processing of unique answers.")
    return enhanced_data

if __name__ == "__main__":
    mapping_file = "././experiments_validation_results/experiment_mapping_configuration.json"
    output_file = "././experiments_validation_results/enhanced_unique_answers_oa_check.json"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load mapping configuration
    mapping_config = load_mapping_configuration(mapping_file)

    # Process unique answers with progress feedback
    enhanced_data = process_unique_answers(mapping_config)

    # Save enhanced data
    save_enhanced_data(enhanced_data, output_file)

    logging.info(f"Enhanced author data saved to {output_file}")