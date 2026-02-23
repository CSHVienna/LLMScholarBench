import json
import requests
import time
from tqdm import tqdm
import random

def get_institution_stats(openalex_id, retries=3):
    url = f"https://api.openalex.org/institutions/I{openalex_id}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'id': openalex_id,
                    'display_name': data.get('display_name'),
                    'works_count': data.get('works_count'),
                    'cited_by_count': data.get('cited_by_count'),
                    'summary_stats': data.get('summary_stats', {}),
                    'country_code': data.get('country_code'),
                    'city': data.get('geo', {}).get('city'),
                    'error': None
                }
            else:
                print(f"Error {response.status_code} for institution ID: {openalex_id}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"Request exception for institution ID: {openalex_id}. Retrying in {2 ** attempt} seconds... Exception: {e}")
            time.sleep(2 ** attempt)
    return {'id': openalex_id, 'error': 'Failed to retrieve after multiple attempts'}

def process_institutions(input_file, output_file, sample_size=None):
    with open(input_file, 'r') as infile:
        institution_ids = [line.strip() for line in infile]
    
    if sample_size:
        institution_ids = random.sample(institution_ids, min(sample_size, len(institution_ids)))
    
    with open(output_file, 'w') as outfile:
        for institution_id in tqdm(institution_ids, desc="Processing institutions"):
            result = get_institution_stats(institution_id)
            json.dump(result, outfile)
            outfile.write('\n')
            time.sleep(0.1)  # To avoid hitting rate limits

if __name__ == "__main__":
    sample_size = 100  # Set to None to process all institutions

    if sample_size:
        input_file = 'data/output/unique_institution_ids_fake.txt'
        output_file = 'data/output/openalex_institutions_data_sample_{sample_size}.json'

    else:
        input_file = 'data/output/unique_institution_ids_fake.txt'
        output_file = 'data/output/openalex_institutions_data.json'
    
    process_institutions(input_file, output_file, sample_size)