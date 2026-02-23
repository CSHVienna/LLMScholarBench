import json
import requests
import time
from tqdm import tqdm
import random

def fetch_author_data(author_id, retries=3, backoff_factor=0.5):
    url = f"https://api.openalex.org/authors/{author_id}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'author_id': author_id,
                    'affiliations': data.get('affiliations'),
                    'cited_by_count': data.get('cited_by_count'),
                    'counts_by_year': data.get('counts_by_year'),
                    'created_date': data.get('created_date'),
                    'display_name': data.get('display_name'),
                    'display_name_alternatives': data.get('display_name_alternatives'),
                    'summary_stats': data.get('summary_stats'),
                    'updated_date': data.get('updated_date'),
                    'works_count': data.get('works_count'),
                    'x_concepts': data.get('x_concepts')
                }
            else:
                print(f"Failed to fetch data for author {author_id}. Status code: {response.status_code}")
                time.sleep(backoff_factor * (2 ** attempt))
        except requests.RequestException as e:
            print(f"Request error for author {author_id}: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    
    return {'author_id': author_id, 'error': 'Failed to retrieve after multiple attempts'}

def process_authors(input_file, output_file, sample_size=None):
    with open(input_file, 'r') as infile:
        author_ids = [line.strip() for line in infile]
    
    if sample_size:
        author_ids = random.sample(author_ids, min(sample_size, len(author_ids)))
    
    with open(output_file, 'w') as outfile:
        for author_id in tqdm(author_ids, desc="Processing authors"):
            result = fetch_author_data(author_id)
            json.dump(result, outfile)
            outfile.write('\n')
            time.sleep(0.1)  # To avoid hitting rate limits

if __name__ == "__main__":
    input_file = 'data/output/unique_author_ids.txt'
    output_file = 'data/output/openalex_authors_data.json'
    sample_size = 100  # Set to None to process all authors
    process_authors(input_file, output_file, sample_size)