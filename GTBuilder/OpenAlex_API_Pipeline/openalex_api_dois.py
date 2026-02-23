import csv
import json
import requests
import time
from tqdm import tqdm
import random

def fetch_openalex_data(doi, retries=3):
    url = f"https://api.openalex.org/works?filter=doi:{doi}"

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    work = data['results'][0]
                    return {
                        'doi': doi,
                        'primary_topic': work.get('primary_topic'),
                        'authors': work.get('authorships', []),
                        'title': work.get('title'),
                        'cited_by_count': work.get('cited_by_count'),
                        'language': work.get('language'),
                        'publication_date': work.get('publication_date'),
                        'referenced_works': work.get('referenced_works'),
                        'error': None
                    }
                else:
                    return {'doi': doi, 'error': 'No result found'}
            else:
                print(f"Error {response.status_code} for DOI: {doi}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"Request exception for DOI: {doi}. Retrying in {2 ** attempt} seconds... Exception: {e}")
            time.sleep(2 ** attempt)
    return {'doi': doi, 'error': 'Failed to retrieve after multiple attempts'}

def process_publications(input_file, output_file, sample_size=None):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        reader = csv.DictReader(infile)
        total_rows = sum(1 for row in infile)
        infile.seek(0)
        next(reader)  # Skip header

        if sample_size:
            rows = random.sample(list(reader), sample_size)
        else:
            rows = reader

        for row in tqdm(rows, total=sample_size or total_rows, desc="Processing DOIs"):
            doi = row['doi']
            result = fetch_openalex_data(doi)
            json.dump(result, outfile)
            outfile.write('\n')
            time.sleep(0.1)  # To avoid hitting rate limits

if __name__ == "__main__":
    sample_size = 100  # Set to None to process all DOIs

    if sample_size: 
        input_file = 'data/input/publications.csv'
        output_file = 'data/output/openalex_dois_data_sample_{sample_size}.json'

    else: 
        input_file = 'data/input/publications.csv'
        output_file = 'data/output/openalex_dois_data.json'
    
    process_publications(input_file, output_file, sample_size)