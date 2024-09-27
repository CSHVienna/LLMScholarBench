import json
from tqdm import tqdm

def extract_institution_ids(input_file, output_file):
    institution_ids = set()
    
    # First, count the total number of lines in the file
    with open(input_file, 'r') as infile:
        total_lines = sum(1 for _ in infile)
    
    with open(input_file, 'r') as infile:
        for line in tqdm(infile, total=total_lines, desc="Extracting institution IDs"):
            author_data = json.loads(line)
            if author_data.get('affiliations'):
                for affiliation in author_data['affiliations']:
                    if isinstance(affiliation, dict) and 'institution' in affiliation and 'id' in affiliation['institution']:
                        institution_ids.add(affiliation['institution']['id'])
    
    with open(output_file, 'w') as outfile:
        for institution_id in institution_ids:
            outfile.write(f"{institution_id}\n")
    
    print(f"Extracted {len(institution_ids)} unique institution IDs")

if __name__ == "__main__":
    input_file = 'data/output/openalex_authors_data.json'
    output_file = 'data/output/unique_institution_ids_fake.txt'
    extract_institution_ids(input_file, output_file)