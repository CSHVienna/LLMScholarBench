import json
from tqdm import tqdm

def extract_author_ids(input_file, output_file):
    author_ids = set()
    
    # First, count the total number of lines in the file
    with open(input_file, 'r') as infile:
        total_lines = sum(1 for _ in infile)
    
    with open(input_file, 'r') as infile:
        for line in tqdm(infile, total=total_lines, desc="Extracting author IDs"):
            record = json.loads(line)
            if record.get('authors'):
                for author in record['authors']:
                    if isinstance(author, dict) and 'author' in author and 'id' in author['author']:
                        author_ids.add(author['author']['id'])
    
    with open(output_file, 'w') as outfile:
        for author_id in author_ids:
            outfile.write(f"{author_id}\n")
    
    print(f"Extracted {len(author_ids)} unique author IDs")

if __name__ == "__main__":
    input_file = 'data/output/openalex_dois_data.json'
    output_file = 'data/output/unique_author_ids_fake.txt'
    extract_author_ids(input_file, output_file)