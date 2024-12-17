import json
from tqdm import tqdm

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_aps_authors(file_path):
    print("Loading APS authors...")
    aps_authors = load_jsonl(file_path)
    print(f"Loaded {len(aps_authors)} APS authors")
    return set(author['openalex_id'] for author in aps_authors if author['openalex_id'])

def check_author_in_aps(author_info, aps_authors):
    if not author_info['is_in_openalex']:
        return False
    return author_info['key'] in aps_authors

def process_unique_answers(unique_answers, aps_authors):
    for content_hash, data in tqdm(unique_answers.items(), desc="Processing unique answers"):
        enhanced_authors = data['enhanced_authors']
        for author in enhanced_authors:
            if not author['is_in_openalex']:
                author['has_published_in_aps'] = False
                author['any_candidate_in_aps'] = False
                author['candidate_aps_ratio'] = 0
            else:
                author['has_published_in_aps'] = check_author_in_aps(author, aps_authors)
                
                candidates_in_aps = sum(check_author_in_aps({'is_in_openalex': True, 'key': candidate['id']}, aps_authors) 
                                        for candidate in author['candidates'])
                author['any_candidate_in_aps'] = candidates_in_aps > 0
                author['candidate_aps_ratio'] = candidates_in_aps / len(author['candidates']) if author['candidates'] else 0

    return unique_answers

def main():
    # File paths
    aps_authors_path = "././organised_data/authors.json"
    enhanced_answers_path = "././experiments_validation_results/enhanced_unique_answers_oa_check.json"
    output_path = "././experiments_validation_results/enhanced_unique_answers_aps_check.json"

    # Load data
    aps_authors = load_aps_authors(aps_authors_path)

    print("Loading enhanced unique answers...")
    enhanced_answers = load_json(enhanced_answers_path)

    # Process unique answers
    print("Processing unique answers...")
    updated_answers = process_unique_answers(enhanced_answers, aps_authors)

    # Save results
    print(f"Saving results to {output_path}")
    save_json(updated_answers, output_path)

    print("Processing complete!")

if __name__ == "__main__":
    main()
    