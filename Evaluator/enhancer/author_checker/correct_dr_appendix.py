import os
import json
import requests

input_dir = "./experiments_validation_results/updated_organized_results"
authors_file_path = "./organised_data/authors.json"
cache = {}

def load_aps_authors(authors_file_path):
    aps_authors = set()
    with open(authors_file_path, 'r') as file:
        for line in file:
            author_data = json.loads(line)
            aps_authors.add(author_data['openalex_id'])
    return aps_authors

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
            key = primary_author['id'].split('/')[-1]
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

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {
            'is_in_openalex': False,
            'key': None,
            'display_name': None,
            'candidates': []
        }

def update_enhanced_authors(authors, aps_authors):
    for author in authors:
        name = author.get("Name")
        if name and name.startswith("Dr. "):
            stripped_name = name.replace("Dr. ", "").strip()
            # Run the OpenAlex check without the "Dr." prefix
            openalex_result = check_author(stripped_name, cache)
            # Update the author information
            author.update(openalex_result)
            
            # Update APS information
            key = openalex_result.get("key")
            if key:
                author["has_published_in_aps"] = key in aps_authors
            else:
                author["has_published_in_aps"] = False

            # Update candidates' APS information
            any_candidate_in_aps = False
            candidate_count = len(openalex_result.get("candidates", []))
            published_count = 0

            for candidate in author["candidates"]:
                candidate_id = candidate.get("id")
                if candidate_id:
                    candidate["has_published_in_aps"] = candidate_id in aps_authors
                    if candidate["has_published_in_aps"]:
                        any_candidate_in_aps = True
                        published_count += 1
                else:
                    candidate["has_published_in_aps"] = False

            author["any_candidate_in_aps"] = any_candidate_in_aps
            author["candidate_aps_ratio"] = published_count / candidate_count if candidate_count > 0 else 0

def process_use_case(file_path, aps_authors):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Check if enhanced_authors need updating
    if "enhanced_authors" in data:
        authors = data["enhanced_authors"]
        if any(author.get("Name", "").startswith("Dr. ") for author in authors):
            print(f"Updating OpenAlex data for file: {file_path}")
            update_enhanced_authors(authors, aps_authors)
            # Write updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=2)

def process_run(run_path, aps_authors):
    for root, _, files in os.walk(run_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                process_use_case(file_path, aps_authors)

def process_configuration(config_path, aps_authors):
    for run_name in os.listdir(config_path):
        run_path = os.path.join(config_path, run_name)
        if os.path.isdir(run_path):
            print(f"Processing run: {run_name}")
            process_run(run_path, aps_authors)

def main():
    # Load the APS authors data from the JSON file
    aps_authors = load_aps_authors(authors_file_path)

    for config_name in os.listdir(input_dir):
        config_path = os.path.join(input_dir, config_name)
        if os.path.isdir(config_path):
            print(f"Processing configuration: {config_name}")
            process_configuration(config_path, aps_authors)

if __name__ == "__main__":
    main()
