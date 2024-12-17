import os
import json
import logging
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_newline_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logging.warning(f"Error decoding JSON in {file_path}, line: {line.strip()}. Error: {e}")
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_disciplines(disciplines_file):
    disciplines = load_newline_json(disciplines_file)
    return {d['label']: d['new_id_discipline'] for d in disciplines}

def load_publications(publications_file):
    publications = load_newline_json(publications_file)
    publication_map = defaultdict(set)
    for pub in publications:
        new_id_publication = pub.get('new_id_publication')
        id_topic = pub.get('id_topic')
        if new_id_publication and id_topic:
            publication_map[new_id_publication].add(id_topic)
        else:
            logging.warning(f"Skipping publication classification due to missing data: {pub}")
    
    if not publication_map:
        logging.error(f"No valid publication classifications found in {publications_file}")
        logging.error(f"First few entries in the file: {publications[:5]}")
    else:
        logging.info(f"Loaded classifications for {len(publication_map)} unique publications")
    
    return publication_map

def get_discipline_code(field, discipline_codes):
    field_mapping = {
        'PER': 'Physics Education Research',
        'CM&MP': 'Condensed Matter & Materials Physics'
    }
    full_field_name = field_mapping.get(field, field)
    return discipline_codes.get(full_field_name)

def process_files(root_dir, publications_map, discipline_codes):
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
                    continue

                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                logging.info(f"    Processing use case '{use_case}'...")
                field = use_case.split('_')[1].upper()  # Extract field from use case name
                correct_discipline_code = get_discipline_code(field, discipline_codes)

                if not correct_discipline_code:
                    logging.warning(f"No discipline code found for field: {field}")
                    continue

                for file_name in os.listdir(use_case_path):
                    if not file_name.startswith('validation_result_') or not file_name.endswith('.json'):
                        continue

                    file_path = os.path.join(use_case_path, file_name)
                    data = load_json(file_path)

                    enhanced_publications = data.get('enhanced_publications', [])

                    if not enhanced_publications:
                        logging.info(f"      No 'enhanced_publications' in file '{file_name}', skipping.")
                        continue

                    for pub in enhanced_publications:
                        new_id_publication = pub.get('new_id_publication')
                        if not new_id_publication:
                            pub['correct_field'] = False
                            continue

                        pub_discipline_codes = publications_map.get(new_id_publication, set())
                        pub['correct_field'] = correct_discipline_code in pub_discipline_codes


                    enhanced_authors = data.get('enhanced_authors', [])

                    if not enhanced_authors:
                        logging.info(f"      No 'enhanced_authors' in file '{file_name}', skipping.")
                        continue

                    for author in enhanced_authors:
                        if author.get('has_published_in_aps'):
                            author_publications = author.get('publications', [])
                            author['author_correct_field'] = any(
                                correct_discipline_code in publications_map.get(pub_id, set())
                                for pub_id in author_publications
                            )
                        else:
                            author['author_correct_field'] = False
                        

                    save_json(data, file_path)
                    logging.info(f"      Updated data saved to '{file_name}'")

    logging.info("Processing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_directory = "././experiments_validation_results/updated_organized_results"
    disciplines_file = "././organised_data/aps_disciplines.json"
    classifications_file = "././organised_data/aps_publication_classifications.json"

    logging.info("Loading disciplines...")
    discipline_codes = load_disciplines(disciplines_file)
    logging.info(f"Loaded {len(discipline_codes)} disciplines.")

    logging.info("Loading publications and their classifications...")
    publications_map = load_publications(classifications_file)
    logging.info(f"Loaded classifications for {len(publications_map)} publications.")

    process_files(root_directory, publications_map, discipline_codes)