import os
import json
import hashlib
from collections import defaultdict
from tqdm import tqdm

def hash_content(content):
    return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()

def get_last_attempt(directory):
    attempts = [f for f in os.listdir(directory) if f.startswith('attempt') and f.endswith('.json')]
    return max(attempts, key=lambda x: int(x.split('_')[0].replace('attempt', ''))) if attempts else None

def process_experiments(root_dir):
    mapping = defaultdict(lambda: defaultdict(list))
    unique_answers = {}
    answer_counts = defaultdict(int)
    answer_runs = defaultdict(set)

    for config in tqdm(os.listdir(root_dir), desc="Processing configs"):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path) or config == 'tryals_runs':
            continue

        for run in tqdm(os.listdir(config_path), desc=f"Processing runs in {config}", leave=False):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            for use_case in os.listdir(run_path):
                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                last_attempt = get_last_attempt(use_case_path)
                if not last_attempt:
                    continue

                attempt_file_path = os.path.join(use_case_path, last_attempt)
                try:
                    with open(attempt_file_path, 'r') as f:
                        data = json.load(f)
                    
                    validation_result = data.get('validation_result', {})
                    extracted_data = validation_result.get('extracted_data')
                    
                    # If extracted_data is missing or empty, use an empty list
                    if not extracted_data:
                        extracted_data = []
                    
                    # Hash the extracted_data
                    content_hash = hash_content(extracted_data)

                    relative_path = os.path.join(config, run, use_case, last_attempt)
                    mapping[config][use_case].append({
                        'run': run,
                        'file': relative_path,
                        'hash': content_hash
                    })

                    if content_hash not in unique_answers:
                        unique_answers[content_hash] = {
                            'reference_file': relative_path,
                            'validation_result': validation_result
                        }
                    
                    answer_counts[content_hash] += 1
                    answer_runs[content_hash].add((config, run))

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {attempt_file_path}")
                except Exception as e:
                    print(f"Error processing file {attempt_file_path}: {str(e)}")

    # Add count and runs information to unique_answers
    for content_hash, answer_data in unique_answers.items():
        answer_data['count'] = answer_counts[content_hash]
        answer_data['runs'] = list(answer_runs[content_hash])

    return mapping, unique_answers

def save_mapping_configuration(mapping, unique_answers, output_file):
    result = {
        'mapping': mapping,
        'unique_answers': unique_answers
    }
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    experiments_dir = "./experiments"
    output_file = "./experiments_validation_results/experiment_mapping_configuration.json"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mapping, unique_answers = process_experiments(experiments_dir)
    save_mapping_configuration(mapping, unique_answers, output_file)
    print(f"Mapping configuration saved to {output_file}")