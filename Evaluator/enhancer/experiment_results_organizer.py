import os
import json
import shutil
from tqdm import tqdm

def load_mapping_configuration(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_validation_results(mapping, unique_answers, source_dir, target_dir):
    for config in tqdm(mapping.keys(), desc="Processing configs"):
        for use_case, runs in tqdm(mapping[config].items(), desc=f"Processing use cases in {config}", leave=False):
            for run_info in runs:
                source_file = os.path.join(source_dir, run_info['file'])
                target_path = os.path.join(target_dir, config, run_info['run'], use_case)
                os.makedirs(target_path, exist_ok=True)

                content_hash = run_info['hash']
                target_file = os.path.join(target_path, f"validation_result_{content_hash}.json")

                if os.path.exists(source_file):
                    with open(source_file, 'r') as f:
                        data = json.load(f)
                    
                    result = {
                        'validation_result': unique_answers[content_hash]['validation_result'],
                        'content_hash': content_hash,
                        'original_file': run_info['file'],
                        'reference_file': unique_answers[content_hash]['reference_file'],
                        'count': unique_answers[content_hash]['count'],
                        'runs': unique_answers[content_hash]['runs']
                    }

                    with open(target_file, 'w') as f:
                        json.dump(result, f, indent=2)
                else:
                    print(f"Warning: Source file not found - {source_file}")

if __name__ == "__main__":
    mapping_file = "./experiments_validation_results/experiment_mapping_configuration.json"
    source_dir = "./experiments"
    target_dir = "./experiments_validation_results/organized_results"

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    config = load_mapping_configuration(mapping_file)
    create_validation_results(config['mapping'], config['unique_answers'], source_dir, target_dir)
    
    # Copy the mapping configuration to the new directory
    shutil.copy(mapping_file, os.path.join(target_dir, os.path.basename(mapping_file)))

    print(f"Validation results organized in {target_dir}")