import argparse
import os
import json
from experiments.runner import ExperimentRunner
from config.loader import load_llm_setup
from datetime import datetime

def create_experiment_config(model_name):
    config = load_llm_setup(model_name)
    
    # Create a base directory for this model configuration if it doesn't exist
    base_config_dir = f"experiments/config_{model_name}"
    os.makedirs(base_config_dir, exist_ok=True)
    
    # Copy the configuration file to the base directory
    config_file_path = os.path.join(base_config_dir, "llm_setup.json")
    if not os.path.exists(config_file_path):
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create a new directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_config_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir, config

def run_experiment(model_name):
    run_dir, config = create_experiment_config(model_name)
    runner = ExperimentRunner(run_dir, config)
    runner.run_experiment()
    print(f"Experiment completed. Results saved in {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM experiments")
    parser.add_argument("--model", type=str, required=True, choices=["gemma2-9b", "llama3-70b", "llama3-8b", "mixtral-8x7b", "llama-3.1-8b", "llama-3.1-70b"],
                        help="Specify the model to use for the experiment")
    args = parser.parse_args()

    run_experiment(args.model)
