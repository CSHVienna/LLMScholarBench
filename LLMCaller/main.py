import argparse
import os
import json
from experiments.runner import ExperimentRunner
from config.loader import load_llm_setup, get_available_models
from datetime import datetime

def create_experiment_config(model_name, output_dir=None):
    config = load_llm_setup(model_name)
    
    # Get output directory from config or override
    if output_dir is None:
        output_dir = config.get('global', {}).get('output_dir', 'experiments')
    
    # Create a base directory for this model configuration if it doesn't exist
    base_config_dir = os.path.join(output_dir, f"config_{model_name}")
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

def run_experiment(model_name, output_dir=None, category=None, variable=None):
    run_dir, config = create_experiment_config(model_name, output_dir)
    runner = ExperimentRunner(run_dir, config)
    if category and variable:
        runner.run_single_experiment(category, variable)
    else:
        runner.run_experiment()
    print(f"Experiment completed. Results saved in {run_dir}")

def run_all_models(output_dir=None, category=None, variable=None):
    models = get_available_models()
    
    for i, model in enumerate(models):
        print(f"\n=== Running experiment for {model} ({i+1}/{len(models)}) ===")
        run_experiment(model, output_dir, category, variable)
        
        # Add delay between models to ensure rate limiting works properly
        if i < len(models) - 1:  # Don't sleep after the last model
            print(f"Waiting 2 seconds before next model...")
            import time
            time.sleep(2)

if __name__ == "__main__":
    available_models = get_available_models()
    parser = argparse.ArgumentParser(description="Run LLM experiments")
    parser.add_argument("--model", type=str, 
                        choices=available_models,
                        help="Specify the model to use for the experiment")
    parser.add_argument("--all-models", action="store_true", 
                        help="Run experiments for all models sequentially")
    parser.add_argument("--output-dir", type=str, 
                        help="Override output directory (default from config)")
    parser.add_argument("--category", type=str,
                        choices=["top_k", "epoch", "field", "twins", "seniority"],
                        help="Run single category experiment")
    parser.add_argument("--variable", type=str,
                        help="Run single variable experiment (requires --category)")
    
    args = parser.parse_args()
    
    # Validation
    if not args.model and not args.all_models:
        parser.error("Either --model or --all-models is required")
    if args.model and args.all_models:
        parser.error("Cannot specify both --model and --all-models")
    if args.variable and not args.category:
        parser.error("--variable requires --category")
    
    if args.all_models:
        run_all_models(args.output_dir, args.category, args.variable)
    else:
        run_experiment(args.model, args.output_dir, args.category, args.variable)
