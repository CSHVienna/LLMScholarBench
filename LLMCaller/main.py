import argparse
import os
import json
import asyncio
from experiments.runner import ExperimentRunner
from experiments.runner_smart import SmartExperimentRunner, MultiModelSmartRunner
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

def run_experiment(model_name, output_dir=None, category=None, variable=None, use_smart_queue=False):
    run_dir, config = create_experiment_config(model_name, output_dir)
    
    if use_smart_queue:
        runner = SmartExperimentRunner(run_dir, config)
    else:
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

def run_all_models_smart(output_dir=None, category=None, variable=None):
    """Run all models using the smart queue system for optimal efficiency"""
    models = get_available_models()
    runner = MultiModelSmartRunner(output_dir)
    asyncio.run(runner.run_all_models_smart(models, category, variable))

if __name__ == "__main__":
    available_models = get_available_models()
    parser = argparse.ArgumentParser(description="Run LLM experiments")
    parser.add_argument("--model", type=str, 
                        choices=available_models,
                        help="Specify the model to use for the experiment")
    parser.add_argument("--all-models", action="store_true", 
                        help="Run experiments for all models sequentially")
    parser.add_argument("--all-models-smart", action="store_true",
                        help="Run experiments for all models using smart queue (RECOMMENDED for efficiency)")
    parser.add_argument("--smart", action="store_true",
                        help="Use smart queue system for single model (better retry handling)")
    parser.add_argument("--output-dir", type=str, 
                        help="Override output directory (default from config)")
    parser.add_argument("--category", type=str,
                        choices=["top_k", "epoch", "field", "twins", "seniority"],
                        help="Run single category experiment")
    parser.add_argument("--variable", type=str,
                        help="Run single variable experiment (requires --category)")
    
    args = parser.parse_args()
    
    # Validation
    model_options = [args.model, args.all_models, args.all_models_smart]
    if sum(bool(x) for x in model_options) != 1:
        parser.error("Exactly one of --model, --all-models, or --all-models-smart is required")
    if args.variable and not args.category:
        parser.error("--variable requires --category")
    
    if args.all_models_smart:
        print("🚀 Using smart queue system for optimal cross-model batching!")
        run_all_models_smart(args.output_dir, args.category, args.variable)
    elif args.all_models:
        print("📝 Using legacy sequential processing (consider --all-models-smart for better efficiency)")
        run_all_models(args.output_dir, args.category, args.variable)
    else:
        if args.smart:
            print("🧠 Using smart queue system for better retry handling!")
        run_experiment(args.model, args.output_dir, args.category, args.variable, args.smart)
