import argparse
import os
import json
import asyncio
from experiments.runner import ExperimentRunner
from experiments.runner_smart import SmartExperimentRunner, MultiModelSmartRunner
from config.loader import load_llm_setup, get_available_models
from run_gemini_concurrent import run_gemini_concurrent
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

def run_experiment(model_name, output_dir=None, category=None, variable=None, use_smart_queue=False, batch_size=15):
    run_dir, config = create_experiment_config(model_name, output_dir)
    
    if use_smart_queue:
        runner = SmartExperimentRunner(run_dir, config, batch_size=batch_size)
    else:
        runner = ExperimentRunner(run_dir, config, batch_size=batch_size)
    
    if category and variable:
        runner.run_single_experiment(category, variable)
    else:
        runner.run_experiment()
    print(f"Experiment completed. Results saved in {run_dir}")

def run_all_models(output_dir=None, category=None, variable=None, batch_size=15):
    models = get_available_models()
    
    for i, model in enumerate(models):
        print(f"\n=== Running experiment for {model} ({i+1}/{len(models)}) ===")
        run_experiment(model, output_dir, category, variable, False, batch_size)
        
        # Add delay between models to ensure rate limiting works properly
        if i < len(models) - 1:  # Don't sleep after the last model
            print(f"Waiting 2 seconds before next model...")
            import time
            time.sleep(2)

def run_all_models_smart(output_dir=None, category=None, variable=None, batch_size=15):
    """Run all models using the smart queue system for optimal efficiency"""
    models = get_available_models()
    runner = MultiModelSmartRunner(output_dir, batch_size=batch_size)
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
    parser.add_argument("--batch-size", type=int, default=15,
                        help="Set batch size for API calls (default: 15)")
    parser.add_argument("--provider", type=str,
                        choices=["openrouter", "gemini"],
                        help="Filter models by provider (openrouter or gemini)")
    
    args = parser.parse_args()
    
    # Validation
    model_options = [args.model, args.all_models, args.all_models_smart]
    if sum(bool(x) for x in model_options) != 1:
        parser.error("Exactly one of --model, --all-models, or --all-models-smart is required")
    if args.variable and not args.category:
        parser.error("--variable requires --category")
    
    if args.all_models_smart:
        # Get models filtered by provider
        models = get_available_models(provider_filter=args.provider)

        if not models:
            print(f"❌ No models found for provider: {args.provider}")
            exit(1)

        # Route to appropriate execution strategy
        if args.provider == 'gemini':
            print(f"🧠 Running {len(models)} Gemini models concurrently (no rate limits)")
            asyncio.run(run_gemini_concurrent(models, args.output_dir, args.category, args.variable))
        elif args.provider == 'openrouter':
            print(f"🚀 Running {len(models)} OpenRouter models with smart queue system")
            print(f"   Batch size: {args.batch_size}")
            runner = MultiModelSmartRunner(args.output_dir, batch_size=args.batch_size)
            asyncio.run(runner.run_all_models_smart(models, args.category, args.variable))
        else:
            # No provider specified - check for mixed providers
            openrouter_models = get_available_models(provider_filter='openrouter')
            gemini_models = get_available_models(provider_filter='gemini')

            if openrouter_models and gemini_models:
                print("⚠️  Mixed providers detected! For optimal performance, use --provider to run them separately:")
                print(f"   --provider openrouter  ({len(openrouter_models)} models)")
                print(f"   --provider gemini      ({len(gemini_models)} models)")
                print("   Running all with OpenRouter strategy (suboptimal for Gemini)...")

            print(f"🚀 Using smart queue system for {len(models)} models")
            print(f"   Batch size: {args.batch_size}")
            runner = MultiModelSmartRunner(args.output_dir, batch_size=args.batch_size)
            asyncio.run(runner.run_all_models_smart(models, args.category, args.variable))
    elif args.all_models:
        print("📝 Using legacy sequential processing (consider --all-models-smart for better efficiency)")
        print(f"   Batch size: {args.batch_size}")
        run_all_models(args.output_dir, args.category, args.variable, args.batch_size)
    else:
        if args.smart:
            print("🧠 Using smart queue system for better retry handling!")
            print(f"   Batch size: {args.batch_size}")
        run_experiment(args.model, args.output_dir, args.category, args.variable, args.smart, args.batch_size)
