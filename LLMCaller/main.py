#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import json
from experiments.runner import ExperimentRunner
from config.loader import load_llm_setup, get_available_disciplines, get_discipline_info
from datetime import datetime
from env_config import get_groq_api_key

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def list_available_models():
    """List available LLM models."""
    from config.loader import load_config
    try:
        llm_config = load_config(os.path.join(PROJECT_ROOT, "config", "llm_setup.json"))
        print("Available models:")
        for model in llm_config.keys():
            print(f"  - {model}")
    except Exception as e:
        print(f"Error loading model configurations: {e}")

def list_disciplines():
    """List available academic disciplines."""
    try:
        disciplines = get_available_disciplines()
        print("Available academic disciplines:")
        for discipline in disciplines:
            info = get_discipline_info(discipline)
            if info:
                print(f"  - {discipline} ({info['category_name']})")
            else:
                print(f"  - {discipline}")
    except Exception as e:
        print(f"Error loading discipline configurations: {e}")

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

def run_experiment(model_name: str, discipline: str = "physics"):
    """Run experiment with specified model and academic discipline."""
    try:
        print(f"Starting experiment with model: {model_name}, discipline: {discipline}")
        
        # Validate discipline
        available_disciplines = get_available_disciplines()
        if discipline not in available_disciplines:
            print(f"Error: '{discipline}' is not a valid discipline.")
            print("Use --list-disciplines to see available options.")
            return False
        
        # Create experiment configuration and directories
        run_dir, config = create_experiment_config(model_name)
        
        # Get API key
        api_key = get_groq_api_key()
        if not api_key:
            print("Error: GROQ_API_KEY environment variable not set.")
            return False
        
        # Initialize and run experiment
        runner = ExperimentRunner(run_dir, config, api_key, discipline)
        runner.run_experiment()
        
        print(f"Experiment completed successfully for {discipline} using {model_name}")
        return True
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        print(f"Experiment failed: {e}")
        return False

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description='Run LLM experiments across multiple academic disciplines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model deepseek-r1-distill-llama-70b                    # Run physics (default)
  python main.py --model gemma2-9b --discipline psychology               # Run psychology
  python main.py --model llama-3.3-70b --discipline computer_science     # Run computer science
  python main.py --list-models                                          # List available models
  python main.py --list-disciplines                                     # List available disciplines
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='LLM model to use for experiments'
    )
    
    parser.add_argument(
        '--discipline', '-d',
        type=str,
        default='physics',
        help='Academic discipline for the experiment (default: physics)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available LLM models'
    )
    
    parser.add_argument(
        '--list-disciplines',
        action='store_true',
        help='List all available academic disciplines'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()
    
    # Handle list commands
    if args.list_models:
        list_available_models()
        return
        
    if args.list_disciplines:
        list_disciplines()
        return
    
    # Validate required arguments
    if not args.model:
        parser.error("Model name is required. Use --list-models to see available options.")
    
    # Run the experiment
    success = run_experiment(args.model, args.discipline)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
