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

def run_scholar_search(model_name: str, scholar_name: str, affiliation: str = None):
    """Run scholar search with specified model for a specific scholar."""
    try:
        print(f"Searching for scholar: {scholar_name} using model: {model_name}")
        if affiliation:
            print(f"Affiliation: {affiliation}")

        # Create experiment configuration and directories
        run_dir, config = create_experiment_config(model_name)
        
        # Get API key
        api_key = get_groq_api_key()
        if not api_key:
            print("Error: GROQ_API_KEY environment variable not set.")
            return False
        
        # Save scholar name and affiliation to runtime config
        runtime_config = {"scholar_name": scholar_name}
        if affiliation:
            runtime_config["affiliation"] = affiliation

        config_dir = os.path.join(PROJECT_ROOT, "config")
        runtime_config_path = os.path.join(config_dir, "runtime_config.json")
        
        with open(runtime_config_path, 'w') as f:
            json.dump(runtime_config, f, indent=2)
        
        # Initialize and run experiment
        runner = ExperimentRunner(run_dir, config, api_key, discipline="all")
        
        # Run the specific category and variable for scholar search
        category = "scholar_search"
        variable = "specific_scholar"
        
        # Run the specific parameter
        runner.run_specific_parameter(category, variable)
        
        print(f"Scholar search completed successfully for {scholar_name} using {model_name}")
        return True
        
    except Exception as e:
        logging.error(f"Scholar search failed: {e}")
        print(f"Scholar search failed: {e}")
        return False

def run_genealogy_search(model_name: str, scholar_name: str, discipline: str = "physics"):
    """Run genealogy/family tree search with specified model."""
    try:
        print(f"Starting genealogy search for: {scholar_name}")
        print(f"Discipline: {discipline}")
        
        # Create experiment configuration and directories
        run_dir, config = create_experiment_config(model_name)
        
        # Get API key
        api_key = get_groq_api_key()
        if not api_key:
            print("Error: GROQ_API_KEY environment variable not set.")
            return False
        
        # Create runtime config for genealogy search
        runtime_config = {
            'scholar_name': scholar_name
        }
        
        # Save runtime config
        config_path = os.path.join(PROJECT_ROOT, "config", "runtime_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(runtime_config, f, indent=2)
        
        # Set environment variable for backward compatibility
        os.environ['SCHOLAR_NAME'] = scholar_name
        
        # Initialize and run experiment
        runner = ExperimentRunner(run_dir, config, api_key, discipline)
        runner.run_specific_category("genealogy")
        
        print(f"Genealogy search completed successfully for {scholar_name}")
        return True
        
    except Exception as e:
        logging.error(f"Genealogy search failed: {e}")
        print(f"Genealogy search failed: {e}")
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
  python main.py --model llama-3.3-70b --scholar "Dr. Albert Einstein"   # Search for scholar info
  python main.py --model llama-3.3-70b --genealogy "Dr. Albert Einstein" # Search genealogy tree
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
        '--scholar', '-s',
        type=str,
        help='Name of scholar to search for comprehensive information'
    )
    
    parser.add_argument(
        '--affiliation', '-a',
        type=str,
        help='Affiliation of the scholar to narrow down the search (e.g., university)'
    )
    
    parser.add_argument(
        '--genealogy', '-g',
        type=str,
        help='Name of scholar to search for genealogy/family tree information'
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
    
    # Run the experiment, scholar search, or genealogy search
    if args.scholar:
        success = run_scholar_search(args.model, args.scholar, args.affiliation)
    elif args.genealogy:
        success = run_genealogy_search(args.model, args.genealogy, args.discipline)
    else:
        success = run_experiment(args.model, args.discipline)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
