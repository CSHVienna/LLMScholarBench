#!/usr/bin/env python3
"""
Unified Genealogy and Biography Test Suite
==========================================

This script runs comprehensive tests for:
1. Biography details (minimal, standard, comprehensive)
2. Genealogy trees (1_down, 1_up, 1_up_1_down, all_down, all_up, all_up_all_down)

For different types of scientists:
- Famous male (alive/dead)
- Famous female (alive/dead) 
- Random male (alive/dead)
- Random female (alive/dead)

Across all available models.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from experiments.runner import ExperimentRunner
from config.loader import load_llm_setup
from env_config import get_groq_api_key

# Load environment variables
load_dotenv('../.env')

# Test subjects configuration
TEST_SUBJECTS = {
    "famous_male_alive": {
        "name": "Dr. Geoffrey Hinton",
        "description": "Famous male scientist (alive) - AI pioneer"
    },
    "famous_female_alive": {
        "name": "Dr. Jennifer Doudna", 
        "description": "Famous female scientist (alive) - CRISPR pioneer"
    },
    "random_male_alive": {
        "name": "Dr. Yann LeCun",
        "description": "Random male scientist (alive) - Deep learning researcher"
    },
    "random_female_alive": {
        "name": "Dr. Fei-Fei Li",
        "description": "Random female scientist (alive) - Computer vision researcher"
    },
    "famous_male_dead": {
        "name": "Dr. Albert Einstein",
        "description": "Famous male scientist (dead) - Theoretical physicist"
    },
    "famous_female_dead": {
        "name": "Dr. Marie Curie",
        "description": "Famous female scientist (dead) - Nobel laureate"
    },
    "random_male_dead": {
        "name": "Dr. Niels Bohr",
        "description": "Random male scientist (dead) - Quantum physicist"
    },
    "random_female_dead": {
        "name": "Dr. Rosalind Franklin",
        "description": "Random female scientist (dead) - DNA structure researcher"
    }
}

# Available models (must match exactly with config/llm_setup.json)
AVAILABLE_MODELS = [
    "llama-3.1-8b",
    "llama-3.3-70b", 
    "gemma2-9b",
    "deepseek-r1-distill-llama-70b",
    "llama3-70b",
    "llama3-8b",
    "qwen-qwq-32b"
]

# Test categories
BIOGRAPHY_VARIABLES = ["minimal_info", "standard_info", "comprehensive_info"]
GENEALOGY_VARIABLES = ["1_down", "1_up", "1_up_1_down", "all_down", "all_up", "all_up_all_down"]

class UnifiedTestRunner:
    def __init__(self, output_dir: str = "unified_test_results"):
        self.output_dir = output_dir
        self.results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the test runner."""
        # Ensure output directory exists before creating log file
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/unified_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_experiment_config(self, model_name: str, subject_key: str = None) -> Tuple[str, Dict]:
        """Create experiment configuration for a model."""
        config = load_llm_setup(model_name)
        
        # Debug: Check if config is loaded correctly
        if not config:
            self.logger.error(f"❌ No configuration found for model: {model_name}")
            raise ValueError(f"No configuration found for model: {model_name}")
        
        self.logger.info(f"✅ Configuration loaded for model: {model_name}")
        
        # Create base directory (include subject_key for organization)
        if subject_key:
            base_config_dir = f"{self.output_dir}/config_{model_name}_{subject_key}"
        else:
            base_config_dir = f"{self.output_dir}/config_{model_name}"
        os.makedirs(base_config_dir, exist_ok=True)
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_config_dir, f"unified_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir, config

    def run_biography_tests(self, model_name: str, subject_key: str, subject_info: Dict, 
                          runner: ExperimentRunner) -> Dict:
        """Run biography tests for a subject."""
        results = {}
        
        for bio_var in BIOGRAPHY_VARIABLES:
            self.logger.info(f"Running biography test: {bio_var} for {subject_info['name']}")
            
            # Set scholar name in environment
            os.environ['SCHOLAR_NAME'] = subject_info['name']
            
            # Create runtime config
            runtime_config = {'scholar_name': subject_info['name']}
            config_path = os.path.join(PROJECT_ROOT, "config", "runtime_config.json")
            with open(config_path, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            
            try:
                runner.run_specific_parameter("biography_detail", bio_var)
                results[bio_var] = "success"
                self.logger.info(f"✅ Biography {bio_var} completed for {subject_info['name']}")
            except Exception as e:
                results[bio_var] = f"error: {str(e)}"
                self.logger.error(f"❌ Biography {bio_var} failed for {subject_info['name']}: {e}")
                
        return results

    def run_genealogy_tests(self, model_name: str, subject_key: str, subject_info: Dict,
                          runner: ExperimentRunner) -> Dict:
        """Run genealogy tests for a subject."""
        results = {}
        
        for gen_var in GENEALOGY_VARIABLES:
            self.logger.info(f"Running genealogy test: {gen_var} for {subject_info['name']}")
            
            # Set scholar name in environment
            os.environ['SCHOLAR_NAME'] = subject_info['name']
            
            # Create runtime config
            runtime_config = {'scholar_name': subject_info['name']}
            config_path = os.path.join(PROJECT_ROOT, "config", "runtime_config.json")
            with open(config_path, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            
            try:
                runner.run_specific_parameter("genealogy", gen_var)
                results[gen_var] = "success"
                self.logger.info(f"✅ Genealogy {gen_var} completed for {subject_info['name']}")
            except Exception as e:
                results[gen_var] = f"error: {str(e)}"
                self.logger.error(f"❌ Genealogy {gen_var} failed for {subject_info['name']}: {e}")
                
        return results

    def run_model_tests(self, model_name: str) -> Dict:
        """Run all tests for a specific model."""
        self.logger.info(f"🚀 Starting tests for model: {model_name}")
        
        # Get API key
        api_key = get_groq_api_key()
        if not api_key:
            self.logger.error(f"❌ GROQ_API_KEY not found for model {model_name}")
            return {"error": "No API key"}
        
        model_results = {}
        
        for subject_key, subject_info in TEST_SUBJECTS.items():
            self.logger.info(f"🧪 Testing subject: {subject_info['name']} ({subject_key})")
            
            try:
                # Create experiment configuration
                run_dir, config = self.create_experiment_config(model_name, subject_key)
                
                # Initialize experiment runner
                runner = ExperimentRunner(run_dir, config, api_key, "physics")
                
                subject_results = {
                    "subject_info": subject_info,
                    "biography_tests": {},
                    "genealogy_tests": {}
                }
                
                # Run biography tests
                subject_results["biography_tests"] = self.run_biography_tests(
                    model_name, subject_key, subject_info, runner
                )
                
                # Run genealogy tests  
                subject_results["genealogy_tests"] = self.run_genealogy_tests(
                    model_name, subject_key, subject_info, runner
                )
                
                model_results[subject_key] = subject_results
                
            except Exception as e:
                self.logger.error(f"❌ Failed tests for {subject_info['name']}: {e}")
                model_results[subject_key] = {"error": str(e)}
        
        return model_results

    def run_all_tests(self, models: List[str] = None) -> Dict:
        """Run unified tests across all models and subjects."""
        if models is None:
            models = AVAILABLE_MODELS
            
        self.logger.info("🎯 Starting Unified Genealogy and Biography Test Suite")
        self.logger.info(f"📊 Testing {len(models)} models across {len(TEST_SUBJECTS)} subjects")
        self.logger.info(f"🧬 {len(BIOGRAPHY_VARIABLES)} biography + {len(GENEALOGY_VARIABLES)} genealogy tests per subject")
        
        all_results = {
            "test_metadata": {
                "start_time": datetime.now().isoformat(),
                "models_tested": models,
                "subjects_tested": list(TEST_SUBJECTS.keys()),
                "biography_variables": BIOGRAPHY_VARIABLES,
                "genealogy_variables": GENEALOGY_VARIABLES,
                "total_tests_per_model": len(TEST_SUBJECTS) * (len(BIOGRAPHY_VARIABLES) + len(GENEALOGY_VARIABLES))
            },
            "results": {}
        }
        
        for model_name in models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 TESTING MODEL: {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                model_results = self.run_model_tests(model_name)
                all_results["results"][model_name] = model_results
                
                # Calculate success rate
                total_tests = 0
                successful_tests = 0
                
                for subject_results in model_results.values():
                    if "error" not in subject_results:
                        for test_category in ["biography_tests", "genealogy_tests"]:
                            for test_result in subject_results[test_category].values():
                                total_tests += 1
                                if test_result == "success":
                                    successful_tests += 1
                
                success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
                self.logger.info(f"✅ Model {model_name} completed: {successful_tests}/{total_tests} tests successful ({success_rate:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ Model {model_name} failed completely: {e}")
                all_results["results"][model_name] = {"error": str(e)}
        
        all_results["test_metadata"]["end_time"] = datetime.now().isoformat()
        
        # Save complete results
        results_file = f"{self.output_dir}/unified_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info(f"\n🎉 UNIFIED TEST SUITE COMPLETED!")
        self.logger.info(f"📁 Results saved to: {results_file}")
        
        return all_results

    def generate_summary_report(self, results: Dict):
        """Generate a summary report of test results."""
        self.logger.info("\n📊 GENERATING SUMMARY REPORT")
        
        summary = {
            "total_models": len(results["results"]),
            "total_subjects": len(TEST_SUBJECTS),
            "model_performance": {}
        }
        
        for model_name, model_results in results["results"].items():
            if "error" in model_results:
                summary["model_performance"][model_name] = {"status": "failed", "error": model_results["error"]}
                continue
                
            total_tests = 0
            successful_tests = 0
            
            for subject_key, subject_results in model_results.items():
                if "error" not in subject_results:
                    for test_category in ["biography_tests", "genealogy_tests"]:
                        for test_result in subject_results[test_category].values():
                            total_tests += 1
                            if test_result == "success":
                                successful_tests += 1
            
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            summary["model_performance"][model_name] = {
                "status": "completed",
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": round(success_rate, 2)
            }
        
        # Save summary
        summary_file = f"{self.output_dir}/test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Summary report saved to: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("🏆 UNIFIED TEST SUITE SUMMARY")
        print("="*80)
        
        for model_name, performance in summary["model_performance"].items():
            if performance["status"] == "completed":
                print(f"🤖 {model_name}: {performance['successful_tests']}/{performance['total_tests']} tests ({performance['success_rate']}%)")
            else:
                print(f"❌ {model_name}: FAILED - {performance.get('error', 'Unknown error')}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Run unified genealogy and biography tests')
    parser.add_argument('--models', nargs='+', default=AVAILABLE_MODELS,
                       help='Models to test (default: all available)')
    parser.add_argument('--output-dir', default='unified_test_results',
                       help='Output directory for results')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with subset of models and subjects')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Quick test: only 2 models and fewer subjects for faster testing
        models = ["llama-3.1-8b", "gemma2-9b"]
        print("🚀 Running QUICK TEST mode")
    else:
        models = args.models
        print("🚀 Running FULL TEST mode")
    
    # Initialize test runner
    test_runner = UnifiedTestRunner(args.output_dir)
    
    # Run all tests
    results = test_runner.run_all_tests(models)
    
    # Generate summary report
    test_runner.generate_summary_report(results)
    
    print(f"\n✅ Test suite completed! Check {args.output_dir}/ for detailed results.")

if __name__ == "__main__":
    main() 