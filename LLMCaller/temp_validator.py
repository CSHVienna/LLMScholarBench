#!/usr/bin/env python3
"""
Simple script to test temperature acceptance for OpenRouter models.
Tests 20 temperature values from 0 to 5 for each model.
"""

import asyncio
import os
import json
import argparse
import numpy as np
from datetime import datetime
from config.loader import get_available_models, load_llm_setup
from api.openrouter_api import OpenRouterAPI

# Default config file (can be overridden via command line)
DEFAULT_CONFIG_FILE = 'llm_setup_pay.json'

# Simple test prompt (short)
TEST_PROMPT = "Hi"

# Generate 5 temperature values from 0 to 2 (including extremes)
TEMPERATURES = np.linspace(0, 2, 5).tolist()

async def test_model_temperature(model_name, temperature, config_file):
    """Test a single model with a specific temperature"""
    try:
        # Load config and override temperature
        config = load_llm_setup(model_name, config_file=config_file)
        config['temperature'] = temperature
        config['max_tokens'] = 2  # Minimize token usage for testing

        # Create API client
        api_client = OpenRouterAPI(config)

        # Generate response
        api_response = await api_client.generate_response(TEST_PROMPT)

        return {
            "model": model_name,
            "temperature": temperature,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "model": model_name,
            "temperature": temperature,
            "success": False,
            "error": str(e)
        }

async def test_model_all_temperatures(model_name, output_dir, config_file):
    """Test one model with all temperature values in parallel"""
    print(f"🧪 Testing {model_name} with {len(TEMPERATURES)} temperatures in parallel...")

    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create all tasks for this model (all temperatures in parallel)
    tasks = []
    for temp in TEMPERATURES:
        task = test_model_temperature(model_name, temp, config_file)
        tasks.append(task)

    # Run all temperature tests in parallel
    print(f"   🚀 Running {len(tasks)} parallel requests...")
    results = await asyncio.gather(*tasks)

    # Save individual results immediately
    for result in results:
        temp = result['temperature']
        temp_file = os.path.join(model_dir, f"temp_{temp:.2f}.json")
        with open(temp_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Save summary for this model
    summary = {
        "model": model_name,
        "total_tests": len(TEMPERATURES),
        "successful": sum(1 for r in results if r['success']),
        "failed": sum(1 for r in results if not r['success']),
        "temperatures_tested": TEMPERATURES,
        "results": results
    }

    summary_file = os.path.join(model_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    success_count = summary['successful']
    fail_count = summary['failed']
    print(f"✅ {model_name}: {success_count} success, {fail_count} failed")

    return summary

async def main(config_file):
    """Main function to test all OpenRouter models"""
    print("🌡️  Temperature Validation Test")
    print(f"   Config file: {config_file}")
    print(f"   Testing {len(TEMPERATURES)} temperatures: {TEMPERATURES[0]:.2f} to {TEMPERATURES[-1]:.2f}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"temp_validation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {output_dir}/")
    print()

    # Get OpenRouter models only from specified config
    openrouter_models = get_available_models(provider_filter='openrouter', config_file=config_file)
    print(f"🚀 Found {len(openrouter_models)} OpenRouter models to test")
    print(f"   Models: {', '.join(openrouter_models)}")
    print()

    # Run ALL tests in parallel (all models, all temperatures)
    print(f"🚀 Running ALL tests in parallel...")
    print(f"   Total requests: {len(openrouter_models)} models × {len(TEMPERATURES)} temps = {len(openrouter_models) * len(TEMPERATURES)} requests")
    print()

    tasks = [test_model_all_temperatures(model, output_dir, config_file) for model in openrouter_models]
    all_summaries = await asyncio.gather(*tasks)

    # Save overall summary
    overall_summary = {
        "test_info": {
            "timestamp": timestamp,
            "total_models": len(openrouter_models),
            "temperatures_tested": TEMPERATURES,
            "models_tested": openrouter_models
        },
        "model_summaries": all_summaries
    }

    overall_file = os.path.join(output_dir, "overall_summary.json")
    with open(overall_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    # Final summary
    total_tests = len(openrouter_models) * len(TEMPERATURES)
    total_success = sum(s['successful'] for s in all_summaries)
    total_failed = sum(s['failed'] for s in all_summaries)

    print("🎉 Temperature validation completed!")
    print(f"   Total tests: {total_tests}")
    print(f"   Total successful: {total_success}")
    print(f"   Total failed: {total_failed}")
    print(f"   Results saved in: {output_dir}/")
    print()

    # Model-by-model summary
    print("📊 Model Summary:")
    for summary in all_summaries:
        model = summary['model']
        success = summary['successful']
        failed = summary['failed']
        total = summary['total_tests']
        success_rate = (success / total) * 100
        print(f"   {model}: {success}/{total} ({success_rate:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test temperature acceptance for OpenRouter models')
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help=f'Config file to use (default: {DEFAULT_CONFIG_FILE})'
    )
    args = parser.parse_args()

    print("🧪 Temperature Validation Script")
    print("=" * 50)
    asyncio.run(main(args.config))