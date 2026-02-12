import argparse
import os
import json
import asyncio
from experiments.runner_openrouter_async import OpenRouterAsyncRunner
from config.loader import load_llm_setup, get_available_models
from run_gemini_concurrent import run_gemini_concurrent
from datetime import datetime

# Old functions removed - use --all-models-async instead

if __name__ == "__main__":
    available_models = get_available_models()
    parser = argparse.ArgumentParser(description="Run LLM experiments")
    parser.add_argument("--all-models-async", action="store_true",
                        required=True,
                        help="Run experiments fully async (no rate limiting)")
    parser.add_argument("--output-dir", type=str, 
                        help="Override output directory (default from config)")
    parser.add_argument("--category", type=str,
                        choices=["top_k", "biased_top_k", "epoch", "field", "twins", "seniority"],
                        help="Run single category experiment")
    parser.add_argument("--variable", type=str,
                        help="Run single variable experiment (requires --category)")
    parser.add_argument("--provider", type=str,
                        choices=["openrouter", "gemini"],
                        help="Filter models by provider (openrouter or gemini)")
    parser.add_argument("--temperature", type=float,
                        help="Override temperature for all models (0.0-2.0)")

    args = parser.parse_args()

    # Validation
    if args.variable and not args.category:
        parser.error("--variable requires --category")

    # Run async experiments (always - it's the only mode now)
    models = get_available_models(provider_filter=args.provider)

    if not models:
        print(f"No models found for provider: {args.provider}")
        exit(1)

    # Route to appropriate async execution strategy
    if args.provider == 'gemini':
        print(f"Running {len(models)} Gemini models concurrently (async)")
        if args.temperature is not None:
            print(f"   Temperature override: {args.temperature}")
        asyncio.run(run_gemini_concurrent(models, args.output_dir, args.category, args.variable, args.temperature))

    elif args.provider == 'openrouter':
        print(f"🚀 Running {len(models)} OpenRouter models fully async (no rate limiting)")
        if args.temperature is not None:
            print(f"   Temperature override: {args.temperature}")
        runner = OpenRouterAsyncRunner(args.output_dir, temperature_override=args.temperature)
        asyncio.run(runner.run_all_models(models, args.category, args.variable))

    else:
        # No provider specified - run both in parallel
        openrouter_models = get_available_models(provider_filter='openrouter')
        gemini_models = get_available_models(provider_filter='gemini')

        async def run_both_providers():
            tasks = []
            if openrouter_models:
                print(f"OpenRouter: {len(openrouter_models)} models (async)")
                or_runner = OpenRouterAsyncRunner(args.output_dir, temperature_override=args.temperature)
                tasks.append(or_runner.run_all_models(openrouter_models, args.category, args.variable))

            if gemini_models:
                print(f"Gemini: {len(gemini_models)} models (async)")
                tasks.append(run_gemini_concurrent(gemini_models, args.output_dir, args.category, args.variable, args.temperature))

            return await asyncio.gather(*tasks)

        print(f"Running both providers in parallel ({len(models)} total models)")
        asyncio.run(run_both_providers())
