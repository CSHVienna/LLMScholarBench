#!/usr/bin/env python3
"""
Test version - just one model with a few temperature values
"""

import asyncio
import os
import json
from datetime import datetime
from config.loader import get_available_models, load_llm_setup
from api.openrouter_api import OpenRouterAPI

# Simple test prompt
TEST_PROMPT = "Hi"

# Just test 5 temperature values for quick test
TEMPERATURES = [0, 1, 2, 3, 4]

async def test_model_temperature(model_name, temperature):
    """Test a single model with a specific temperature"""
    try:
        config = load_llm_setup(model_name)
        config['temperature'] = temperature
        api_client = OpenRouterAPI(config)
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

async def main():
    print("🧪 Quick Temperature Test")

    # Test just first OpenRouter model
    models = get_available_models(provider_filter='openrouter')
    test_model = models[0]
    print(f"Testing model: {test_model}")
    print(f"Testing temperatures: {TEMPERATURES}")

    for temp in TEMPERATURES:
        print(f"Testing temp {temp}...")
        result = await test_model_temperature(test_model, temp)
        if result['success']:
            print(f"  ✅ Success")
        else:
            print(f"  ❌ Failed: {result['error']}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())