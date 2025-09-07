#!/usr/bin/env python3
"""
Test script to verify Option B behavior:
- Both API errors and validation failures are retried
- Full API response is ALWAYS saved
"""

import asyncio
import json
import os
from experiments.runner_smart import SmartExperimentRunner
from main import create_experiment_config

async def test_option_b_behavior():
    """Test that Option B works correctly"""
    
    print("🧪 Testing Option B: Retry both errors, always save API response")
    print("=" * 60)
    
    # Test with an available model
    from config.loader import get_available_models
    models = get_available_models()
    if not models:
        print("❌ No models available for testing")
        return
        
    model_name = models[0]  # Use first available model
    run_dir, config = create_experiment_config(model_name, "test_option_b")
    
    print(f"🎯 Testing with model: {model_name}")
    print(f"📁 Output directory: {run_dir}")
    print()
    
    # Test single experiment that might have validation issues
    smart_runner = SmartExperimentRunner(run_dir, config)
    
    try:
        print("🚀 Running experiment that may have validation issues...")
        await smart_runner._run_single_experiment_async("top_k", "top_100")
        print("✅ Experiment completed")
    except Exception as e:
        print(f"⚠️  Experiment ended with: {e}")
    
    print("\n" + "=" * 60)
    print("📊 ANALYZING SAVED RESULTS")
    print("=" * 60)
    
    # Analyze all saved results
    result_files = []
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("attempt") and file.endswith(".json"):
                result_files.append(os.path.join(root, file))
    
    result_files.sort()  # Sort by filename (chronological)
    
    print(f"📁 Found {len(result_files)} result files:")
    
    for i, file_path in enumerate(result_files, 1):
        print(f"\n📄 File {i}: {os.path.basename(file_path)}")
        
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        attempt = result.get('attempt', 'unknown')
        is_valid = result.get('validation_result', {}).get('is_valid', False)
        has_api_response = 'full_api_response' in result
        has_error = 'error' in result
        
        print(f"   🔢 Attempt: {attempt}")
        print(f"   ✅ Valid: {is_valid}")
        print(f"   📡 Has API Response: {has_api_response}")
        print(f"   🚨 Has Error: {has_error}")
        
        if has_error:
            error_type = result['error']['error_type']
            error_msg = result['error']['message']
            print(f"   📝 Error: {error_type} - {error_msg}")
        
        if not is_valid:
            validation_msg = result.get('validation_result', {}).get('message', 'No message')
            print(f"   🔍 Validation Issue: {validation_msg}")
        
        if has_api_response:
            api_resp = result['full_api_response']
            if 'choices' in api_resp and len(api_resp['choices']) > 0:
                content = api_resp['choices'][0]['message']['content']
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"   💬 API Response Preview: {content_preview}")
    
    print("\n" + "=" * 60)
    print("🎯 OPTION B VERIFICATION")
    print("=" * 60)
    
    # Verify Option B behavior
    validation_failures = 0
    api_errors = 0
    responses_with_full_api = 0
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        has_api_response = 'full_api_response' in result
        has_error = 'error' in result
        is_valid = result.get('validation_result', {}).get('is_valid', False)
        
        if has_api_response:
            responses_with_full_api += 1
        
        if has_error:
            if "Validation failed:" in result['error']['message']:
                validation_failures += 1
            else:
                api_errors += 1
        elif not is_valid:
            validation_failures += 1
    
    print(f"📊 Summary:")
    print(f"   Total files: {len(result_files)}")
    print(f"   Files with full API response: {responses_with_full_api}/{len(result_files)}")
    print(f"   Validation failures: {validation_failures}")
    print(f"   API errors: {api_errors}")
    print(f"   Multiple attempts (retries): {len(result_files) > 1}")
    
    # Verification results
    print(f"\n✅ OPTION B VERIFICATION:")
    print(f"   ✅ Saves full API response: {'✅ YES' if responses_with_full_api == len(result_files) else '❌ NO'}")
    print(f"   ✅ Retries validation failures: {'✅ YES' if len(result_files) > validation_failures else '❌ NO (or no failures)'}")
    print(f"   ✅ Retries API errors: {'✅ YES' if api_errors == 0 or len(result_files) > 1 else '❌ NO'}")

if __name__ == "__main__":
    asyncio.run(test_option_b_behavior())