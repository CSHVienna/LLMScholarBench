#!/usr/bin/env python3

import os
import json
from collections import defaultdict, Counter

def analyze_experiment_results():
    """Quick analysis of the current LLM experiment results"""
    
    print("BiasLLM Experiment Results Analysis")
    print("="*50)
    
    # Analyze the latest experiment run
    experiment_dir = "experiments/config_llama-3.1-8b/run_20250530_211251"
    
    if not os.path.exists(experiment_dir):
        print(" No experiment results found")
        return
    
    # Count categories and success rates
    categories = defaultdict(list)
    total_experiments = 0
    successful_experiments = 0
    validation_results = []
    
    for item in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, item)) and item not in ['__pycache__']:
            total_experiments += 1
            
            # Find the latest attempt
            item_path = os.path.join(experiment_dir, item)
            attempts = [f for f in os.listdir(item_path) if f.startswith('attempt') and f.endswith('.json')]
            
            if attempts:
                latest_attempt = max(attempts, key=lambda x: int(x.split('_')[0].replace('attempt', '')))
                attempt_path = os.path.join(item_path, latest_attempt)
                
                try:
                    with open(attempt_path, 'r') as f:
                        data = json.load(f)
                    
                    validation = data.get('validation_result', {})
                    is_valid = validation.get('is_valid', False)
                    extracted_data = validation.get('extracted_data', [])
                    
                    if is_valid and extracted_data:
                        successful_experiments += 1
                        
                    # Parse category from folder name
                    if '_' in item:
                        category = item.split('_')[0]
                        variable = '_'.join(item.split('_')[1:])
                        categories[category].append({
                            'variable': variable,
                            'success': is_valid and bool(extracted_data),
                            'num_scientists': len(extracted_data) if isinstance(extracted_data, list) else 0
                        })
                    
                except Exception as e:
                    print(f"⚠️  Error reading {attempt_path}: {e}")
    
    # Print summary
    print(f"📊 Overall Results:")
    print(f"   Total Experiments: {total_experiments}")
    print(f"   Successful: {successful_experiments}")
    print(f"   Success Rate: {successful_experiments/total_experiments*100:.1f}%")
    print()
    
    # Category breakdown
    print(f"📈 Results by Category:")
    for category, results in categories.items():
        successful = sum(1 for r in results if r['success'])
        total_scientists = sum(r['num_scientists'] for r in results if r['success'])
        avg_scientists = total_scientists / successful if successful > 0 else 0
        
        print(f"   {category.upper()}:")
        print(f"      Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        if successful > 0:
            print(f"      Avg Scientists per Response: {avg_scientists:.1f}")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            scientists_info = f" ({result['num_scientists']} scientists)" if result['success'] else ""
            print(f"         {status} {result['variable']}{scientists_info}")
        print()
    
    # Bias insights
    print(f"🎯 Bias Analysis Insights:")
    if 'twins' in categories:
        twins_results = categories['twins']
        male_results = [r for r in twins_results if 'male' in r['variable']]
        female_results = [r for r in twins_results if 'female' in r['variable']]
        
        male_success = sum(1 for r in male_results if r['success'])
        female_success = sum(1 for r in female_results if r['success'])
        
        if len(male_results) > 0 and len(female_results) > 0:
            male_rate = male_success / len(male_results) * 100
            female_rate = female_success / len(female_results) * 100
            print(f"   Gender Bias Check (Twins):")
            print(f"      Male variants success: {male_rate:.1f}%")
            print(f"      Female variants success: {female_rate:.1f}%")
            
            if abs(male_rate - female_rate) > 10:
                print(f"      ⚠️  Potential gender bias detected!")
            else:
                print(f"      ✅ No significant gender bias in success rates")
    
    # Sample successful result
    if successful_experiments > 0:
        print(f"\n📝 Sample Successful Result:")
        for category, results in categories.items():
            for result in results:
                if result['success']:
                    print(f"   Category: {category}")
                    print(f"   Variable: {result['variable']}")
                    print(f"   Scientists returned: {result['num_scientists']}")
                    break
            if any(r['success'] for r in results):
                break

if __name__ == "__main__":
    analyze_experiment_results() 