import json
import os
from collections import defaultdict

def analyze_experiment(run_dir, discipline):
    summary_file = os.path.join(run_dir, 'experiment_summary.json')
    if not os.path.exists(summary_file):
        return None
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    stats = {
        'discipline': discipline,
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'success_rate': 0,
        'category_stats': defaultdict(lambda: {'total': 0, 'success': 0}),
        'failed_categories': []
    }
    
    for category, variables in data.items():
        for variable, info in variables.items():
            stats['total_tests'] += 1
            stats['category_stats'][category]['total'] += 1
            
            # Check if any attempt was successful
            attempts = info.get('attempts', [])
            success = any(attempt.get('is_valid', False) for attempt in attempts)
            
            if success:
                stats['successful_tests'] += 1
                stats['category_stats'][category]['success'] += 1
            else:
                stats['failed_tests'] += 1
                stats['failed_categories'].append(f'{category}:{variable}')
    
    if stats['total_tests'] > 0:
        stats['success_rate'] = (stats['successful_tests'] / stats['total_tests']) * 100
    
    return stats

# Analyze the three experiments
experiments = [
    ('experiments/config_deepseek-r1-distill-llama-70b/run_20250603_190339', 'neuroscience'),
    ('experiments/config_deepseek-r1-distill-llama-70b/run_20250603_190739', 'materials_science'),
    ('experiments/config_deepseek-r1-distill-llama-70b/run_20250603_191440', 'decision_sciences')
]

print('🧠 DEEPSEEK MODEL PERFORMANCE ON NEW FIELDS')
print('=' * 60)

for run_dir, discipline in experiments:
    stats = analyze_experiment(run_dir, discipline)
    if stats:
        print(f'\n📊 {discipline.upper().replace("_", " ")}:')
        print(f'   Total Tests: {stats["total_tests"]}')
        print(f'   Successful: {stats["successful_tests"]}')
        print(f'   Failed: {stats["failed_tests"]}')
        print(f'   Success Rate: {stats["success_rate"]:.1f}%')
        
        print(f'   Category Breakdown:')
        for cat, cat_stats in stats['category_stats'].items():
            success_rate = (cat_stats['success'] / cat_stats['total']) * 100 if cat_stats['total'] > 0 else 0
            print(f'     {cat}: {cat_stats["success"]}/{cat_stats["total"]} ({success_rate:.1f}%)')
        
        if stats['failed_categories']:
            print(f'   Failed Tests: {stats["failed_categories"]}')

print(f'\n📈 OVERALL SUMMARY:')
total_tests = sum(analyze_experiment(run_dir, discipline)['total_tests'] for run_dir, discipline in experiments)
total_success = sum(analyze_experiment(run_dir, discipline)['successful_tests'] for run_dir, discipline in experiments)
overall_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
print(f'   Combined Success Rate: {total_success}/{total_tests} ({overall_rate:.1f}%)') 