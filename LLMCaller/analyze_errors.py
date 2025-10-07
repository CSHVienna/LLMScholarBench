#!/usr/bin/env python3
"""
Analyze all errors from experiment results across all models and conditions.
Categorizes errors by type and counts occurrences to understand API vs validation failures.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

class ErrorAnalyzer:
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = experiments_dir
        self.error_stats = defaultdict(lambda: defaultdict(int))
        self.detailed_errors = defaultdict(list)

    def analyze_attempt_file(self, attempt_path, model_name):
        """Analyze a single attempt JSON file"""
        try:
            with open(attempt_path, 'r') as f:
                exp = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to read {attempt_path}: {e}")
            return

        validation = exp.get('validation_result', {})
        is_valid = validation.get('is_valid', False)

        if not is_valid:
            # Categorize the error
            error_type = self._categorize_error(exp)

            # Count it
            self.error_stats[error_type]['total'] += 1
            self.error_stats[error_type][model_name] += 1

            # Store details for analysis
            self.detailed_errors[error_type].append({
                'model': model_name,
                'category': exp.get('category'),
                'variable': exp.get('variable'),
                'attempt': exp.get('attempt'),
                'message': validation.get('message'),
                'api_response_present': 'full_api_response' in exp and 'choices' in exp.get('full_api_response', {}),
                'error_details': exp.get('error', {}),
                'processing_error': exp.get('processing_error', {}),
                'file': str(attempt_path)
            })

    def _categorize_error(self, exp):
        """Categorize error into specific types"""
        validation = exp.get('validation_result', {})
        message = validation.get('message', '')

        # Check for API-level errors (no response from API)
        api_response = exp.get('full_api_response', {})

        # JSONDecodeError - API returned non-JSON
        if 'error_from_exception' in api_response:
            error_msg = api_response.get('error_from_exception', '')
            if 'Expecting value:' in error_msg or 'JSONDecodeError' in api_response.get('exception_type', ''):
                return 'JSONDecodeError (API returned non-JSON)'

        # Processing errors (API succeeded but processing failed)
        if 'processing_error' in exp:
            proc_error = exp.get('processing_error', {})
            error_type = proc_error.get('error_type', 'Unknown')
            return f'Processing Error: {error_type}'

        # API errors (error object in response)
        if 'error' in exp:
            error_obj = exp.get('error', {})
            error_type = error_obj.get('error_type', 'Unknown')
            error_msg = error_obj.get('message', '')

            # Rate limiting
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                return 'Rate Limit (429)'

            # Other API errors
            return f'API Error: {error_type}'

        # Validation errors (response received but invalid format)
        if 'Schema validation failed' in message:
            return 'Schema Validation Failed (response has wrong format)'

        if 'No JSON-like structure found' in message:
            return 'No JSON Found (response is not JSON)'

        if 'Invalid JSON format' in message:
            return 'Invalid JSON Format'

        # Other validation issues
        if message:
            return f'Validation Error: {message[:50]}'

        return 'Unknown Error Type'

    def scan_all_experiments(self):
        """Scan all config_* directories for attempt JSON files"""
        experiments_path = Path(self.experiments_dir)

        # Find all config_* directories
        config_dirs = [d for d in experiments_path.iterdir() if d.is_dir() and d.name.startswith('config_')]

        print(f"🔍 Scanning {len(config_dirs)} model directories...")

        total_files = 0
        for config_dir in config_dirs:
            model_name = config_dir.name.replace('config_', '')

            # Find all run_* directories
            run_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]

            for run_dir in run_dirs:
                # Find all attempt JSON files
                for attempt_file in run_dir.rglob('attempt*.json'):
                    self.analyze_attempt_file(attempt_file, model_name)
                    total_files += 1

        print(f"✅ Scan complete! Analyzed {total_files} attempt files\n")

    def print_report(self):
        """Print comprehensive error analysis report"""
        print("=" * 80)
        print("ERROR ANALYSIS REPORT")
        print("=" * 80)
        print()

        # Sort errors by frequency
        sorted_errors = sorted(self.error_stats.items(), key=lambda x: x[1]['total'], reverse=True)

        total_errors = sum(stats['total'] for _, stats in sorted_errors)

        print(f"📊 TOTAL ERRORS FOUND: {total_errors}")
        print()
        print("=" * 80)
        print("ERROR TYPES (sorted by frequency)")
        print("=" * 80)
        print()

        for error_type, stats in sorted_errors:
            count = stats['total']
            percentage = (count / total_errors * 100) if total_errors > 0 else 0

            print(f"🔴 {error_type}")
            print(f"   Count: {count} ({percentage:.1f}% of all errors)")

            # Check if this is an API error (no actual response)
            sample_details = self.detailed_errors[error_type][:5]
            api_response_count = sum(1 for d in self.detailed_errors[error_type] if d['api_response_present'])
            no_response_count = len(self.detailed_errors[error_type]) - api_response_count

            if no_response_count > 0:
                print(f"   ⚠️  NO API RESPONSE: {no_response_count}/{count} (YOU ARE LIKELY PAYING FOR NOTHING)")
            if api_response_count > 0:
                print(f"   ✅ Has API response: {api_response_count}/{count} (response exists but invalid)")

            # Top affected models
            model_counts = {k: v for k, v in stats.items() if k != 'total'}
            if model_counts:
                top_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   Top affected models: {', '.join(f'{m}({c})' for m, c in top_models)}")

            # Show sample files for investigation
            if sample_details:
                print(f"   Sample files:")
                for detail in sample_details[:3]:
                    print(f"     - {detail['file']}")

            print()

        print("=" * 80)
        print("BILLING IMPACT ANALYSIS")
        print("=" * 80)
        print()

        # Categorize errors by billing impact
        api_failed_errors = []
        api_succeeded_errors = []

        for error_type, details in self.detailed_errors.items():
            no_response = sum(1 for d in details if not d['api_response_present'])
            has_response = sum(1 for d in details if d['api_response_present'])

            if no_response > 0:
                api_failed_errors.append((error_type, no_response))
            if has_response > 0:
                api_succeeded_errors.append((error_type, has_response))

        total_api_failed = sum(count for _, count in api_failed_errors)
        total_api_succeeded = sum(count for _, count in api_succeeded_errors)

        print(f"🚨 LIKELY NOT BILLED (API call failed): {total_api_failed}")
        for error_type, count in sorted(api_failed_errors, key=lambda x: x[1], reverse=True):
            print(f"   - {error_type}: {count}")

        print()
        print(f"💰 LIKELY BILLED (API succeeded but response invalid): {total_api_succeeded}")
        for error_type, count in sorted(api_succeeded_errors, key=lambda x: x[1], reverse=True):
            print(f"   - {error_type}: {count}")

        print()
        print("=" * 80)

    def export_detailed_report(self, output_file="error_analysis_detailed.json"):
        """Export detailed error information to JSON"""
        report = {
            'summary': {
                error_type: {
                    'total': stats['total'],
                    'by_model': {k: v for k, v in stats.items() if k != 'total'}
                }
                for error_type, stats in self.error_stats.items()
            },
            'detailed_errors': {
                error_type: details
                for error_type, details in self.detailed_errors.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Detailed report exported to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze errors from LLM experiments")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Directory containing experiment results")
    parser.add_argument("--export", type=str,
                        help="Export detailed report to JSON file")

    args = parser.parse_args()

    analyzer = ErrorAnalyzer(args.experiments_dir)
    analyzer.scan_all_experiments()
    analyzer.print_report()

    if args.export:
        analyzer.export_detailed_report(args.export)
