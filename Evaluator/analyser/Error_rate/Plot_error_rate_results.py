import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Set the main directory path
main_dir = "./experiments"
output_dir = "././output_results/error_rate"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize lists to store results
expected_requests = []
total_requests_formatted = []
error_rate_formatted = []


# Load model names from the config file
config_file = os.path.join('../model_config.json')
with open(config_file, 'r') as f:
    config_data = json.load(f)
    models = config_data["models"]


# Iterate through the model folders
for model_name in models:
    config_dir = os.path.join(main_dir, f"config_{model_name}")

    if not os.path.exists(config_dir):
        # If the directory does not exist, append NaN values to keep lists the same length
        expected_requests.append(np.nan)
        total_requests_formatted.append("NaN")
        error_rate_formatted.append("NaN")
        continue

    run_folders = [f for f in os.listdir(config_dir) if f.startswith('run_')]
    expected_requests.append(len(run_folders))  # Expected request per task

    total_requests_sum = 0
    error_rates = []

    # Iterate through each run folder
    for run_folder in run_folders:
        run_path = os.path.join(config_dir, run_folder)
        if not os.path.isdir(run_path):
            continue
        
        use_cases = [uc for uc in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, uc))]

        use_case_attempts = []
        errors_in_run = 0

        # Iterate through each use case
        for use_case in use_cases:
            use_case_path = os.path.join(run_path, use_case)

            # Get all attempt files
            attempt_files = [f for f in os.listdir(use_case_path) if f.startswith('attempt') and f.endswith('.json')]
            if not attempt_files:
                continue  # No attempt files in this use case

            # Sort attempt files by number and pick the highest one
            attempt_files_sorted = sorted(attempt_files, key=lambda x: int(x.split('_')[0][7:]), reverse=True)
            latest_attempt_file = attempt_files_sorted[0]
            latest_attempt_path = os.path.join(use_case_path, latest_attempt_file)

            # Track number of attempts per use case
            use_case_attempts.append(len(attempt_files))

            # Open the latest attempt and check if it's valid
            try:
                with open(latest_attempt_path, 'r') as f:
                    data = json.load(f)

                # Check if the latest attempt is valid
                validation_result = data.get("validation_result", {})
                is_valid = validation_result.get("is_valid", False)
                extracted_data = validation_result.get("extracted_data", None)

                # Error counting: 1 error if invalid or empty extracted_data
                if not is_valid or (isinstance(extracted_data, dict) and not extracted_data) or (isinstance(extracted_data, list) and not extracted_data):
                    errors_in_run += 1

            except Exception as e:
                print(f"Error reading {latest_attempt_path}: {e}")
                errors_in_run += 1

        # Normalize the number of attempts by the number of use cases in that run
        if use_case_attempts:
            total_attempts_in_run = sum(use_case_attempts)  # Total attempts across all use cases
            total_requests_sum += total_attempts_in_run / len(use_cases)  # Normalized by the number of use cases

        # Calculate error rate for this run
        if len(use_cases) > 0:
            error_rate_for_run = errors_in_run / len(use_cases)  # Error rate between 0 and 1
            error_rates.append(error_rate_for_run)

    # Sum of total requests across all runs (normalized)
    total_requests_formatted.append(f"{total_requests_sum:.2f}")

    # Calculate mean and std for error rate across all runs
    if error_rates:
        error_mean = np.mean(error_rates)
        error_std = np.std(error_rates)
        error_rate_formatted.append(f"{error_mean:.2f} ± {error_std:.2f}")
    else:
        error_rate_formatted.append("NaN")

# Create DataFrame with models as columns and formatted values
results_df = pd.DataFrame({
    'Expected Request Per Task': expected_requests,
    'Total Request': total_requests_formatted,
    'Error Rate': error_rate_formatted
}, index=models).transpose()

# Save as LaTeX
latex_file = os.path.join(output_dir, 'error_rate_results.tex')
with open(latex_file, 'w') as f:
    f.write(results_df.to_latex(index=True))

# Plot the table and save as PNG
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size if necessary
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

png_file = os.path.join(output_dir, 'error_rate_results.png')
plt.savefig(png_file, bbox_inches='tight', dpi=300)

print(f"Error rate results saved in {output_dir} as LaTeX and PNG.")
