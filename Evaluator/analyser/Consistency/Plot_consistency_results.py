import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# Configuration
#CONFIGURATIONS = ["llama3-8b", "gemma2-9b", "mixtral-8x7b", "llama3-70b"]
BASE_PATH = "./experiments"  
OUTPUT_PATH = "./output_results/consistency/answers_consitency"



def count_unique_answers(config):
    config_path = os.path.join(BASE_PATH, f"config_{config}")
    use_case_answers = defaultdict(set)     # Store unique answers for each use case
    use_case_attempts = defaultdict(int)    # Store total number of attempts per use case
    use_case_valid_answers = defaultdict(set)  # Store valid unique answers per use case

    if not os.path.isdir(config_path):
        print(f"Config path does not exist: {config_path}")
        return {}

    for run_folder in os.listdir(config_path):
        run_path = os.path.join(config_path, run_folder)
        if not os.path.isdir(run_path):
            continue
        # Run folders start with 'run_'
        if not run_folder.startswith('run_'):
            continue

        # Iterate over use cases (category_variable folders)
        for use_case_folder in os.listdir(run_path):
            use_case_path = os.path.join(run_path, use_case_folder)
            if not os.path.isdir(use_case_path):
                continue

            use_case = use_case_folder  # [category]_[variable]

            # Get all attempt files
            attempt_files = [f for f in os.listdir(use_case_path) if f.startswith('attempt') and f.endswith('.json')]
            if not attempt_files:
                continue  # No attempt files in this use case

            # Iterate over all attempt files
            for attempt_file in attempt_files:
                attempt_file_path = os.path.join(use_case_path, attempt_file)
                # Read the JSON file
                try:
                    with open(attempt_file_path, 'r') as f:
                        data = json.load(f)
                    # Check if 'error' key exists
                    if 'error' in data:
                        print(f"Skipped attempt due to error in {attempt_file_path}")
                        continue  # Skip attempts that resulted in an error

                    use_case_attempts[use_case] += 1

                    # Extract the answer content
                    response_content = data.get("full_api_response", {}).get("choices", [{}])[0].get("message", {}).get("content", None)
                    if response_content:
                        response_content = response_content.strip()
                        # Add the answer to the set of answers for this use case
                        use_case_answers[use_case].add(response_content)

                        # Check if the answer is valid
                        validation_result = data.get("validation_result", {})
                        is_valid = validation_result.get("is_valid", False)
                        extracted_data = validation_result.get("extracted_data", None)

                        if is_valid and extracted_data:
                            # Check if extracted_data is not empty
                            if isinstance(extracted_data, dict) and extracted_data:
                                use_case_valid_answers[use_case].add(response_content)
                            elif isinstance(extracted_data, list) and extracted_data:
                                use_case_valid_answers[use_case].add(response_content)
                            else:
                                pass  # extracted_data is empty
                        else:
                            pass  # Not valid or extracted_data is None/empty
                    else:
                        print(f"No content found in {attempt_file_path}")
                except Exception as e:
                    print(f"Error reading {attempt_file_path}: {e}")

    # Prepare the result
    result = {}
    for use_case in use_case_attempts.keys():
        total_attempts = use_case_attempts[use_case]
        unique_answers = len(use_case_answers[use_case])
        valid_unique_answers = len(use_case_valid_answers[use_case])
        print(f"For use case '{use_case}' in config '{config}': {unique_answers} unique answers out of {total_attempts} attempts, {valid_unique_answers} valid unique answers")
        result[use_case] = {
            'expected_attempts': total_attempts,
            'unique_answers': unique_answers,
            'total_attempts': total_attempts,
            'valid_unique_answers': valid_unique_answers
        }
    return result

def create_unique_answer_summary(all_answer_data, models):
    # Define the new column order
    metrics = ['Total Requests', 'Unique Answers', 'Valid Unique Answers']

    for config in models:
        config_data = all_answer_data.get(config, {})
        if not config_data:
            print(f"No data available for configuration: {config}")
            continue
        
        # Get all use cases for the current config
        all_use_cases = sorted(config_data.keys())

        # Create a DataFrame for this configuration
        df = pd.DataFrame(index=all_use_cases, columns=metrics)

        # Fill in the DataFrame with the appropriate metrics
        for use_case in all_use_cases:
            use_case_data = config_data.get(use_case, {'unique_answers': 0, 'total_attempts': 0, 'valid_unique_answers': 0})
            
            # Populate the metrics
            df.loc[use_case, 'Total Requests'] = use_case_data['total_attempts']
            df.loc[use_case, 'Unique Answers'] = use_case_data['unique_answers']
            df.loc[use_case, 'Valid Unique Answers'] = use_case_data['valid_unique_answers']

        # Calculate the mean and standard deviation for each column
        avg_std_row = {}
        for column in metrics:
            avg = df[column].astype(float).mean()
            std = df[column].astype(float).std()
            avg_std_row[column] = f"{avg:.2f} ± {std:.2f}"

        # Add the row with avg ± std to the DataFrame
        df.loc['Average ± Std'] = avg_std_row

        # Set the index name
        df.index.name = 'Use Case'

        # Save each configuration DataFrame as a CSV file
        output_csv_path = os.path.join(OUTPUT_PATH, f'unique_answer_summary_{config}.csv')
        df.to_csv(output_csv_path)
        
        # For LaTeX table, apply bold formatting where valid_unique_answers == unique_answers
        df_latex = df.copy()

        # Function to apply bold formatting
        def bold_valid_equals_unique(valid_unique_answers, unique_answers):
            try:
                valid_unique_answers = int(valid_unique_answers)
                unique_answers = int(unique_answers)
            except (ValueError, TypeError) as e:
                print(f"Error converting to integer: {e}")
                return False

            if valid_unique_answers == unique_answers and unique_answers > 0:
                return True
            else:
                return False

        # Apply formatting to Valid Unique Answers columns
        for use_case in all_use_cases:
            valid_unique_answers = df_latex.loc[use_case, 'Valid Unique Answers']
            unique_answers = df_latex.loc[use_case, 'Unique Answers']
            if bold_valid_equals_unique(valid_unique_answers, unique_answers):
                df_latex.loc[use_case, 'Valid Unique Answers'] = f"\\textbf{{{valid_unique_answers}}}"

        # Save each LaTeX table
        output_latex_path = os.path.join(OUTPUT_PATH, f'unique_answer_summary_{config}.tex')
        with open(output_latex_path, 'w') as f:
            f.write(df_latex.to_latex(escape=False, multicolumn=True, multicolumn_format='c'))

        # Plot each table and save as PNG and PDF
        fig, ax = plt.subplots(figsize=(len(df.columns)*2, len(df)*0.5 + 1))
        ax.axis('off')
        table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1)
        plt.title(f"Unique Answers Summary - {config}")

        # Save as PNG and PDF
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f'unique_answer_summary_{config}.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_PATH, f'unique_answer_summary_{config}.pdf'))
        plt.close()

        # Print the DataFrame for each config
        print(df)

def create_transposed_summary(all_answer_data, models):
    # Define the metrics
    metrics = ['Total Requests', 'Unique Answers', 'Valid Unique Answers']

    # Create a DataFrame to hold the transposed summary for all configurations
    summary_df = pd.DataFrame(index=metrics, columns=models)

    for config in models:
        config_data = all_answer_data.get(config, {})
        if not config_data:
            print(f"No data available for configuration: {config}")
            continue

        # Get all use cases for the current config
        all_use_cases = sorted(config_data.keys())

        # Create a temporary DataFrame to hold the data for this config
        df = pd.DataFrame(index=all_use_cases, columns=metrics)

        # Fill in the DataFrame with the appropriate metrics
        for use_case in all_use_cases:
            use_case_data = config_data.get(use_case, {'unique_answers': 0, 'total_attempts': 0, 'valid_unique_answers': 0})
            df.loc[use_case, 'Total Requests'] = use_case_data['total_attempts']
            df.loc[use_case, 'Unique Answers'] = use_case_data['unique_answers']
            df.loc[use_case, 'Valid Unique Answers'] = use_case_data['valid_unique_answers']

        # Calculate the mean and std for each metric
        for column in df.columns:
            avg = df[column].astype(float).mean()
            std = df[column].astype(float).std()
            summary_df.loc[column, config] = f"{avg:.2f} ± {std:.2f}"

    # Save the transposed summary table as CSV
    output_csv_path = os.path.join(OUTPUT_PATH, 'transposed_unique_answer_summary.csv')
    summary_df.to_csv(output_csv_path)

    # Save as LaTeX for appendix
    output_latex_path = os.path.join(OUTPUT_PATH, 'transposed_unique_answer_summary.tex')
    with open(output_latex_path, 'w') as f:
        f.write(summary_df.to_latex(escape=False))

    # Print the transposed DataFrame
    print(summary_df)

    # Plot the summary table
    fig, ax = plt.subplots(figsize=(len(summary_df.columns)*2, len(summary_df)*0.5 + 1))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, rowLabels=summary_df.index, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)
    plt.title("Transposed Unique Answers Summary")

    # Save as PNG and PDF
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'transposed_unique_answer_summary.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_PATH, 'transposed_unique_answer_summary.pdf'))
    plt.close()

    # Return the transposed summary DataFrame
    return summary_df

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    all_answer_data = {}
    
    # Load model names from the config file
    config_file = os.path.join('../model_config.json')
    with open(config_file, 'r') as f:
        config_data = json.load(f)
        models = config_data["models"]
        
    # Generate data for each configuration
    for config in models:
        print(f"Analyzing unique answers for configuration: {config}")
        config_unique_answers = count_unique_answers(config)
        if not config_unique_answers:
            print(f"No data found for configuration: {config}")
        all_answer_data[config] = config_unique_answers

    # Create the individual tables for each configuration
    create_unique_answer_summary(all_answer_data, models)

    # Create and save the transposed summary table for all configurations
    print("Generating transposed summary table...")
    create_transposed_summary(all_answer_data, models)

    # Save raw data
    with open(os.path.join(OUTPUT_PATH, 'unique_answer_raw_data.json'), 'w') as f:
        json.dump(all_answer_data, f, indent=2)

    print(f"Unique answer analysis complete. Results saved in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
