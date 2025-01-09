import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# Function to load JSON data (standard JSON)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to compute Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1  # If both sets are empty, similarity is considered 1
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to compute Jaccard average and std for a list of sets
def compute_jaccard_for_runs(all_author_sets):
    """
    Compute the Jaccard similarity for all combinations of sets in all_author_sets.
    Return the average, standard deviation, and the list of Jaccard values.
    """
    if len(all_author_sets) < 2:
        return "--", "--", []  # Not enough data to compute Jaccard

    jaccard_values = []
    for set1, set2 in combinations(all_author_sets, 2):
        jaccard_values.append(jaccard_similarity(set1, set2))

    avg_jaccard = np.mean(jaccard_values)
    std_jaccard = np.std(jaccard_values)
    return f"{avg_jaccard:.2f}", f"{std_jaccard:.2f}", jaccard_values

# Function to process data and gather author information
def process_data(root_dir):
    results = {}
    for config in os.listdir(root_dir):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path):
            continue

        # Initialize dictionary to store use case results for each config
        use_case_summary = {}

        for run in os.listdir(config_path):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            for use_case in os.listdir(run_path):
                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                unique_to_total_ratios = []
                all_author_names_sets = []

                for file in os.listdir(use_case_path):
                    if file.startswith("validation_result_") and file.endswith(".json"):
                        file_path = os.path.join(use_case_path, file)
                        data = load_json(file_path)

                        if not data.get('enhanced_authors', []):
                            continue  # Skip if there are no authors recommended


                        # Extract author names
                        author_names = [author['Name'] for author in data.get('enhanced_authors', [])]

                        # Calculate unique-to-total ratio for the current run
                        if len(author_names) > 0:
                            unique_to_total_ratio = len(set(author_names)) / len(author_names)
                            unique_to_total_ratios.append(unique_to_total_ratio)

                        # Store the set of author names for Jaccard similarity calculation
                        all_author_names_sets.append(set(author_names))

                # Store the results for the current use case
                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {
                        'unique_to_total_ratios': [],
                        'author_name_sets': []
                    }

                use_case_summary[use_case]['unique_to_total_ratios'].extend(unique_to_total_ratios)
                use_case_summary[use_case]['author_name_sets'].extend(all_author_names_sets)

        # Store the summary for each config
        results[config] = use_case_summary

    return results


def compute_jaccard_for_runs(all_author_sets):
    """
    Compute the Jaccard similarity for all combinations of sets in all_author_sets.
    Return the average, standard deviation, and the list of Jaccard values.
    """
    if len(all_author_sets) < 2:
        return "--", "--", []  # Not enough data to compute Jaccard

    jaccard_values = []
    for set1, set2 in combinations(all_author_sets, 2):
        jaccard_values.append(jaccard_similarity(set1, set2))

    avg_jaccard = np.mean(jaccard_values)
    std_jaccard = np.std(jaccard_values)
    return f"{avg_jaccard:.2f}", f"{std_jaccard:.2f}", jaccard_values

def create_summary_tables(results, output_folder):
    # Define the models in the required order
    model_order = ["llama3-8b", "gemma2-9b", "mixtral-8x7b", "llama3-70b"]

    # Clean config names (remove 'config_' prefix)
    cleaned_configs = [config.replace('config_', '') for config in results.keys()]

    # Initialize DataFrames to store the summaries
    uniqueness_data = []
    jaccard_data = []

    # Initialize dictionaries to collect all ratios and Jaccard values for final rows
    config_unique_ratios = {config: [] for config in cleaned_configs}
    config_jaccard_values = {config: [] for config in cleaned_configs}

    # Get all the use cases
    use_cases = sorted(set(use_case for config in results.values() for use_case in config.keys()))

    for use_case in use_cases:
        uniqueness_row = {'Use Case': use_case}
        jaccard_row = {'Use Case': use_case}

        for config in cleaned_configs:
            original_config = 'config_' + config  # Convert back to the original config key with 'config_' prefix
            if use_case in results[original_config]:
                values = results[original_config][use_case]

                # Calculate average and standard deviation of unique-to-total ratios
                if values['unique_to_total_ratios']:
                    avg_unique_ratio = np.mean(values['unique_to_total_ratios'])
                    std_unique_ratio = np.std(values['unique_to_total_ratios'])
                    unique_ratio_result = f"{avg_unique_ratio:.2f} ± {std_unique_ratio:.2f}"
                    config_unique_ratios[config].extend(values['unique_to_total_ratios'])
                else:
                    unique_ratio_result = "--"

                # Calculate average and standard deviation of Jaccard index
                if len(values['author_name_sets']) > 1:
                    avg_jaccard, std_jaccard, jaccard_values = compute_jaccard_for_runs(values['author_name_sets'])
                    config_jaccard_values[config].extend(jaccard_values)
                else:
                    avg_jaccard, std_jaccard = "--", "--"

                jaccard_result = f"{avg_jaccard} ± {std_jaccard}"

                # Append the metrics for this configuration
                uniqueness_row[config] = unique_ratio_result
                jaccard_row[config] = jaccard_result
            else:
                uniqueness_row[config] = "--"
                jaccard_row[config] = "--"

        uniqueness_data.append(uniqueness_row)
        jaccard_data.append(jaccard_row)

    # Convert to DataFrames
    df_uniqueness = pd.DataFrame(uniqueness_data)
    df_jaccard = pd.DataFrame(jaccard_data)

    # Compute the final rows for each metric
    final_uniqueness_row = {'Use Case': 'Final Avg ± Std'}
    final_jaccard_row = {'Use Case': 'Final Avg ± Std'}

    for config in cleaned_configs:
        # For uniqueness ratios
        if config_unique_ratios[config]:
            final_avg_unique = np.mean(config_unique_ratios[config])
            final_std_unique = np.std(config_unique_ratios[config])
            final_uniqueness_row[config] = f"{final_avg_unique:.2f} ± {final_std_unique:.2f}"
        else:
            final_uniqueness_row[config] = "--"

        # For Jaccard values
        if config_jaccard_values[config]:
            final_avg_jaccard = np.mean(config_jaccard_values[config])
            final_std_jaccard = np.std(config_jaccard_values[config])
            final_jaccard_row[config] = f"{final_avg_jaccard:.2f} ± {final_std_jaccard:.2f}"
        else:
            final_jaccard_row[config] = "--"

    # Append final rows to each DataFrame
    df_uniqueness = pd.concat([df_uniqueness, pd.DataFrame([final_uniqueness_row])], ignore_index=True)
    df_jaccard = pd.concat([df_jaccard, pd.DataFrame([final_jaccard_row])], ignore_index=True)

    # Reorder columns based on the model order
    df_uniqueness = df_uniqueness[['Use Case'] + [config for config in model_order if config in df_uniqueness.columns]]
    df_jaccard = df_jaccard[['Use Case'] + [config for config in model_order if config in df_jaccard.columns]]

    # Function to save table as CSV, LaTeX, and PNG
    def save_table(df, filename, title):
        # Save as CSV
        csv_path = os.path.join(output_folder, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {title} as CSV at {csv_path}")

        # Save as LaTeX
        latex_path = os.path.join(output_folder, f"{filename}.tex")
        with open(latex_path, 'w') as latex_file:
            latex_file.write(df.to_latex(index=False, caption=title, label=filename))
        print(f"Saved {title} as LaTeX at {latex_path}")

        # Save as PNG
        if not df.empty:
            fig, ax = plt.subplots(figsize=(15, len(df) * 0.5 + 1))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width(col=list(range(len(df.columns))))
            
            # Add title
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            png_path = os.path.join(output_folder, f"{filename}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {title} as PNG at {png_path}")
        else:
            print(f"No data available to create PNG for {title}")

    # Save Uniqueness table
    save_table(df_uniqueness, "author_uniqueness_summary", "Author Uniqueness Ratio (Unique/Total)")

    # Save Jaccard Similarity table
    save_table(df_jaccard, "answer_similarity_summary", "Answer Similarity (Jaccard Index)")


def main():
    input_dir = "././experiments_validation_results/updated_organized_results"
    output_folder = "././output_results/consistency/names_consistency"
    os.makedirs(output_folder, exist_ok=True)

    print("Processing data for consistency metrics...")
    results = process_data(input_dir)

    print("Creating summary tables...")
    create_summary_tables(results, output_folder)

    print(f"Results saved in {output_folder}")

if __name__ == "__main__":
    main()
