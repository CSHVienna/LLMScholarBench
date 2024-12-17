import os
import json
import numpy as np
import pandas as pd
import random
import itertools
import re
from scipy.stats import chi2
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Function to compute Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0  # Handle case where both sets are empty
    return intersection / union

def process_data(root_dir, author_network_data):
    # replace with id_author (if not present is 'not in aps')
    author_collaborators = {entry['id_author']: set(entry['aps_co_authors']) for entry in author_network_data}

    results = {}

    for config in os.listdir(root_dir):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path):
            continue

        use_case_summary = {}

        for run in os.listdir(config_path):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            for use_case in os.listdir(run_path):
                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                for file in os.listdir(use_case_path):
                    if file.startswith("validation_result_") and file.endswith(".json"):
                        file_path = os.path.join(use_case_path, file)
                        data = load_json(file_path)
                        author_coauthors_sets = {}
                        author_coauthors_sets[run] = {}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if len(enhanced_authors) == 0:
                            continue

                        for author in enhanced_authors:
                            id_author = author.get('id_author', None)
                            if id_author and id_author != 'not in aps':
                                collaborators = author_collaborators.get(id_author, set())
                                author_coauthors_sets[run][id_author] = collaborators

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {'coauthors_dictionary': []}

                if author_coauthors_sets[run]:
                    use_case_summary[use_case]['coauthors_dictionary'].append(author_coauthors_sets)

        results[config] = use_case_summary

    return results

def compute_bootstrap_p_value(observed_avg, co_author_network, num_authors, seed=43, n_iterations=1000):
    if seed is not None:
        np.random.seed(seed)

    bootstrap_means = []

    for _ in range(n_iterations):
        bootstrapped_jaccard_scores = []

        for i in range(len(num_authors)):
            sampled_authors = random.sample(co_author_network, num_authors[i])
            sampled_coauthors = {author['id_author']: set(author['aps_co_authors']) for author in sampled_authors}
            
            author_pairs = list(itertools.combinations(sampled_coauthors.keys(), 2))
            jaccard_scores = [
                jaccard_similarity(sampled_coauthors[a1], sampled_coauthors[a2])
                for a1, a2 in author_pairs
            ]
            bootstrapped_jaccard_scores.extend(jaccard_scores)
            
        if bootstrapped_jaccard_scores:
            bootstrap_means.append(np.mean(bootstrapped_jaccard_scores))

    p_value = np.mean(np.array(bootstrap_means) >= observed_avg)
    return p_value, bootstrap_means

# Fisher's method to combine p-values
def fishers_method(p_values):
    """
    Apply Fisher's method to combine p-values.
    
    Parameters:
        p_values (list): List of p-values to combine
    
    Returns:
        combined_p (float): Combined p-value from Fisher's method
    """
    # Remove "No Data" or None from p-values
    p_values = [p for p in p_values if p not in ['No Data', None]]
    
    if len(p_values) == 0:
        return None  # Return None if no valid p-values

    # Step 1: Calculate the Fisher's combined statistic X^2
    X2_statistic = -2 * np.sum(np.log(p_values))
    
    # Step 2: Compute the degrees of freedom (2k, where k is the number of p-values)
    df = 2 * len(p_values)
    
    # Step 3: Compute the p-value from the chi-squared distribution with df degrees of freedom
    combined_p = chi2.sf(X2_statistic, df)
    
    return combined_p

def compute_jaccard_with_bootstrap_and_store_in_dataframe(results, co_author_network):
    table_data = {}
    p_values = {}
    bootstrapped_means = {}  # Dictionary to store the bootstrapped means
    overall_bootstrap_means = []  # List to store all bootstrapped means for overall aggregation

    total_use_cases = sum(len(use_cases) for use_cases in results.values())
    progress_bar = tqdm(total=total_use_cases, desc="Processing Configs and Use Cases")  # Progress bar

    for config, use_cases in results.items():
        table_data[config] = {}  # Initialize a dictionary for avg and std
        p_values[config] = {}  # Initialize a dictionary for p-values
        bootstrapped_means[config] = {}  # Initialize for bootstrapped Jaccard
        all_scores_for_config = []
        all_bootstrap_means_for_config = []  # Collect bootstrapped means across use cases
        config_p_values = []  # Collect p-values to apply Fisher's method later

        for use_case, data in use_cases.items():
            all_jaccard_scores = []
            num_authors = []

            for coauthors_dict in data['coauthors_dictionary']:
                for run, coauthors_list in coauthors_dict.items():
                    # Get the number of authors for this run
                    num_authors.append(len(coauthors_list.keys()))
                    
                    # Get all author combinations for this run
                    author_pairs = list(itertools.combinations(coauthors_list.keys(), 2))
                    # Compute Jaccard similarity for each pair
                    jaccard_scores = [
                        jaccard_similarity(coauthors_list[a1], coauthors_list[a2])
                        for a1, a2 in author_pairs
                    ]
                    # Append the scores to the list for this use case
                    all_jaccard_scores.extend(jaccard_scores)

                    #print(f"Use Case: {use_case}, Config: {config}, Run: {run}, Authors: {len(coauthors_list)} - Jaccard Scores: {len(jaccard_scores)}")

            # Calculate the observed average Jaccard similarity for this use case
            if all_jaccard_scores:
                observed_avg = np.mean(all_jaccard_scores)
                observed_std = np.std(all_jaccard_scores)

                p_value, bootstrap_means_list = compute_bootstrap_p_value(observed_avg, co_author_network, num_authors)

                # Store the avg ± std and p-value for this use case
                table_data[config][use_case] = f"{observed_avg:.4f} ± {observed_std:.2f}"
                p_values[config][use_case] = f"{p_value:.4f}"
                
                # Add p-value to the list for Fisher's method
                config_p_values.append(float(p_value))

                # Add bootstrapped mean for this use case
                bootstrap_jaccard_avg = np.mean(bootstrap_means_list) if bootstrap_means_list else 0.0
                bootstrapped_means[config][use_case] = f"{bootstrap_jaccard_avg:.4f} ± {np.std(bootstrap_means_list):.2f}"
                
                # Collect bootstrapped means for overall aggregation
                all_bootstrap_means_for_config.extend(bootstrap_means_list)
                overall_bootstrap_means.extend(bootstrap_means_list)  # Add to global list for overall mean
                
                # Add scores to the overall scores for this config
                all_scores_for_config.extend(all_jaccard_scores)

            else:
                table_data[config][use_case] = "No Data"
                p_values[config][use_case] = "No Data"

            # Update the progress bar
            progress_bar.update(1)

        # Compute the overall avg and std for the config, aggregating all scores across use cases
        if all_scores_for_config:
            overall_avg = np.mean(all_scores_for_config)
            overall_std = np.std(all_scores_for_config)

            # Store only avg and std, no p-value for overall
            table_data[config]['Overall'] = f"{overall_avg:.4f} ± {overall_std:.2f}"
            
            # Apply Fisher's method to combine p-values and place it in the Overall row
            combined_p_value = fishers_method(config_p_values)
            p_values[config]['Overall'] = f"{combined_p_value:.4f}" 

            # Compute overall bootstrapped Jaccard mean and std for this config
            if all_bootstrap_means_for_config:
                overall_bootstrap_avg = np.mean(all_bootstrap_means_for_config)
                overall_bootstrap_std = np.std(all_bootstrap_means_for_config)
                bootstrapped_means[config]['Overall'] = f"{overall_bootstrap_avg:.4f} ± {overall_bootstrap_std:.2f}"
            else:
                bootstrapped_means[config]['Overall'] = "No Data"
        else:
            table_data[config]['Overall'] = "No Data"
            p_values[config]['Overall'] = "No Data"
            bootstrapped_means[config]['Overall'] = "No Data"

    progress_bar.close()  # Close the progress bar after completion

    # Compute the global overall bootstrapped Jaccard mean and std across all configs
    if overall_bootstrap_means:
        global_bootstrap_avg = np.mean(overall_bootstrap_means)
        global_bootstrap_std = np.std(overall_bootstrap_means)
        print(f"Global Bootstrapped Jaccard Mean: {global_bootstrap_avg:.4f} ± {global_bootstrap_std:.2f}")

    # Create three DataFrames: one for avg ± std, one for p-values, and one for bootstrapped Jaccard
    df_avg_std = pd.DataFrame(table_data)
    df_p_values = pd.DataFrame(p_values)
    df_bootstrapped = pd.DataFrame(bootstrapped_means)

    # Concatenate the three DataFrames
    df_combined = pd.concat([df_avg_std, df_bootstrapped, df_p_values], axis=1, keys=['Avg ± Std', 'Bootstrapped Jaccard', 'p-value'])

    # Flatten the multi-level columns
    df_combined.columns = [' '.join(col).strip() for col in df_combined.columns.values]

    # Sort the columns so that each config's "Avg ± Std", "Bootstrapped Jaccard", and "p-value" are next to each other
    sorted_columns = []
    for config in table_data.keys():
        sorted_columns.append(f'Avg ± Std {config}')
        sorted_columns.append(f'Bootstrapped Jaccard {config}')
        sorted_columns.append(f'p-value {config}')

    # Select only the sorted columns from df_combined
    df_combined = df_combined[sorted_columns]

    return df_combined

def save_to_latex(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def escape_latex_special_chars(text):
        return re.sub(r"([_&])", r"\\\1", text)

    with open(output_path, 'w') as f:
        f.write("\\begin{table*}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Analysis Results}\n")
        f.write("\\label{tbl:analysis_results}\n")
        f.write("\\begin{tabular}{lR{2.4cm}R{2.4cm}R{2.4cm}R{2.4cm}R{2.4cm}R{2.4cm}R{2.4cm}}\n")
        f.write("\\toprule\n")
        
        headers = ["Metric"] + [escape_latex_special_chars(str(col)) for col in df.columns]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        for index, row in df.iterrows():
            index_escaped = escape_latex_special_chars(str(index))
            row_data = [escape_latex_special_chars(str(value)) for value in row.astype(str).tolist()]
            f.write(index_escaped + " & " + " & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

def main():
    # Set paths
    input_dir = "./experiments_validation_results/updated_organized_results"
    author_network_file_path = "./organised_data/aps_coauthor_networks.json"
    output_path = "./output_results/analysis/aps_coauthors_jaccard_similarity/table_output.tex"

    # Load data
    author_network_data = load_ndjson(author_network_file_path)
    
    # Process data
    results = process_data(input_dir, author_network_data)

    # Compute Jaccard similarities
    df_jaccard = compute_jaccard_with_bootstrap_and_store_in_dataframe(results, author_network_data)

    # Sort and organize DataFrame
    over_all_row = df_jaccard.sort_index().iloc[0]
    dropped_over_all_row = df_jaccard.drop('Overall', axis=0)
    sorted_df = pd.concat([dropped_over_all_row.sort_index(), over_all_row.to_frame().T], axis=0)

    # Reorder columns
    columns_order = [
        'Avg ± Std config_llama3-8b', 'Bootstrapped Jaccard config_llama3-8b', 'p-value config_llama3-8b',
        'Avg ± Std config_gemma2-9b', 'Bootstrapped Jaccard config_gemma2-9b', 'p-value config_gemma2-9b',
        'Avg ± Std config_mixtral-8x7b', 'Bootstrapped Jaccard config_mixtral-8x7b', 'p-value config_mixtral-8x7b',
        'Avg ± Std config_llama3-70b', 'Bootstrapped Jaccard config_llama3-70b', 'p-value config_llama3-70b'
    ]
    sorted_df = sorted_df[columns_order]

    # Save to LaTeX
    save_to_latex(sorted_df, output_path)

    print("Analysis complete. Results saved to", output_path)

if __name__ == "__main__":
    main()