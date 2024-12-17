import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2
from tqdm import tqdm
import re
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def calculate_cosine_similarity_with_min_max(vectors):
    if len(vectors) < 2:
        return []
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(vectors)
    similarity_matrix = cosine_similarity(normalized_vectors)
    upper_triangle_values = similarity_matrix[np.triu_indices(len(vectors), k=1)]
    return upper_triangle_values.tolist() if len(upper_triangle_values) > 0 else []

def create_feature_vector(author_id, author_info, author_networks_dict, author_affiliations_dict, institutions_stats_dict):
    h_index = author_info.get('aps_h_index', 0)
    num_pub = author_info.get('aps_works_count', 0)
    num_cit = author_info.get('aps_cited_by_count', 0)
    academic_age = author_info.get('aps_career_age', 0)
    e_index = author_info.get('aps_e_index', 0)
    num_collaborators = author_networks_dict.get(author_id, {}).get('collaboration_counts', 0)
    
    # Computed metrics
    cit_per_pub_academic_age = (num_cit / num_pub) * academic_age if num_pub > 0 else 0
    
    # Highest H-index among affiliations
    highest_h_index = 0
    if author_id in author_affiliations_dict:
        affiliations = author_affiliations_dict[author_id]['affiliations']
        affiliation_ids = [aff['id_affiliation'] for aff in affiliations]
        highest_h_index = max(
            [institutions_stats_dict.get(aff_id, {}).get('h_index', 0) for aff_id in affiliation_ids],
            default=0
        )
    
    return [h_index, num_pub, num_cit, academic_age, num_collaborators, 
            cit_per_pub_academic_age, e_index, highest_h_index]

def process_data(root_dir, author_stats_dict, author_networks_dict, author_affiliations_dict, institutions_stats_dict):
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
                        feature_vector_dict = {run: {}}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if not enhanced_authors:
                            continue

                        for author in enhanced_authors:
                            if not author.get('has_published_in_aps', False):
                                continue

                            author_id = author.get('id_author')
                            author_info = author_stats_dict[author_id]
                            feature_vector = create_feature_vector(
                                author_id, author_info, author_networks_dict,
                                author_affiliations_dict, institutions_stats_dict
                            )
                            feature_vector_dict[run][author_id] = feature_vector

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {'feature_vector_dictionary': []}

                if feature_vector_dict[run]:
                    use_case_summary[use_case]['feature_vector_dictionary'].append(feature_vector_dict)

        results[config] = use_case_summary

    return results

def fishers_method(p_values):
    p_values = [p for p in p_values if p not in ['No Data', None]]
    if not p_values:
        return None
    
    min_p_value = 1e-10
    p_values = [max(p, min_p_value) for p in p_values]
    X2_statistic = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = chi2.sf(X2_statistic, df)
    return combined_p

def compute_bootstrap_p_value(observed_avg, author_stats, author_networks_dict, author_affiliations_dict, 
                            institutions_stats_dict, num_authors, seed=43, n_iterations=1000):
    author_stats_dict = {entry['id_author']: entry for entry in author_stats}
    if seed is not None:
        np.random.seed(seed)

    bootstrap_means = []

    for _ in range(n_iterations):
        bootstrapped_similarity_scores = []

        for num_author in num_authors:
            sampled_authors = random.sample(author_stats, int(num_author))
            feature_vectors = []
            
            for author in sampled_authors:
                author_id = author['id_author']
                feature_vector = create_feature_vector(
                    author_id, author_stats_dict[author_id], author_networks_dict,
                    author_affiliations_dict, institutions_stats_dict
                )
                feature_vectors.append(feature_vector)

            similarity_scores = calculate_cosine_similarity_with_min_max(feature_vectors)
            bootstrapped_similarity_scores.extend(similarity_scores)

        if bootstrapped_similarity_scores:
            bootstrap_means.append(np.mean(bootstrapped_similarity_scores))

    p_value = np.mean(np.array(bootstrap_means) >= observed_avg)
    return p_value, bootstrap_means


def compute_similarity_with_bootstrap_and_store_in_dataframe(results, author_stats, author_networks_dict, author_affiliations_dict, institutions_stats_dict):
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
        config_p_values = []  # List to collect all p-values for Fisher's method

        for use_case, data in use_cases.items():
            all_similarity_scores = []
            num_authors = []  # To store the number of authors for each run

            for feature_vector_dictionary in data['feature_vector_dictionary']:
                for run, authors_feature_vector_mapping in feature_vector_dictionary.items():
                    author_feature_vectors = []
                    for author, feature_vector in authors_feature_vector_mapping.items():
                        author_feature_vectors.append(feature_vector)

                    # Get the number of authors for this run
                    num_authors.append(len(feature_vector_dictionary[run].keys()))
                    
                    similarity_scores = calculate_cosine_similarity_with_min_max(author_feature_vectors)
                    
                    all_similarity_scores.extend(similarity_scores)

            # Calculate the observed average Jaccard similarity for this use case
            if all_similarity_scores:
                observed_avg = np.mean(all_similarity_scores)
                observed_std = np.std(all_similarity_scores)
                
                # Compute the p-value and the full bootstrap_means using bootstrapping
                p_value, bootstrap_means_list = compute_bootstrap_p_value(observed_avg, author_stats, author_networks_dict, author_affiliations_dict, institutions_stats_dict, num_authors)
                # Store the avg ± std, p-value, and bootstrapped Jaccard for this use case
                table_data[config][use_case] = f"{observed_avg:.4f} ± {observed_std:.2f}"
                p_values[config][use_case] = f"{p_value:.4f}"
                
                # Add bootstrapped mean for this use case
                bootstrap_jaccard_avg = np.mean(bootstrap_means_list) if bootstrap_means_list else 0.0
                bootstrapped_means[config][use_case] = f"{bootstrap_jaccard_avg:.4f} ± {np.std(bootstrap_means_list):.2f}"
                
                # Collect bootstrapped means for overall aggregation
                all_bootstrap_means_for_config.extend(bootstrap_means_list)
                overall_bootstrap_means.extend(bootstrap_means_list)  # Add to global list for overall mean

                # Add the p-value to config_p_values for Fisher's method
                if p_value is not None:
                    config_p_values.append(float(p_value))
                
                # Add scores to the overall scores for this config
                all_scores_for_config.extend(all_similarity_scores)
            else:
                table_data[config][use_case] = "No Data"
                p_values[config][use_case] = "No Data"
                bootstrapped_means[config][use_case] = "No Data"

            # Update the progress bar
            progress_bar.update(1)

        # Compute the overall avg and std for the config, aggregating all scores across use cases
        if all_scores_for_config:
            overall_avg = np.mean(all_scores_for_config)
            overall_std = np.std(all_scores_for_config)

            # Store only avg and std, no p-value for overall
            table_data[config]['Overall'] = f"{overall_avg:.4f} ± {overall_std:.2f}"

            # Compute overall bootstrapped Jaccard mean and std for this config
            if all_bootstrap_means_for_config:
                overall_bootstrap_avg = np.mean(all_bootstrap_means_for_config)
                overall_bootstrap_std = np.std(all_bootstrap_means_for_config)
                bootstrapped_means[config]['Overall'] = f"{overall_bootstrap_avg:.4f} ± {overall_bootstrap_std:.2f}"
            else:
                bootstrapped_means[config]['Overall'] = "No Data"

            # Combine p-values for this config using Fisher's method
            if config_p_values:
                combined_p_value = fishers_method(config_p_values)
                p_values[config]['Overall'] = f"{combined_p_value:.4f}"  # Insert combined p-value into Overall row
            else:
                p_values[config]['Overall'] = "No Data"
        else:
            table_data[config]['Overall'] = "No Data"
            p_values[config]['Overall'] = "No Data"
            bootstrapped_means[config]['Overall'] = "No Data"

    progress_bar.close()  # Close the progress bar after completion

    # Compute the global overall bootstrapped Jaccard mean and std across all configs
    if overall_bootstrap_means:
        global_bootstrap_avg = np.mean(overall_bootstrap_means)
        global_bootstrap_std = np.std(overall_bootstrap_means)
        print(f"Global Bootstrapped Similarity Mean: {global_bootstrap_avg:.4f} ± {global_bootstrap_std:.2f}")

    # Create three DataFrames: one for avg ± std, one for p-values, and one for bootstrapped Jaccard
    df_avg_std = pd.DataFrame(table_data)
    df_p_values = pd.DataFrame(p_values)
    df_bootstrapped = pd.DataFrame(bootstrapped_means)

    # Concatenate the three DataFrames
    df_combined = pd.concat([df_avg_std, df_bootstrapped, df_p_values], axis=1, keys=['Avg ± Std', 'Bootstrapped Similarity', 'p-value'])

    # Flatten the multi-level columns
    df_combined.columns = [' '.join(col).strip() for col in df_combined.columns.values]

    # Sort the columns so that each config's "Avg ± Std", "Bootstrapped Jaccard", and "p-value" are next to each other
    sorted_columns = []
    for config in table_data.keys():
        sorted_columns.append(f'Avg ± Std {config}')
        sorted_columns.append(f'Bootstrapped Similarity {config}')
        sorted_columns.append(f'p-value {config}')

    # Select only the sorted columns from df_combined

    df_combined = df_combined[sorted_columns]

    # sort rows alphabetically but keep overall as last one
    df_combined = df_combined.reindex(sorted(df_combined.index[:-1]) + [df_combined.index[-1]])

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
    author_stats_file = "./organised_data/aps_author_stats.json"
    author_networks_file = "./organised_data/aps_coauthor_networks.json"
    author_affiliations_file = "./organised_data/author_affiliations.json"
    institutions_stats_file = "./organised_data/institutions_stats.json"
    output_path = "./output_results/analysis/numerical_metrics_jaccard_similarity/table_output.tex"

    # Load data
    print("Loading data files...")
    author_stats = load_ndjson(author_stats_file)
    author_networks = load_ndjson(author_networks_file)
    author_affiliations = load_ndjson(author_affiliations_file)
    institutions_stats = load_ndjson(institutions_stats_file)

    # Create lookup dictionaries
    author_stats_dict = {entry['id_author']: entry for entry in author_stats}
    author_networks_dict = {entry['id_author']: entry for entry in author_networks}
    author_affiliations_dict = {entry['id_author']: entry for entry in author_affiliations}
    institutions_stats_dict = {entry['id_affiliation']: entry for entry in institutions_stats}

    # Process data and compute similarities
    print("Processing data and computing similarities...")
    results = process_data(input_dir, author_stats_dict, author_networks_dict, 
                         author_affiliations_dict, institutions_stats_dict)

    # Compute similarity metrics and create DataFrame
    df_similarity = compute_similarity_with_bootstrap_and_store_in_dataframe(
        results, author_stats, author_networks_dict, author_affiliations_dict, institutions_stats_dict
    )

    # Sort and organize DataFrame
    over_all_row = df_similarity.sort_index().iloc[0]
    dropped_over_all_row = df_similarity.drop('Overall', axis=0)
    sorted_df = pd.concat([dropped_over_all_row.sort_index(), over_all_row.to_frame().T], axis=0)

    # Reorder columns
    columns_order = [
        'Avg ± Std config_llama3-8b', 'Bootstrapped Similarity config_llama3-8b', 'p-value config_llama3-8b',
        'Avg ± Std config_gemma2-9b', 'Bootstrapped Similarity config_gemma2-9b', 'p-value config_gemma2-9b',
        'Avg ± Std config_mixtral-8x7b', 'Bootstrapped Similarity config_mixtral-8x7b', 'p-value config_mixtral-8x7b',
        'Avg ± Std config_llama3-70b', 'Bootstrapped Similarity config_llama3-70b', 'p-value config_llama3-70b'
    ]
    sorted_df = sorted_df[columns_order]

    # Save to LaTeX
    print("Saving results to LaTeX file...")
    save_to_latex(sorted_df, output_path)
    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()