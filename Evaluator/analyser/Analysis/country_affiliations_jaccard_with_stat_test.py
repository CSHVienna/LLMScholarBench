import os
import json
import numpy as np
import pandas as pd
import random
import itertools
import re
from scipy.stats import chi2
from tqdm import tqdm

# Function to load JSON data (standard JSON)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to load NDJSON data (one JSON per line)
def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Function to compute Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1  # If both sets are empty, similarity is considered 1
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to process data with country affiliations
def process_data(root_dir, authors_affiliations_data, affiliations_data):
    # Create a dictionary to map openalex_author_id to a set of affiliations
    author_affiliations = {entry['openalex_author_id']: {aff['id_affiliation'] for aff in entry['affiliations']} 
                         for entry in authors_affiliations_data}

    # Create a dictionary to map affiliation_id to its country_code
    affiliation_to_country = {entry['id_affiliation']: entry['country_code'] for entry in affiliations_data}

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
                        author_affiliation_country_sets = {}
                        author_affiliation_country_sets[run] = {}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if len(enhanced_authors) == 0:
                            continue

                        for author in enhanced_authors:
                            author_key = author.get('key')
                            if author_key and author_key in author_affiliations:
                                affiliations = author_affiliations.get(author_key, set())
                                countries = {affiliation_to_country[aff] for aff in affiliations 
                                           if aff in affiliation_to_country}
                                author_affiliation_country_sets[run][author_key] = countries

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {
                        'affiliations_dictionary': []
                    }

                if author_affiliation_country_sets[run]:
                    use_case_summary[use_case]['affiliations_dictionary'].append(author_affiliation_country_sets)

        results[config] = use_case_summary

    return results

# Function to perform bootstrapping and compute p-value
def compute_bootstrap_p_value(observed_avg, author_affiliations, affiliation_to_country, num_authors, seed=43, n_iterations=1000):
    if seed is not None:
        np.random.seed(seed)

    bootstrap_means = []

    for _ in range(n_iterations):
        bootstrapped_jaccard_scores = []

        for i in range(len(num_authors)):
            sampled_authors = random.sample(author_affiliations, int(num_authors[i]))
            author_country_mapping = {}

            for author in sampled_authors:
                sampled_author_affiliations = author['affiliations']
                list_institutions = [affiliation['id_affiliation'] for affiliation in sampled_author_affiliations]
                country_codes = {affiliation_to_country.get(affiliation_id) for affiliation_id in list_institutions}
                author_country_mapping[author['id_author']] = country_codes

            author_pairs = list(itertools.combinations(author_country_mapping.keys(), 2))
            jaccard_scores = [
                jaccard_similarity(set(author_country_mapping[a1]), set(author_country_mapping[a2]))
                for a1, a2 in author_pairs
            ]
            bootstrapped_jaccard_scores.extend(jaccard_scores)

        if bootstrapped_jaccard_scores:
            bootstrap_means.append(np.mean(bootstrapped_jaccard_scores))

    p_value = np.mean(np.array(bootstrap_means) >= observed_avg)
    
    return p_value, bootstrap_means

# Fisher's method to combine p-values
def fishers_method(p_values):
    p_values = [p for p in p_values if p not in ['No Data', None]]
    
    if len(p_values) == 0:
        return None

    X2_statistic = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = chi2.sf(X2_statistic, df)
    
    return combined_p

def compute_jaccard_with_bootstrap_and_store_in_dataframe(results, author_affiliations, affiliation_to_country):
    table_data = {}
    p_values = {}
    bootstrapped_means = {}
    overall_bootstrap_means = []

    total_use_cases = sum(len(use_cases) for use_cases in results.values())
    progress_bar = tqdm(total=total_use_cases, desc="Processing Configs and Use Cases")

    for config, use_cases in results.items():
        table_data[config] = {}
        p_values[config] = {}
        bootstrapped_means[config] = {}
        all_scores_for_config = []
        all_bootstrap_means_for_config = []
        config_p_values = []

        for use_case, data in use_cases.items():
            all_jaccard_scores = []
            num_authors = []

            for affiliation_dict in data['affiliations_dictionary']:
                for run, authors_affiliations in affiliation_dict.items():
                    num_authors.append(len(authors_affiliations.keys()))
                    author_pairs = list(itertools.combinations(authors_affiliations.keys(), 2))
                    jaccard_scores = [
                        jaccard_similarity(authors_affiliations[a1], authors_affiliations[a2])
                        for a1, a2 in author_pairs
                    ]
                    all_jaccard_scores.extend(jaccard_scores)

            if all_jaccard_scores:
                observed_avg = np.mean(all_jaccard_scores)
                observed_std = np.std(all_jaccard_scores)

                p_value, bootstrap_means_list = compute_bootstrap_p_value(
                    observed_avg, author_affiliations, affiliation_to_country, num_authors
                )

                table_data[config][use_case] = f"{observed_avg:.4f} ± {observed_std:.2f}"
                p_values[config][use_case] = f"{p_value:.4f}"
                
                config_p_values.append(float(p_value))

                bootstrap_jaccard_avg = np.mean(bootstrap_means_list) if bootstrap_means_list else 0.0
                bootstrapped_means[config][use_case] = f"{bootstrap_jaccard_avg:.4f} ± {np.std(bootstrap_means_list):.2f}"
                
                all_bootstrap_means_for_config.extend(bootstrap_means_list)
                overall_bootstrap_means.extend(bootstrap_means_list)
                all_scores_for_config.extend(all_jaccard_scores)
            else:
                table_data[config][use_case] = "No Data"
                p_values[config][use_case] = "No Data"
                bootstrapped_means[config][use_case] = "No Data"

            progress_bar.update(1)

        if all_scores_for_config:
            overall_avg = np.mean(all_scores_for_config)
            overall_std = np.std(all_scores_for_config)
            table_data[config]['Overall'] = f"{overall_avg:.4f} ± {overall_std:.2f}"

            if all_bootstrap_means_for_config:
                overall_bootstrap_avg = np.mean(all_bootstrap_means_for_config)
                overall_bootstrap_std = np.std(all_bootstrap_means_for_config)
                bootstrapped_means[config]['Overall'] = f"{overall_bootstrap_avg:.4f} ± {overall_bootstrap_std:.2f}"
            else:
                bootstrapped_means[config]['Overall'] = "No Data"

            combined_p_value = fishers_method(config_p_values)
            p_values[config]['Overall'] = f"{combined_p_value:.4f}"
        else:
            table_data[config]['Overall'] = "No Data"
            p_values[config]['Overall'] = "No Data"
            bootstrapped_means[config]['Overall'] = "No Data"

    progress_bar.close()

    if overall_bootstrap_means:
        global_bootstrap_avg = np.mean(overall_bootstrap_means)
        global_bootstrap_std = np.std(overall_bootstrap_means)
        print(f"Global Bootstrapped Jaccard Mean: {global_bootstrap_avg:.4f} ± {global_bootstrap_std:.2f}")

    df_avg_std = pd.DataFrame(table_data)
    df_p_values = pd.DataFrame(p_values)
    df_bootstrapped = pd.DataFrame(bootstrapped_means)

    df_combined = pd.concat([df_avg_std, df_bootstrapped, df_p_values], axis=1, keys=['Avg ± Std', 'Bootstrapped Jaccard', 'p-value'])
    df_combined.columns = [' '.join(col).strip() for col in df_combined.columns.values]

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
        
        headers = [escape_latex_special_chars(str(col)) for col in df.columns]
        f.write(" & ".join(["Metric"] + headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        for index, row in df.iterrows():
            index_escaped = escape_latex_special_chars(str(index))
            row_data = [escape_latex_special_chars(str(value)) for value in row.astype(str).tolist()]
            f.write(index_escaped + " & " + " & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

def main():
    input_dir = "././experiments_validation_results/updated_organized_results"
    affiliations_file_path = "././organised_data/author_affiliations.json"
    institutions_file_path = "././organised_data/affiliations.json"
    output_latex_path = "./output_results/analysis/country_affiliation_jaccard/table_output.tex"

    # Load data
    authors_affiliations_data = load_ndjson(affiliations_file_path)
    affiliations_data = load_ndjson(institutions_file_path)
    affiliation_to_country = {inst['id_affiliation']: inst['country_code'] for inst in affiliations_data}

    # Process the data
    results = process_data(input_dir, authors_affiliations_data, affiliations_data)

    # Compute Jaccard similarities and create DataFrame
    df_jaccard = compute_jaccard_with_bootstrap_and_store_in_dataframe(
        results, authors_affiliations_data, affiliation_to_country
    )

    # Sort and organize the DataFrame
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

    # Replace 'No Data' with '--' for better presentation
    sorted_df = sorted_df.replace('No Data', '--').replace(np.nan, '--')

    # Save to LaTeX
    save_to_latex(sorted_df, output_latex_path)

    print("Analysis complete. Results have been saved to LaTeX file.")

if __name__ == "__main__":
    main()