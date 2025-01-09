import os
import json
import numpy as np
import pandas as pd
import random
import itertools
from scipy.stats import chi2
from tqdm import tqdm
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def get_author_decades(author_stats):
    author_decade = {}
    for author in author_stats:
        author_decade[author['id_author']] = author['aps_years_of_activity'][0] // 10 * 10
    return author_decade

def get_nobel_laureates(nobel_file_path):
    nobel_laureates = set()
    with open(nobel_file_path, 'r') as f:
        nobel_data = json.load(f)
        for laureate in nobel_data:
            nobel_laureates.add(laureate['openalex_id'])
            for candidate in laureate.get('candidates', []):
                nobel_laureates.add(candidate['id'])
    return nobel_laureates

def process_data(root_dir):
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
                        author_metrics_sets = {}
                        author_metrics_sets[run] = {}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if len(enhanced_authors) == 0:
                            continue

                        for author in enhanced_authors:
                            if not author.get('has_published_in_aps', False):
                                continue

                            demographics = set()
                            gender = author.get('gender')
                            ethnicity = author.get('ethnicity')
                            decade_first_pub = author.get('decade_first_publication')
                            is_nobel = author.get('is_nobel_laureate')

                            if gender:
                                demographics.add(f"gender_{gender}")
                            if ethnicity:
                                demographics.add(f"ethnicity_{ethnicity}")
                            if decade_first_pub:
                                demographics.add(f"decade_{decade_first_pub}")
                            if is_nobel is not None:
                                demographics.add(f"nobel_{'yes' if is_nobel else 'no'}")

                            author_metrics_sets[run][author.get('id_author')] = demographics

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {'metrics_dictionary': []}

                if author_metrics_sets[run]:
                    use_case_summary[use_case]['metrics_dictionary'].append(author_metrics_sets)

        results[config] = use_case_summary

    return results

def compute_bootstrap_p_value(observed_avg, authors_demographics, author_decade, nobel_laureates, num_authors, seed=43, n_iterations=1000):
    if seed is not None:
        np.random.seed(seed)

    bootstrap_means = []

    for _ in range(n_iterations):
        bootstrapped_jaccard_scores = []

        for i in range(len(num_authors)):
            sampled_authors = random.sample(authors_demographics, int(num_authors[i]))
            authors_categorical_data = {}

            for author in sampled_authors:
                author_id = author['id_author']
                openalex_id = author['openalex_id']
                demographics = set()

                if author.get("gender"):
                    demographics.add(f"gender_{author['gender']}")
                if author.get("ethnicity"):
                    demographics.add(f"ethnicity_{author['ethnicity']}")
                if decade := author_decade.get(author_id):
                    demographics.add(f"decade_{decade}")
                
                demographics.add(f"nobel_{'yes' if openalex_id in nobel_laureates else 'no'}")
                authors_categorical_data[author_id] = demographics

            author_pairs = list(itertools.combinations(authors_categorical_data.keys(), 2))
            jaccard_scores = [
                jaccard_similarity(set(authors_categorical_data[a1]), set(authors_categorical_data[a2]))
                for a1, a2 in author_pairs
            ]
            bootstrapped_jaccard_scores.extend(jaccard_scores)

        if bootstrapped_jaccard_scores:
            bootstrap_means.append(np.mean(bootstrapped_jaccard_scores))

    p_value = np.mean(np.array(bootstrap_means) >= observed_avg)
    return p_value, bootstrap_means

def compute_jaccard_with_bootstrap_and_store_in_dataframe(results, demographics, author_decade, nobel_laureates):
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
            all_jaccard_scores = []
            num_authors = []  # To store the number of authors for each run

            for affiliation_dict in data['metrics_dictionary']:
                for run, authors_affiliations in affiliation_dict.items():
                    # Get the number of authors for this run
                    num_authors.append(len(authors_affiliations.keys()))
                    
                    # Get all author combinations for this run
                    author_pairs = list(itertools.combinations(authors_affiliations.keys(), 2))
                    # Compute Jaccard similarity for each pair
                    jaccard_scores = [
                        jaccard_similarity(authors_affiliations[a1], authors_affiliations[a2])
                        for a1, a2 in author_pairs
                    ]
                    # Append the scores to the list for this use case
                    all_jaccard_scores.extend(jaccard_scores)

            # Calculate the observed average Jaccard similarity for this use case
            if all_jaccard_scores:
                observed_avg = np.mean(all_jaccard_scores)
                observed_std = np.std(all_jaccard_scores)
                
                # Compute the p-value and the full bootstrap_means using bootstrapping
                p_value, bootstrap_means_list = compute_bootstrap_p_value(observed_avg, demographics, author_decade, nobel_laureates, num_authors)

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
                all_scores_for_config.extend(all_jaccard_scores)
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

    # sort rows alphabetically but keep overall as last one
    df_combined = df_combined.reindex(sorted(df_combined.index[:-1]) + [df_combined.index[-1]])

    return df_combined

def fishers_method(p_values):
    p_values = [p for p in p_values if p not in ['No Data', None]]
    if len(p_values) == 0:
        return None

    X2_statistic = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = chi2.sf(X2_statistic, df)
    return combined_p

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
    demographics_file_path = "./organised_data/authors_demographics.json"
    author_stats_file_path = "./organised_data/aps_author_stats.json"
    nobel_file_path = "./organised_data/nobel_prize_with_openalex.json"
    output_path = "./output_results/analysis/categorical_metrics_jaccard_similarity/table_output.tex"

    # Load data
    demographics = load_ndjson(demographics_file_path)
    author_stats = load_ndjson(author_stats_file_path)
    author_decade = get_author_decades(author_stats)
    nobel_laureates = get_nobel_laureates(nobel_file_path)

    # Process data and compute Jaccard similarities
    results = process_data(input_dir)
    df_jaccard = compute_jaccard_with_bootstrap_and_store_in_dataframe(
        results, demographics, author_decade, nobel_laureates
    )

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