import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def get_author_percentiles(aps_author_ranking):
    return {
        'rr1': {entry['id_author']: entry['rr1_rank_publications_percentile'] for entry in aps_author_ranking},
        'rr2': {entry['id_author']: entry['rr2_rank_citations_percentile'] for entry in aps_author_ranking},
        'rr3': {entry['id_author']: entry['rr3_rank_h_index_percentile'] for entry in aps_author_ranking},
        'rr4': {entry['id_author']: entry['rr4_rank_i10_index_percentile'] for entry in aps_author_ranking},
        'rr5': {entry['id_author']: entry['rr5_rank_e_index_percentile'] for entry in aps_author_ranking},
        'rr6': {entry['id_author']: entry['rr6_rank_citation_publication_age_percentile'] for entry in aps_author_ranking},
        # 'rr7': {entry['id_author']: entry['rr7_rank_mean_citedness_2yr_percentile'] for entry in aps_author_ranking}
    }

def process_data(root_dir, aps_author_ranking):
    results = {}
    percentiles = get_author_percentiles(aps_author_ranking)

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
                        author_rankings_sets = {run: {}}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if not enhanced_authors:
                            continue

                        for author in enhanced_authors:
                            if not author.get('has_published_in_aps', False):
                                continue

                            author_id = author.get('id_author')
                            rankings = [
                                percentiles['rr1'].get(author_id, 0),
                                percentiles['rr2'].get(author_id, 0),
                                percentiles['rr3'].get(author_id, 0),
                                percentiles['rr4'].get(author_id, 0),
                                percentiles['rr5'].get(author_id, 0),
                                percentiles['rr6'].get(author_id, 0)
                            ]
                            author_rankings_sets[run][author_id] = rankings

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {'rankings_dictionary': []}

                if author_rankings_sets[run]:
                    use_case_summary[use_case]['rankings_dictionary'].append(author_rankings_sets)

        results[config] = use_case_summary

    return results

def create_ranking_tables_avg_std_with_overall(results, config_list):
    ranking_columns = [
        'rr1_rank_publications_percentile',
        'rr2_rank_citations_percentile',
        'rr3_rank_h_index_percentile',
        'rr4_rank_i10_index_percentile',
        'rr5_rank_e_index_percentile',
        'rr6_rank_citation_publication_age_percentile'
    ]

    summary_data = []

    for config in config_list:
        use_cases = results.get(config, {})
        all_use_cases = sorted(use_cases.keys())

        ranking_summary = {use_case: {ranking: '-' for ranking in ranking_columns} 
                         for use_case in all_use_cases}
        overall_ranking_list = {ranking: [] for ranking in ranking_columns}

        for use_case, use_case_data in use_cases.items():
            run_rankings = {ranking: [] for ranking in ranking_columns}

            for run_data in use_case_data['rankings_dictionary']:
                for run_id, author_data in run_data.items():
                    for author_id, rankings in author_data.items():
                        for idx, ranking_column in enumerate(ranking_columns):
                            run_rankings[ranking_column].append(rankings[idx])
                            overall_ranking_list[ranking_column].append(rankings[idx])

            for ranking_column in ranking_columns:
                if run_rankings[ranking_column]:
                    mean_ranking = np.mean(run_rankings[ranking_column])
                    std_ranking = np.std(run_rankings[ranking_column])
                    ranking_summary[use_case][ranking_column] = f"{mean_ranking:.3f} ± {std_ranking:.3f}"
                else:
                    ranking_summary[use_case][ranking_column] = '--'

        overall_row = {}
        for ranking_column in ranking_columns:
            if overall_ranking_list[ranking_column]:
                overall_mean = np.mean(overall_ranking_list[ranking_column])
                overall_std = np.std(overall_ranking_list[ranking_column])
                overall_row[ranking_column] = f"{overall_mean:.3f} ± {overall_std:.3f}"
            else:
                overall_row[ranking_column] = '--'
        ranking_summary['Overall'] = overall_row

        df = pd.DataFrame.from_dict(ranking_summary, orient='index').reset_index()
        df = df.rename(columns={'index': 'Use Case'})
        summary_data.append(df)

    return summary_data

def save_table_to_latex(df, output_path, model_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def escape_latex_special_chars(text):
        return re.sub(r"([_&])", r"\\\1", text)

    with open(output_path, 'w') as f:
        f.write("\\begin{table*}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Ranking Analysis Results for {model_name}}}\n")
        f.write("\\label{tbl:ranking_analysis_results}\n")
        f.write("\\begin{tabular}{l|ccccc}\n")
        f.write("\\toprule\n")
        
        # Write headers
        headers = [escape_latex_special_chars(col) for col in df.columns]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Write data rows
        for _, row in df.iterrows():
            row_data = [escape_latex_special_chars(str(val)) for val in row]
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

def main():
    # Set paths
    input_dir = "./experiments_validation_results/updated_organized_results"
    rank_with_percentile_path = "./organised_data/aps_author_rankings_with_percentile.json"
    output_dir = "./output_results/rankings"
    config_path = "../model_config.json"

    # Load data
    print("Loading ranking data...")
    aps_author_ranking = load_ndjson(rank_with_percentile_path)

    # Process data
    print("Processing data...")
    results = process_data(input_dir, aps_author_ranking)

    # Generate tables
    with open(config_path) as f:
        config_list = [f"config_{model}" for model in json.load(f)['models']]
    
    ranking_tables = create_ranking_tables_avg_std_with_overall(results, config_list)

    # Save tables
    print("Saving tables...")
    for i, (config, table) in enumerate(zip(config_list, ranking_tables)):
        model_name = config.replace('config_', '')
        output_path = os.path.join(output_dir, f"table_output_{model_name}.tex")
        save_table_to_latex(table, output_path, model_name)
        print(f"Saved table for {model_name}")

    print("Analysis complete.")

if __name__ == "__main__":
    main()