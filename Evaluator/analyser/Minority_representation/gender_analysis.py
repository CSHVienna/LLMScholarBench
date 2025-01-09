import os
import json
import numpy as np
import pandas as pd

from itertools import combinations

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

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
                        author_metrics_sets = {run: {}}

                        enhanced_authors = data.get('enhanced_authors', [])
                        if not enhanced_authors:
                            continue

                        for author in enhanced_authors:
                            if not author.get('has_published_in_aps', False):
                                continue

                            demographics = set()
                            gender = author.get('gender')
                            ethnicity = author.get('ethnicity')

                            if gender:
                                demographics.add(f"gender_{gender}")
                            if ethnicity:
                                demographics.add(f"ethnicity_{ethnicity}")

                            author_metrics_sets[run][author.get('id_author')] = demographics

                if use_case not in use_case_summary:
                    use_case_summary[use_case] = {'metrics_dictionary': []}

                if author_metrics_sets[run]:
                    use_case_summary[use_case]['metrics_dictionary'].append(author_metrics_sets)

        results[config] = use_case_summary

    return results

def create_gender_ethnicity_tables_avg_std_with_overall(results, config_list, table_type='gender'):
    label_mapping = {
        'white': 'White',
        'hispanic': 'Hispanic',
        'Hispanic or Latino': 'Hispanic',
        'Androgynous': 'Unknown',
    }

    summary_data = []

    for config in config_list:
        use_cases = results.get(config, {})
        all_use_cases = sorted(use_cases.keys())

        if table_type == 'gender':
            label_key_prefix = 'gender_'
            categories = ['Male', 'Female', 'Unknown']
        else:
            label_key_prefix = 'ethnicity_'
            categories = []
            for use_case_data in use_cases.values():
                for metrics in use_case_data['metrics_dictionary']:
                    for run_data in metrics.values():
                        for demographics in run_data.values():
                            for label in demographics:
                                if label.startswith(label_key_prefix):
                                    category = label.split('_')[1]
                                    category = label_mapping.get(category, category)
                                    categories.append(category)
            categories = sorted(set(categories))

        proportions_summary = {use_case: {category: '-' for category in categories} 
                             for use_case in all_use_cases}
        overall_run_ratios = {category: [] for category in categories}

        for use_case, use_case_data in use_cases.items():
            run_ratios = {category: [] for category in categories}

            for metrics in use_case_data['metrics_dictionary']:
                for run_data in metrics.values():
                    total_authors = 0
                    category_counts = {category: 0 for category in categories}

                    for author_id, demographics in run_data.items():
                        total_authors += 1
                        for label in demographics:
                            if label.startswith(label_key_prefix):
                                category = label.split('_')[1]
                                category = label_mapping.get(category, category)
                                category_counts[category] += 1

                    if total_authors > 0:
                        for category in categories:
                            proportion = category_counts[category] / total_authors
                            run_ratios[category].append(proportion)
                            overall_run_ratios[category].append(proportion)

            for category in categories:
                if run_ratios[category]:
                    mean_ratio = np.mean(run_ratios[category])
                    std_ratio = np.std(run_ratios[category])
                    proportions_summary[use_case][category] = f"{mean_ratio:.3f} ± {std_ratio:.3f}"
                else:
                    proportions_summary[use_case][category] = '--'

        overall_row = {}
        for category in categories:
            if overall_run_ratios[category]:
                overall_mean = np.mean(overall_run_ratios[category])
                overall_std = np.std(overall_run_ratios[category])
                overall_row[category] = f"{overall_mean:.3f} ± {overall_std:.3f}"
            else:
                overall_row[category] = '--'
        proportions_summary['Overall'] = overall_row

        df = pd.DataFrame.from_dict(proportions_summary, orient='index').reset_index()
        df = df.rename(columns={'index': 'Use Case'})
        summary_data.append(df)

    return summary_data

def generate_latex_table(data, config, table_type, folder_path):
    """Generate LaTeX table for gender or ethnicity data"""
    table_caption = f"{table_type} representation across use cases for {config}."
    table_label = f"tbl:{table_type.lower()}_{config}"

    if table_type.lower() == 'ethnicity':
        column_mapping = {
            'asian': 'Asian',
            'black': 'Black'
        }
        data = data.rename(columns={col: column_mapping.get(col.lower(), col) for col in data.columns})
        column_order = ['Use Case', 'Asian', 'White', 'Hispanic', 'Black', 'Unknown']
        data = data[[col for col in column_order if col in data.columns]]

    def escape_latex_special_chars(text):
        return text.replace('_', '\\_').replace('&', '\\&')

    def format_float(value):
        try:
            if '±' in str(value):
                parts = value.split('±')
                mean = float(parts[0])
                std = float(parts[1])
                return f"{mean:.3f} ± {std:.3f}"
            return value
        except:
            return value

    latex_str = (
        "\\begin{table*}[h]\n"
        "\\centering\n"
        f"\\caption{{{escape_latex_special_chars(table_caption)}}}\n"
        f"\\label{{{escape_latex_special_chars(table_label)}}}\n"
        "\\begin{tabular}{l" + "".join(["R{2.2cm}" for _ in range(len(data.columns)-1)]) + "}\n"
        "\\toprule\n"
    )

    # Headers
    headers = [escape_latex_special_chars(col) for col in data.columns]
    latex_str += " & ".join(headers) + " \\\\\n\\midrule\n"

    # Data rows
    for _, row in data.iterrows():
        row_values = [escape_latex_special_chars(str(row[col])) for col in data.columns]
        latex_str += " & ".join(row_values) + " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table*}"

    # Save to file
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{table_type.lower()}_{config}.tex")
    with open(file_path, "w") as f:
        f.write(latex_str)

def main():
    # Set paths
    input_dir = "./experiments_validation_results/updated_organized_results"
    output_folder = "./output_results/minority_representation"

    # Process data
    print("Processing demographic data...")
    results = process_data(input_dir)

    # Generate tables
    with open('model_config.json') as f:
        config_list = [f"config_{model}" for model in json.load(f)['models']]

    print("Generating gender tables...")
    gender_tables = create_gender_ethnicity_tables_avg_std_with_overall(
        results, config_list, table_type='gender'
    )

    print("Generating ethnicity tables...")
    ethnicity_tables = create_gender_ethnicity_tables_avg_std_with_overall(
        results, config_list, table_type='ethnicity'
    )

    # Save tables
    print("Saving tables...")
    for idx, (config, gender_table, ethnicity_table) in enumerate(zip(config_list, gender_tables, ethnicity_tables)):
        generate_latex_table(gender_table, config, 'Gender', output_folder)
        generate_latex_table(ethnicity_table, config, 'Ethnicity', output_folder)

    print(f"Analysis complete. Results saved to {output_folder}")

if __name__ == "__main__":
    main()