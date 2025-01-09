import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load JSON data (standard JSON)
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_nobel_data(root_dir):
    results = {}
    all_use_cases = set()  # Collect all unique use cases across all configurations

    for config in os.listdir(root_dir):
        config_path = os.path.join(root_dir, config)
        if not os.path.isdir(config_path):
            continue
        
        use_case_summary = {}
        
        # Loop through each run in the config directory
        for run in os.listdir(config_path):
            run_path = os.path.join(config_path, run)
            if not os.path.isdir(run_path):
                continue

            # Loop through each use case in the run directory
            for use_case in os.listdir(run_path):
                use_case_path = os.path.join(run_path, use_case)
                if not os.path.isdir(use_case_path):
                    continue

                all_use_cases.add(use_case)  # Add to the set of all use cases

                nobel_count = 0
                total_authors = 0
                nobel_years = []
                all_physics = True
                has_valid_authors = False

                # Check each file in the use_case directory
                for file in os.listdir(use_case_path):
                    if file.startswith("validation_result_") and file.endswith(".json"):
                        file_path = os.path.join(use_case_path, file)
                        data = load_json(file_path)

                        enhanced_authors = data.get('enhanced_authors', [])

                        if len(enhanced_authors) == 0:
                            continue

                        # If we reach this point, we have valid authors
                        has_valid_authors = True
                        total_authors += len(enhanced_authors)

                        for author in enhanced_authors:
                            if author.get('is_nobel_laureate', False):
                                nobel_count += 1
                                nobel_years.append(author.get('nobel_year', '--'))

                                if author.get('nobel_category') != "Physics":
                                    all_physics = False

                # Store the results for the use case
                if has_valid_authors:
                    if use_case not in use_case_summary:
                        use_case_summary[use_case] = {
                            'nobel_count': [],
                            'total_authors': [],
                            'nobel_years': [],
                            'all_physics': []
                        }

                    use_case_summary[use_case]['nobel_count'].append(nobel_count)
                    use_case_summary[use_case]['total_authors'].append(total_authors)
                    use_case_summary[use_case]['nobel_years'].append(nobel_years)
                    use_case_summary[use_case]['all_physics'].append(all_physics)

        results[config] = use_case_summary
    return results, all_use_cases


# Function to create summary table and save as CSV and PNG
# Function to create summary table and save as CSV, LaTeX, and PNG
def create_summary_table(results, all_use_cases, output_folder):
    for config, use_cases in results.items():
        data = []

        # Overall data lists to collect all valid entries across use cases
        overall_nobel_counts = []
        overall_total_authors = []
        overall_nobel_years = []
        overall_ratios = []
        all_physics_consistent = True  # Start as True, but flip to False if any use case is not Physics

        for use_case in all_use_cases:  # Iterate over all possible use cases
            if use_case in use_cases:
                values = use_cases[use_case]
                nobel_counts = values['nobel_count']
                total_authors = values['total_authors']
                nobel_years = [year for years in values['nobel_years'] for year in years]  # Flatten years

                # If valid authors were found but no Nobel prizes were awarded
                if all(v == 0 for v in nobel_counts):
                    avg_nobel = "0"
                    std_nobel = "0"
                    median_nobel = "0"
                    avg_years = "--"
                    std_years = "--"
                    median_years = "--"
                    all_physics = "False"
                    all_physics_consistent = False  # If any use case is False, set overall to False
                    
                    if not total_authors or all(v == 0 for v in total_authors):
                        avg_ratio = "--"
                        std_ratio = "--"
                    else:
                        ratios = [
                            count / total if total != 0 else 0
                            for count, total in zip(nobel_counts, total_authors)
                        ]
                        avg_ratio = f"{np.mean(ratios):.2f} ± {np.std(ratios):.2f}"
                        overall_ratios.extend(ratios)  # Collect valid ratios
                else:
                    # Valid authors and some Nobel laureates found
                    avg_nobel = f"{np.mean(nobel_counts):.2f} ± {np.std(nobel_counts):.2f}"
                    median_nobel = f"{np.median(nobel_counts):.2f}"

                    if nobel_years:
                        avg_years = f"{np.mean(nobel_years):.0f} ± {np.std(nobel_years):.0f}"
                        median_years = f"{np.median(nobel_years):.0f}"
                    else:
                        avg_years = "--"
                        std_years = "--"
                        median_years = "--"

                    all_physics = "True" if all(values['all_physics']) else "False"
                    if all_physics == "False":
                        all_physics_consistent = False
                    
                    if not total_authors or all(v == 0 for v in total_authors):
                        avg_ratio = "--"
                        std_ratio = "--"
                    else:
                        ratios = [
                            count / total if total != 0 else 0
                            for count, total in zip(nobel_counts, total_authors)
                        ]
                        avg_ratio = f"{np.mean(ratios):.2f} ± {np.std(ratios):.2f}"
                        overall_ratios.extend(ratios)  # Collect valid ratios

                # Collect overall data for this use case
                overall_nobel_counts.extend(nobel_counts)
                overall_total_authors.extend(total_authors)
                overall_nobel_years.extend(nobel_years)

            else:
                # If the use case is not found for this config, fill in with '--'
                avg_nobel = "--"
                std_nobel = "--"
                median_nobel = "--"
                avg_years = "--"
                std_years = "--"
                median_years = "--"
                all_physics = "--"
                avg_ratio = "--"
                std_ratio = "--"

            # Append data for this use case
            data.append([
                use_case,
                avg_nobel,
                median_nobel,
                avg_ratio,
                f"{avg_years}, {median_years}",
                all_physics
            ])

        # Create DataFrame for the current config
        df = pd.DataFrame(data, columns=[
            'Use Case', 'Nobel (avg ± std)', 'Nobel (median)', 'Ratio (avg ± std)',
            'Year Awarded (avg ± std, median)', 'All Physics?'
        ])

        # Sort the DataFrame by use case names
        df.sort_values(by=['Use Case'], inplace=True)

        # Add the overall row at the end after sorting
        if overall_nobel_counts:
            overall_avg_nobel = f"{np.mean(overall_nobel_counts):.2f} ± {np.std(overall_nobel_counts):.2f}"
            overall_median_nobel = f"{np.median(overall_nobel_counts):.2f}"
        else:
            overall_avg_nobel = overall_median_nobel = "--"

        if overall_nobel_years:
            overall_avg_years = f"{np.mean(overall_nobel_years):.0f} ± {np.std(overall_nobel_years):.0f}"
            overall_median_years = f"{np.median(overall_nobel_years):.0f}"
        else:
            overall_avg_years = overall_median_years = "--"

        if overall_ratios:
            overall_avg_ratio = f"{np.mean(overall_ratios):.2f} ± {np.std(overall_ratios):.2f}"
        else:
            overall_avg_ratio = "--"

        overall_physics = "True" if all_physics_consistent else "False"

        # Append the overall row to the DataFrame
        df.loc[len(df.index)] = [
            'Overall',
            overall_avg_nobel,
            overall_median_nobel,
            overall_avg_ratio,
            f"{overall_avg_years}, {overall_median_years}",
            overall_physics
        ]

        # Save the DataFrame as a CSV file
        output_file_csv = os.path.join(output_folder, f"summary_table_nobel_{config}.csv")
        df.to_csv(output_file_csv, index=False)
        print(f"Saved summary table for {config} at {output_file_csv}")

        # Save the DataFrame as a LaTeX file
        output_file_latex = os.path.join(output_folder, f"summary_table_nobel_{config}.tex")
        with open(output_file_latex, 'w') as latex_file:
            latex_file.write(df.to_latex(index=False, longtable=True))
        print(f"Saved summary table for {config} as LaTeX at {output_file_latex}")

        # Save the DataFrame as a PNG file
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        output_file_png = os.path.join(output_folder, f"summary_table_nobel_{config}.png")
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary table as PNG for {config} at {output_file_png}")


# Main function
def main():
    input_dir = "././experiments_validation_results/updated_organized_results"
    output_folder = "././output_results/analysis/nobel_prizes"
   
    os.makedirs(output_folder, exist_ok=True)

    print("Processing data...")
    results, all_use_cases = process_nobel_data(input_dir)

    print("Creating summary tables...")
    create_summary_table(results, all_use_cases, output_folder)

    print(f"Results saved in {output_folder}")

if __name__ == "__main__":
    main()
