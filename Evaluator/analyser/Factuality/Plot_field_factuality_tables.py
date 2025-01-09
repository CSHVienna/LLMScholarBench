import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load the JSON file
file_path = "././experiments_validation_results/updated_organized_results/descriptive_statistics_by_run.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to compute avg, std, median, mode, min, max after applying mask
def compute_statistics(values):
    if not values:
        return '--', '--', '--', '--', '--'
    
    avg = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    mode = pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else '--'
    min_val = np.min(values)
    max_val = np.max(values)
    
    return f'{avg:.2f} ± {std:.2f}', f'Median: {median}, Mode: {mode}, Min: {min_val}, Max: {max_val}'

# Function to create a table for each configuration and field use case
def create_field_table(config_data):
    rows = []
    overall_dois_list = []
    overall_unique_dois_list = []
    overall_openalex_pub_list = []
    overall_aps_pub_list = []
    overall_correct_authorship_list = []
    overall_correct_field_list = []
    overall_author_correct_field_list = []
    overall_ratio_author_correct_field_list = []
    overall_ratio_correct_field_list = []

    for use_case, metrics in config_data.items():
        if 'field' in use_case:  # Only process field use cases
            row = {'Field Use Case': use_case}
            
            # Apply mask based on total_dois > 0
            mask = [dois != 0 for dois in metrics['total_dois']]
            
            # Clean the lists using the mask
            cleaned_dois = [dois for dois, m in zip(metrics['total_dois'], mask) if m]
            cleaned_unique_dois = [doi for doi, m in zip(metrics['unique_dois'], mask) if m]
            cleaned_openalex_pub = [oa for oa, m in zip(metrics['count_in_openalex_pub'], mask) if m]
            cleaned_aps_pub = [aps for aps, m in zip(metrics['count_aps_publication'], mask) if m]
            cleaned_correct_authorship = [ca for ca, m in zip(metrics['count_correct_authorship'], mask) if m]
            cleaned_correct_field = [cf for cf, m in zip(metrics['count_correct_field'], mask) if m]
            cleaned_author_correct_field = [acf for acf, m in zip(metrics['author_correct_field'], mask) if m]

            # Calculate and store statistics for each metric
            dois_stats = compute_statistics(cleaned_dois)
            row['Total DOIs Avg'] = dois_stats[0]
            row['Total DOIs (Median, Mode, Min, Max)'] = dois_stats[1]

            unique_dois_stats = compute_statistics(cleaned_unique_dois)
            row['Unique DOIs Avg'] = unique_dois_stats[0]
            row['Unique DOIs (Median, Mode, Min, Max)'] = unique_dois_stats[1]

            openalex_pub_stats = compute_statistics(cleaned_openalex_pub)
            row['OpenAlex Publications Avg'] = openalex_pub_stats[0]
            row['OpenAlex Publications (Median, Mode, Min, Max)'] = openalex_pub_stats[1]

            aps_pub_stats = compute_statistics(cleaned_aps_pub)
            row['APS Publications Avg'] = aps_pub_stats[0]
            row['APS Publications (Median, Mode, Min, Max)'] = aps_pub_stats[1]

            correct_authorship_stats = compute_statistics(cleaned_correct_authorship)
            row['Correct Authorship Avg'] = correct_authorship_stats[0]
            row['Correct Authorship (Median, Mode, Min, Max)'] = correct_authorship_stats[1]

            correct_field_stats = compute_statistics(cleaned_correct_field)
            row['Correct Field Avg'] = correct_field_stats[0]
            row['Correct Field (Median, Mode, Min, Max)'] = correct_field_stats[1]

            author_correct_field_stats = compute_statistics(cleaned_author_correct_field)
            row['Author Correct Field Avg'] = author_correct_field_stats[0]
            row['Author Correct Field (Median, Mode, Min, Max)'] = author_correct_field_stats[1]

            # Calculate ratios and add new columns
            ratio_author_correct_field = [acf / name for acf, name, m in zip(metrics['author_correct_field'], metrics['count_names'], mask) if m]
            ratio_correct_field = [cf / dois for cf, dois, m in zip(metrics['count_correct_field'], metrics['total_dois'], mask) if m]

            ratio_author_correct_field_stats = compute_statistics(ratio_author_correct_field)
            ratio_correct_field_stats = compute_statistics(ratio_correct_field)

            row['Ratio Author Correct Field / Number of Names Avg'] = ratio_author_correct_field_stats[0]
            row['Ratio Author Correct Field / Number of Names (Median, Mode, Min, Max)'] = ratio_author_correct_field_stats[1]

            row['Ratio Pub Correct Field / Total DOIs Avg'] = ratio_correct_field_stats[0]
            row['Ratio Pub Correct Field / Total DOIs (Median, Mode, Min, Max)'] = ratio_correct_field_stats[1]


            # Append to the overall lists for calculating overall stats
            overall_dois_list.extend(cleaned_dois)
            overall_unique_dois_list.extend(cleaned_unique_dois)
            overall_openalex_pub_list.extend(cleaned_openalex_pub)
            overall_aps_pub_list.extend(cleaned_aps_pub)
            overall_correct_authorship_list.extend(cleaned_correct_authorship)
            overall_correct_field_list.extend(cleaned_correct_field)
            overall_author_correct_field_list.extend(cleaned_author_correct_field)
            overall_ratio_author_correct_field_list.extend(ratio_author_correct_field)
            overall_ratio_correct_field_list.extend(ratio_correct_field)

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add the 'Overall' row
    overall_row = {
        'Field Use Case': 'Overall',
        'Total DOIs Avg': compute_statistics(overall_dois_list)[0],
        'Total DOIs (Median, Mode, Min, Max)': compute_statistics(overall_dois_list)[1],
        'Unique DOIs Avg': compute_statistics(overall_unique_dois_list)[0],
        'Unique DOIs (Median, Mode, Min, Max)': compute_statistics(overall_unique_dois_list)[1],
        'OpenAlex Publications Avg': compute_statistics(overall_openalex_pub_list)[0],
        'OpenAlex Publications (Median, Mode, Min, Max)': compute_statistics(overall_openalex_pub_list)[1],
        'APS Publications Avg': compute_statistics(overall_aps_pub_list)[0],
        'APS Publications (Median, Mode, Min, Max)': compute_statistics(overall_aps_pub_list)[1],
        'Correct Authorship Avg': compute_statistics(overall_correct_authorship_list)[0],
        'Correct Authorship (Median, Mode, Min, Max)': compute_statistics(overall_correct_authorship_list)[1],
        'Correct Field Avg': compute_statistics(overall_correct_field_list)[0],
        'Correct Field (Median, Mode, Min, Max)': compute_statistics(overall_correct_field_list)[1],
        'Author Correct Field Avg': compute_statistics(overall_author_correct_field_list)[0],
        'Author Correct Field (Median, Mode, Min, Max)': compute_statistics(overall_author_correct_field_list)[1],
        'Ratio Author Correct Field / Number of Names Avg': compute_statistics(overall_ratio_author_correct_field_list)[0],
        'Ratio Author Correct Field / Number of Names (Median, Mode, Min, Max)': compute_statistics(overall_ratio_author_correct_field_list)[1],
        'Ratio Pub Correct Field / Total DOIs Avg': compute_statistics(overall_ratio_correct_field_list)[0],
        'Ratio Pub Correct Field / Total DOIs (Median, Mode, Min, Max)': compute_statistics(overall_ratio_correct_field_list)[1],
    }
    
    # Append the overall row
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    
    return df

# Function to save table as PNG
def save_as_png(df, filename):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Function to save table as TEX
# Function to save table as TEX
def save_as_tex(df, filename, caption):
    with open(filename, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\label{tab:" + os.path.splitext(os.path.basename(filename))[0] + "}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        tex = df.to_latex(index=False, escape=True)
        f.write(tex)
        f.write("}\n")
        f.write("\\end{table}\n")

# Main logic to create the tables for each configuration
if __name__ == "__main__":
    # Create main output folder under 'output_results/factuality/factuality_field'
    factuality_field_folder = os.path.join('output_results/factuality', 'factuality_field')
    os.makedirs(factuality_field_folder, exist_ok=True)

    # Create subfolders for full, avg, and other metrics
    full_table_folder = os.path.join(factuality_field_folder, 'full_table')
    avg_table_folder = os.path.join(factuality_field_folder, 'avg_table')
    other_metrics_table_folder = os.path.join(factuality_field_folder, 'other_metrics_table')

    os.makedirs(full_table_folder, exist_ok=True)
    os.makedirs(avg_table_folder, exist_ok=True)
    os.makedirs(other_metrics_table_folder, exist_ok=True)

    # Process each configuration in the data and create the respective tables
    for config_name, config_data in data.items():
        # Create the table for each configuration
        table = create_field_table(config_data)
        
        # Generate filenames for saving the tables
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{config_name.replace(' ', '_')}_{timestamp}"

        # Full table (all metrics)
        full_png_filename = os.path.join(full_table_folder, f"{base_filename}.png")
        full_tex_filename = os.path.join(full_table_folder, f"{base_filename}.tex")

        # Save the full table
        save_as_png(table, full_png_filename)
        caption = f"Full descriptive statistics for {config_name}"
        save_as_tex(table, full_tex_filename, caption)

        # Avg only table
        avg_df = table[['Field Use Case', 'Total DOIs Avg', 'Unique DOIs Avg', 'OpenAlex Publications Avg', 
                        'APS Publications Avg', 'Correct Authorship Avg', 'Correct Field Avg', 
                        'Author Correct Field Avg', 'Ratio Author Correct Field / Number of Names Avg', 
                        'Ratio Pub Correct Field / Total DOIs Avg']]
        avg_png_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.png")
        avg_tex_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.tex")

        # Save the avg table
        save_as_png(avg_df, avg_png_filename)
        caption_avg = f"Average statistics for {config_name}"
        save_as_tex(avg_df, avg_tex_filename, caption_avg)

        # Other metrics (median, mode, min/max)
        other_metrics_df = table[['Field Use Case', 'Total DOIs (Median, Mode, Min, Max)', 'Unique DOIs (Median, Mode, Min, Max)', 
                                  'OpenAlex Publications (Median, Mode, Min, Max)', 'APS Publications (Median, Mode, Min, Max)', 
                                  'Correct Authorship (Median, Mode, Min, Max)', 'Correct Field (Median, Mode, Min, Max)', 
                                  'Author Correct Field (Median, Mode, Min, Max)', 'Ratio Author Correct Field / Number of Names (Median, Mode, Min, Max)', 
                                  'Ratio Pub Correct Field / Total DOIs (Median, Mode, Min, Max)']]
        other_metrics_png_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.png")
        other_metrics_tex_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.tex")

        # Save the other metrics table
        save_as_png(other_metrics_df, other_metrics_png_filename)
        caption_other = f"Other metrics (Median, Mode, Min/Max) for {config_name}"
        save_as_tex(other_metrics_df, other_metrics_tex_filename, caption_other)

        print(f"Saved full, avg, and other metrics tables for {config_name}")

    print(f"All tables have been saved in {factuality_field_folder}.")

