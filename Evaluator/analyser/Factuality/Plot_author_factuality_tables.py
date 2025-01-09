import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load the JSON file
file_path =  "././experiments_validation_results/updated_organized_results/descriptive_statistics_by_run.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to compute avg, std, median, mode, min, max without removing zeros inside
def compute_statistics(values):
    if not values:
        return '--', '--', '--', '--', '--'
    
    avg = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    mode = pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else '--'
    min_val = np.min(values)
    max_val = np.max(values)
    
    return f'{avg:.2f} ± {std:.2f}', median, mode, (min_val, max_val)


# Function to compute the ratio and its statistics
def compute_ratio(count_names, count_other):
    ratios = []
    for name, other in zip(count_names, count_other):
        if name != 0:  # Ensure only valid entries are considered
            ratios.append(other / name)  # Keep 0 in the division
    
    if not ratios:
        return '--'
    
    avg = np.mean(ratios)
    std = np.std(ratios)
    return f'{avg:.2f} ± {std:.2f}'

# Function to create the "Overall" row
def compute_overall_column(combined_list):
    if not combined_list:
        return '--'
    avg = np.mean(combined_list)
    std = np.std(combined_list)
    return f'{avg:.2f} ± {std:.2f}'

# Create a table for each configuration
def create_table(config_data):
    rows = []
    overall_names_list = []
    overall_oa_list = []
    overall_aps_list = []
    overall_ratio_oa_list = []
    overall_ratio_aps_list = []
    
    for use_case, metrics in config_data.items():
        row = {'Use Case': use_case}
        
        # Apply mask: Only consider entries where count_names > 0
        mask = [name != 0 for name in metrics['count_names']]
        
        # Apply mask to the metrics and then compute statistics
        cleaned_names = [name for name, m in zip(metrics['count_names'], mask) if m]
        cleaned_oa = [oa for oa, m in zip(metrics['count_in_oa'], mask) if m]
        cleaned_aps = [aps for aps, m in zip(metrics['count_in_aps'], mask) if m]

        # Calculate statistics for count_names (after applying mask)
        names_stats = compute_statistics(cleaned_names)
        row['Total Names (avg ± std)'] = names_stats[0]
        row['Total Names Median'] = names_stats[1]
        row['Total Names Mode'] = names_stats[2]
        row['Total Names (min, max)'] = names_stats[3]

        # Calculate statistics for count_in_oa (after applying mask)
        oa_stats = compute_statistics(cleaned_oa)
        row['Present in OpenAlex (avg ± std)'] = oa_stats[0]
        row['Present in OpenAlex Median'] = oa_stats[1]
        row['Present in OpenAlex Mode'] = oa_stats[2]
        row['Present in OpenAlex (min, max)'] = oa_stats[3]

        # Calculate statistics for count_in_aps (after applying mask)
        aps_stats = compute_statistics(cleaned_aps)
        row['Present in APS (avg ± std)'] = aps_stats[0]
        row['Present in APS Median'] = aps_stats[1]
        row['Present in APS Mode'] = aps_stats[2]
        row['Present in APS (min, max)'] = aps_stats[3]

        # Calculate the ratio for Present in OA / Total Names (after applying mask)
        cleaned_ratios_oa = [oa / name for name, oa, m in zip(metrics['count_names'], metrics['count_in_oa'], mask) if m]
        ratio_oa = compute_statistics(cleaned_ratios_oa)[0]
        row['Ratio Present in OA (avg ± std)'] = ratio_oa

        # Calculate the ratio for Present in APS / Total Names (after applying mask)
        cleaned_ratios_aps = [aps / name for name, aps, m in zip(metrics['count_names'], metrics['count_in_aps'], mask) if m]
        ratio_aps = compute_statistics(cleaned_ratios_aps)[0]
        row['Ratio Present in APS (avg ± std)'] = ratio_aps
        
        # Append to overall lists
        overall_names_list.extend(cleaned_names)
        overall_oa_list.extend(cleaned_oa)
        overall_aps_list.extend(cleaned_aps)
        overall_ratio_oa_list.extend(cleaned_ratios_oa)
        overall_ratio_aps_list.extend(cleaned_ratios_aps)

        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add the 'Overall' row
    overall_row = {
        'Use Case': 'Overall',
        'Total Names (avg ± std)': compute_statistics(overall_names_list)[0],
        'Total Names Median': '--',
        'Total Names Mode': '--',
        'Total Names (min, max)': '--',
        'Present in OpenAlex (avg ± std)': compute_statistics(overall_oa_list)[0],
        'Present in OpenAlex Median': '--',
        'Present in OpenAlex Mode': '--',
        'Present in OpenAlex (min, max)': '--',
        'Present in APS (avg ± std)': compute_statistics(overall_aps_list)[0],
        'Present in APS Median': '--',
        'Present in APS Mode': '--',
        'Present in APS (min, max)': '--',
        'Ratio Present in OA (avg ± std)': compute_statistics(overall_ratio_oa_list)[0],
        'Ratio Present in APS (avg ± std)': compute_statistics(overall_ratio_aps_list)[0]
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


# Create main output folder under 'output_results/factuality'
base_output_folder = '././output_results/factuality'
os.makedirs(base_output_folder, exist_ok=True)

# Subfolder for author tables (full tables, averages, and other metrics)
author_folder = os.path.join(base_output_folder, 'factuality_author')
os.makedirs(author_folder, exist_ok=True)

full_table_folder = os.path.join(author_folder, 'full_table')
avg_only_folder = os.path.join(author_folder, 'avg_only')
rest_of_metrics_folder = os.path.join(author_folder, 'rest_of_metrics')

os.makedirs(full_table_folder, exist_ok=True)
os.makedirs(avg_only_folder, exist_ok=True)
os.makedirs(rest_of_metrics_folder, exist_ok=True)

# Process the data and create a table for each configuration
for config_name, config_data in data.items():
    table = create_table(config_data)
    
    # Generate filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{config_name.replace(' ', '_')}_{timestamp}"
    
    # Save full table
    full_png_filename = os.path.join(full_table_folder, f"{base_filename}.png")
    full_tex_filename = os.path.join(full_table_folder, f"{base_filename}.tex")
    save_as_png(table, full_png_filename)
    caption = f"Full descriptive statistics for {config_name}"
    save_as_tex(table, full_tex_filename, caption)
    
    # Save avg only
    avg_df = table[['Use Case', 'Total Names (avg ± std)', 'Present in OpenAlex (avg ± std)', 'Present in APS (avg ± std)', 'Ratio Present in OA (avg ± std)', 'Ratio Present in APS (avg ± std)']]
    avg_png_filename = os.path.join(avg_only_folder, f"{base_filename}_avg.png")
    avg_tex_filename = os.path.join(avg_only_folder, f"{base_filename}_avg.tex")
    save_as_png(avg_df, avg_png_filename)
    caption_avg = f"Average descriptive statistics for {config_name}"
    save_as_tex(avg_df, avg_tex_filename, caption_avg)
    
    rest_df = table[table['Use Case'] != 'Overall'][['Use Case', 'Total Names Median', 'Total Names Mode', 'Total Names (min, max)',
                                                 'Present in OpenAlex Median', 'Present in OpenAlex Mode', 'Present in OpenAlex (min, max)',
                                                 'Present in APS Median', 'Present in APS Mode', 'Present in APS (min, max)']]

    rest_png_filename = os.path.join(rest_of_metrics_folder, f"{base_filename}_rest.png")
    rest_tex_filename = os.path.join(rest_of_metrics_folder, f"{base_filename}_rest.tex")
    save_as_png(rest_df, rest_png_filename)
    caption_rest = f"Median, mode, min and max statistics for {config_name}"
    save_as_tex(rest_df, rest_tex_filename, caption_rest)

    print(f"Saved full table, avg only, and rest of metrics for {config_name}")

print("All tables have been saved in respective folders.")

