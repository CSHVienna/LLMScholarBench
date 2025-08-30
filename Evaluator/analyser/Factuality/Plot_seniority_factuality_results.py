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

# Function to compute just avg and std for errors like mean_career_age_error
def compute_error_statistics(values):
    if not values:
        return '--', '--'
    
    avg = np.mean(values)
    std = np.std(values)
    return f'{avg:.2f}', f'{std:.2f}'

# Function to create a table for each configuration and seniority use case
def create_seniority_table(config_data):
    rows = []
    overall_names_list = []
    overall_correct_author_seniority_list = []
    overall_career_age_error_list = []  # For mean career age error
    overall_ratio_author_seniority_list = []
    
    for use_case, metrics in config_data.items():
        if 'seniority' in use_case:  # Only process seniority use cases
            row = {'Seniority Use Case': use_case}
            
            # Apply mask based on count_names > 0
            mask = [name not in [0, None] and pd.notna(name) for name in metrics['count_names']]
            
            # Clean the lists using the mask
            cleaned_names = [name for name, m in zip(metrics['count_names'], mask) if m]
            cleaned_correct_author_seniority = [cas for cas, m in zip(metrics['count_correct_author_seniority'], mask) if m]
            
            
            # Check if 'mean_career_age_error' key exists before accessing it
            if 'mean_career_age_error' in metrics:
                cleaned_career_age_error = [error for error, m in zip(metrics['mean_career_age_error'], mask) if m and pd.notna(error) and error is not None]
            else:
                # If the key is missing, return an empty list or handle as necessary
                cleaned_career_age_error = []

            # Calculate and store statistics for each metric
            names_stats = compute_statistics(cleaned_names)
            row['Total Names Avg'] = names_stats[0]
            row['Total Names (Median, Mode, Min, Max)'] = names_stats[1]

            correct_author_seniority_stats = compute_statistics(cleaned_correct_author_seniority)
            row['Correct Author Seniority Avg'] = correct_author_seniority_stats[0]
            row['Correct Author Seniority (Median, Mode, Min, Max)'] = correct_author_seniority_stats[1]

            # Calculate error for career age and store the mean and std
            career_age_error_stats = compute_error_statistics(cleaned_career_age_error)
            row['Career Age Error (Mean ± Std)'] = f'{career_age_error_stats[0]} ± {career_age_error_stats[1]}'
            
            # Calculate ratios and add new columns
            ratio_author_seniority = [cas / name for cas, name, m in zip(metrics['count_correct_author_seniority'], metrics['count_names'], mask) if m]

            ratio_author_seniority_stats = compute_statistics(ratio_author_seniority)

            row['Ratio Correct Author Seniority / Total Names Avg'] = ratio_author_seniority_stats[0]
            row['Ratio Correct Author Seniority / Total Names (Median, Mode, Min, Max)'] = ratio_author_seniority_stats[1]

            # Append to the overall lists for calculating overall stats
            overall_names_list.extend(cleaned_names)
            overall_correct_author_seniority_list.extend(cleaned_correct_author_seniority)
            overall_career_age_error_list.extend(cleaned_career_age_error)
            overall_ratio_author_seniority_list.extend(ratio_author_seniority)

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add the 'Overall' row
    overall_row = {
        'Seniority Use Case': 'Overall',
        'Total Names Avg': compute_statistics(overall_names_list)[0],
        'Total Names (Median, Mode, Min, Max)': compute_statistics(overall_names_list)[1],
        'Correct Author Seniority Avg': compute_statistics(overall_correct_author_seniority_list)[0],
        'Correct Author Seniority (Median, Mode, Min, Max)': compute_statistics(overall_correct_author_seniority_list)[1],
        'Career Age Error (Mean ± Std)': f"{compute_error_statistics(overall_career_age_error_list)[0]} ± {compute_error_statistics(overall_career_age_error_list)[1]}",
        'Ratio Correct Author Seniority / Total Names Avg': compute_statistics(overall_ratio_author_seniority_list)[0],
        'Ratio Correct Author Seniority / Total Names (Median, Mode, Min, Max)': compute_statistics(overall_ratio_author_seniority_list)[1],
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

# Main logic to create the tables for each configuration
if __name__ == "__main__":
    # Create main output folder under 'output_results/factuality/factuality_seniority'
    factuality_seniority_folder = os.path.join('output_results/factuality', 'factuality_seniority')
    os.makedirs(factuality_seniority_folder, exist_ok=True)

    # Create subfolders for full, avg, and other metrics
    full_table_folder = os.path.join(factuality_seniority_folder, 'full_table')
    avg_table_folder = os.path.join(factuality_seniority_folder, 'avg_table')
    other_metrics_table_folder = os.path.join(factuality_seniority_folder, 'other_metrics_table')

    os.makedirs(full_table_folder, exist_ok=True)
    os.makedirs(avg_table_folder, exist_ok=True)
    os.makedirs(other_metrics_table_folder, exist_ok=True)


    # Process each configuration in the data and create the respective tables
    for config_name, config_data in data.items():
        # Create the table for each configuration
        table = create_seniority_table(config_data)
        
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
                # Avg only table
        avg_df = table[['Seniority Use Case', 'Total Names Avg', 'Correct Author Seniority Avg', 
                        'Career Age Error (Mean ± Std)', 'Ratio Correct Author Seniority / Total Names Avg']]
        avg_png_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.png")
        avg_tex_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.tex")

        # Save the avg table
        save_as_png(avg_df, avg_png_filename)
        caption_avg = f"Average statistics for {config_name}"
        save_as_tex(avg_df, avg_tex_filename, caption_avg)

        # Other metrics (median, mode, min/max)
        other_metrics_df = table[['Seniority Use Case', 'Total Names (Median, Mode, Min, Max)', 
                                  'Correct Author Seniority (Median, Mode, Min, Max)', 
                                  'Ratio Correct Author Seniority / Total Names (Median, Mode, Min, Max)', 
                                  'Career Age Error (Mean ± Std)']]
        other_metrics_png_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.png")
        other_metrics_tex_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.tex")

        # Save the other metrics table
        save_as_png(other_metrics_df, other_metrics_png_filename)
        caption_other = f"Other metrics (Median, Mode, Min/Max) for {config_name}"
        save_as_tex(other_metrics_df, other_metrics_tex_filename, caption_other)

        print(f"Saved full, avg, and other metrics tables for {config_name}")

    print(f"All tables have been saved in {factuality_seniority_folder}.")

                       
