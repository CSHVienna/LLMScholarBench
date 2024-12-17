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

# Function to create a table for each configuration and epoch use case
def create_epoch_table(config_data):
    rows = []
    overall_names_list = []
    overall_correct_author_epoch_list = []
    overall_correct_recommended_epoch_list = []
    overall_ratio_author_epoch_list = []
    overall_ratio_recommended_epoch_list = []
    
    for use_case, metrics in config_data.items():
        if 'epoch' in use_case:  # Only process epoch use cases
            row = {'Epoch Use Case': use_case}
            
            # Apply mask based on count_names > 0
            mask = [name != 0 for name in metrics['count_names']]
            
            # Clean the lists using the mask
            cleaned_names = [name for name, m in zip(metrics['count_names'], mask) if m]
            cleaned_correct_author_epoch = [cae for cae, m in zip(metrics['count_correct_author_epoch'], mask) if m]
            cleaned_correct_recommended_epoch = [cre for cre, m in zip(metrics['count_correct_recommended_epoch'], mask) if m]
            
            # Calculate and store statistics for each metric
            names_stats = compute_statistics(cleaned_names)
            row['Total Names Avg'] = names_stats[0]
            row['Total Names (Median, Mode, Min, Max)'] = names_stats[1]

            correct_author_epoch_stats = compute_statistics(cleaned_correct_author_epoch)
            row['Correct Author Epoch Avg'] = correct_author_epoch_stats[0]
            row['Correct Author Epoch (Median, Mode, Min, Max)'] = correct_author_epoch_stats[1]

            correct_recommended_epoch_stats = compute_statistics(cleaned_correct_recommended_epoch)
            row['Correct Recommended Epoch Avg'] = correct_recommended_epoch_stats[0]
            row['Correct Recommended Epoch (Median, Mode, Min, Max)'] = correct_recommended_epoch_stats[1]

            # Calculate ratios and add new columns
            ratio_author_epoch = [cae / name for cae, name, m in zip(metrics['count_correct_author_epoch'], metrics['count_names'], mask) if m]
            ratio_recommended_epoch = [cre / name for cre, name, m in zip(metrics['count_correct_recommended_epoch'], metrics['count_names'], mask) if m]

            ratio_author_epoch_stats = compute_statistics(ratio_author_epoch)
            ratio_recommended_epoch_stats = compute_statistics(ratio_recommended_epoch)

            row['Ratio Correct Author Epoch / Total Names Avg'] = ratio_author_epoch_stats[0]
            row['Ratio Correct Author Epoch / Total Names (Median, Mode, Min, Max)'] = ratio_author_epoch_stats[1]

            row['Ratio Correct Recommended Epoch / Total Names Avg'] = ratio_recommended_epoch_stats[0]
            row['Ratio Correct Recommended Epoch / Total Names (Median, Mode, Min, Max)'] = ratio_recommended_epoch_stats[1]

            # Append to the overall lists for calculating overall stats
            overall_names_list.extend(cleaned_names)
            overall_correct_author_epoch_list.extend(cleaned_correct_author_epoch)
            overall_correct_recommended_epoch_list.extend(cleaned_correct_recommended_epoch)
            overall_ratio_author_epoch_list.extend(ratio_author_epoch)
            overall_ratio_recommended_epoch_list.extend(ratio_recommended_epoch)

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add the 'Overall' row
    overall_row = {
        'Epoch Use Case': 'Overall',
        'Total Names Avg': compute_statistics(overall_names_list)[0],
        'Total Names (Median, Mode, Min, Max)': compute_statistics(overall_names_list)[1],
        'Correct Author Epoch Avg': compute_statistics(overall_correct_author_epoch_list)[0],
        'Correct Author Epoch (Median, Mode, Min, Max)': compute_statistics(overall_correct_author_epoch_list)[1],
        'Correct Recommended Epoch Avg': compute_statistics(overall_correct_recommended_epoch_list)[0],
        'Correct Recommended Epoch (Median, Mode, Min, Max)': compute_statistics(overall_correct_recommended_epoch_list)[1],
        'Ratio Correct Author Epoch / Total Names Avg': compute_statistics(overall_ratio_author_epoch_list)[0],
        'Ratio Correct Author Epoch / Total Names (Median, Mode, Min, Max)': compute_statistics(overall_ratio_author_epoch_list)[1],
        'Ratio Correct Recommended Epoch / Total Names Avg': compute_statistics(overall_ratio_recommended_epoch_list)[0],
        'Ratio Correct Recommended Epoch / Total Names (Median, Mode, Min, Max)': compute_statistics(overall_ratio_recommended_epoch_list)[1],
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
    # Create main output folder under 'output_results/factuality/factuality_epoch'
    factuality_epoch_folder = os.path.join('output_results/factuality', 'factuality_epoch')
    os.makedirs(factuality_epoch_folder, exist_ok=True)

    # Create subfolders for full, avg, and other metrics
    full_table_folder = os.path.join(factuality_epoch_folder, 'full_table')
    avg_table_folder = os.path.join(factuality_epoch_folder, 'avg_table')
    other_metrics_table_folder = os.path.join(factuality_epoch_folder, 'other_metrics_table')

    os.makedirs(full_table_folder, exist_ok=True)
    os.makedirs(avg_table_folder, exist_ok=True)
    os.makedirs(other_metrics_table_folder, exist_ok=True)

    # Process each configuration in the data and create the respective tables
    for config_name, config_data in data.items():
        # Create the table for each configuration
        table = create_epoch_table(config_data)
        
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
        avg_df = table[['Epoch Use Case', 'Total Names Avg', 'Correct Author Epoch Avg', 
                        'Correct Recommended Epoch Avg', 'Ratio Correct Author Epoch / Total Names Avg', 
                        'Ratio Correct Recommended Epoch / Total Names Avg']]
        avg_png_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.png")
        avg_tex_filename = os.path.join(avg_table_folder, f"{base_filename}_avg.tex")

        # Save the avg table
        save_as_png(avg_df, avg_png_filename)
        caption_avg = f"Average statistics for {config_name}"
        save_as_tex(avg_df, avg_tex_filename, caption_avg)

        # Other metrics (median, mode, min/max)
        other_metrics_df = table[['Epoch Use Case', 'Total Names (Median, Mode, Min, Max)', 
                                  'Correct Author Epoch (Median, Mode, Min, Max)', 
                                  'Correct Recommended Epoch (Median, Mode, Min, Max)', 
                                  'Ratio Correct Author Epoch / Total Names (Median, Mode, Min, Max)', 
                                  'Ratio Correct Recommended Epoch / Total Names (Median, Mode, Min, Max)']]
        other_metrics_png_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.png")
        other_metrics_tex_filename = os.path.join(other_metrics_table_folder, f"{base_filename}_other_metrics.tex")

        # Save the other metrics table
        save_as_png(other_metrics_df, other_metrics_png_filename)
        caption_other = f"Other metrics (Median, Mode, Min/Max) for {config_name}"
        save_as_tex(other_metrics_df, other_metrics_tex_filename, caption_other)

        print(f"Saved full, avg, and other metrics tables for {config_name}")

    print(f"All tables have been saved in {factuality_epoch_folder}.")

