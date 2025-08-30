from itertools import combinations
import numpy as np
from tqdm import tqdm
import pandas as pd

def compute_jaccard_similarity(lists_of_names):
    # 0: The two sets have no elements in common.
    # 1: The two sets are identical (all elements are shared).

    # Function to compute Jaccard Similarity
    def jaccard_similarity(list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    # Compute pairwise similarity scores
    n = len(lists_of_names)
    similarity_scores = np.zeros((n, n))

    for (i, j) in combinations(range(n), 2):
        score = jaccard_similarity(lists_of_names[i], lists_of_names[j])
        similarity_scores[i, j] = score
        similarity_scores[j, i] = score  # Symmetric matrix

    # Average similarity score
    data = similarity_scores[np.triu_indices(n, k=1)] # only the upper triangle because it's symmetric
    average_score = np.mean(data) 
    standard_dev = np.std(data)
    return similarity_scores, average_score, standard_dev


def run_consistency_jaccard_similarity(df):

    # Initialize results
    results = {}

    # Group by model, task, and param
    grouped = df.groupby(['model', 'task_name', 'task_param'], observed=False)

    # Process each chunk with tqdm for progress tracking
    for (model, task, param), group in tqdm(grouped, desc="Processing Groups"):
        # Sort by date and time
        group = group.sort_values(['date', 'time'])

        # Group by date and time and collect names
        grouped_names = group.groupby(['date', 'time'], observed=False)['clean_name'].apply(list).tolist()

        sim, avg, std = compute_jaccard_similarity(grouped_names)

        # Store the result
        results[(model, task, param)] = (avg, std)

    # Convert results to a DataFrame or keep as dictionary
    results_df = pd.DataFrame(results.items(), columns=['model_task_param', 'jaccard_similarity_mean_std'])

    # Split the column into three separate columns
    results_df[['model', 'task_name', 'task_param']] = results_df['model_task_param'].apply(pd.Series)
    results_df[['jaccard_similarity_mean', 'jaccard_similarity_std']] = results_df['jaccard_similarity_mean_std'].apply(pd.Series)
    
    # Drop the original column and reorder
    results_df.drop(columns=['model_task_param','jaccard_similarity_mean_std'], inplace=True)
    cols = results_df.columns.tolist() 
    results_df = results_df[cols[1:] + cols[:1]]

    return results_df


def run_consistency_uniqueness(df):

    # Initialize results
    results = {}

    # Group by model, task, and param
    results = df.groupby(['model', 'task_name', 'task_param', 'date', 'time'], observed=True).agg(
                                                        unique_names=('clean_name', lambda x: x.nunique()),
                                                        total_names=('clean_name', 'count')
                                                    ).reset_index()
    results.loc[:, 'duplicates'] = results.apply(lambda row: row.total_names - row.unique_names, axis=1)
    results.loc[:, 'duplicates_pct'] = results.apply(lambda row: row.duplicates / row.total_names, axis=1)

    results.loc[:, 'uniqueness'] = results.apply(lambda row: row.unique_names, axis=1)
    results.loc[:, 'uniqueness_pct'] = results.apply(lambda row: row.uniqueness / row.total_names, axis=1)

    return results