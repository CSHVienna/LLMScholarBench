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
    average_score = np.mean(similarity_scores[np.triu_indices(n, k=1)])

    return similarity_scores, average_score


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

        sim, avg = compute_jaccard_similarity(grouped_names)

        # Store the result
        results[(model, task, param)] = avg

    # Convert results to a DataFrame or keep as dictionary
    results_df = pd.DataFrame(results.items(), columns=['model_task_param', 'average_jaccard_similarity'])

    # Split the column into three separate columns
    results_df[['model', 'task_name', 'task_param']] = results_df['model_task_param'].apply(pd.Series)
    results_df.drop(columns=['model_task_param'], inplace=True)
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

    return results