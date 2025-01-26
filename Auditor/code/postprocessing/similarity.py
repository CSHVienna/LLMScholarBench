from ast import mod
from operator import add
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def get_components_by_model(df_responses_sample, df_author_population, label_population, metrics_col, include_task_param=False, n_components=2):

    oa_id = 'id_author_oa'
    task_cols = ['task_name', 'task_param']
    inf = [np.inf, -np.inf]

    additional_cols = [oa_id]
    if include_task_param:
        additional_cols += task_cols

    metric_author_col =  metrics_col + [oa_id]
    needed_cols = metrics_col + additional_cols
    results = {}

    metadata_N = df_author_population[metric_author_col].copy()
    metadata_N = metadata_N.fillna({'e_index':0, 'citations_per_paper_age':0}).replace({'e_index':inf, 'citations_per_paper_age':inf}, 0)
    metadata_N['label'] = label_population
    metadata_N['task_name'] = None
    metadata_N['task_param'] = None
    
    df_sample_clean = df_responses_sample.dropna(subset=[oa_id])
    df_sample_clean = df_sample_clean.fillna({'e_index':0, 'citations_per_paper_age':0}).replace({'e_index':inf, 'citations_per_paper_age':inf}, 0)

    groupby = 'model' if not include_task_param else ['model', 'task_name', 'task_param']

    for group, df in df_sample_clean.groupby(groupby):
        
        model = group if not include_task_param else group[0]
        task_name = None if not include_task_param else group[1]
        task_param = None if not include_task_param else group[2]

        metadata_n = df[needed_cols].drop_duplicates(subset=additional_cols).copy()
        metadata_n['label'] = model
        metadata_n['task_name'] = task_name
        metadata_n['task_param'] = task_param

        # Combine the datasets
        combined_data = pd.concat([metadata_N, metadata_n], ignore_index=True)

        # Extract the required columns
        data = combined_data[metrics_col]

        # Handle missing values (if any)
        data = data.fillna(data.mean())

        # Standardize the data
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        
        # Apply dimensionality reduction (e.g., PCA)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data_normalized)

        # Add reduced dimensions to the combined DataFrame
        for i in range(n_components):
            combined_data[f'dim{i+1}'] = reduced_data[:, i]
            
        # appending the results
        results[group] = {'reduction':combined_data, 
                          'variance':pca.explained_variance_, 
                          'variance_ratio':pca.explained_variance_ratio_}

    return results



def compute_simpson_diversity(vector):
    '''
    2. Simpson's Diversity Index
    Simpson's Index measures the probability that two individuals randomly selected from the dataset belong to the same category.
    '''

    # Compute proportions
    counts = Counter(vector)
    total = sum(counts.values())
    proportions = [count / total for count in counts.values()]

    # Simpson's Index
    simpson_index = 1 - sum(p**2 for p in proportions)
    return simpson_index


def compute_average_pairwise_cosine_similarity(array):
    '''
    Compute the similarity of numeric data using the cosine similarity metric.
    '''
    array = array.replace([np.inf, -np.inf], 0,)
    array = array.fillna(0)
    
    # Standardize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(array)

    # Compute the cosine similarity
    similarity_matrix = cosine_similarity(data_normalized)

    return np.mean(similarity_matrix)


def compute_average_cosine_zscore_similarity(vector):
    '''
    Compute the similarity of numeric data using the cosine similarity metric.
    '''

    clean_vector = vector.dropna()

    if clean_vector.empty or clean_vector.shape[0] <= 1:
        return None
    
    cosine_sim = cosine_similarity(clean_vector.values.reshape(-1, 1))
    average_similarity = cosine_sim.mean()
    return average_similarity


