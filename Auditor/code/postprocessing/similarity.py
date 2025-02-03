from ast import mod
from operator import add
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

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


def gini_coefficient(data):
    clean_vector = data.dropna()

    if clean_vector.empty or clean_vector.shape[0] <= 1:
        return None
    
    sorted_data = np.sort(clean_vector)
    n = len(clean_vector)
    cumulative_sum = np.cumsum(sorted_data) / np.sum(sorted_data)
    cumulative_sum = np.insert(cumulative_sum, 0, 0)
    gini = 1 - (2 / n) * np.sum(cumulative_sum[:-1] * (1 / n + cumulative_sum[1:] - cumulative_sum[:-1]))
    return gini

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


# def compute_average_cosine_zscore_similarity(vector):
#     '''
#     Compute the similarity of numeric data using the cosine similarity metric.
#     '''

#     clean_vector = vector.dropna()

#     if clean_vector.empty or clean_vector.shape[0] <= 1:
#         return None
    
#     cosine_sim = cosine_similarity(clean_vector.values.reshape(-1, 1))
#     average_similarity = cosine_sim.mean()
#     return average_similarity


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def compute_average_jaccard_similarity(df_items_per_author):
    '''
    Compute the average Jaccard similarity between authors based on their items_per_author.
    '''
    
    if df_items_per_author.empty or df_items_per_author.shape[0] <= 1:
        return None
    
    
    # Compute pairwise Jaccard similarity
    vals = []
    for id_i, items_i in df_items_per_author.iterrows():
        for id_j, items_j in df_items_per_author.iterrows():
    
    # TODO: check maybe this is faster
    # n = df_items_per_author.shape[0]
    # for i in range(n):
    #     for j in range(i+1, n):
    #         id_i, items_i = df_items_per_author.iloc[i]
    #         id_j, items_j = df_items_per_author.iloc[j]

            if id_i == id_j:
                continue

            s = jaccard_similarity(set(items_i['_items']), set(items_j['_items']))

            vals.append(s)
    return np.mean(vals)


def get_items_by_author(id_institutions_by_author, df_all, column_item):

    df_items_by_author = pd.DataFrame()
    for id_author_oa, id_institution_oa in id_institutions_by_author.items():
        items = df_all.query('id_institution_oa in @id_institution_oa')[column_item].unique()
        df_items_by_author = pd.concat([df_items_by_author, pd.DataFrame({'id_author_oa': [id_author_oa], '_items': [items]})], ignore_index=True)
        
    return df_items_by_author.set_index('id_author_oa')