import unicodedata
import re
import recordlinkage
import pandas as pd
import jellyfish
import numpy as np

from . import constants

def flatten_name_str(s: str) -> str:
     # Match: {"Name":{"Name":"<value>"}}  (works with ' or " and optional spaces)
    pattern = r'\{\s*(?P<q>"|\')Name(?P=q)\s*:\s*\{\s*(?P=q)Name(?P=q)\s*:\s*(?P=q)(?P<val>[^"\']*?)(?P=q)\s*\}\s*\}'
    replacement = r'{\g<q>Name\g<q>: \g<q>\g<val>\g<q>}'
    return re.sub(pattern, replacement, s)

def clean_name(text, lower=True):
    if text is None or pd.isnull(text):
        return None  # Handle None gracefully

    # remove patterns
    clean_text = re.sub(r"^(?:Dr|Mr|Mrs|Ms|PhD)\.\s*", "", text)
    clean_text = re.sub(r"\s*\([^)]*\)", "", clean_text)
                 
    # Normalize the text to decompose accents and diacritics
    normalized_text = unicodedata.normalize('NFD', clean_text)
    # Remove accents by filtering out combining characters
    no_accents = ''.join(char for char in normalized_text if not unicodedata.combining(char))
    # Special case: Replace apostrophes followed by a vowel with an empty string
    no_accents = re.sub(r"'(?=[aeiouAEIOU])", '', no_accents)
    # Replace special characters with whitespace using regex
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', no_accents)
    # Remove extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    

    return clean_text.lower() if lower else clean_text


def find_matching_texts(dfA, dfB, column_block, column_pairs_to_evaluate, threshold=5):

    # Indexation step
    indexer = recordlinkage.Index()
    indexer.block(column_block)
    candidate_links = indexer.index(dfA, dfB)

    # Comparison step
    compare_cl = recordlinkage.Compare()

    for (colA, colB, method, pair_threshold, label) in column_pairs_to_evaluate:
        compare_cl.string(colA, colB, method=method, threshold=pair_threshold, label=label)

    # compute matches
    features = compare_cl.compute(candidate_links, dfA, dfB)

    # Classification step
    features.loc[:,'total_matches'] = features.apply(lambda row: row.sum(), axis=1)
    
    # valid matches
    valid_matches = features.query("total_matches >= @threshold").sort_values(['total_matches'], ascending=[False])
    valid_matches = valid_matches.reset_index().drop_duplicates(subset='level_0', keep="first").set_index('level_0') # keeping only best & first matches

    return valid_matches


def replace(txt, replacements, lower=True):
    for k, v in replacements.items():
        if lower:
            txt = txt.lower().replace(k.lower(), v)
        else:
            txt = txt.replace(k, v)
    return txt


def compute_similarity_list_of_text(list_of_text):
    '''
    Jaro-Winkler Similarity
    A metric that gives more weight to characters at the start of strings.
    Higher values indicate more similar strings.
    '''
    
    if len(list_of_text) < 2:
        return None
    
    similarities = []
    # Compute pairwise similarity
    for i in range(len(list_of_text)):
        for j in range(i + 1, len(list_of_text)):
            name_i = list_of_text[i]
            name_j = list_of_text[j]
            sim = jellyfish.jaro_winkler_similarity(name_i, name_j)
            similarities.append(sim)
    return np.mean(similarities) if len(similarities) > 1 else None