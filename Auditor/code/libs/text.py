import unicodedata
import re
import recordlinkage
import pandas as pd

def clean_name(text, lower=True):
    if text is None or pd.isnull(text):
        return None  # Handle None gracefully

    # Normalize the text to decompose accents and diacritics
    normalized_text = unicodedata.normalize('NFD', text)
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