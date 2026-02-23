##### Demographix
from tqdm import tqdm
from demographicx import EthnicityEstimator
from ethnicolr import census_ln
import pandas as pd

from libs import constants

tqdm.pandas()

############################################################################################################
# GENERAL
############################################################################################################

def choose_ethnicity(row, col_k1, col_k2):
    if pd.notna(row[col_k1]):
        return row[col_k1]
    else:
        return row[col_k2]

    
############################################################################################################
# DEMOGRAPHICX
############################################################################################################
ethnicity_estimator = EthnicityEstimator()

def _predict_ethnicity(name):
    ethnicity_prediction = ethnicity_estimator.predict(name)
    most_likely_ethnicity = max(ethnicity_prediction, key=ethnicity_prediction.get)

    # Check if the highest probability is above 0.50
    if ethnicity_prediction[most_likely_ethnicity] > 0.50:
        return most_likely_ethnicity
    else:
        return 'Unknown'

def dx_predict_ethnicity(df, col_name, new_col):
    df.loc[:,new_col] = df.loc[:,col_name].progress_apply(_predict_ethnicity)
    return df


############################################################################################################
# ETHNICOLOR
############################################################################################################

def ec_predict_ethnicity(df, col_lastname, new_col):
    
    if col_lastname is None:
        raise Exception("There should be one column name")
    
    data = df[[col_lastname]].drop_duplicates().reset_index(drop=True)

    # infer gender and make numeric the probabilities
    data = census_ln(data, col_lastname, 2010)
    data[constants.ETHNICOLOR_ETHNICITY_COLS] = data[constants.ETHNICOLOR_ETHNICITY_COLS].apply(pd.to_numeric, errors='coerce')

    # Get the max column (gender with max probability)
    ethnicity_values = data[constants.ETHNICOLOR_ETHNICITY_COLS]
    max_column = ethnicity_values.idxmax(axis=1)

    # convert max probabilities to label
    data.loc[:,new_col] = max_column.progress_apply(lambda x: constants.ETHNICOLOR_ETHNICITY_MAPPING_CODE.get(x, constants.UNKNOWN_STR))

    # removing percentage columns
    data.drop(columns=constants.ETHNICOLOR_ETHNICITY_COLS, inplace=True)

    df = df.merge(data, on=col_lastname, how='left')
    return df