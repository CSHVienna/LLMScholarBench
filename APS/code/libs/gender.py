from asyncio import constants
import gender_guesser.detector as gender
from tqdm import tqdm

from libs import constants


def determine_gender(group, col_name):
    """
    Determines the gender for an id_author based on the following rules:
    1. Remove instances of 'Unknown'.
    2. If all values are the same, return that value.
    3. If there are multiple distinct values:
       - If there is a majority, return the one that occurs more times.
       - If there is a tie, return 'Unisex'.
    """
    # Access the 'gender' column
    gender_series = group[col_name]
    
    # Remove 'Unknown' values
    filtered_genders = gender_series[gender_series != constants.UNKNOWN_STR]
    
    # Count occurrences of each gender
    gender_counts = filtered_genders.value_counts()
    
    if len(gender_counts) == 0:
        return constants.UNKNOWN_STR
    
    if len(gender_counts) == 1:
        return gender_counts.index[0]  # All values are the same
    
    # Check for majority
    max_count = gender_counts.max()
    majority_genders = gender_counts[gender_counts == max_count].index.tolist()
    
    if len(majority_genders) == 1:
        return majority_genders[0]  # A clear majority exists
    
    return constants.GENDER_UNISEX  # Tie between two or more genders


def predict_gender_with_gender_guesser(first_name):
    d = gender.Detector()

    gender_prediction = d.get_gender(first_name)

    # Mapping gender-guesser results to a simpler form
    if gender_prediction in ['male', 'mostly_male']:
        return constants.GENDER_MALE
    elif gender_prediction in ['female', 'mostly_female']:
        return constants.GENDER_FEMALE
    elif gender_prediction == 'andy':
        return constants.GENDER_UNISEX
    else:
        return constants.UNKNOWN_STR
    
def assign_combined_gender(row, col_ethnicity, col_gender, col_firstname):

    if row[col_ethnicity] in constants.ETHNICITIES_TO_PREDICT_GENDER:
        # Apply gender_guesser only if necessary (i.e., not "Other" ethnicity)
        return predict_gender_with_gender_guesser(row[col_firstname])
    else:
        # Use the pre-existing Gender inferred from Image
        return row[col_gender]