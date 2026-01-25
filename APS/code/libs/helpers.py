import re
import numpy as np
import pandas as pd
from datetime import datetime

from libs import io
from libs import constants
from libs import helpers

def is_none(value):
    return value is None or pd.isna(value)

def get_longest_name(lst):
    cleaned_list = [item for item in lst if item is not None and item != ''  and not pd.isna(item)]
    if isinstance(cleaned_list, list) and len(cleaned_list) > 0 and cleaned_list is not None:
        return max(cleaned_list, key=len)
    return None


def convert_YYYYMMDD_to_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception as e:
        io.printf(e)
        return None
    
def convert_HHMM_to_time(time_str):
    try:
        h = datetime.strptime(time_str, "%H%M%S").strftime("%H")
        h = f"{h}:00"
        return h
    except Exception as e:
        io.printf(e)
        return None
    

def pivot_model_tasks(df, index_col='task_name', columns_col='model', values_col='mean', x_order=None, hue_order=None):
    tmp = df.pivot_table(index=index_col, columns=columns_col, values=values_col, observed=False).T.reset_index()
    if x_order is not None:
        tmp = tmp[[columns_col] + x_order]
    if hue_order is not None:
        tmp[columns_col] = io.pd.Categorical(tmp[columns_col], categories=hue_order, ordered=True)
    tmp.sort_values(columns_col, inplace=True)
    return tmp


def assign_ax_to_task_param(row):
    if constants.EXPERIMENT_TASK_TWINS in row.task_name:
        return constants.TASK_TWINS_GENDER_ORDER.index(row.task_param.split("_")[1])
    return constants.TASK_PARAMS_BY_TASK[row.task_name].index(row.task_param)


def add_mean_values_rows_columns_of_pivot(df):
    # Function to extract main value before parentheses
    def extract_main_value(cell):
        if isinstance(cell, str) and cell != '-':
            match = re.match(r'([0-9.]+)', cell)
            if match:
                return float(match.group(1))
        return np.nan

    # Extract only numeric values
    value_df = df.iloc[:, 2:].map(extract_main_value)

    # Compute row means (skip NaN)
    row_means = value_df.mean(axis=1, skipna=True)

    # Add as new column
    df['Row Mean'] = row_means.map('{:.2f}'.format)

    # Compute column means (skip NaN)
    col_means = value_df.mean(axis=0, skipna=True)

    # Format means
    col_means_formatted = col_means.map('{:.2f}'.format)

    # Create a new row for column means
    summary_row = pd.DataFrame(
        [['Mean', '-'] + col_means_formatted.tolist() + ['-']],
        columns=df.columns
    )

    # Append the summary row
    df = pd.concat([df, summary_row], ignore_index=True)
    return df


def assign_model_size_class(row):
    model_name = row.get('model', None)

    if model_name is not None:
        for size_class, models in constants.LLMS_SIZE_CATEGORIES.items():
            if model_name in models:
                return size_class
    return None

def detect_refusal(msg):
    if type(msg) == str:
        if any(re.search(p, msg) for p in constants.REFUSAL_KEYWORDS):
            return constants.REFUSAL_TRUE
        return constants.REFUSAL_FALSE
    return constants.REFUSAL_UNKNOWN

def add_quantiles(df, cols_dict={'works_count':'pub', 'cited_by_count':'cit'}):
    for col, label in cols_dict.items():
        if col in df.columns:
            df[f"prominence_{label}"] = io.pd.qcut(
                                                df[col],
                                                q=[0, 0.5, 0.8, 0.95, 1.0],
                                                labels=["low", "mid", "high", "elite"]
                                            )
    return df
                                            
def add_infrastructure_columns(df):
    df.loc[:, 'model_access'] = df.model.map(constants.LLM_ACCESS_CATEGORIES_INV)
    df.loc[:, 'model_size'] = df.model.map(constants.LLMS_SIZE_CATEGORIES_INV)
    df.loc[:, 'model_class'] = df.model.map(constants.LLM_CLASS_CATEGORIES_INV)
    df.loc[:, 'model_provider'] = df.model.str.split('-').str[0]

    if 'result_original_message' in df.columns and 'is_refusal' not in df.columns:
        df.loc[:, 'is_refusal'] = df.result_original_message.apply(lambda msg: helpers.detect_refusal(msg))

    # model size (merging open and proprietary models)
    df["model_size"] = df["model_size"].str.replace(r"\s*\(.*\)", "", regex=True)

    if 'id_author_oa' in df.columns and 'id_author_aps_list' in df.columns:
        df.loc[:, 'author_exists'] = df.id_author_oa.notnull() | df.id_author_aps_list.notnull()

    
    return df

    