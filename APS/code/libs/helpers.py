import pandas as pd
from datetime import datetime

from libs import io
from libs import constants

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
    tmp = df.pivot_table(index=index_col, columns=columns_col, values=values_col).T.reset_index()
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

