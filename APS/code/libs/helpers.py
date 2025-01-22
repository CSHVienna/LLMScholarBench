import pandas as pd
from datetime import datetime

from libs import io

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
        return datetime.strptime(time_str, "%H%M%S").strftime("%H:%M")
    except Exception as e:
        io.printf(e)
        return None
    