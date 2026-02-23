import pandas as pd

from libs import io
from libs import constants

def read_responses(valid_responses_dir: str, model: str, task: str) -> pd.DataFrame:
    """
    Read responses from valid_responses_dir and return a DataFrame
    """
    try:
        fn = io.get_files(io.path_join(valid_responses_dir), f"{model}.csv")[0]
        df = io.read_csv(fn, index_col=0, low_memory=False)
        if task is not None and task != constants.FACTUALITY_AUTHOR:
            df = df.query("task_name == @task")
        
        io.printf(f"Data loaded from {fn}")
        io.printf(f"Data shape: {df.shape}")

    except Exception as e:
        io.printf(f"Error: {e}")
        df = None
    
    return df