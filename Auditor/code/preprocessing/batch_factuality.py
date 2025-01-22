# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from tqdm import tqdm
import pandas as pd
# from multiprocessing import Pool
# from functools import partial

from libs.factuality.author import FactualityAuthor
from libs.factuality.topk import FactualityTopK
from libs.factuality.field import FactualityField
from libs.factuality.epoch import FactualityEpoch
from libs.factuality.seniority import FactualitySeniority
from libs.factuality.twins import FactualityTwins
from libs import io
from libs import constants

def run(aps_os_data_tar_gz: str, valid_responses_dir: str, model: str, task_name: str|None,  max_workers: int, output_dir: str):
    
    # Validate author factuality across all tasks
    fact_author = FactualityAuthor(aps_os_data_tar_gz=aps_os_data_tar_gz, valid_responses_dir=valid_responses_dir, model=model, max_workers=max_workers, output_dir=output_dir)
    if fact_author.already_checked():
        fact_author.load_valid_responses_with_factuality_check()
    else:
        fact_author.run_factuality_check()
        fact_author.save_valid_responses_with_factuality_check()
    
    # Validate factuality for the the other tasks
    fact_task = None
    if task_name == constants.EXPERIMENT_TASK_FIELD:
        fact_task = FactualityField(aps_os_data_tar_gz=aps_os_data_tar_gz, valid_responses_dir=None, model=model, max_workers=max_workers, output_dir=output_dir)

    elif task_name == constants.EXPERIMENT_TASK_EPOCH:
        fact_task = FactualityEpoch(aps_os_data_tar_gz=aps_os_data_tar_gz, valid_responses_dir=None, model=model, max_workers=max_workers, output_dir=output_dir)

    elif task_name == constants.EXPERIMENT_TASK_SENIORITY:
        fact_task = FactualitySeniority(aps_os_data_tar_gz=aps_os_data_tar_gz, valid_responses_dir=None, model=model, max_workers=max_workers, output_dir=output_dir)


    if fact_task is not None:
        if fact_task.already_checked():
            fact_task.load_valid_responses_with_factuality_check()
        else:
            fact_task.set_valid_responses(fact_author.df_valid_responses)
            fact_task.run_factuality_check()
            fact_task.save_valid_responses_with_factuality_check()

    print(f"Final size of the valid responses (Auhthors): {fact_author.df_valid_responses.shape}")
    if fact_task is not None:
        print(f"Final size of the valid responses ({task_name}): {fact_task.df_valid_responses.shape}")
    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aps_os_data_tar_gz", required=True, type=str, help="final_dataset.tar.gz")
    parser.add_argument("--valid_responses_dir", type=str, help="Directory where the valid responses are stored")
    parser.add_argument("--model", type=str, required=True, choices=constants.LLMS, help="Model to analyse (eg., gemma2-9b, llama-3.1-8b, llama-3.1-70b, llama3-8b, llama3-70b, mixtral-8x7b)")
    parser.add_argument("--task_name", type=str, choices=constants.FACTUALITY_TASKS, help="Tasks to analyse (e.g., field, epoch, seniority)")
    parser.add_argument("--max_workers", type=int, default=1, help="How many jobs to run in parallel maximum")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.aps_os_data_tar_gz, args.valid_responses_dir, args.model, args.task_name, args.max_workers, args.output_dir)

    