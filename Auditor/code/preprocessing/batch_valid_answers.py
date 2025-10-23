# export PYTHONPATH="${PYTHONPATH}:."

import argparse
from operator import index
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from functools import partial

from libs import experiments
from libs import io
from libs import constants
from libs import text

def run(experiments_dir: str, model: str, max_workers: int, output_dir: str):
   # process experiments
   results = experiments.read_experiments(experiments_dir, model, max_workers)
   df = pd.DataFrame(results).drop(columns=['result_answer']).sort_values(by=['temperature', 'date', 'time', 'task_name', 'task_param'])

   # process each reponse (json)
   df_valid_responses = parallel_processing(results, max_workers).sort_values(by=['temperature', 'date', 'time', 'task_name', 'task_param'])
   df_valid_responses.loc[:,'clean_name'] = df_valid_responses.loc[:,'name'].apply(lambda x: text.clean_name(text.replace(x, constants.EXPERIMENT_AUDIT_FACTUALITY_AUTHOR_NAME_REPLACEMENTS)))

   # Update responses with valid_attempt that end up invalid
   col_ids = ['model', 'temperature', 'llm_provider', 'llm_model','date','time','task_name','task_param','task_attempt']
   df = df.set_index(col_ids)
   tmp = df.join(df_valid_responses[col_ids + ['name']].set_index(col_ids), how='left')
   ids = tmp.query("@io.pd.isnull(name) and result_valid_flag in @constants.EXPERIMENT_OUTPUT_VALID_FLAGS").index
   df.loc[ids, 'result_valid_flag'] = constants.EXPERIMENT_OUTPUT_INVALID
   df.loc[ids, 'result_is_valid'] = False
   df.loc[ids, 'valid_attempt'] = False
   df = df.reset_index()

   # store summary (csv)
   df = experiments.set_attempt_validity(df)
   fn = io.path_join(output_dir, 'summaries', f"experiments_{model}.csv")
   io.validate_path(fn)
   io.save_csv(df, fn, index=False)

   # Add metadata to valid responses 
   df_valid_responses = df_valid_responses.merge(df[['model','temperature','llm_model','date','time','task_name','task_param','task_attempt','valid_attempt']], on=['model', 'temperature', 'llm_model', 'date','time','task_name','task_param','task_attempt'], how='left')
   df_valid_responses = df_valid_responses.query("valid_attempt == True")
   fn = io.path_join(output_dir, 'valid_responses', f"{model}.csv")
   io.validate_path(fn)
   io.save_csv(df_valid_responses, fn, index=True)

def parallel_processing(results, n_chunks=1):
    chunks = [results[i::n_chunks] for i in range(n_chunks)]

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=n_chunks) as pool:
        results = list(tqdm(pool.map(process_results, chunks), total=n_chunks))
    
    return pd.concat(results, ignore_index=True)

def _final_validation_response(result_answer):
    result_valid_flag = None
    new_result_dict = []
    for obj_rec in result_answer:
        
        if type(obj_rec) == set:
            # fix
            obj_rec = {'Name': list(obj_rec)[0]}
            result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON

        # if it is not a dict
        if type(obj_rec) != dict:
            # skip
            result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM
            continue

        # If Name is missing, but '' is available
        if 'Name' not in obj_rec and '' in obj_rec:
            # fix
            obj_rec['Name'] = obj_rec['']
            del(obj_rec[''])
            result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON

        # If any key is not valid
        if any(x not in constants.AUDITOR_RESPONSE_DICT_KEYS for x in obj_rec.keys()):
            # skip
            result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM
            continue
            
        # if the value is too long (from the LLMCaller keys)
        if any(len(obj_rec[key].split(' ')) > constants.MAX_WORDS_RESPONSE or len(obj_rec[key]) > constants.MAX_LETTERS_RESPONSE for key in constants.LLMCALLER_DICT_KEYS if key in obj_rec and type(obj_rec[key]) == str):
            # skip
            result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM
            continue

        if len(obj_rec) > 0:
            new_result_dict.append(obj_rec)

    return new_result_dict, result_valid_flag

def process_results(results):
    rows = []

    for obj in results:

        result_valid_flag = obj['result_valid_flag']
        result_answer = obj['result_answer']
            
        if result_valid_flag in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:  # Only iterate if the answer list is not empty
            if result_answer is None:
                continue
            
            ### Verify Name is in the key list of each dict
            new_result_dict, result_valid_flag = _final_validation_response(result_answer)
            result_valid_flag = obj['result_valid_flag'] if result_valid_flag is None else result_valid_flag
            
            ### Prepare summary per recommendation
            for answer_id, answer in enumerate(new_result_dict):

                name = answer.get("Name", None)

                if name is None or pd.isnull(name) or len(name) >= constants.MAX_LETTERS_RESPONSE or name.lower() in constants.EXPERIMENTS_BAD_CONTENT_TEXTS_LC:
                    io.printf(f"skipping recommendation (one name): {obj['temperature']} | {obj['model']} | {obj['date']} | {obj['time']} | {obj['task_name']} | {obj['task_param']} | {obj['task_attempt']} | {answer_id+1} of {len(result_answer)} | {result_valid_flag}")
                    continue

                rows.append({
                    "date": obj['date'],
                    "time": obj['time'],
                    "model": obj['model'],
                    "temperature": obj['temperature'],
                    "llm_provider": obj['llm_provider'],
                    "llm_model": obj['llm_model'],
                    "task_name": obj['task_name'],
                    "task_param": obj['task_param'],
                    "task_attempt": obj['task_attempt'],
                    'result_valid_flag':result_valid_flag,
                    "name": name,
                    "years": answer.get("Years", None),
                    "doi": answer.get("DOI", None),
                    "career_age": answer.get("Career Age", None),
                })

    if len(rows) == 0:
        return None
    
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", required=True, type=str, help="Directory where the experiment data is stored (root of temperature_<value>/)")
    parser.add_argument("--model", required=True, type=str, choices=constants.LLMS, help=f"Model to analyse (ie., {', '.join(constants.LLMS)})")
    parser.add_argument("--max_workers", type=int, default=1, help="How many jobs to run in parallel maximum")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.experiments_dir, args.model, args.max_workers, args.output_dir)

    