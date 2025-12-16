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
   df_valid_responses = parallel_processing(results, max_workers)
   
   if df_valid_responses is None:
       io.printf(f"{model} has no valid responses.")
       return 
   
   df_valid_responses = df_valid_responses.sort_values(by=['temperature', 'date', 'time', 'task_name', 'task_param'])
   df_valid_responses.loc[:,'clean_name'] = df_valid_responses.loc[:,'name'].apply(lambda x: text.clean_name(text.replace(x, constants.EXPERIMENT_AUDIT_FACTUALITY_AUTHOR_NAME_REPLACEMENTS)))

   # Update responses with valid_attempt that end up invalid
   col_ids = ['model', 'temperature', 'grounded', 'llm_provider', 'llm_model','date','time','task_name','task_param','task_attempt']
   df.loc[:, 'valid_attempt'] = True
   df.set_index(col_ids, inplace=True)
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
   col_ids = ['model','temperature','grounded','llm_model','date','time','task_name','task_param','task_attempt']
   df_valid_responses = df_valid_responses.set_index(col_ids).join(df.set_index(col_ids)[['valid_attempt']], how='left').reset_index()
   df_valid_responses = df_valid_responses.query("valid_attempt == True")
   fn = io.path_join(output_dir, 'valid_responses', f"{model}.csv")
   io.validate_path(fn)
   io.save_csv(df_valid_responses, fn, index=True)


def parallel_processing(results, n_chunks=1):
    chunks = [results[i::n_chunks] for i in range(n_chunks)]

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=n_chunks) as pool:
        results = list(tqdm(pool.map(process_results, chunks), total=n_chunks))
    
    if  all(x is None for x in results):
        return None
    
    return pd.concat(results, ignore_index=True)

def process_results(results):
    rows = []

    for obj in results:

        result_valid_flag = obj['result_valid_flag']
        result_answer = obj['result_answer']
            
        if result_valid_flag in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:  # Only iterate if the answer list is not empty
            
            if result_answer is None:
                continue
            
            ### Prepare summary per recommendation
            for answer_id, answer in enumerate(result_answer):

                name = answer.get("Name", None)

                if type(name) == dict:
                    name = name['Name']
                
                for m in [", 7th duc de Broglie", "Dr. "]:
                    if m in name:
                        name = name.split(m)[0]
                        result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

                if name is None or pd.isnull(name) or len(name.split('(')[0]) >= constants.MAX_LETTERS_RESPONSE or name.lower() in constants.EXPERIMENTS_BAD_CONTENT_TEXTS_LC or (any(m.lower() in name.lower() for m in constants.NO_OUTPUT_KEYWORDS)) or ('colleague' in name.lower() and 'student' in name.lower()) or ('colleague' in name.lower() and 'contemporary' in name.lower()) or (name.lower().count('colleague') >= 2) or ' but her colleague' in name.lower():
                    io.printf(f"skipping recommendation (one name): {obj['temperature']} | {obj['model']} | {obj['date']} | {obj['time']} | {obj['task_name']} | {obj['task_param']} | {obj['task_attempt']} | {answer_id+1} of {len(result_answer)} | {result_valid_flag} | {name}")
                    result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM
                    continue
                
                for m in ["'s contemporary, ", "'s collaborator, ", "instead, ", "'s colleague, "]:
                    if m in name:
                        name = name.split(m)[1]
                        result_valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON
                        break

                rows.append({
                    "date": obj['date'],
                    "time": obj['time'],
                    "model": obj['model'],
                    "grounded": obj['grounded'],
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

    