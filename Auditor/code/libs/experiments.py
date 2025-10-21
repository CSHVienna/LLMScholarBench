import re
import ast
from ssl import OP_NO_RENEGOTIATION
import time
import concurrent.futures
from datetime import datetime
import pandas as pd

from libs import io
from libs import helpers
from libs import constants

def is_valid_extracted_data(extracted_data):
    if extracted_data is None:
        return False
    if not extracted_data or all(not bool(d) for d in extracted_data) or len(extracted_data) == 0:        
        return False
    return True

def extract_and_convert_to_dict(result_api, file_path):
    if result_api.get('choices', None) is None:
        return None, constants.EXPERIMENT_OUTPUT_PROVIDER_ERROR

    input_string = result_api.get('choices', [{}])[0].get('message', {}).get('content', None)

    error_keywords_lc = ['"error"', 'fatalerror', '.runners(position', 'taba_keydown', 'unable to provide', 'addtogroup', 'onitemclick', 'getinstance', "i'm stuck", '.datasource', 'getclient', 'phone.toolstrip', 'actordatatype', 'baseactivity', 'setcurrent_company', '.clearfest', 'getdata_suffix', '.texture_config', 'translator_concurrent']
    if input_string in [None, ''] or any(keyword in input_string.lower() for keyword in error_keywords_lc):
        return None, constants.EXPERIMENT_OUTPUT_INVALID
    
    input_string = input_string.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace("\\", '').replace('...', '').replace('```json', ' ')
    valid_flag = None
    
    for bad_content in constants.EXPERIMENTS_BAD_CONTENT_TEXTS:
        if bad_content.lower() in input_string.lower():
            return None, constants.EXPERIMENT_OUTPUT_INVALID   

    input_string = input_string.replace('"Years": 2', '"Years": "2').replace('0},', '0"},').replace(".}",'."}')
    
    if 'Note: This response is illustrative' in input_string:
        input_string = input_string.split('Note: This response is illustrative')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_ILLUSTRATIVE

    if '### Updated Output ###' in input_string:
        input_string = input_string.split("### Updated Output ###")[1].split("###")[0]

    if '### Output ###' in input_string:
        input_string = input_string.replace("### Output ###","").split("]")[0]

    if '### Reasoning Explanation ###' in input_string:
        input_string = input_string.split("### Reasoning Explanation ###")[0]

    if '### Reasoning Explanation  The' in input_string:
        input_string = input_string.split("### Reasoning Explanation  The")[0]
    
    if "### Output in JSON Array Format: ```" in input_string:
        input_string = input_string.split("### Output in JSON Array Format: ```")[1].split("```")[0]

    for c in ['[{"Physicists": [', '[{\"Physicists\": [', '{"physicists": [', '{"Physicists": [', '{"Scientists": [']:
        if c in input_string:
            input_string = input_string.replace(c, "[")

    if '} ]}  The scientists' in input_string:
        input_string = input_string.split('} ]}  The scientists')[0]
        input_string = f"{input_string}{'}'}]"

    if 'is as follows:' in input_string:
        input_string = f"[{input_string.split('is as follows:')[1]}]"

    if ':  {"Name": "' in input_string:
        input_string = input_string.split(':  {')[-1]
        input_string = f"[{'{'}{input_string}"

    if input_string.startswith("[  [ {"):
        input_string = input_string.replace("[  [ {", "[ {")

    if input_string.endswith("} ]}"):
        input_string = input_string.replace("} ]}", "}]")

    if input_string.strip().replace(' ','').startswith("{"):
        input_string = f"[{input_string}"

    if input_string.strip().replace(' ','').endswith("}"):
        input_string = f"{input_string}]"

    try:
        # Extract substring between "[" and "]"
        start_index = input_string.find("[")
        end_index = input_string.find("]")
        
        if start_index != -1:  # "[" found
            if end_index != -1:  # "]" found
                # valid JSON in verbosed response
                substring = input_string[start_index:end_index+1]
                valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED if valid_flag is None else valid_flag
            else:  # "]" not found
                last_curly = input_string.rfind("}")
                if last_curly != -1:
                    # valid JSON after fixing truncated JSON
                    substring = input_string[start_index:last_curly + 1] + "]"
                    valid_flag = constants.EXPERIMENT_OUTPUT_FIXED if valid_flag is None else valid_flag
                else:
                    # io.printf(f"\n{input_string}\nNo matching ']' or ')' found.\n")
                    return None, constants.EXPERIMENT_OUTPUT_INVALID
        else:
            # io.printf(f"\n{input_string}\nNo '[' found in the string.\n")
            return None, constants.EXPERIMENT_OUTPUT_INVALID

        # Convert the substring to a dictionary
        result_dict = ast.literal_eval(substring)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        return result_dict, valid_flag
    
    except Exception as e:
        io.printf(f"Error: {e} | {file_path}")
        return None, constants.EXPERIMENT_OUTPUT_INVALID
    
    
def _get_extracted_data(result_api, validation_result, file_path):
    is_valid = validation_result.get('is_valid', False)
    original_extracted_data = validation_result.get('extracted_data', None)

    # if it is not valid (at first) - try to extract the data (if possible)
    if not is_valid:
        output, valid_flag = extract_and_convert_to_dict(result_api, file_path)

        if output is None:
            return None, valid_flag # constants.EXPERIMENT_OUTPUT_INVALID
        elif len(output) == 0:
            return None, valid_flag # constants.EXPERIMENT_OUTPUT_INVALID
        
        if isinstance(output, list):
            if isinstance(output[0], dict):
                return output, valid_flag
        
        return None, constants.EXPERIMENT_OUTPUT_INVALID
    
    return original_extracted_data, constants.EXPERIMENT_OUTPUT_VALID
    
def _process_file(file_path):
    # metadata
    datetime_str = file_path.split('/run_')[-1].split('/')[0]
    date = helpers.convert_YYYYMMDD_to_date(datetime_str.split('_')[0])
    time = helpers.convert_HHMM_to_time(datetime_str.split('_')[1])

    tprefix = f'/{constants.TEMPERATURE_FOLDER_PREFIX}'
    temperature = float(file_path.split(tprefix)[-1].split('/')[0]) if tprefix in file_path else None
    model = file_path.split('/config_')[-1].split('/')[0]
    model = model.replace('-it','').replace('-instant','').replace('-versatile','').replace('-8192','').replace('-32768','')

    # content
    try:
        # Replace this with the actual processing logic
        data = io.read_json_file(file_path)
        result = {}
    except Exception as e:
        io.printf(f"Error reading file {file_path}: {e}")
        return None
    
    try:
        # IF ERRONEUS
        if 'full_api_response' not in data:
            original_message = data.get('error', {}).get('message', None)
            obj = {'date':date,
                   'time':time,
                   'task_name':data['category'],
                   'task_param':data['variable'],
                   'task_attempt':data['attempt'],
                   'model':model,
                   'temperature': temperature,
                   'llm_model': None,
                   'llm_provider': None,
                   'llm_completion_tokens':None,
                   'llm_prompt_tokens':None,
                   'llm_total_tokens':None,
                   'result_is_valid':False,
                   'result_valid_flag':constants.EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR if 'internal_server_error' in original_message else 
                                       constants.EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT if 'rate_limit_exceeded' in original_message else 
                                       constants.EXPERIMENT_OUTPUT_INVALID,
                   'result_original_message': original_message,
                   'result_answer':None,
                   'file_path':file_path,
                   }
            return obj
    except Exception as e:
        io.printf(f"Error processing empty outpu {file_path}: {e}")
        return None
    
    try:
        # general
        result['date'] = date
        result['time'] = time

        # task metadata
        result['task_name'] = data['category']
        result['task_param'] = data['variable']
        result['task_attempt'] = data['attempt']

        # llm metadata
        _result_api = data.get('full_api_response', None)
        result['model'] = model
        result['temperature'] = temperature
        result['llm_model'] = _result_api.get('model', None)
        result['llm_provider'] = _result_api.get('provider', None)
        

        _result_usage = _result_api.get('usage', None) if _result_api is not None else None
        if _result_usage is None:
            _result_usage = _result_api.get('response', {}).get('usage_metadata',{})
            result['llm_completion_tokens'] = _result_usage.get('candidates_token_count', None)
            result['llm_prompt_tokens'] = _result_usage.get('prompt_token_count', None)
            result['llm_total_tokens'] = _result_usage.get('total_token_count', None)
        else:
            result['llm_completion_tokens'] = _result_usage.get('completion_tokens', None) if _result_usage is not None else None
            result['llm_prompt_tokens'] = _result_usage.get('prompt_tokens', None) if _result_usage is not None else None
            result['llm_total_tokens'] = _result_usage.get('total_tokens', None) if _result_usage is not None else None
            
        # results
        _result = data.get('validation_result', None)
        extracted_data, valid_flag = _get_extracted_data(_result_api, _result, file_path)
        is_valid = is_valid_extracted_data(extracted_data)
        
        
        full_answer = _result_api.get('choices', [{}])[0].get('message', {}).get('content', None) if _result_api.get('choices', None) is not None else None
        result_message = full_answer if full_answer is not None and valid_flag in [constants.EXPERIMENT_OUTPUT_INVALID, constants.EXPERIMENT_OUTPUT_PROVIDER_ERROR] else _result.get('message', '')
        result['result_is_valid'] = is_valid
        result['result_valid_flag'] = constants.EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT if 'rate_limit_exceeded' in result_message else valid_flag
        result['result_original_message'] = result_message
        result['result_answer'] = extracted_data

        # for tracking results
        result['file_path'] = file_path
        
        return result
    
    except Exception as e:
        io.printf(f"Error processing file {file_path}: {e}")
        return None


def read_experiments(experiments_dir: str, model: str, max_workers: int = 1):
    file_paths = io.get_files(experiments_dir, f"temperature_*/config_{model}/run_*_*/*/attempt*_*_*.json")
    io.printf(f"Found {len(file_paths)} files to process.")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_file = {executor.submit(_process_file, file): file for file in file_paths}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result is None:
                    continue
                results.append(result)
            except Exception as e:
                print(f"Error processing file in parallel {file}: {e}")
    return results


def set_attempt_validity(df):
    df.loc[:,'valid_attempt'] = False
    
    df_valid_attempts = df.query("result_valid_flag in @constants.EXPERIMENT_OUTPUT_VALID_FLAGS").groupby(['model','task_name','task_param','date','time']).task_attempt.count().reset_index(name='count')

    # Those with only one valid attempt
    df_to_choose_one = df_valid_attempts.query("count == 1").reset_index(drop=True)
    for i, row in df_to_choose_one.iterrows():
        tmp_valid = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VALID").sort_values('task_attempt', ascending=True)

        # if there is at leat one valid response, then we can choose that one
        if tmp_valid.shape[0] > 0:
            # choose the first valid one
            df.loc[tmp_valid.index[0], 'valid_attempt'] = True
        else:
            # choose among the verbosed ones
            tmp_verbosed = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VERBOSED").sort_values('task_attempt', ascending=True)
            if tmp_verbosed.shape[0] > 0:
                df.loc[tmp_verbosed.index[0], 'valid_attempt'] = True

            # choose among the fixed ones
            if constants.EXPERIMENT_OUTPUT_FIXED in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:
                tmp_fixed = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_FIXED").sort_values('task_attempt', ascending=True)
                if tmp_fixed.shape[0] > 0:
                    df.loc[tmp_fixed.index[0], 'valid_attempt'] = True
                                        
    # Those with more than one valid attempt
    df_to_choose_one = df_valid_attempts.query("count > 1").reset_index(drop=True)
    for i, row in df_to_choose_one.iterrows():
        tmp_valid = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VALID").sort_values('task_attempt', ascending=True)

        # if there is at leat one valid response, then we can choose that one
        if tmp_valid.shape[0] > 0:
            # choose the first valid one
            df.loc[tmp_valid.index[0], 'valid_attempt'] = True
        else:
            # choose among the verbosed ones
            tmp_verbosed = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VERBOSED").sort_values('task_attempt', ascending=True)
            if tmp_verbosed.shape[0] > 0:
                df.loc[tmp_verbosed.index[0], 'valid_attempt'] = True

            # choose among the fixed ones
            if constants.EXPERIMENT_OUTPUT_FIXED in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:
                tmp_fixed = df.query("model==@row.model and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_FIXED").sort_values('task_attempt', ascending=True)
                if tmp_fixed.shape[0] > 0:
                    df.loc[tmp_fixed.index[0], 'valid_attempt'] = True
        
    return df
