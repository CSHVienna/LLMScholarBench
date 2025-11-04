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
from libs import text

def is_valid_extracted_data(extracted_data):
    if extracted_data is None:
        return False
    if not extracted_data or all(not bool(d) for d in extracted_data) or len(extracted_data) == 0:        
        return False
    return True

def extract_and_convert_to_dict(result_api, file_path):
    gemini = result_api.get('model','') in constants.LLMS_GEMINI

    if gemini:
        if 'full_api_response' in result_api:
            input_string = result_api.get('full_api_response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [None,None,None])

            if type(input_string) == list:
                if len(input_string) == 1:
                    input_string = input_string[0].get('text', '')
                    if '"Tsunami":' in str(input_string):
                        input_string = str(input_string).replace('"Tsunami":', '"Name":')
                else:
                    input_string = input_string[1 if len(input_string) == 3 else 2 if len(input_string) == 4 else -2].get('text', None)
            else:
                input_string = None
        else:
            input_string = result_api.get('response', None)
    else:
        if result_api.get('choices', None) is None:
            return None, constants.EXPERIMENT_OUTPUT_PROVIDER_ERROR
        input_string = result_api.get('choices', [{}])[0].get('message', {}).get('content', None)

    ### Decoding 
    input_string = input_string.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace("\\", '').replace('...', '').replace('```json', ' ')
    
    valid_flag = None

    ### Fixing specific cases (invalid JSON): 
    
    input_string = input_string.replace('"Years": 2', '"Years": "2').replace('0},', '0"},').replace(".}",'."}')
    
    ### FIXING WEIRD CASES
    for name_str in ['"Namename"', '"Noame"', '"Nime"', "'Näme'"]:
        if name_str in input_string:
            input_string = input_string.replace(name_str, '"Name"')
            valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '"), {"Name"' in input_string:
        input_string = input_string.replace('"), {"Name"', '"}, {"Name"')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '"}, "Years":' in input_string:
        input_string = input_string.replace('"}, "Years":', '", "Years":')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if ',}, ]' in input_string:
        input_string = input_string.replace(',}, ]', '"}]')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '",   {"Name":' in input_string:
        input_string = input_string.replace('",   {"Name":', '"}, {"Name":')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '{"Name": {"Name": "Freeman Dyson", "Years": "1950-1960"}}' in input_string:
        input_string = input_string.replace('{"Name": {"Name": "Freeman Dyson", "Years": "1950-1960"}}', '{"Name": "Freeman Dyson", "Years": "1950-1960"}')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '{"Name": {"Name":' in input_string:
        input_string = text.flatten_name_str(input_string)
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if ',     {"Name": "Tatiana "Tanya" Evans (astronomer)"},' in input_string:
        input_string = input_string.split(',     {"Name": "Tatiana "Tanya" Evans (astronomer)"},')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM

    if '{})  AngiePhraseIn MIN nan--;' in input_string:
        input_string = input_string.split('{})  AngiePhraseIn MIN nan--;')[0] + '"'
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '{"Name": "Marie G緎 bite.Job Pamela Gay' in input_string:
        input_string = input_string.split('{"Name": "Marie G緎 bite.Job Pamela Gay')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM

    if ',   {"Name": "Sin‑Iou‌లు DörfEchtthank}^{remotepoint.scatter/blob)' in input_string:
        input_string = input_string.split(',   {"Name": "Sin‑Iou‌లు DörfEchtthank}^{remotepoint.scatter/blob)')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM

    if ',   {"Name":"Walter H"}' in input_string:
        input_string = input_string.split(',   {"Name":"Walter H"}')[0].replace("Kü},  // placeholder accidental", 'Kü"}, ')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM

    if ':  {"Name": "' in input_string:
        input_string = input_string.split(':  {')[-1]
        input_string = f"[{'{'}{input_string}"
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON
        
    if '{"nomyfut": -12.3}' in input_string:
        input_string = input_string.split(', {"nomyfut": -12.3}')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM

    if "Namer:  P" in input_string:
        input_string = input_string.replace("Namer:  P", 'Name": "P')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if "Name " in input_string:
        input_string = input_string.replace("Name ", '"Name": ')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if "NameMaria" in input_string:
        input_string = input_string.replace("NameMaria", 'Name": "Maria')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if "NameCharles" in input_string:
        input_string = input_string.replace("NameCharles", 'Name": "Charles')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if "NameClaude" in input_string:
        input_string = input_string.replace("NameClaude", 'Name": "Claude')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if "NamePhilip" in input_string:
        input_string = input_string.replace("NamePhilip", 'Name": "Philip')
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    ### SPLITTING VERBOSE CONTENT

    if 'While many distinguished individuals' in input_string:
        input_string = input_string.split('While many distinguished individuals')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED
        
    if 'Given the critique' in input_string:
        input_string = input_string.split('Given the critique')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if 'Note: This response is illustrative' in input_string:
        input_string = input_string.split('Note: This response is illustrative')[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_ILLUSTRATIVE

    if "**Reasoning and Verification:**" in input_string:
        input_string = input_string.split("**Reasoning and Verification:**")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if '### Updated Output ###' in input_string:
        input_string = input_string.split("### Updated Output ###")[1].split("###")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if '### Output ###' in input_string:
        input_string = input_string.replace("### Output ###","").split("]")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if '### Reasoning Explanation ###' in input_string:
        input_string = input_string.split("### Reasoning Explanation ###")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if '### Reasoning Explanation  The' in input_string:
        input_string = input_string.split("### Reasoning Explanation  The")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if "### Output in JSON Array Format: ```" in input_string:
        input_string = input_string.split("### Output in JSON Array Format: ```")[1].split("```")[0]
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    for c in ['[{"Physicists": [', '[{\"Physicists\": [', '{"physicists": [', '{"Physicists": [', '{"Scientists": [']:
        if c in input_string:
            input_string = input_string.replace(c, "[")
            valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if '} ]}  The scientists' in input_string:
        input_string = input_string.split('} ]}  The scientists')[0]
        input_string = f"{input_string}{'}'}]"
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if 'is as follows:' in input_string:
        input_string = f"[{input_string.split('is as follows:')[1]}]"
        valid_flag = constants.EXPERIMENT_OUTPUT_VERBOSED

    if input_string.startswith("[  [ {"):
        input_string = input_string.replace("[  [ {", "[ {")
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if input_string.endswith("} ]}"):
        input_string = input_string.replace("} ]}", "}]")
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if input_string.strip().replace(' ','').startswith("{"):
        input_string = f"[{input_string}"
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON

    if input_string.strip().replace(' ','').endswith("}"):
        input_string = f"{input_string}]"
        valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON
    

    try:
        # Extract substring between "[" and "]"
        start_index = input_string.find("[")
        end_index = input_string.find("]")
        
        if start_index != -1:  # "[" found
            if end_index != -1:  # "]" found 
                
                try:
                    # valid JSON in verbosed response
                    substring = input_string[start_index:end_index+1]
                    tmp = ast.literal_eval(substring)
                    if len(tmp) == 0:
                        # invalid JSON
                        return None, constants.EXPERIMENT_OUTPUT_INVALID
                    valid_flag = constants.EXPERIMENT_OUTPUT_VALID if start_index == 0 and valid_flag is None else constants.EXPERIMENT_OUTPUT_VERBOSED if valid_flag is None else valid_flag
                except Exception as e:
                    # invalid JSON
                    return None, constants.EXPERIMENT_OUTPUT_INVALID

            else:  # "]" not found
                last_curly = input_string.rfind("}")

                if last_curly == -1 or (last_curly < start_index) or (start_index + last_curly > len(input_string)):
                    # invalid JSON
                    return None, constants.EXPERIMENT_OUTPUT_INVALID
                else:
                    # valid JSON after fixing truncated JSON
                    substring = input_string[start_index:last_curly + 1] + "]"
                    valid_flag = constants.EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON if valid_flag is None else valid_flag
                    
        else:
            return None, constants.EXPERIMENT_OUTPUT_INVALID
            
        ### Convert the substring to a dictionary
        try:
            result_dict = ast.literal_eval(substring)
        except Exception as e:
            result_dict = None
            valid_flag = constants.EXPERIMENT_OUTPUT_INVALID
        
        ### Return
        return result_dict, valid_flag
    
    except Exception as e:
        io.printf(f"Error: {e} | {file_path}")
        return None, constants.EXPERIMENT_OUTPUT_INVALID
    
    
def _get_extracted_data(result_api, validation_result, file_path):
    is_valid = validation_result.get('is_valid', False)
    original_extracted_data = validation_result.get('extracted_data', None)
    message = validation_result.get('message', None)
        
    
    if file_path.endswith('temperature_2.0/config_deepseek-chat-v3.1/run_20251009_200926/epoch_1950s/attempt2_20251009_201134.json'):
        # print(input_string)
        # print(input_string.find("["))
        # print(input_string.find("]"))
        # print(input_string.rfind("}"))
        
        print(is_valid)
        print(message)
        print('full_api_response' in result_api)

    
    # If the flag is invalid from the LLMCaller
    if not is_valid and 'full_api_response' in result_api:
        # special case: potentially parsable
        pass
    
    if not is_valid:        
        # SEMI_VALID: potentially reparsable
        if message.startswith(constants.LLMCALLER_OUTPUT_SCHEMA_FAILED) or \
                message.startswith(constants.LLMCALLER_OUTPUT_PARSING_ERROR) or \
                message.startswith(constants.LLMCALLER_OUTPUT_EXPECTING_VALUE) or \
                message.startswith(constants.LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE_COMMA):
            # to be fixed
            pass
        else:
            # INVALID: hard or impossible to parse (or garbage data)
            if message == constants.LLMCALLER_OUTPUT_INVALID_JSON_EMPTY:
                return None, constants.EXPERIMENT_OUTPUT_INVALID
            # if message.startswith(constants.LLMCALLER_OUTPUT_INVALID_JSON_EXTRA_DATA) or \
            if message.startswith(constants.LLMCALLER_OUTPUT_INVALID_JSON_PROPERTY_NAME) or \
                    message.startswith(constants.LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE) or \
                    message.startswith(constants.LLMCALLER_OUTPUT_INVALID_JSON_CHAR):
                return None, constants.EXPERIMENT_OUTPUT_INVALID
        
    # if it is not valid (at first) - try to extract the data (if possible)
    if not is_valid:
        output, valid_flag = extract_and_convert_to_dict(result_api, file_path)
        
        if output is None:
            return None, valid_flag # constants.EXPERIMENT_OUTPUT_INVALID
        elif len(output) == 0:
            return None, valid_flag # constants.EXPERIMENT_OUTPUT_INVALID
        
        if isinstance(output, list):
            if isinstance(output[0], dict):
                
                ### Verify Name is in the key list of each dict
                new_output, result_valid_flag = _final_validation_response(output)
                result_valid_flag = valid_flag if result_valid_flag is None else result_valid_flag

                if not all('Name' in obj.keys() for obj in new_output) or any('error' in obj.keys() for obj in new_output):
                    # when the key name is missind, or error is present in any item of the objects in the list
                    return None, constants.EXPERIMENT_OUTPUT_INVALID

                if all(obj['Name'].startswith('Scientist') for obj in new_output):
                    # generic answer
                    return None, constants.EXPERIMENT_OUTPUT_INVALID
                
                if len(new_output) == 0:
                    return None, constants.EXPERIMENT_OUTPUT_INVALID
                
                return new_output, result_valid_flag
        
        return None, constants.EXPERIMENT_OUTPUT_INVALID
    
    return original_extracted_data, constants.EXPERIMENT_OUTPUT_VALID
    
def _final_validation_response(result_answer):
    result_valid_flag = None
    new_result_dict = []
    for obj_rec in result_answer:
        
        # if only a string
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
        if any(x not in constants.AUDITOR_RESPONSE_DICT_KEYS + ['Reasoning'] for x in obj_rec.keys()):
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
        data = io.read_json_file(file_path)
        result = {}
    except Exception as e:
        io.printf(f"Error reading file {file_path}: {e}")
        return None
    
    try:
        # IF ERRONEUS
        if 'full_api_response' not in data:
            if 'error' in data:
                if type(data['error']) == dict:
                    original_message = data.get('error', {}).get('message', None)
                else:
                    original_message = data.get('error', None)

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
        io.printf(f"Error processing empty output {file_path}: {e}")
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

        llm_provider = _result_api.get('provider', None)
        llm_provider = _result_api.get('error', {}).get('metadata',{}).get('provider_name',None) if 'error' in _result_api and llm_provider is None else llm_provider
        result['llm_provider'] = llm_provider
        result['grounded'] = _result_api.get('grounded', False) # websearch or not

        ### LLM metadata on token usage
        _result_usage = _result_api.get('usage', None) if _result_api is not None else None
        if _result_usage is None:
            
            _result_usage = _result_api.get('response', None)

            if type(_result_usage) == str: 
                _result_usage = _result_usage.split("usage_metadata")[-1].split('model_version')[0].replace(":",'":').replace("prompt_", '"prompt_').replace("candidates_", ', "candidates_').replace("total_", ', "total_')
                try:
                    _result_usage = ast.literal_eval(_result_usage)
                except:
                    _result_usage = None

            if _result_usage is None:
                _result_usage = _result_api.get('full_vertex_response', None)

                if type(_result_usage) == str: 
                    _result_usage = _result_usage.split("usage_metadata")[-1].split('model_version')[0].replace(":",'":').replace("prompt_", '"prompt_').replace("candidates_", ', "candidates_').replace("total_", ', "total_')
                    try:
                        _result_usage = ast.literal_eval(_result_usage)
                    except:
                        _result_usage = None
            
            if _result_usage is None:

                if 'full_api_response' in _result_api:
                    # special case gemini
                    _result_api_data = _result_api.get('full_api_response', {})
                    _result_usage = _result_api_data.get('usageMetadata', {})
                    result['llm_completion_tokens'] = _result_usage.get('candidatesTokenCount', None)
                    result['llm_prompt_tokens'] = _result_usage.get('promptTokenCount', None)
                    result['llm_total_tokens'] = _result_usage.get('totalTokenCount', None)
                    result['llm_tool_tokens'] = _result_usage.get('toolUsePromptTokenCount', None)
                    result['llm_thoughts_tokens'] = _result_usage.get('thoughtsTokenCount', None)

                elif _result_usage is None or (type(_result_usage) == list and len(_result_usage) == 0):
                    result['llm_completion_tokens'] = None
                    result['llm_prompt_tokens'] = None
                    result['llm_total_tokens'] = None
                    result['llm_tool_tokens'] = None
                    result['llm_thoughts_tokens'] = None

                else:
                    result['llm_completion_tokens'] = _result_usage.get('candidates_token_count', None)
                    result['llm_prompt_tokens'] = _result_usage.get('prompt_token_count', None)
                    result['llm_total_tokens'] = _result_usage.get('total_token_count', None)
                    result['llm_tool_tokens'] = None
                    result['llm_thoughts_tokens'] = None
        else:
            result['llm_completion_tokens'] = _result_usage.get('completion_tokens', None) if _result_usage is not None else None
            result['llm_prompt_tokens'] = _result_usage.get('prompt_tokens', None) if _result_usage is not None else None
            result['llm_total_tokens'] = _result_usage.get('total_tokens', None) if _result_usage is not None else None
            result['llm_tool_tokens'] = None
            result['llm_thoughts_tokens'] = None


        ### LLM output processing
        _result = data.get('validation_result', None)
        extracted_data, valid_flag = _get_extracted_data(_result_api, _result, file_path)
        is_valid = is_valid_extracted_data(extracted_data)
        
        ### Finalize result
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
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing file in parallel {file}: {e}")
    return results


def set_attempt_validity(df):
    df.loc[:,'valid_attempt'] = False
    
    df_valid_attempts = df.query("result_valid_flag in @constants.EXPERIMENT_OUTPUT_VALID_FLAGS").groupby(['model','temperature','grounded','task_name','task_param','date','time']).task_attempt.count().reset_index(name='count')

    # Those with only one valid attempt
    df_to_choose_one = df_valid_attempts.query("count == 1").reset_index(drop=True)
    for i, row in df_to_choose_one.iterrows():
        tmp_valid = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VALID").sort_values('task_attempt', ascending=True)

        # if there is at leat one valid response, then we can choose that one
        if tmp_valid.shape[0] > 0:
            # choose the first valid one
            df.loc[tmp_valid.index[0], 'valid_attempt'] = True
        else:
            # choose among the verbosed ones
            tmp_verbosed = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VERBOSED").sort_values('task_attempt', ascending=True)
            if tmp_verbosed.shape[0] > 0:
                df.loc[tmp_verbosed.index[0], 'valid_attempt'] = True

            # choose among the fixed ones
            for fixed_label in [constants.EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON, constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM, constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON]:
                if fixed_label in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:
                    tmp_fixed = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@fixed_label").sort_values('task_attempt', ascending=True)
                    if tmp_fixed.shape[0] > 0:
                        df.loc[tmp_fixed.index[0], 'valid_attempt'] = True
                                        
    # Those with more than one valid attempt
    df_to_choose_one = df_valid_attempts.query("count > 1").reset_index(drop=True)
    for i, row in df_to_choose_one.iterrows():
        tmp_valid = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VALID").sort_values('task_attempt', ascending=True)

        # if there is at leat one valid response, then we can choose that one
        if tmp_valid.shape[0] > 0:
            # choose the first valid one
            df.loc[tmp_valid.index[0], 'valid_attempt'] = True
        else:
            # choose among the verbosed ones
            tmp_verbosed = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@constants.EXPERIMENT_OUTPUT_VERBOSED").sort_values('task_attempt', ascending=True)
            if tmp_verbosed.shape[0] > 0:
                df.loc[tmp_verbosed.index[0], 'valid_attempt'] = True

            # choose among the fixed ones
            for fixed_label in [constants.EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON, constants.EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM, constants.EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON]:
                if fixed_label in constants.EXPERIMENT_OUTPUT_VALID_FLAGS:
                    tmp_fixed = df.query("model==@row.model and temperature==@row.temperature and grounded==@row.grounded and task_name==@row.task_name and task_param==@row.task_param and date==@row.date and time==@row.time and result_valid_flag==@fixed_label").sort_values('task_attempt', ascending=True)
                    if tmp_fixed.shape[0] > 0:
                        df.loc[tmp_fixed.index[0], 'valid_attempt'] = True
        
    return df
