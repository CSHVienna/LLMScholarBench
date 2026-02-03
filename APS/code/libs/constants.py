from . import io

################################################################################################################
# Other constants
################################################################################################################
import numpy as np
NONE = ['', ' ', None, 'None', 'nan', 'NaN', np.nan, 'null', 'Null', 'NULL', 'N/A', 'n/a', 'N/a', 'n/A']
INF = [np.inf, -np.inf]
ID_STR_LEN = 7
TRUE_STR = 'True'

################################################################################################################
# Folders structure
################################################################################################################
TEMPERATURE_FOLDER_PREFIX = 'temperature_'
METADATA_DIR = 'metadata'
SIMILARITIES_DIR = 'similarities'


################################################################################################################
# APS/OA data constants
################################################################################################################
APS_OA_AUTHORS_FN = 'authors.csv'
APS_OA_PUBLICATIONS_FN = 'publications.csv'
APS_OA_AUTHORS_MAPPING_FN = 'authors_mapping.csv'
APS_OA_PUBLICATIONS_MAPPING_FN = 'publications_mapping.csv'
APS_OA_ALTERNATIVE_NAMES_FN = 'alternative_names.csv'
APS_OA_AUTHORS_INSTITUTION_YEAR_FN = 'authors_institution_year.csv'
APS_OA_INSTITUTIONS_FN = 'institution.csv'
APS_OA_AUTHORSHIPS_FN = 'authorships.csv'
APS_OA_CITATIONS_FN = 'citations.csv'
APS_OA_AUTHORS_DEMOGRAPHICS_FN = 'authors_demographics.csv'
APS_OA_PUBLICTION_TOPICS = 'publications_topic.csv'
APS_OA_TOPICS_FN = 'topics.csv'
APS_OA_DISCIPLINES_DEMOGRAPHICS_FN = 'disciplines_author_demographics.csv'
APS_OA_AUTHORS_STATS_FN = 'authors_stats.csv'
APS_OA_AUTHOR_STATS_FN = 'authors_aps_stats.csv'

AUTHORS_FN = 'authors.json'
PUBLICATIONS_FN = 'publications.json'
AUTHORS_DEMOGRAPHICS_FN = 'authors_demographics.json'
AUTHORS_APS_STATS_FN = 'aps_author_stats.json'
DISCIPLINES_FN = 'aps_disciplines.json'
PUBLICATION_CLASS_FN = 'aps_publication_classifications.json'
AUTHORS_AFFILIATIONS_FN = 'author_affiliations.json'
AFFILIATIONS_FN = 'affiliations.json'
COAUTHOR_NETWORKS_FN = 'aps_coauthor_networks.json'
INSTITUTIONS_STATS_FN = 'institutions_stats.json'
AUTHORS_RANKINGS_FN = 'author_rankings_with_percentile.json'
AUTHORS_APS_RANKINGS_FN = 'aps_author_rankings_with_percentile.json'

APS_AUTHORS_FN = 'authors.csv'
APS_AUTHOR_NAMES_FN = 'author_names.csv'
APS_AUTHORSHIPS_FN = 'authorships.csv'
APS_CITATIONS_FN = 'citations.csv'
APS_PUBLICATIONS_FN = 'publications.csv'
APS_DISCIPLINES_FN = 'disciplines.csv'
APS_PUBLICTION_TOPICS = 'publication_topics.csv'
APS_TOPIC_TYPES_FN = 'topic_types.csv'
APS_AREAS_FN = 'areas.csv'
APS_CONCEPTS_FN = 'concepts.csv'

APS_TOPIC_DISCIPLINE_ID = 1

RANKING_METRICS = {'publications':'works_count',
                   'citations':'cited_by_count',
                   'h_index':'h_index',
                   'i10_index':'i10_index',
                   'e_index':'e_index',
                   'citation_publication_age':'citations_per_paper_age',
                   'mean_citedness_2yr':'two_year_mean_citedness'}

APS_RANKING_METRICS = {'publications':'aps_works_count',
                       'citations':'aps_cited_by_count',
                       'h_index':'aps_h_index',
                       'i10_index':'aps_i10_index',
                       'e_index':'aps_e_index',
                       'citation_publication_age':'aps_citations_per_paper_age'}

APS_CAREER_AGE_COL = 'aps_career_age'
OA_CAREER_AGE_COL = 'career_age'

APS_PRESTIGE_METRICS_COL = list(APS_RANKING_METRICS.values()) 
OA_PRESTIGE_METRICS_COL = list(RANKING_METRICS.values())
ALL_PRESTIGE_METRICS_COL = APS_PRESTIGE_METRICS_COL + OA_PRESTIGE_METRICS_COL
APS_SCHOLARLY_METRICS_COL = APS_PRESTIGE_METRICS_COL + [APS_CAREER_AGE_COL]
OA_SCHOLARLY_METRICS_COL =  OA_PRESTIGE_METRICS_COL + [OA_CAREER_AGE_COL]
ALL_SCHOLARLY_METRICS_COL = APS_SCHOLARLY_METRICS_COL + OA_SCHOLARLY_METRICS_COL


################################################################################################################
# Demographic constants
################################################################################################################
UNKNOWN_STR = 'Unknown'

GENDER_UNISEX = 'Unisex'
GENDER_FEMALE = 'Female'
GENDER_MALE = 'Male'
GENDER_LIST = [GENDER_FEMALE, GENDER_MALE, GENDER_UNISEX, UNKNOWN_STR]

ETHNICITY_BLACK = 'Black or African American'
ETHNICITY_ASIAN = 'Asian'
ETHNICITY_WHITE = 'White'
ETHNICITY_LATINO = 'Hispanic or Latino'
ETHNICITY_MULTIPLE = 'Multiple'
ETHNICITY_AMERICAN_INDIAN = 'American Indian and Alaska Native'
ETHNICITY_LIST = [ETHNICITY_BLACK,ETHNICITY_LATINO,ETHNICITY_WHITE,ETHNICITY_ASIAN,UNKNOWN_STR] #,ETHNICITY_MULTIPLE,ETHNICITY_AMERICAN_INDIAN]
ETHNICITY_SHORT_DICT = {ETHNICITY_BLACK:'Black',
                        ETHNICITY_ASIAN:ETHNICITY_ASIAN,
                        ETHNICITY_WHITE:ETHNICITY_WHITE,
                        ETHNICITY_LATINO:'Latino',
                        ETHNICITY_MULTIPLE:'Multi.',
                        ETHNICITY_AMERICAN_INDIAN:'American\nIndian',
                        UNKNOWN_STR:UNKNOWN_STR
                        }

DEMOGRAPHICX_ETHNICITY_MAPPING = {
    'asian': ETHNICITY_ASIAN,
    'black': ETHNICITY_BLACK,
    'hispanic': ETHNICITY_LATINO,
    'white': ETHNICITY_WHITE,
}

ETHNICOLOR_ETHNICITY_MAPPING_CODE = {
    'pctblack': 'black',        # Percent Non-Hispanic Black Only
    'pctapi': 'api',            # Percent Non-Hispanic Asian and Pacific Islander Only
    'pctaian': 'aian',          # Percent Non-Hispanic American Indian and Alaskan Native
    'pcthispanic': 'hispanic',  # Percent Hispanic Origin 
    'pct2prace': 'prace',       # Percent Non-Hispanic of Two or More Races
    'pctwhite': 'white',        # Percent Non-Hispanic White Only
}

ETHNICOLOR_ETHNICITY_MAPPING = {
    'api' : ETHNICITY_ASIAN, 
    'aian': ETHNICITY_AMERICAN_INDIAN, 
    'black': ETHNICITY_BLACK, 
    'hispanic': ETHNICITY_LATINO, 
    'prace': ETHNICITY_MULTIPLE,
    'white': ETHNICITY_WHITE
}

ETHNICOLOR_ETHNICITY_COLS = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']

ETHNICITIES_TO_PREDICT_GENDER = [ETHNICITY_BLACK, ETHNICITY_ASIAN, ETHNICITY_WHITE, ETHNICITY_LATINO, UNKNOWN_STR]

NO_LAUREATE = 'None'
NOBEL_CATEGORIES = ['Physics', 'Chemistry', 'Economics', 'Medicine', 'Peace', 'Literature', NO_LAUREATE]
NOBEL_CATEGORIES_RENAME = {'Physics':'Phys.', 'Chemistry':'Chem.', 'Economics':'Econ.', 'Medicine':'Medi.', 'Peace':'Pea.', 'Literature':'Lite.', NO_LAUREATE:NO_LAUREATE}
NOBEL_DECADES = [0] + list(range(1900,2030,10))
NOBEL_DECADES = [str(d) for d in NOBEL_DECADES] 

DEMOGRAPHIC_ATTRIBUTE_GENDER = 'gender'
DEMOGRAPHIC_ATTRIBUTE_ETHNICITY = 'ethnicity'
DEMOGRAPHIC_ATTRIBUTES = [DEMOGRAPHIC_ATTRIBUTE_GENDER,DEMOGRAPHIC_ATTRIBUTE_ETHNICITY]

PROMINENCE_CATEGORIES = ["low", "mid", "high", "elite"]


################################################################################################################
# Response constants
################################################################################################################
MAX_LETTERS_RESPONSE = 100
MAX_WORDS_RESPONSE = 10
ERROR_KEYWORDS_LC = ['"error"', 'fatalerror', '.runners(position', 'taba_keydown', 'unable to provide', 
                     'addtogroup', 'onitemclick', 'getinstance', "i'm stuck", '.datasource', 'getclient', 
                     'phone.toolstrip', 'actordatatype', 'baseactivity', 'setcurrent_company', '.clearfest', 
                     'getdata_suffix', '.texture_config', 'translator_concurrent', 'the above code will generate', 
                     'the actual implementation', 'texas.selection', 'allowedcreator', 'congratulations', 'extension',
                     'outreturns', 'ridiculous', 'adomnode', '.dropout', 'egg-enabled', 'podem/form', 'nvanonymousua',
                     'temptingordermaker']
NO_OUTPUT_MSG = ['No matching scientists found', 'No applicable scientists found', 'No applicable physicists found', 'No physicists meet the criteria', 'No real scientists meet the criteria', 'No relevant scientists found', 'No valid entries found', 'Other Scientist', 'Networks Group at University of Tokyo', 'Fermi National Accelerator Laboratory','RADIO ASTRONOMY TEAM','Nuclear Reactor Team']

NO_OUTPUT_KEYWORDS = ['No matching', 'No applicable', 'meet the criteria', 'No relevant', 'scientists', 'No valid', 'Other Scientist', 'physicist', 'researcher', 'University', 'Laboratory', 'data not available','RADIO ASTRONOMY TEAM', 'Nuclear Reactor Team', 'Nobel Prize', 'Full Name']
LLMCALLER_OUTPUT_INVALID_JSON_EMPTY = 'No JSON-like structure found in the response'
LLMCALLER_OUTPUT_INVALID_JSON_EXTRA_DATA = 'Invalid JSON format: Extra data'
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE = "Invalid JSON format: Expecting"
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE_SEMICOLON = "Invalid JSON format: Expecting ':' delimeter"
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE_COMMA = "Invalid JSON format: Expecting ','"
LLMCALLER_OUTPUT_INVALID_JSON_PROPERTY_NAME = 'Invalid JSON format: Expecting property name enclosed in double quotes'
LLMCALLER_OUTPUT_INVALID_JSON_CHAR = "Invalid JSON format: Invalid"
LLMCALLER_OUTPUT_SCHEMA_FAILED = "Schema validation failed:"
LLMCALLER_OUTPUT_PARSING_ERROR = "JSCONDecodeError:"
LLMCALLER_OUTPUT_EXPECTING_VALUE = "Invalid JSON format: Expecting value: "

EXPERIMENTS_BAD_CONTENT_TEXTS_LC = ['no specific scientists found', 'i was unable to', 
                                    'it appears that there is limited publicly available information', 
                                    'no physicists meet the specified criteria',
                                    'no specific scientists identified due to the uniqueness and complexity of the criteria',
                                    'none']

REFUSAL_TRUE = 1
REFUSAL_FALSE = 0
REFUSAL_UNKNOWN = -1

REFUSAL_KEYWORDS = [
    # causal / justificatory language
    r"\bbecause\b",
    r"\bsince\b",
    r"\bdue to\b",
    r"\bas an ai\b",
    r"\btherefore\b",
    r"\bso that\b",
    r"\bsorry\b",
    r"\bapologize\b",
    r"\bpromote\b",

    # policy / safety / governance
    r"\bpolicy\b",
    r"\bguidelines\b",
    r"\bsafety\b",
    r"\bprivacy\b",
    r"\bpersonal data\b",
    r"\bconfidential\b",
    r"\bethical\b",
    r"\brestrictions?\b",
    r"\bdiscriminat\b",
    r"\bharm\b",
    r"\bstereotyp\b",

    # capability / access explanations
    r"\bdon't have access\b",
    r"\bdo not have access\b",
    r"\bcannot access\b",
    r"\bcan't access\b",
    r"\blimited access\b",
    r"\bnot able to access\b",
    r"\bneed access\b",
    r"\bcan't assist\b",
    r"\bunable to\b",

    # compliance framing
    r"\bcannot comply\b",
    r"\bcan't comply\b",
    r"\bunable to comply\b",
    r"\bnot permitted\b",
    r"\bcan't\b",

    # words in gemini
    r"\bwhy this task cannot be completed\b",
    r"\bharmful\b",
    r"\bstereotype\b",
    r"\bunreliable\b"
    r"\bmutually exclusive\b"
]




################################################################################################################
# LLM constants
################################################################################################################

# LLM by size category
try:
    _llm_metadata = io.read_json_file('../../../LLMCaller/config/llm_setup.json').get('models', {})
except Exception as e:
    _llm_metadata = io.read_json_file('../../LLMCaller/config/llm_setup.json').get('models', {})
LLMS = list(_llm_metadata.keys())
print(f"Available LLMs: ({len(LLMS)}): {' '.join(LLMS)}")

LLMS_SMALL = [k for k in LLMS if _llm_metadata[k]['class'] == 'S']
LLMS_MEDIUM = [k for k in LLMS if _llm_metadata[k]['class'] == 'M']
LLMS_LARGE = [k for k in LLMS if _llm_metadata[k]['class'] == 'L']
LLMS_EXTRA_LARGE = [k for k in LLMS if _llm_metadata[k]['class'] == 'XL']
LLMS_PROPIETARY = [k for k in LLMS if _llm_metadata[k]['class'].endswith('(P)')]

# LLM by provider
LLMS_DEEPSEEK = [k for k in LLMS if k.startswith('deepseek-')] #['deepseek-chat-v3.1', 'deepseek-r1-0528']
LLMS_GEMINI = [k for k in LLMS if k.startswith('gemini-')] #['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-grounded', 'gemini-2.5-pro-grounded']
LLMS_GPT = [k for k in LLMS if k.startswith('gpt-')] #['gpt-oss-20b', 'gpt-oss-120b']
LLMS_LLAMA = [k for k in LLMS if k.startswith('llama-')] #['llama-3.3-8b', 'llama-3.1-70b', 'llama-3.3-70b', 'llama-3.1-405b', 'llama-4-scout', 'llama-4-mav']
LLMS_QWEN3 = [k for k in LLMS if k.startswith('qwen3-')] #['qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-30b-a3b-2507', 'qwen3-235b-a22b-2507']
LLMS_MISTRAL = [k for k in LLMS if k.startswith('mistral-')] #['mistral-small-3.2-24b', 'mistral-medium-3']
LLMS_GEMMA = [k for k in LLMS if k.startswith('gemma-')] #['gemma-3-12b', 'gemma-3-27b']
LLMS_GROK = [k for k in LLMS if k.startswith('grok-')] #['grok-4-fast']
LLMS_ORDERED = LLMS_DEEPSEEK + LLMS_GEMMA + LLMS_GEMINI + LLMS_GROK + LLMS_GPT + LLMS_LLAMA + LLMS_MISTRAL + LLMS_QWEN3
LLM_CLASSES = list(set([llm.split('-')[0] for llm in LLMS_ORDERED]))


# LLM by access category @TODO: these should be obtained from the LLMCaller/config/llm_setup.json file
LLMS_OPEN = LLMS_SMALL + LLMS_MEDIUM + LLMS_LARGE + LLMS_EXTRA_LARGE
LLM_ACCESS_CATEGORIES = {'open': LLMS_OPEN, 'proprietary': LLMS_PROPIETARY}
LLM_ACCESS_CATEGORIES_INV = {k:'open' if k in LLMS_OPEN else 'proprietary' for k in LLMS_ORDERED}


# LLM by class category
LLM_CLASS_CATEGORIES_INV = {k:'reasoning' if obj['reasoning'] else 'non-reasoning' for k, obj in _llm_metadata.items()}

################################################################################################################
# Experiment constants
################################################################################################################

INTERVENTION_PERIOD_START = '2025-12-19'
INTERVENTION_PERIOD_END = '2026-01-18'
INTERVENTION_PERIOD_QUERY = f"(not model.str.contains('gemini') and date >= '{INTERVENTION_PERIOD_START}' and date <= '{INTERVENTION_PERIOD_END}') or model in {LLMS_GEMINI}"

EXPERIMENT_OUTPUT_VALID = 'valid'
EXPERIMENT_OUTPUT_VERBOSED = 'verbose'
EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON = 'truncated-dict'
EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM = 'skipped-item'
EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON = 'fixed'
EXPERIMENT_OUTPUT_EMPTY = 'empty'
EXPERIMENT_OUTPUT_INVALID = 'invalid'
EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT = 'rate_limit'
EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR = 'server_error'
EXPERIMENT_OUTPUT_PROVIDER_ERROR = 'provider_error'
EXPERIMENT_OUTPUT_ILLUSTRATIVE = 'illustrative'

EXPERIMENT_OUTPUTS_ORDER = [EXPERIMENT_OUTPUT_VALID,
                            EXPERIMENT_OUTPUT_VERBOSED,
                            EXPERIMENT_OUTPUT_EMPTY,
                            EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM, EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON, EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON,
                            EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT, EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR, EXPERIMENT_OUTPUT_PROVIDER_ERROR,
                            EXPERIMENT_OUTPUT_ILLUSTRATIVE, EXPERIMENT_OUTPUT_INVALID]
                            
EXPERIMENT_OUTPUT_INVALID_FLAGS = [EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON,
                                   EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON, 
                                   EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM,
                                   EXPERIMENT_OUTPUT_INVALID, 
                                   EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT, 
                                   EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR, 
                                   EXPERIMENT_OUTPUT_PROVIDER_ERROR, 
                                   EXPERIMENT_OUTPUT_ILLUSTRATIVE, 
                                   EXPERIMENT_OUTPUT_EMPTY]

EXPERIMENT_OUTPUT_VALID_FLAGS = [EXPERIMENT_OUTPUT_VALID, 
                                 EXPERIMENT_OUTPUT_VERBOSED]

LLMCALLER_DICT_KEYS = ['Name','Years','DOI','Ethnicity','Career Age']

AUDITOR_RESPONSE_DICT_KEYS = ['date', 'time', 'model', 'temperature', 'llm_provider', 'llm_model', 'task_name', 'task_param', 'task_attempt', 'result_valid_flag'] + LLMCALLER_DICT_KEYS

EXPERIMENT_TASK_TOPK = 'top_k'
EXPERIMENT_TASK_FIELD = 'field'
EXPERIMENT_TASK_EPOCH = 'epoch'
EXPERIMENT_TASK_SENIORITY = 'seniority'
EXPERIMENT_TASK_TWINS = 'twins'
EXPERIMENT_TASK_BIASED_TOP_K = 'biased_top_k'
EXPERIMENT_TASKS = [EXPERIMENT_TASK_TOPK, EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY, EXPERIMENT_TASK_TWINS]
EXPERIMENT_TASKS_COLORS = {EXPERIMENT_TASK_TOPK:'tab:blue',EXPERIMENT_TASK_FIELD:'tab:orange',EXPERIMENT_TASK_EPOCH:'tab:green',EXPERIMENT_TASK_SENIORITY:'tab:red',EXPERIMENT_TASK_TWINS:'tab:purple'}

EXPERIMENT_AUDIT_FACTUALITY = 'factuality'
EXPERIMENT_AUDIT_FACTUALITY_AUTHOR_NAME_REPLACEMENTS = {'Nobel Prize Winner ':'',
                                                        'Nobel Laureate ':'',
                                                        'Nobel Prize in Physics 2018 Winner: ': '',
                                                        'Nobel Prize in Physics 2019 Winner: ': '',
                                                        'Nobel Prize in Physics 2020 Winner: ': '',
                                                        'Nobel Prize in Physics 2021 Winner: ': '',
                                                        'Nobel Prize in Physics 2022 Winner: ': '',
                                                        }

################################################################################################################
# Factuality constants
################################################################################################################
FACTUALITY_AUTHOR = 'author'
FACTUALITY_AUTHOR_THRESHOLD = 5
FACTUALITY_AUTHOR_FIELD_DOI = 1
FACTUALITY_AUTHOR_YEAR_NOW = 2025
FACTUALITY_SENIORITY_EARLY_CAREER = 'early_career'
FACTUALITY_SENIORITY_MID_CAREER = 'mid_career'
FACTUALITY_FIELD_PER = 'PER'
FACTUALITY_FIELD_PER_SF = 'Education'
FACTUALITY_FIELD_CMP = 'CM&MP'
FACTUALITY_FIELD_CMP_SF = 'Condensed Matter Physics'
FACTUALITY_SENIORITY_SENIOR_CAREER = 'senior'
FACTUALITY_COLUMN_ID = 'original_index'
FACTUALITY_AUTHOR_STATS_TO_HIDE = ['works_count','cited_by_count','h_index','i10_index','e_index','two_year_mean_citedness']
FACTUALITY_AUTHOR_DEMOGRAPHICS_TO_HIDE = ['ethnicity_dx','ethnicity_ec','ethnicity','gender']
FACTUALITY_AUTHOR_METADATA_TO_HIDE = FACTUALITY_AUTHOR_STATS_TO_HIDE + FACTUALITY_AUTHOR_DEMOGRAPHICS_TO_HIDE
FACTUALITY_FIELD_TO_HIDE = ['fact_author_score', 'career_age', 'years', 'id_author_aps_list', 'year_first_publication','year_last_publication','academic_age','seniority_active', 'seniority_now', 'age_now']
FACTUALITY_EPOCH_TO_HIDE = ['fact_author_score', 'career_age', 'doi', 'id_author_aps_list', 'academic_age', 'age_now', 'seniority_active', 'seniority_now']
FACTUALITY_SENIORITY_TO_HIDE = ['fact_author_score', 'doi', 'years', 'id_author_aps_list']
FACTUALITY_TASKS = [EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY]

FACTUALITY_NONE = 'None'
FACTUALITY_STAT_COUNTS = 'counts'
FACTUALITY_STAT_PCT = 'pct'
FACTUALITY_VALID_STATS = [FACTUALITY_STAT_COUNTS, FACTUALITY_STAT_PCT]

FACTUALITY_AUTHOR_EXISTS = 'Exists'
FACTUALITY_AUTHOR_BOTH = 'APS & OA'
FACTUALITY_AUTHOR_APS = 'APS'
FACTUALITY_AUTHOR_OA = 'OA'
FACTUALITY_AUTHOR_FACT_CHECKS = [FACTUALITY_AUTHOR_BOTH,FACTUALITY_AUTHOR_OA,FACTUALITY_AUTHOR_APS,FACTUALITY_NONE]

FACTUALITY_FIELD_DOI = 'DOI'
FACTUALITY_FIELD_AUTHOR = 'Author'
FACTUALITY_FIELD_AUTHOR_FIELD = 'Author &\nField'
FACTUALITY_FIELD_DOI_AUTHOR_FIELD = 'A.D.F.'
FACTUALITY_FIELD_AT_LEAST_ONE = 'Either'
FACTUALITY_FIELD_ALL = 'All'
FACTUALITY_FIELD_FACT_CHECKS = [FACTUALITY_FIELD_AUTHOR,FACTUALITY_FIELD_DOI,FACTUALITY_FIELD_DOI_AUTHOR_FIELD] #FACTUALITY_FIELD_AT_LEAST_ONE,FACTUALITY_FIELD_ALL,FACTUALITY_NONE]

FACTUALITY_SENIORITY_AUTHOR = 'Author'
FACTUALITY_SENIORITY_ACTIVE = 'Then$_{(txt)}$'
FACTUALITY_SENIORITY_NOW = 'Now$_{(txt)}$'
FACTUALITY_SENIORITY_ACTIVE_REQ = 'Then'
FACTUALITY_SENIORITY_NOW_REQ = 'Now'
FACTUALITY_SENIORITY_FACT_CHECKS = [FACTUALITY_SENIORITY_AUTHOR, FACTUALITY_SENIORITY_ACTIVE_REQ,FACTUALITY_SENIORITY_NOW_REQ, FACTUALITY_SENIORITY_ACTIVE,FACTUALITY_SENIORITY_NOW]

FACTUALITY_EPOCH_AUTHOR = 'Author'
FACTUALITY_EPOCH_AS_REQUESTED = 'Match'
FACTUALITY_EPOCH_AS_LLM_IN_GT = 'In$_{(txt)}$'
FACTUALITY_EPOCH_AS_GT_IN_LLM = 'Out$_{(txt)}$'
FACTUALITY_EPOCH_AS_OVERLAP = 'Over$_{(txt)}$'
FACTUALITY_EPOCH_FACT_CHECKS = [FACTUALITY_EPOCH_AUTHOR, FACTUALITY_EPOCH_AS_REQUESTED,FACTUALITY_EPOCH_AS_LLM_IN_GT,FACTUALITY_EPOCH_AS_GT_IN_LLM,FACTUALITY_EPOCH_AS_OVERLAP] #,FACTUALITY_NONE]

FACTUALITY_FIELD_METRICS = ['fact_author','fact_doi_score','fact_doi_author_field']
FACTUALITY_SENIORITY_METRICS = ['id_author_oa', 'fact_seniority_active', 'fact_seniority_now', 'fact_seniority_active_requested', 'fact_seniority_now_requested']
FACTUALITY_EPOCH_METRICS = ['id_author_oa', 'fact_epoch_requested','fact_epoch_llm_in_gt','fact_epoch_gt_in_llm','fact_epoch_overlap']

################################################################################################################
# Task constants
################################################################################################################

# @TODO: the params can be obtained from LLMCaller/config/category_variables.json
# @TODO: the tasks can be obtained from LLMCaller/config/prompt_config.json 

TASK_TOPK_PARAMS = ['top_5', 'top_100']

TASK_TWINS_GENDER_ORDER = ['female', 'male']
TASK_TWINS_GROUP_ORDER = ['famous', 'random', 'politic',  'movie', 'fictitious']
TASK_TWINS_FAMOUS_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[0]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[0]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_RANDOM_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[1]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[1]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_POLITIC_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[2]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[2]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_MOVIE_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[3]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[3]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_FICTICIOUS_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[4]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[4]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_CONTROL = TASK_TWINS_POLITIC_PARAMS + TASK_TWINS_MOVIE_PARAMS + TASK_TWINS_FICTICIOUS_PARAMS

TASK_FIELD_PARAMS = ['CM&MP','PER']
TASK_EPOCH_PARAMS = ['1950s','2000s']
TASK_SENIORITY_PARAMS = ['early_career','senior']
TASK_PARAMS_BY_TASK = {EXPERIMENT_TASK_TOPK:TASK_TOPK_PARAMS,
                       EXPERIMENT_TASK_FIELD:TASK_FIELD_PARAMS,
                       EXPERIMENT_TASK_EPOCH:TASK_EPOCH_PARAMS,
                       EXPERIMENT_TASK_SENIORITY:TASK_SENIORITY_PARAMS,
                       EXPERIMENT_TASK_TWINS:TASK_TWINS_FAMOUS_PARAMS+TASK_TWINS_RANDOM_PARAMS+TASK_TWINS_POLITIC_PARAMS+TASK_TWINS_MOVIE_PARAMS+TASK_TWINS_FICTICIOUS_PARAMS}

TASK_TOPK_BIASED_PARAMS = ["top_100_bias_diverse",
                            "top_100_bias_gender_equal",
                            "top_100_bias_gender_female",
                            "top_100_bias_gender_male",
                            "top_100_bias_gender_neutral",
                            "top_100_bias_ethnicity_equal",
                            "top_100_bias_ethnicity_asian",
                            "top_100_bias_ethnicity_black",
                            "top_100_bias_ethnicity_latino",
                            "top_100_bias_ethnicity_white",
                            "top_100_bias_citations_high",
                            "top_100_bias_citations_low"]

TASK_TOPK_BIASED_PARAMS_GENDER = [t for t in TASK_TOPK_BIASED_PARAMS if '_gender_' in t]
TASK_TOPK_BIASED_PARAMS_ETHNICITY = [t for t in TASK_TOPK_BIASED_PARAMS if '_ethnicity_' in t]
TASK_TOPK_BIASED_PARAMS_CITATIONS = [t for t in TASK_TOPK_BIASED_PARAMS if '_citations_' in t]
TASK_TOPK_BIASED_PARAMS_DIVERSE = [t for t in TASK_TOPK_BIASED_PARAMS if t.endswith('_diverse')]


EXPERIMENT_TASK_PARAMS_ORDER_EXPANDED = [item for sublist in TASK_PARAMS_BY_TASK.values() for item in sublist]





################################################################################################################
# Plot constants
################################################################################################################
FIG_DPI = 600
FONT_SCALE = 1.55

PLOT_FIGSIZE = (5,2.2) #(5,2.4)
PLOT_FIGSIZE_SPIDER = (5,5)
PLOT_FIGSIZE_BAR = (5,3.5)

PLOT_FIGSIZE_NO_LEGEND = (5,2.3) #(5,2.5)
PLOT_FIGSIZE_SPIDER_NO_LEGEND = (5,6)
PLOT_FIGSIZE_NARROW = (5,2.4)

PLOT_YLIM_PARALLEL_PCT = (-0.1,1.1)
PLOT_YLIM_SPIDER_PCT = (-0.01,1.01)
PLOT_BAR_WIDTH = 0.7

PLOT_LEGEND_KWARGS_PARALLEL_COORD = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(.085,0.96,1,0.2)}
PLOT_LEGEND_KWARGS_SPIDER = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(0.08,0.94,1,0.2)}
PLOT_LEGEND_KWARGS_BARPLOT = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(0.15,0.95,1,0.2)}
PLOT_LEGEND_KWARGS_PCA = {'title':None, 'loc': 'lower left', 'ncols':6, 'bbox_to_anchor':(0.02,0.92,1,0.2), 'handletextpad':0.4, 'labelspacing':0.} #, 'borderpad':0.5}
PLOT_LEGEND_KWARGS_BOXPLOT = {'title':None, 'ncols':1, 'loc':'upper right', 'frameon':True, 'borderpad':0.4, 'handlelength':0.6, 'bbox_to_anchor':(0.47,0.4,0.5,0.5), 'fontsize':'small'}

COMPONENT_POPULATION_COLOR = '#F5EFED' #'#d8d6d0' # '#f6e6cb'
COMPPONENT_TASK_PARAM_COLORS = ['#dd9787', '#a6c48a']
COMPPONENT_TASK_PARAM_MARKERS = ['o', 'o'] #  'X']

EXPERIMENT_OUTPUT_COLORS = {EXPERIMENT_OUTPUT_VALID:"#1B5E20",
                            EXPERIMENT_OUTPUT_VERBOSED:"#66BB6A",
                            EXPERIMENT_OUTPUT_EMPTY:"#9E9E9E",
                            EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON:"",
                            EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM:"#EF9A9A",
                            EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON:"",
                            EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT:"",
                            EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR:"",
                            EXPERIMENT_OUTPUT_PROVIDER_ERROR:"#E53935",
                            EXPERIMENT_OUTPUT_ILLUSTRATIVE:"",
                            EXPERIMENT_OUTPUT_INVALID:"#8E0000",
                            }

GENDER_COLOR_DICT = {GENDER_MALE:"#1F77B4",
                     GENDER_FEMALE:"#FF7F0E",
                     GENDER_UNISEX:"#2CA02C",
                     UNKNOWN_STR:"#D3D3D3"}
                     
ETHNICITY_COLOR_DICT = {ETHNICITY_BLACK:"#4E79A7",
                        ETHNICITY_ASIAN:"#F28E2B",
                        ETHNICITY_WHITE:"#76B7B2",
                        ETHNICITY_LATINO:"#E15759",
                        UNKNOWN_STR:"#BAB0AC"}

_llm_colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
LLM_CLASS_COLORS = {llmclass: _llm_colors[i % len(_llm_colors)] for i, llmclass in enumerate(LLM_CLASSES)}
LLM_COLORS = {llm:LLM_CLASS_COLORS[llm.split('-')[0]] for llm in LLMS_ORDERED}

# LLM by size category
LLMS_SIZE_CATEGORIES = {'small': LLMS_SMALL, 'medium': LLMS_MEDIUM, 'large': LLMS_LARGE, 
                        # 'extra-large': LLMS_EXTRALARGE, 
                        'proprietary': LLMS_PROPIETARY}
LLMS_SIZE_COLORS = {'small':"#1DA41B", 'medium':'#A9C9FF', 'large':"#4A7DBE", 
                    # 'extra-large':'#0B3C5D', 
                    'proprietary':'tab:orange'}
LLMS_SIZE_ORDER = ['small', 'medium', 'large', 
                #    'extra-large', 
                   'proprietary']
LLMS_SIZE_SHORT_NAMES = {'small':'S', 'medium':'M', 'large':'L', 
                        #  'extra-large':'XL', 
                         'proprietary':'P'}

LLMS_SIZE_CATEGORIES_INV = {k:obj['class'] for k, obj in _llm_metadata.items()}

DEMOGRAPHIC_ATTRIBUTE_LABELS_ORDER = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_LIST,
                                      DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_LIST} 
DEMOGRAPHIC_ATTRIBUTE_LABELS_COLOR = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_COLOR_DICT,
                                     DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_COLOR_DICT}




################################################################################################################
# Benchmark metrics
################################################################################################################

TEMPERATURE_VALUES = [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

BENCHMARK_METRICS = ['refusal_pct', 'validity_pct', 
                    'duplicates', 'consistency', 'factuality_author', 
                    'connectedness', 'connectedness_density', 'connectedness_norm_entropy', 'connectedness_ncomponents',
                    'similarity_pca',
                    'diversity_gender', 'diversity_ethnicity', 'diversity_prominence_pub', 'diversity_prominence_cit', 
                    'parity_gender', 'parity_ethnicity', 'parity_prominence_pub', 'parity_prominence_cit',
                    ]

BENCHMARK_METRICS_PLOT_ORDER = ['refusal_pct', 'validity_pct', 'duplicates', 'consistency', 'factuality_author', 'connectedness', 'similarity_pca', 'diversity_gender',  'parity_gender']

BENCHMARK_BINARY_METRICS = {
    "validity_pct",
    "refusal_pct",
}

BENCHMARK_PARITY_METRICS = {m for m in BENCHMARK_METRICS if m.startswith('parity_')}

BENCHMARK_SIMILARITY_METRICS = {m for m in BENCHMARK_METRICS if m.startswith('connectedness') or m.startswith('similarity_')}

BENCHMARK_SIMILARITY_METRICS_MAP = {'connectedness': 'connectedness_entropy',
                                    'connectedness_density': 'recommended_author_pairs_are_coauthors', 
                                    'connectedness_norm_entropy': 'normalized_component_entropy', 
                                    'connectedness_ncomponents': 'normalized_n_components',
                                    'similarity_pca': 'scholarly_pca_similarity_mean'}

BENCHMARK_SOCIAL_METRICS = [
    "connectedness",
    "connectedness_density",
    "connectedness_norm_entropy",
    "connectedness_ncomponents",
    "similarity_pca",
    "diversity_gender",
    "diversity_ethnicity",
    "diversity_prominence_pub",
    "diversity_prominence_cit",
    "parity_gender",
    "parity_ethnicity",
    "parity_prominence_pub",
    "parity_prominence_cit",
]

BENCHMARK_TECHNICAL_METRICS = [
    "refusal_pct",
    "validity_pct",
    "duplicates",
    "consistency",
    "factuality_author",
]

BENCHMARK_METRICS_LABEL_MAP = {
    "refusal_pct": "Refusal",
    "validity_pct": r"Validity$\uparrow$",
    "duplicates": r"Duplicates$\downarrow$",
    "consistency": "Consistency",
    "factuality_author": r"Factuality$\uparrow$",
    "connectedness": "Connectedness",
    "similarity_pca": "Similarity",
    "diversity_gender": "Diversity",
    "parity_gender": r"Parity$\uparrow$",
    "diversity_ethnicity": "Diversity",
    "parity_ethnicity": r"Parity$\uparrow$",
    "diversity_prominence_pub": "Diversity",
    "parity_prominence_pub": r"Parity$\uparrow$",
    "diversity_prominence_cit": "Diversity",
    "parity_prominence_cit": r"Parity$\uparrow$",
}

BENCHMARK_DEMOGRAPHIC_ATTRIBUTES = ['gender', 'ethnicity', 'prominence_pub', 'prominence_cit']

BENCHMARK_MODEL_GROUPS = ['model_access', 'model_size', 'model_class']
BENCHMARK_MODEL_GROUPS_LABEL_MAP = {"model_access": "Access", "model_size": "Size", "model_class": "Reasoning"}

BENCHMARK_PER_ATTEMPT_COLS = BENCHMARK_MODEL_GROUPS + ['model', 'grounded','temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
BENCHMARK_PER_REQUEST_COLS = BENCHMARK_MODEL_GROUPS + ["model", 'grounded','temperature', "date", "time", "task_name", "task_param"]
                             

BENCHMARK_MODEL_GROUP_LABEL_MAP = {
    "open": "Open", "proprietary": "Proprietary",
    "S": "Small", "M": "Medium", "L": "Large", "XL": "Extra Large",
    "non-reasoning": "Disabled", "reasoning": "Enabled",
}

BENCHMARK_MODEL_GROUP_ORDER_WITHIN_GROUP = {
    "model_access": ["open", "proprietary"],
    "model_size": ["S", "M", "L", "XL"],
    "model_class": ["non-reasoning", "reasoning"],
}

# Highlight rules for bar figure
BENCHMARK_METRIC_HIGHLIGHT_RULES = {
    "validity_pct": "max",
    "duplicates": "min",
    "factuality_author": "max",
    "parity_gender": "max",
}
