
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

UNKNOWN_STR = 'Unknown'

GENDER_UNISEX = 'Unisex'
GENDER_FEMALE = 'Female'
GENDER_MALE = 'Male'
GENDER_LIST = [GENDER_FEMALE, GENDER_MALE, GENDER_UNISEX, UNKNOWN_STR]
GENDER_COLOR_DICT = {GENDER_MALE:"#1F77B4",
                     GENDER_FEMALE:"#FF7F0E",
                     GENDER_UNISEX:"#2CA02C",
                     UNKNOWN_STR:"#D3D3D3"}

ETHNICITY_BLACK = 'Black or African American'
ETHNICITY_ASIAN = 'Asian'
ETHNICITY_WHITE = 'White'
ETHNICITY_LATINO = 'Hispanic or Latino'
ETHNICITY_MULTIPLE = 'Multiple'
ETHNICITY_AMERICAN_INDIAN = 'American Indian and Alaska Native'
ETHNICITY_LIST = [ETHNICITY_BLACK,ETHNICITY_LATINO,ETHNICITY_WHITE,ETHNICITY_ASIAN,UNKNOWN_STR] #,ETHNICITY_MULTIPLE,ETHNICITY_AMERICAN_INDIAN]
# ETHNICITY_LIST = [ETHNICITY_ASIAN,ETHNICITY_WHITE,ETHNICITY_LATINO,ETHNICITY_BLACK,UNKNOWN_STR] #,ETHNICITY_MULTIPLE,ETHNICITY_AMERICAN_INDIAN]
ETHNICITY_SHORT_DICT = {ETHNICITY_BLACK:'Black',
                        ETHNICITY_ASIAN:ETHNICITY_ASIAN,
                        ETHNICITY_WHITE:ETHNICITY_WHITE,
                        ETHNICITY_LATINO:'Latino',
                        ETHNICITY_MULTIPLE:'Multi.',
                        ETHNICITY_AMERICAN_INDIAN:'American\nIndian',
                        UNKNOWN_STR:UNKNOWN_STR
                        }
ETHNICITY_COLOR_DICT = {ETHNICITY_BLACK:"#4E79A7",
                        ETHNICITY_ASIAN:"#F28E2B",
                        ETHNICITY_WHITE:"#76B7B2",
                        ETHNICITY_LATINO:"#E15759",
                        UNKNOWN_STR:"#BAB0AC"}

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

ID_STR_LEN = 7
TRUE_STR = 'True'

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



### EXPERIMETS ###
# LLMS = ['llama3-8b', 'llama-3.1-8b', 'gemma2-9b', 'mixtral-8x7b', 'llama3-70b', 'llama-3.1-70b']
# LLMS = ["deepseek-chat-v3.1", "deepseek-r1-0528",
#         "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-grounded", "gemini-2.5-pro-grounded",
#         "gpt-oss-20b", "gpt-oss-120b",
#         "llama-3.3-8b", "llama-3.1-70b", "llama-3.3-70b", "llama-3.1-405b", "llama-4-scout", "llama-4-mav",
#         "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b-2507", "qwen3-235b-a22b-2507", 
#         "mistral-small-3.2-24b", "mistral-medium-3", 
#         "gemma-3-12b-it", "gemma-3-27b-it",
#         "grok-4-fast"]

LLMS_DEEPSEEK = ['deepseek-chat-v3.1', 'deepseek-r1-0528']
LLMS_GEMINI = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-grounded', 'gemini-2.5-pro-grounded']
LLMS_GPT = ['gpt-oss-20b', 'gpt-oss-120b']
LLMS_LLAMA = ['llama-3.3-8b', 'llama-3.1-70b', 'llama-3.3-70b', 'llama-3.1-405b', 'llama-4-scout', 'llama-4-mav']
LLMS_QWEN3 = ['qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-30b-a3b-2507', 'qwen3-235b-a22b-2507']
LLMS_MISTRAL = ['mistral-small-3.2-24b', 'mistral-medium-3']
LLMS_GEMMA = ['gemma-3-12b-it', 'gemma-3-27b-it']
LLMS_GROK = ['grok-4-fast']

LLMS = LLMS_DEEPSEEK + LLMS_GEMINI + LLMS_GPT + LLMS_LLAMA + LLMS_QWEN3 + LLMS_MISTRAL + LLMS_GEMMA + LLMS_GROK

# deepseek-chat-v3.1 deepseek-r1-0528 gemini-2.5-flash gemini-2.5-pro gemini-2.5-flash-grounded gemini-2.5-pro-grounded gpt-oss-20b gpt-oss-120b llama-3.3-8b llama-3.1-70b llama-3.3-70b llama-3.1-405b llama-4-scout llama-4-mav qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b-2507 qwen3-235b-a22b-2507 mistral-small-3.2-24b mistral-medium-3 gemma-3-12b-it gemma-3-27b-it grok-4-fast

EXPERIMENT_OUTPUT_VALID = 'valid'
EXPERIMENT_OUTPUT_VERBOSED = 'verbose'
EXPERIMENT_OUTPUT_FIXED = 'fixed'
EXPERIMENT_OUTPUT_INVALID = 'invalid'
EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT = 'rate_limit'
EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR = 'server_error'
EXPERIMENT_OUTPUT_PROVIDER_ERROR = 'provider_error'
EXPERIMENT_OUTPUT_ILLUSTRATIVE = 'illustrative'

LLMCALLER_OUTPUT_NO_JSON = 'No JSON-like structure found in the response'
LLMCALLER_OUTPUT_INVALID_JSON = 'Invalid JSON format: Extra data'
LLMCALLER_OUTPUT_EXPECTING_SEMICOLON = "Invalid JSON format: Expecting ':' delimeter"

EXPERIMENTS_BAD_CONTENT_TEXTS = ['no specific scientists found', 'i was unable to', 
                                 'it appears that there is limited publicly available information', 
                                 'no physicists meet the specified criteria',
                                 'no specific scientists identified due to the uniqueness and complexity of the criteria']

EXPERIMENT_OUTPUT_VALIDATION_FLAGS = [EXPERIMENT_OUTPUT_VALID,EXPERIMENT_OUTPUT_VERBOSED,EXPERIMENT_OUTPUT_FIXED,
                                      EXPERIMENT_OUTPUT_INVALID,EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT,EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR,
                                      EXPERIMENT_OUTPUT_PROVIDER_ERROR,EXPERIMENT_OUTPUT_ILLUSTRATIVE]
EXPERIMENT_OUTPUT_VALIDATION_FLAGS_COLORS = {EXPERIMENT_OUTPUT_VALID : (0.0, 0.23529411764705882, 0.18823529411764706, 1.0),
                                             EXPERIMENT_OUTPUT_VERBOSED : (0.0878892733564014, 0.479123414071511, 0.44775086505190315, 1.0),
                                             EXPERIMENT_OUTPUT_FIXED : (0.9636293733179546, 0.9237985390234525, 0.8185313341022683, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID : (0.8572856593617839, 0.7257977700884274, 0.4471357170319107, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT : (0.6313725490196078, 0.3951557093425605, 0.09573241061130335, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR : (0.32941176470588235, 0.18823529411764706, 0.0196078431372549, 1.0)}
EXPERIMENT_OUTPUT_VALID_FLAGS = [EXPERIMENT_OUTPUT_VALID,EXPERIMENT_OUTPUT_VERBOSED] #,EXPERIMENT_OUTPUT_FIXED, EXPERIMENT_OUTPUT_ILLUSTRATIVE

# LLMS_COLORS = {'llama3-8b':'tab:blue', 
#                'llama-3.1-8b':'tab:orange', 
#                'gemma2-9b':'tab:green', 
#                'mixtral-8x7b':'tab:red', 
#                'llama3-70b':'tab:purple', 
#                'llama-3.1-70b':'tab:brown'}

LLM_COLOR = {'deepseek': 'tab:blue',
             'gemini': 'tab:orange',
             'gpt': 'tab:green',
             'llama': 'tab:red',
             'qwen3': 'tab:purple',
             'mistral': 'tab:brown',
             'gemma': 'tab:pink',
             'grok': 'tab:gray'}
LLMS_COLORS = {LLM_COLOR[llm.split("-")[0]] for llm in LLMS}


EXPERIMENT_TASK_TOPK = 'top_k'
EXPERIMENT_TASK_FIELD = 'field'
EXPERIMENT_TASK_EPOCH = 'epoch'
EXPERIMENT_TASK_SENIORITY = 'seniority'
EXPERIMENT_TASK_TWINS = 'twins'
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

# FACTUALITY_FIELD_CORRECT = 'Correct'
# FACTUALITY_FIELD_INCORRECT = 'Incorrect'
# FACTUALITY_FIELD_PARTIAL = 'Partial'
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
# FACTUALITY_SENIORITY_FACT_CHECKS = [FACTUALITY_SENIORITY_ACTIVE_REQ,FACTUALITY_SENIORITY_NOW_REQ,FACTUALITY_SENIORITY_ACTIVE,FACTUALITY_SENIORITY_NOW,FACTUALITY_NONE]
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

NO_LAUREATE = 'None'
NOBEL_CATEGORIES = ['Physics', 'Chemistry', 'Economics', 'Medicine', 'Peace', 'Literature', NO_LAUREATE]
NOBEL_CATEGORIES_RENAME = {'Physics':'Phys.', 'Chemistry':'Chem.', 'Economics':'Econ.', 'Medicine':'Medi.', 'Peace':'Pea.', 'Literature':'Lite.', NO_LAUREATE:NO_LAUREATE}
NOBEL_DECADES = [0] + list(range(1900,2030,10))
NOBEL_DECADES = [str(d) for d in NOBEL_DECADES] 


DEMOGRAPHIC_ATTRIBUTE_GENDER = 'gender'
DEMOGRAPHIC_ATTRIBUTE_ETHNICITY = 'ethnicity'
DEMOGRAPHIC_ATTRIBUTES = [DEMOGRAPHIC_ATTRIBUTE_GENDER,DEMOGRAPHIC_ATTRIBUTE_ETHNICITY]
DEMOGRAPHIC_ATTRIBUTE_LABELS_ORDER = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_LIST,
                                      DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_LIST} 
DEMOGRAPHIC_ATTRIBUTE_LABELS_COLOR = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_COLOR_DICT,
                                     DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_COLOR_DICT}

EXPERIMENT_TASK_PARAMS_ORDER_EXPANDED = [item for sublist in TASK_PARAMS_BY_TASK.values() for item in sublist]

COMPONENT_POPULATION_COLOR = '#F5EFED' #'#d8d6d0' # '#f6e6cb'
COMPPONENT_TASK_PARAM_COLORS = ['#dd9787', '#a6c48a']
COMPPONENT_TASK_PARAM_MARKERS = ['o', 'o'] #  'X']

APS_CAREER_AGE_COL = 'aps_career_age'
OA_CAREER_AGE_COL = 'career_age'

APS_PRESTIGE_METRICS_COL = list(APS_RANKING_METRICS.values()) 
OA_PRESTIGE_METRICS_COL = list(RANKING_METRICS.values())
ALL_PRESTIGE_METRICS_COL = APS_PRESTIGE_METRICS_COL + OA_PRESTIGE_METRICS_COL
APS_SCHOLARLY_METRICS_COL = APS_PRESTIGE_METRICS_COL + [APS_CAREER_AGE_COL]
OA_SCHOLARLY_METRICS_COL =  OA_PRESTIGE_METRICS_COL + [OA_CAREER_AGE_COL]
ALL_SCHOLARLY_METRICS_COL = APS_SCHOLARLY_METRICS_COL + OA_SCHOLARLY_METRICS_COL

METADATA_DIR = 'metadata'
SIMILARITIES_DIR = 'similarities'

import numpy as np
NONE = ['', None, 'None', 'nan', 'NaN', np.nan, 'null', 'Null', 'NULL', 'N/A', 'n/a', 'N/a', 'n/A']
INF = [np.inf, -np.inf]

TEMPERATURE_FOLDER_PREFIX = 'temperature_'

ERROR_KEYWORDS_LC = ['"error"', 'fatalerror', '.runners(position', 'taba_keydown', 'unable to provide', 
                     'addtogroup', 'onitemclick', 'getinstance', "i'm stuck", '.datasource', 'getclient', 
                     'phone.toolstrip', 'actordatatype', 'baseactivity', 'setcurrent_company', '.clearfest', 
                     'getdata_suffix', '.texture_config', 'translator_concurrent', 'the above code will generate', 
                     'the actual implementation', 'texas.selection', 'allowedcreator', 'congratulations', 'extension',
                     'outreturns', 'ridiculous', 'adomnode', '.dropout', 'egg-enabled', 'podem/form', 'nvanonymousua',
                     'temptingordermaker']