APS_OA_AUTHORS_FN = 'authors.csv'
APS_OA_PUBLICATIONS_FN = 'publications.csv'
APS_OA_AUTHORS_MAPPING_FN = 'authors_mapping.csv'
APS_OA_PUBLICATIONS_MAPPING_FN = 'publications_mapping.csv'
APS_OA_ALTERNATIVE_NAMES_FN = 'alternative_names.csv'
APS_OA_AUTHORS_INSTITUTION_YEAR_FN = 'author_institution_year.csv'
APS_OA_INSTITUTIONS_FN = 'institution.csv'
APS_OA_AUTHORSHIPS_FN = 'authorships.csv'
APS_OA_CITATIONS_FN = 'citations.csv'
APS_OA_AUTHORS_DEMOGRAPHICS_FN = 'author_demographics.csv'
APS_OA_PUBLICTION_TOPICS = 'publications_topic.csv'
APS_OA_TOPICS_FN = 'topics.csv'

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

GENDER_UNISEX = 'Unisex'
GENDER_FEMALE = 'Female'
GENDER_MALE = 'Male'

UNKNOWN_STR = 'Unknown'

ETHNICITY_BLACK = 'Black or African American'
ETHNICITY_ASIAN = 'Asian'
ETHNICITY_WHITE = 'White'
ETHNICITY_LATINO = 'Hispanic or Latino'
ETHNICITY_MULTIPLE = 'Multiple'
ETHNICITY_AMERICAN_INDIAN = 'American Indian and Alaska Native'

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
LLMS = ['llama3-8b', 'llama-3.1-8b', 'gemma2-9b', 'mixtral-8x7b', 'llama3-70b', 'llama-3.1-70b']

EXPERIMENT_OUTPUT_VALID = 'valid'
EXPERIMENT_OUTPUT_VERBOSED = 'verbosed'
EXPERIMENT_OUTPUT_FIXED = 'fixed'
EXPERIMENT_OUTPUT_INVALID = 'invalid'
EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT = 'rate_limit'
EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR = 'server_error'

EXPERIMENTS_BAD_CONTENT_TEXTS = ['no specific scientists found', 'i was unable to', 
                                 'it appears that there is limited publicly available information', 
                                 'no physicists meet the specified criteria',
                                 'no specific scientists identified due to the uniqueness and complexity of the criteria']

EXPERIMENT_OUTPUT_VALIDATION_FLAGS = [EXPERIMENT_OUTPUT_VALID,EXPERIMENT_OUTPUT_VERBOSED,EXPERIMENT_OUTPUT_FIXED,EXPERIMENT_OUTPUT_INVALID,EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT,EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR]
EXPERIMENT_OUTPUT_VALIDATION_FLAGS_COLORS = {EXPERIMENT_OUTPUT_VALID : (0.0, 0.23529411764705882, 0.18823529411764706, 1.0),
                                             EXPERIMENT_OUTPUT_VERBOSED : (0.0878892733564014, 0.479123414071511, 0.44775086505190315, 1.0),
                                             EXPERIMENT_OUTPUT_FIXED : (0.9636293733179546, 0.9237985390234525, 0.8185313341022683, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID : (0.8572856593617839, 0.7257977700884274, 0.4471357170319107, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT : (0.6313725490196078, 0.3951557093425605, 0.09573241061130335, 1.0),
                                             EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR : (0.32941176470588235, 0.18823529411764706, 0.0196078431372549, 1.0)}
EXPERIMENT_OUTPUT_VALID_FLAGS = [EXPERIMENT_OUTPUT_VALID,EXPERIMENT_OUTPUT_VERBOSED] #,EXPERIMENT_OUTPUT_FIXED

# EXPERIMENT_OUTPUT_VALIDATION_FLAGS_COLORS = {EXPERIMENT_OUTPUT_VALID : (0.0, 0.23529411764705882, 0.18823529411764706, 1.0),
#                                              EXPERIMENT_OUTPUT_VERBOSED : (0.207843137254902, 0.592156862745098, 0.5607843137254902, 1.0),
#                                              EXPERIMENT_OUTPUT_FIXED : (0.7803921568627453, 0.9176470588235294, 0.8980392156862746, 1.0),
#                                              EXPERIMENT_OUTPUT_INVALID : (0.9647058823529412, 0.9098039215686274, 0.7647058823529411, 1.0),
#                                              EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT : (0.7490196078431373, 0.5058823529411764, 0.17647058823529413, 1.0),
#                                              EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR : (0.32941176470588235, 0.18823529411764706, 0.0196078431372549, 1.0)}


# LLMS_COLORS = {'llama3-8b':(0.984313725490196, 0.7058823529411765, 0.6823529411764706, 1.0), 
#                'llama-3.1-8b':(0.7019607843137254, 0.803921568627451, 0.8901960784313725, 1.0), 
#                'gemma2-9b':(0.8, 0.9215686274509803, 0.7725490196078432, 1.0), 
#                'mixtral-8x7b':(0.8705882352941177, 0.796078431372549, 0.8941176470588236, 1.0), 
#                'llama3-70b':(0.996078431372549, 0.8509803921568627, 0.6509803921568628, 1.0), 
#                'llama-3.1-70b':(1.0, 1.0, 0.8, 1.0)}

LLMS_COLORS = {'llama3-8b':'tab:blue', 
               'llama-3.1-8b':'tab:orange', 
               'gemma2-9b':'tab:green', 
               'mixtral-8x7b':'tab:red', 
               'llama3-70b':'tab:purple', 
               'llama-3.1-70b':'tab:brown'}


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
FACTUALITY_AUTHOR_METADATA_TO_HIDE = ['two_year_mean_citedness','h_index','i10_index','works_count','cited_by_count','ethnicity_dx','ethnicity_ec','ethnicity','gender']
FACTUALITY_FIELD_TO_HIDE = ['fact_author_score', 'career_age', 'years', 'id_author_aps_list', 'year_first_publication','year_last_publication','academic_age','seniority_active', 'seniority_now', 'age_now']
FACTUALITY_EPOCH_TO_HIDE = ['fact_author_score', 'career_age', 'doi', 'id_author_aps_list', 'academic_age', 'age_now', 'seniority_active', 'seniority_now']
FACTUALITY_SENIORITY_TO_HIDE = ['fact_author_score', 'doi', 'years', 'id_author_aps_list']
FACTUALITY_TASKS = [EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY]

FACTUALITY_NONE = 'None'

FACTUALITY_AUTHOR_EXISTS = 'Exists'
FACTUALITY_AUTHOR_BOTH = 'APS & OA'
FACTUALITY_AUTHOR_APS = 'APS'
FACTUALITY_AUTHOR_OA = 'OA'
FACTUALITY_AUTHOR_FACT_CHECKS = [FACTUALITY_AUTHOR_BOTH,FACTUALITY_AUTHOR_OA,FACTUALITY_AUTHOR_APS,FACTUALITY_NONE]

# FACTUALITY_FIELD_CORRECT = 'Correct'
# FACTUALITY_FIELD_INCORRECT = 'Incorrect'
# FACTUALITY_FIELD_PARTIAL = 'Partial'
FACTUALITY_FIELD_DOI = 'DOI'
FACTUALITY_FIELD_DOI_AUTHOR = 'DOI &\nAuthor'
FACTUALITY_FIELD_AUTHOR_FIELD = 'Author &\nField'
FACTUALITY_FIELD_AT_LEAST_ONE = 'At least one'
FACTUALITY_FIELD_ALL = 'All'
FACTUALITY_FIELD_FACT_CHECKS = [FACTUALITY_NONE,FACTUALITY_FIELD_AT_LEAST_ONE,FACTUALITY_FIELD_DOI,FACTUALITY_FIELD_DOI_AUTHOR,FACTUALITY_FIELD_AUTHOR_FIELD,FACTUALITY_FIELD_ALL]

FACTUALITY_SENIORITY_ACTIVE = 'Seniority\n(while active)'
FACTUALITY_SENIORITY_NOW = 'Seniority\n(now)'
FACTUALITY_SENIORITY_ACTIVE_REQ = 'Seniority\nas requested\n(while active)'
FACTUALITY_SENIORITY_NOW_REQ = 'Seniority\nas requested\n(now)'
FACTUALITY_SENIORITY_FACT_CHECKS = [FACTUALITY_SENIORITY_ACTIVE,FACTUALITY_SENIORITY_NOW,FACTUALITY_SENIORITY_ACTIVE_REQ,FACTUALITY_SENIORITY_NOW_REQ,FACTUALITY_NONE]


FACTUALITY_EPOCH_AS_REQUESTED = 'Epoch\nas requested'
FACTUALITY_EPOCH_AS_LLM_IN_GT = 'Valid epoch\n(within)'
FACTUALITY_EPOCH_AS_GT_IN_LLM = 'Overestimated'
FACTUALITY_EPOCH_AS_OVERLAP = 'Overlap'
FACTUALITY_EPOCH_FACT_CHECKS = [FACTUALITY_EPOCH_AS_REQUESTED,FACTUALITY_EPOCH_AS_LLM_IN_GT,FACTUALITY_EPOCH_AS_GT_IN_LLM,FACTUALITY_EPOCH_AS_OVERLAP,FACTUALITY_NONE]