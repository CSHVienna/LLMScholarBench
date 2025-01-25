from libs import constants
from libs import io

def get_factuality_authors_either_OA_APS(df_authors):
    df_fact_authors = df_authors.drop_duplicates(subset=['model','task_name','task_param','date','time','task_attempt', 'clean_name']).copy()       # we remove duplicated answers in the same request
    df_fact_authors = df_authors[['model','task_name','task_param','date','time','task_attempt','id_author_oa','id_author_aps_list']].copy()        # keeping necesary columns
    df_fact_authors.loc[:,constants.FACTUALITY_AUTHOR_OA] = df_fact_authors['id_author_oa'].notnull()                                               # adding column with boolean value if author is in oa
    df_fact_authors.loc[:,constants.FACTUALITY_AUTHOR_APS] = df_fact_authors['id_author_aps_list'].notnull()                                        # adding column with boolean value if author is in aps
    df_fact_authors.loc[:,'factuality_author'] = df_fact_authors.apply(lambda x: constants.FACTUALITY_AUTHOR_EXISTS if x[constants.FACTUALITY_AUTHOR_OA] or x[constants.FACTUALITY_AUTHOR_APS] else constants.FACTUALITY_NONE, axis=1)
    df_fact_authors.drop(columns=['id_author_oa','id_author_aps_list','task_attempt', constants.FACTUALITY_AUTHOR_OA, constants.FACTUALITY_AUTHOR_APS], inplace=True)                                               # removing unnecessary columns
    df_fact_authors = df_fact_authors.groupby(['model','task_name', 'task_param','date','time', 'factuality_author'], observed=False).size().reset_index(name='counts').sort_values(by='counts', ascending=False)    # grouping per instance how many authhors are real
    grouped = df_fact_authors.groupby(['model','task_name','task_param','date','time','factuality_author'], observed=False).sum()                                                                                    # summing to compute percentage between real and doesn't exist authors
    grouped['percentage'] = grouped['counts'] / grouped.groupby(level=['model','task_name','task_param','date','time'], observed=False)['counts'].transform('sum')                                                    # computing percentage 
    df_fact_authors_either = grouped.query("factuality_author == @constants.FACTUALITY_AUTHOR_EXISTS").groupby(['model','task_name'], observed=False)['percentage'].agg(['mean','std']).reset_index()                        # getting mean and std of percentage of real authors per task and model
    return df_fact_authors_either


def get_factuality_author_for_each_case(df_authors):
    df_fact_authors = df_authors.drop_duplicates(subset=['model','task_name','task_param','date','time','task_attempt', 'clean_name']).copy()       # we remove duplicated answers in the same request
    df_fact_authors = df_authors[['model','task_name','task_param','date','time','task_attempt','id_author_oa','id_author_aps_list']].copy()        # keeping necesary columns
    df_fact_authors.loc[:,constants.FACTUALITY_AUTHOR_OA] = df_fact_authors['id_author_oa'].notnull()                                               # adding column with boolean value if author is in oa
    df_fact_authors.loc[:,constants.FACTUALITY_AUTHOR_APS] = df_fact_authors['id_author_aps_list'].notnull()                                        # adding column with boolean value if author is in aps
    df_fact_authors.loc[:,'factuality_author'] = df_fact_authors.apply(lambda x: constants.FACTUALITY_AUTHOR_BOTH if x[constants.FACTUALITY_AUTHOR_OA] and x[constants.FACTUALITY_AUTHOR_APS] 
                                                                    else constants.FACTUALITY_AUTHOR_OA if x[constants.FACTUALITY_AUTHOR_OA] 
                                                                    else constants.FACTUALITY_AUTHOR_APS if x[constants.FACTUALITY_AUTHOR_APS]
                                                                    else constants.FACTUALITY_NONE, axis=1)
    df_fact_authors.drop(columns=['id_author_oa','id_author_aps_list','task_attempt', constants.FACTUALITY_AUTHOR_OA, constants.FACTUALITY_AUTHOR_APS], inplace=True)                                               # removing unnecessary columns
    df_fact_authors = df_fact_authors.groupby(['model','task_name', 'task_param','date','time', 'factuality_author'], observed=False).size().reset_index(name='counts').sort_values(by='counts', ascending=False)    # grouping per instance how many authhors are real
    grouped = df_fact_authors.groupby(['model','task_name','task_param','date','time','factuality_author'], observed=False).sum()                                                                                    # summing to compute percentage between real and doesn't exist authors
    grouped['percentage'] = grouped['counts'] / grouped.groupby(level=['model','task_name','task_param','date','time'], observed=False)['counts'].transform('sum')                                                    # computing percentage 

    # for each
    df_fact_authors_both = grouped.query("factuality_author == @constants.FACTUALITY_AUTHOR_BOTH").groupby(['model','task_name'], observed=False)['percentage'].agg(['mean','std']).reset_index()                        # getting mean and std of percentage of real authors per task and model
    df_fact_authors_oa = grouped.query("factuality_author == @constants.FACTUALITY_AUTHOR_OA").groupby(['model','task_name'], observed=False)['percentage'].agg(['mean','std']).reset_index()                        # getting mean and std of percentage of real authors per task and model
    df_fact_authors_none = grouped.query("factuality_author == @constants.FACTUALITY_NONE").groupby(['model','task_name'], observed=False)['percentage'].agg(['mean','std']).reset_index()                        # getting mean and std of percentage of real authors per task and model
    return df_fact_authors_both, df_fact_authors_oa, df_fact_authors_none


def get_factuality_field(df_field):
    main_cols = ['model','task_name','task_param','date','time']
    final_fact_col = 'factuality_field'
    final_group = ['model', 'task_name', final_fact_col]
    uniq_cols = ['clean_name','doi']
    fact_cols_map = {'fact_doi_score':constants.FACTUALITY_FIELD_DOI, 'fact_doi_author':constants.FACTUALITY_FIELD_DOI_AUTHOR, 'fact_author_field':constants.FACTUALITY_FIELD_AUTHOR_FIELD}
    fact_cols = list(fact_cols_map.keys())

    # new cols
    new_cols_map = {'none':constants.FACTUALITY_NONE, 'at_least_one':constants.FACTUALITY_FIELD_AT_LEAST_ONE, 'all':constants.FACTUALITY_FIELD_ALL}
    all_cols = fact_cols + list(new_cols_map.keys())
    all_col_map = {**fact_cols_map, **new_cols_map}

    # only for field task
    df_fact_field = df_field.copy()

    # remove duplicated responses
    df_fact_field.drop_duplicates(subset=main_cols + uniq_cols, inplace=True)

    # keep only necessary columns (and format them as integers: 1 if factual and 0 if not)
    # Do this when values include nulls
    df_fact_field.loc[:,'fact_doi_score'] = df_fact_field.fact_doi_score.apply(lambda x: io.pd.notnull(x) and x>0).astype(int)
    df_fact_field.loc[:,'fact_doi_author'] = df_fact_field['fact_doi_author'].notnull()
    df_fact_field.loc[:,'fact_author_field'] = df_fact_field.fact_author_field.apply(lambda x: io.pd.notnull(x) or x > 0)

    # new metrics
    df_fact_field.loc[:,'none'] = df_fact_field.apply(lambda row: sum([row[c] for c in fact_cols])==0, axis=1)
    df_fact_field.loc[:,'at_least_one'] = df_fact_field.apply(lambda row: sum([row[c] for c in fact_cols])>0, axis=1)
    df_fact_field.loc[:,'all'] = df_fact_field.apply(lambda row: sum([row[c] for c in fact_cols])==3, axis=1)

    # get output size per response
    grouped = df_fact_field.groupby(main_cols, observed=False).size().reset_index(name='counts')

    # get counts per factual metric
    df_fact_field = df_fact_field.groupby(main_cols, observed=False)[all_cols].sum().reset_index()

    # normalize counts by total responses (percentage)
    df_fact_field[all_cols] = df_fact_field[all_cols].div(grouped['counts'], axis=0)

    # melt columns to have a single column with the metric and its value
    df_fact_field = df_fact_field.melt(id_vars=main_cols, var_name=final_fact_col, value_name='percentage')
    df_fact_field  = df_fact_field.groupby(final_group, observed=False)['percentage'].agg(['mean','std']).reset_index()
    df_fact_field.drop(columns=['task_name'], inplace=True)
    df_fact_field[final_fact_col] = df_fact_field[final_fact_col].replace(all_col_map)

    return df_fact_field.sort_values('factuality_field')


def get_factuality_seniority(df_seniority):
    main_cols = ['model','task_name','task_param','date','time']
    final_fact_col = 'factuality_field'
    final_group = ['model', 'task_name', final_fact_col]
    uniq_cols = ['clean_name','career_age']
    fact_cols_map = {'fact_seniority_active':constants.FACTUALITY_SENIORITY_ACTIVE, 'fact_seniority_now':constants.FACTUALITY_SENIORITY_NOW, 'fact_seniority_active_requested':constants.FACTUALITY_SENIORITY_ACTIVE_REQ, 'fact_seniority_now_requested':constants.FACTUALITY_SENIORITY_NOW_REQ}
    fact_cols = list(fact_cols_map.keys())

    # new cols
    new_cols_map = {'none':constants.FACTUALITY_NONE}
    all_cols = fact_cols + list(new_cols_map.keys())
    all_col_map = {**fact_cols_map, **new_cols_map}

    # only for field task
    df_fact_seniority = df_seniority.copy()

    # remove duplicated responses
    df_fact_seniority.drop_duplicates(subset=main_cols + uniq_cols, inplace=True)

    df_fact_seniority.loc[:,'fact_seniority_active'] = df_fact_seniority.fact_seniority_active.apply(lambda x: io.pd.notnull(x) and x>0).astype(int)
    df_fact_seniority.loc[:,'fact_seniority_now'] = df_fact_seniority.fact_seniority_now.apply(lambda x: io.pd.notnull(x) and x>0).astype(int)
    df_fact_seniority.loc[:,'fact_seniority_active_requested'] = df_fact_seniority.fact_seniority_active_requested.apply(lambda x: io.pd.notnull(x) and x>0).astype(int)
    df_fact_seniority.loc[:,'fact_seniority_now_requested'] = df_fact_seniority.fact_seniority_now_requested.apply(lambda x: io.pd.notnull(x) and x>0).astype(int)
    df_fact_seniority.loc[:,'none'] = df_fact_seniority.apply(lambda row: sum([row[c] for c in fact_cols])==0, axis=1)

    # get output size per response
    grouped = df_fact_seniority.groupby(main_cols, observed=False).size().reset_index(name='counts')

    # get counts per factual metric
    df_fact_seniority = df_fact_seniority.groupby(main_cols, observed=False)[all_cols].sum().reset_index()

    # # normalize counts by total responses (percentage)
    df_fact_seniority[all_cols] = df_fact_seniority[all_cols].div(grouped['counts'], axis=0)

    # melt columns to have a single column with the metric and its value
    df_fact_seniority = df_fact_seniority.melt(id_vars=main_cols, var_name=final_fact_col, value_name='percentage')
    df_fact_seniority  = df_fact_seniority.groupby(final_group, observed=False)['percentage'].agg(['mean','std']).reset_index()
    df_fact_seniority.drop(columns=['task_name'], inplace=True)
    df_fact_seniority[final_fact_col] = df_fact_seniority[final_fact_col].replace(all_col_map)

    return df_fact_seniority.sort_values('factuality_field')



def get_factuality_epoch(df_epoch):
    main_cols = ['model','task_name','task_param','date','time']
    final_fact_col = 'factuality_field'
    final_group = ['model', 'task_name', final_fact_col]
    uniq_cols = ['clean_name','years']
    fact_cols_map = {'fact_epoch_requested':constants.FACTUALITY_EPOCH_AS_REQUESTED, 'fact_epoch_llm_in_gt':constants.FACTUALITY_EPOCH_AS_LLM_IN_GT, 
                    'fact_epoch_gt_in_llm':constants.FACTUALITY_EPOCH_AS_GT_IN_LLM, 'fact_epoch_overlap':constants.FACTUALITY_EPOCH_AS_OVERLAP}
    fact_cols = list(fact_cols_map.keys())

    # new cols
    new_cols_map = {'none':constants.FACTUALITY_NONE}
    all_cols = fact_cols + list(new_cols_map.keys())
    all_col_map = {**fact_cols_map, **new_cols_map}

    # only for field task
    df_fact_epoch = df_epoch.copy()

    # remove duplicated responses
    df_fact_epoch.drop_duplicates(subset=main_cols + uniq_cols, inplace=True)

    df_fact_epoch.loc[:,'fact_epoch_requested'] = df_fact_epoch.fact_epoch_requested.astype(int)
    df_fact_epoch.loc[:,'fact_epoch_llm_in_gt'] = df_fact_epoch.fact_epoch_llm_in_gt.astype(int)
    df_fact_epoch.loc[:,'fact_epoch_gt_in_llm'] = df_fact_epoch.fact_epoch_gt_in_llm.astype(int)
    df_fact_epoch.loc[:,'fact_epoch_overlap'] = df_fact_epoch.fact_epoch_overlap.astype(int)
    df_fact_epoch.loc[:,'none'] = df_fact_epoch.apply(lambda row: sum([row[c] for c in fact_cols])==0, axis=1)

    # get output size per response
    grouped = df_fact_epoch.groupby(main_cols, observed=False).size().reset_index(name='counts')

    # get counts per factual metric
    df_fact_epoch = df_fact_epoch.groupby(main_cols, observed=False)[all_cols].sum().reset_index()

    # # normalize counts by total responses (percentage)
    df_fact_epoch[all_cols] = df_fact_epoch[all_cols].div(grouped['counts'], axis=0)

    # melt columns to have a single column with the metric and its value
    df_fact_epoch = df_fact_epoch.melt(id_vars=main_cols, var_name=final_fact_col, value_name='percentage')
    df_fact_epoch  = df_fact_epoch.groupby(final_group, observed=False)['percentage'].agg(['mean','std']).reset_index()
    df_fact_epoch.drop(columns=['task_name'], inplace=True)
    df_fact_epoch[final_fact_col] = df_fact_epoch[final_fact_col].replace(all_col_map)

    return df_fact_epoch.sort_values('factuality_field')