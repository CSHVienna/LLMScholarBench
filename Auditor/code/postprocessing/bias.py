
import itertools

from libs import constants
from libs import io


def get_baselined_from_gt(df_gt, cat_col, cat_order):
    df = df_gt.copy()
    df.fillna({cat_col: constants.UNKNOWN_STR}, inplace=True)
    baselines_with_unknownn = df.groupby(cat_col,observed=False).size()
    baselines_with_unknownn = baselines_with_unknownn.reset_index(name='counts')
    baselines_with_unknownn.loc[:,'percentage'] = baselines_with_unknownn.counts.apply(lambda x: x/baselines_with_unknownn.counts.sum())
    baselines_with_unknownn[cat_col] = io.pd.Categorical(baselines_with_unknownn[cat_col], categories=cat_order, ordered=True)
    baselines_with_unknownn.set_index(cat_col, inplace=True)

    df_clean = df.dropna(subset=[cat_col]).query(f"{cat_col}!=@constants.UNKNOWN_STR")
    baselines = df_clean.groupby(cat_col,observed=False).size()
    baselines = baselines.reset_index(name='counts')
    baselines.loc[:,'percentage'] = baselines.counts.apply(lambda x: x/baselines.counts.sum())
    baselines[cat_col] = io.pd.Categorical(baselines[cat_col], categories=cat_order, ordered=True)
    baselines.set_index(cat_col, inplace=True)

    return baselines, baselines_with_unknownn


def get_mean_percentages(df_authors, cat_cols=['gender'], cat_orders=None):
    main_groups = ['model','task_name']
    main_cols = main_groups + ['task_param','date','time']
    dup_cols = ['clean_name']
    cat_cols = [cat_cols] if not isinstance(cat_cols, list) else cat_cols

    # remove duplicates
    df_authors_oa = df_authors.drop_duplicates(subset=main_cols + dup_cols).copy()

    # remove authors not found in OA
    df_authors_oa = df_authors_oa.query("@io.pd.notnull(id_author_oa)").copy()

    if cat_orders is not None:
        for cat_col, cat_order in cat_orders.items():
            df_authors_oa[cat_col] = io.pd.Categorical(df_authors_oa[cat_col], categories=cat_order, ordered=True)

    # percentage per request
    df_authors_oa = df_authors_oa.groupby(main_cols + cat_cols, observed=False).size().reset_index(name='counts').sort_values(by='counts', ascending=False) 
    grouped = df_authors_oa.groupby(main_cols + cat_cols, observed=False).sum()
    grouped['percentage'] = grouped['counts'] / grouped.groupby(level=main_cols, observed=False)['counts'].transform('sum')
    grouped = grouped.reset_index()

    # mean percentage by model and task
    grouped_model = grouped.groupby(['model'] + cat_cols, observed=False).percentage.agg(['mean','std']).reset_index()
    grouped_model_tasks = grouped.groupby(['model','task_name'] + cat_cols, observed=False).percentage.agg(['mean','std']).reset_index()
    return grouped_model, grouped_model_tasks


def get_nobel_prize_stats(df_real_authors, df_nobel_prize_winners):
    df_nobel_prize_winners_clean = df_nobel_prize_winners.copy()
    df_nobel_prize_winners_clean.dropna(subset=['openalex_id'], inplace=True)
    df_nobel_prize_winners_clean.loc[:,'openalex_id'] = df_nobel_prize_winners_clean.loc[:,'openalex_id'].apply(lambda x: int(x.replace('A','')))
    df_nobel_prize_winners_clean.rename(columns={'category':'nobel_category', 'year':'nobel_year'}, inplace=True)

    df_authors_nobel = df_real_authors.merge(df_nobel_prize_winners_clean[['openalex_id','nobel_category','nobel_year']], left_on='id_author_oa', right_on='openalex_id', how='left')
    df_authors_nobel = df_authors_nobel[['model','task_name','task_param','date','time', 'id_author_oa', 'clean_name', 'nobel_category', 'nobel_year']]
    df_authors_nobel.loc[:,'is_nobel'] = df_authors_nobel.loc[:,'nobel_category'].notnull()
    df_authors_nobel.fillna({'nobel_category':constants.NO_LAUREATE, 'nobel_year':0}, inplace=True)
    df_authors_nobel['nobel_decade'] = (df_authors_nobel['nobel_year'] // 10) * 10
    df_authors_nobel['nobel_decade'] = df_authors_nobel['nobel_decade'].apply(lambda x: str(int(x)))
    df_authors_nobel.fillna({'nobel_decade':0}, inplace=True)
    return df_authors_nobel


def get_counts_and_percentage(df_data, group_col):
    df = io.pd.DataFrame()
    for attribute in group_col:
        _stats = df_data.groupby(attribute).size().reset_index(name='count')
        _stats['attribute'] = attribute
        _stats.rename(columns={attribute: 'label'}, inplace=True)
        _stats['percentage'] = _stats['count'] / _stats['count'].sum()
        df = io.pd.concat([df, _stats], ignore_index=True)
    return df


def get_demographic_counts_per_task_param(df_llm_real_authors, attribute):
    def assign_ax(row):
        if constants.EXPERIMENT_TASK_TWINS in row.task_name:
            return constants.TASK_TWINS_GENDER_ORDER.index(row.task_param.split("_")[1])
        return constants.TASK_PARAMS_BY_TASK[row.task_name].index(row.task_param)

    all_models = constants.LLMS
    all_task_params = constants.TASK_PARAMS_BY_TASK
    all_labels = constants.DEMOGRAPHIC_ATTRIBUTE_LABELS_ORDER[attribute]

    data = df_llm_real_authors.drop_duplicates(subset=['model', 'task_name', 'task_param', 'id_author_oa'], keep='first')

    # Create a full cartesian product of all possible combinations
    combinations = []
    for task_name, task_params in all_task_params.items():
        for combo in itertools.product(all_models, [task_name], task_params, all_labels):
            combinations.append(combo)

    # Convert the cartesian product to a DataFrame
    full_index = io.pd.DataFrame(combinations, columns=["model", "task_name", "task_param", attribute])

    # split statistical twinns
    full_index.loc[:,"task_name"] = full_index.apply(lambda row: f"{row['task_name']}_{row['task_param'].split('_')[0]}" if row["task_name"] == constants.EXPERIMENT_TASK_TWINS else row["task_name"], axis=1)
    data.loc[:,"task_name"] = data.apply(lambda row: f"{row['task_name']}_{row['task_param'].split('_')[0]}" if row["task_name"] == constants.EXPERIMENT_TASK_TWINS else row["task_name"], axis=1)

    # Group by and count records in the original data
    grouped = data.groupby(["model", "task_name", "task_param", attribute]).size().reset_index(name="counts")

    # Merge with the full cartesian product to ensure all combinations are included
    result = full_index.merge(grouped, on=["model", "task_name", "task_param", attribute], how="left").fillna(0)

    # Calculate percentages
    result["percentage"] = result.groupby(["model", "task_name", "task_param"])["counts"].transform(lambda x: x / x.sum())

    # Fill missing percentages with 0 (if sum was 0 for the group)
    result["percentage"] = result["percentage"].fillna(0)
    result['ax'] = result.apply(assign_ax, axis=1)
    result.rename(columns={attribute: 'attribute_label'}, inplace=True)

    return result, all_labels


def get_data_by_attribute_and_metric(df, attribute, metric):
    #attribute = gender, ethnicity
    #metric = counts, fractions
    cols_2 = [('label','')] + [(c,v) for c, v in df.columns if c.startswith(attribute) and c.endswith(metric)]
    cols_1 = ['label'] + [v for c, v in df.columns if c.startswith(attribute) and c.endswith(metric)]
    result = df.reset_index()
    result = result[cols_2]
    result.columns = cols_1
    result.set_index('label', inplace=True)
    cols = constants.GENDER_LIST if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER else constants.ETHNICITY_LIST
    result = result[cols]
    return result