from libs import io
from libs.metrics import aggregators
from libs import helpers
from libs import constants


def get_plot_fn(metric, path, prefix=None):
    pre = '' if prefix is None else f'{prefix}_'
    return io.path_join(path, f'{pre}{metric}.pdf')
    
# def get_per_attempt_table_fn(metric, path, prefix=None):
#     pre = '' if prefix is None else f'{prefix}_'
#     return io.path_join(path, f'{pre}per_attempt_{metric}.csv')

def get_per_attempt_fn(model, metric, path, prefix=None):
    pre_1 = '' if prefix is None else f'{prefix}_'
    pre_2 = '' if model is None else f'{model}_'
    pre_3 = '' if metric is None else f'{metric}'
    fn = f'{pre_1}per_attempt_{pre_2}{pre_3}.csv'.strip()
    return io.path_join(path, fn)

def load_per_attempt(metric, df, fn, save=False, **kwargs):

    if metric not in constants.BENCHMARK_METRICS:
        raise ValueError(f'Metric {metric} not supported')

    overwrite = kwargs.pop('overwrite', False)
    if io.exists(fn) and not overwrite:
        return io.read_csv(fn, index_col=0)

    gt = kwargs.get('gt', None)
    df_similarity = kwargs.get('df_similarity', None)
    metric_similarity = kwargs.get('metric_similarity', None)
    
    if metric == 'validity_pct':
        per_attempt = aggregators.aggregate_validity(df)
    elif metric == 'refusal_pct':
        per_attempt = aggregators.aggregate_refusal(df)
    elif metric == 'duplicates':
        per_attempt = aggregators.aggregate_duplicates(df)
    elif metric == 'consistency':
        per_attempt = aggregators.aggregate_consistency(df)

    elif metric == 'factuality_author':
        per_attempt = aggregators.aggregate_factuality_author(df)

    elif metric == 'connectedness_density':
        per_attempt = aggregators.aggregate_similarity(df, df_similarity=df_similarity, metric_similarity=metric_similarity)
    elif metric == 'connectedness_entropy':
        per_attempt = aggregators.aggregate_similarity(df, df_similarity=df_similarity, metric_similarity=metric_similarity)
    elif metric == 'connectedness_components':
        per_attempt = aggregators.aggregate_similarity(df, df_similarity=df_similarity, metric_similarity=metric_similarity)
    
    elif metric == 'similarity_pca':
        per_attempt = aggregators.aggregate_similarity(df, df_similarity=df_similarity, metric_similarity=metric_similarity)

    elif metric == 'diversity_gender':
        per_attempt = aggregators.aggregate_diversity_gender(df)
    elif metric == 'diversity_ethnicity':
        per_attempt = aggregators.aggregate_diversity_ethnicity(df)
    elif metric == 'diversity_prominence_pub':
        per_attempt = aggregators.aggregate_diversity_prominence_pub(df)
    elif metric == 'diversity_prominence_cit':
        per_attempt = aggregators.aggregate_diversity_prominence_cit(df)

    elif metric == 'parity_gender':
        per_attempt = aggregators.aggregate_parity_gender(df, gt=gt)
    elif metric == 'parity_ethnicity':
        per_attempt = aggregators.aggregate_parity_ethnicity(df, gt=gt)
    elif metric == 'parity_prominence_pub':
        per_attempt = aggregators.aggregate_parity_prominence_pub(df, gt=gt)
    elif metric == 'parity_prominence_cit':
        per_attempt = aggregators.aggregate_parity_prominence_cit(df, gt=gt)
    else:
        raise ValueError(f'Metric {metric} not supported')

    per_attempt.rename(columns={'metric':'metric_value'}, inplace=True)
    per_attempt.loc[:, 'metric_name'] = metric
    cols_order = per_attempt.columns.tolist()[:-2] + ['metric_name', 'metric_value']
    per_attempt = per_attempt[cols_order]

    if save:
        verbose = kwargs.get('verbose', True)
        io.save_csv(per_attempt, fn, verbose=verbose)
        
    return per_attempt


def load_infrastructure_data(query, tables_path, prefix='infrastructure', include_in_group_by=None):
    posfix = '' if include_in_group_by is None else f"_{'_'.join(include_in_group_by)}"
    fn_attempt = io.path_join(tables_path, f'{prefix}_per_attempt{posfix}.csv')
    fn_group = io.path_join(tables_path, f'{prefix}_per_group{posfix}.csv')

    if not io.exists(fn_group) or not io.exists(fn_attempt):

        df_summary_infrastructure_attempt = io.pd.DataFrame()
        df_summary_infrastructure_group = io.pd.DataFrame()

        for metric in constants.BENCHMARK_METRICS:
            # per attempt
            prefix_pa = prefix if prefix == 'temperature' else None
            per_attempt = load_per_attempt(metric, None, tables_path, prefix_pa).query(query)
            per_attempt.loc[:, 'metric_name'] = metric
            per_attempt.rename(columns={'metric':'metric_value'}, inplace=True)
            
            per_attempt = helpers.add_infrastructure_columns(per_attempt)
            cols_order = ['model_access','model_size','model_class','model_provider','model','grounded','temperature','date','time','task_name','task_param','task_attempt','metric_name','metric_value']
            for c in cols_order:
                if c not in per_attempt.columns:
                    per_attempt[c] = None
            per_attempt = per_attempt[cols_order]
            df_summary_infrastructure_attempt = io.pd.concat([df_summary_infrastructure_attempt, per_attempt], axis=0, ignore_index=True)
            
            
            for group in constants.BENCHMARK_MODEL_GROUPS:
            
                _group = [group] + include_in_group_by if include_in_group_by is not None and type(include_in_group_by) == list and len(include_in_group_by) > 0 else group

                is_bernoulli = metric in ['validity_pct', 'refusal_pct']
                per_group = aggregators.aggregate_per_group(per_attempt, _group, alpha=0.05, is_bernoulli=is_bernoulli, metric_col_name='metric_value')
                per_group.rename(columns={group: 'model_kind'}, inplace=True)
                per_group.loc[:, 'metric_name'] = metric
                per_group.loc[:, 'model_group'] = group
                cols_order = ['model_group','model_kind']
                
                if include_in_group_by is not None and type(include_in_group_by) == list and len(include_in_group_by) > 0:
                    cols_order.extend(include_in_group_by)
                
                cols_order.extend(['n','metric_name','mean','std','median','sum','ci','ci_low','ci_high'])
                per_group = per_group[cols_order]
                df_summary_infrastructure_group = io.pd.concat([df_summary_infrastructure_group, per_group], axis=0, ignore_index=True)

        df_summary_infrastructure_attempt.reset_index(drop=True, inplace=True)
        io.save_csv(df_summary_infrastructure_attempt, fn_attempt)

        df_summary_infrastructure_group.reset_index(drop=True, inplace=True)
        io.save_csv(df_summary_infrastructure_group, fn_group)

    else:
        df_summary_infrastructure_attempt = io.read_csv(fn_attempt, index_col=0)
        df_summary_infrastructure_group = io.read_csv(fn_group, index_col=0)

    return df_summary_infrastructure_attempt, df_summary_infrastructure_group