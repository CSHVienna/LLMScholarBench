from libs import io
from libs.metrics import aggregators

METRICS = ['validity_pct', 'refusal_pct', 
           'duplicates', 'consistency', 'factuality_author', 
           'connectedness',
           'diversity_gender', 'diversity_ethnicity', 'diversity_prominence_pub', 'diversity_prominence_cit', 
           'parity_gender', 'parity_ethnicity', 'parity_prominence_pub', 'parity_prominence_cit',
           ]


def get_plot_fn(metric, path, prefix=None):
    pre = '' if prefix is None else f'{prefix}_'
    return io.path_join(path, f'{pre}{metric}.pdf')
    
def get_per_attempt_table_fn(metric, path, prefix=None):
    pre = '' if prefix is None else f'{prefix}_'
    return io.path_join(path, f'{pre}per_attempt_{metric}.csv')

def load_per_attempt(metric, df, path, prefix=None, **kwargs):
    if metric not in METRICS:
        raise ValueError(f'Metric {metric} not supported')

    fn = get_per_attempt_table_fn(metric, path, prefix)
    if io.exists(fn):
        return io.read_csv(fn, index_col=0)

    gt = kwargs.get('gt', None)
    df_coauthorships_in_recommendations = kwargs.get('df_coauthorships_in_recommendations', None)
    

    if metric == 'validity_pct':
        per_attempt = aggregators.aggregate_validity(df)
    elif metric == 'refusal_pct':
        per_attempt = aggregators.aggregate_refusal(df)
    elif metric == 'factuality_author':
        per_attempt = aggregators.aggregate_factuality_author(df)
    elif metric == 'duplicates':
        per_attempt = aggregators.aggregate_duplicates(df)
    elif metric == 'consistency':
        per_attempt = aggregators.aggregate_consistency(df)
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
    elif metric == 'connectedness':
        per_attempt = aggregators.aggregate_connectedness(df, df_coauthorships_in_recommendations=df_coauthorships_in_recommendations)
    else:
        raise ValueError(f'Metric {metric} not supported')

    io.save_csv(per_attempt, fn)
    return per_attempt
