import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint
from tqdm.auto import tqdm
tqdm.pandas()

from libs import constants
from libs.network import fragmentation

def aggregate_per_attempt(df, group_cols, metric_agg):

    def pct(x):
        return x["n_counts"] / x["n_total"]
    
    def pct_inverse(x):
        return 1 - x["n_counts"] / x["n_total"]
    
    if type(metric_agg) == tuple:
        per_attempt = (
            df
            .groupby(group_cols, as_index=False)
            .agg(
                metric = metric_agg
            )
            .reset_index(drop=True)
        )
    else:
        k = [k for k in metric_agg.keys() if k != 'ntotal']

        if len(k) == 1:
            k = k[0]
            per_attempt = (
                df
                .groupby(group_cols, as_index=False)
                .agg(
                    n_total = metric_agg['ntotal'],
                    n_counts = metric_agg[k]
                )
                .assign(metric=lambda x: pct(x) if k in ['npositive'] else pct_inverse(x))
                .reset_index(drop=True)
            )

    return per_attempt

def aggregate_per_group(per_attempt, main_col_group, alpha=0.05, is_bernoulli=False, metric_col_name='metric'):

    per_group = (
        per_attempt
        .groupby(main_col_group)
        .agg(
            n = (metric_col_name, "count"),
            mean = (metric_col_name, "mean"),
            std = (metric_col_name, "std"),
            median = (metric_col_name, "median"),
            sum = (metric_col_name, "sum"),
            )
    )

    # calculate confidence interval (means margin of error, t-distribution for non-bernoulli distributions)
    per_group["ci"] = (
        t.ppf(1 - alpha/2, per_group["n"] - 1) * per_group["std"] / np.sqrt(per_group["n"])
    )

    if is_bernoulli:
        # For bernoulli distributions, use the Wilson score interval (i.e., refusal and validity)
        per_group["ci_low"], per_group["ci_high"] = proportion_confint(count=per_group["sum"],
                                                                        nobs=per_group["n"],
                                                                        method="wilson")
    else:
        per_group["ci_low"] = per_group["mean"] - per_group["ci"]
        per_group["ci_high"] = per_group["mean"] + per_group["ci"]

    per_group.reset_index(inplace=True)

    return per_group


# @DEPRECATED: 
# def _aggregate(df, group_cols, main_col_group, metric_agg, alpha=0.05, is_bernoulli=False):
#     per_attempt = aggregate_per_attempt(df, group_cols, metric_agg)
#     per_group = aggregate_per_group(per_attempt, main_col_group, alpha, is_bernoulli)
#     return per_attempt, per_group

def aggregate_factuality_author(df_factuality_author):
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean.author_exists = df_factuality_author_clean.author_exists.astype(int)
    group_cols = ['model','grounded','temperature','date','time','task_name','task_param','task_attempt'] 
    metric_agg  = ("author_exists", "mean")
    per_attempt = aggregate_per_attempt(df_factuality_author_clean, group_cols, metric_agg)
    return per_attempt

def aggregate_validity(df_summary):
    group_cols = ["model",'grounded','temperature', "date", "time", "task_name", "task_param"]
    metric_agg  = ("valid_attempt", "any")
    per_attempt = aggregate_per_attempt(df_summary, group_cols, metric_agg)
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt

def aggregate_duplicates(df_factuality_author):
    group_cols = ['model','grounded', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
    metric_agg  = {'ntotal':("clean_name", "size"), 'nunique':("clean_name", "nunique")}
    per_attempt = aggregate_per_attempt(df_factuality_author, group_cols, metric_agg)
    return per_attempt

def aggregate_diversity(df_factuality_author, attribute=None):
    attrs = ['gender', 'ethnicity', 'prominence_pub', 'prominence_cit']
    if attribute not in attrs:
        raise ValueError(f"attribute must be one of {attrs}")

    def normalized_entropy(s):
        p = s.value_counts(normalize=True)
        h = -(p * np.log(p)).sum()
        return h / np.log(len(p)) if len(p) > 1 else 0.0

    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(id_author_oa) and @pd.notnull(id_author_oa) and @pd.notna(@attribute) and @pd.notnull(@attribute)")

    group_cols = ['model', 'grounded','temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
    
    per_attempt = (
        df_factuality_author_clean
        .groupby(group_cols)[attribute]
        .apply(normalized_entropy)
        .reset_index(name="metric")
    )

    return per_attempt

def aggregate_diversity_gender(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='gender')

def aggregate_diversity_ethnicity(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='ethnicity')

def aggregate_diversity_prominence_pub(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='prominence_pub')

def aggregate_diversity_prominence_cit(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='prominence_cit')

def aggregate_refusal(df_summary):
    group_cols = ['model','grounded', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
    metric_agg  = {'ntotal':("is_refusal", "size"), 'npositive':("is_refusal", lambda s: (s == constants.REFUSAL_TRUE).sum())}
    per_attempt = aggregate_per_attempt(df_summary, group_cols, metric_agg)
    return per_attempt

def aggregate_consistency(df_factuality_author):
    def jaccard(a: set, b: set) -> float:
        u = a | b
        return (len(a & b) / len(u)) if len(u) else 1.0

    def set_transition_stability(name_sets: pd.Series) -> float:
        sets = list(name_sets)
        if len(sets) <= 1:
            return np.nan
        vals = [jaccard(sets[i], sets[i-1]) for i in range(1, len(sets))]
        return float(np.mean(vals))
        
    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(clean_name) and @pd.notnull(clean_name) and clean_name != ''")

    # sort by date and time
    group_cols = ["model", 'grounded','temperature', "task_name", "task_param", "date", "time"]
    df_factuality_author_clean = df_factuality_author_clean.sort_values(group_cols)

    # name_set per timestamp within each group
    per_time = (
        df_factuality_author_clean.groupby(group_cols, sort=True)["name"]
        .agg(lambda s: set(s.dropna().astype(str)))
        .reset_index(name="name_set")
    )

    # final
    group_cols = ["model",'grounded', 'temperature', "task_name", "task_param"]
    metric_agg  = ("name_set",set_transition_stability)
    per_attempt = aggregate_per_attempt(per_time, group_cols, metric_agg)
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt


def aggregate_parity(df_factuality_author, attribute=None, **kwargs):
    
    attrs = ['gender', 'ethnicity', 'prominence_pub', 'prominence_cit']
    if attribute not in attrs:
        raise ValueError(f"attribute must be one of {attrs}")

    df_gt = kwargs.get('gt', None)
    if df_gt is None:
        raise ValueError("gt must be provided")

    def parity(series):
        # compares the fraction of each group from the results against the fraction of each group from the ground truth
        # this is 1 - TV (the total variation distance)
        p = series.value_counts(normalize=True)
        p_gt = df_gt[attribute].value_counts(normalize=True)
        return 1 - (1/2 * sum(abs(p - p_gt)))
    
    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()

    group_cols = ['model','grounded', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
    metric_agg  = (attribute, parity)
    per_attempt = aggregate_per_attempt(df_factuality_author_clean, group_cols, metric_agg)

    return per_attempt

def aggregate_parity_gender(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='gender', **kwargs)

def aggregate_parity_ethnicity(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='ethnicity', **kwargs)

def aggregate_parity_prominence_pub(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='prominence_pub', **kwargs)

def aggregate_parity_prominence_cit(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='prominence_cit', **kwargs) 



def aggregate_connectedness(df_factuality_author, **kwargs):
    df_coauthorships_in_recommendations = kwargs.get('df_coauthorships_in_recommendations', None)
    if df_coauthorships_in_recommendations is None:
        raise ValueError("df_coauthorships_in_recommendations must be provided")

    group_cols = ['model','grounded','temperature','date','time','task_name','task_param','task_attempt']
    results = []

    g = df_factuality_author.groupby(group_cols)

    for group, df in tqdm(g, total=g.ngroups, desc="Processing groups"):
    
            rec_ids = df.id_author_oa.dropna().unique()

            connectedness = fragmentation.norm_entropy_R_from_edgelist(rec_ids,
                                                                    df_coauthorships_in_recommendations,
                                                                    src_col = "src",
                                                                    dst_col = "dst")

            obj = {c: group[i] for i, c in enumerate(group_cols)}
            obj |= {'nrecs': connectedness.n,
                    'n_components': connectedness.n_components,
                    'metric': connectedness.norm_entropy,
                    'n_edges_rows': connectedness.n_edges_rows,
                    'n_edges_undirected_unique': connectedness.n_edges_undirected_unique
                    }
            
            results.append(obj)

    return pd.DataFrame(results)
    

def aggregate_scholarly_similarity(df_factuality_author, **kwargs):
    # similarity:
    # 1. log-transfor count variables (work_counts, cited_by_counts, h_index), eg. x = log(x + 1)
    # 2. standardize the variables, eg. z = (x - mean) / std (mean and std are computed on the entire dataset)
    # 3. compute the PCA of the standardized variables (2 components) - retain components explaining 80%-90% variance
    # 4. compute the cosine similarity between the PCA components (yi, yj: person i and j) where yi and yj are the PCA components of person i and j
    # 4. compute the average cosine similarity
    return None

