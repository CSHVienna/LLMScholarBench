import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint

from libs import constants

def aggregate_per_group(per_attempt, main_col_group, alpha=0.05, is_bernoulli=False):

    per_group = (
        per_attempt
        .groupby(main_col_group)
        .agg(
            n = ('metric', "count"),
            mean = ('metric', "mean"),
            std = ('metric', "std"),
            median = ('metric', "median"),
            sum = ('metric', "sum"),
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


def _aggregate(df, group_cols, main_col_group, metric_agg, alpha=0.05, is_bernoulli=False):
    
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
            .reset_index()
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
                .reset_index()
            )
        
    per_group = aggregate_per_group(per_attempt, main_col_group, alpha, is_bernoulli)

    return per_attempt, per_group

def aggregate_factuality_author(df_factuality_author, main_col_group='model_access', alpha=0.05):
    if main_col_group not in df_factuality_author.columns:
        raise ValueError(f"main_col_group must be a column in df_factuality_author")

    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=[main_col_group, 'model','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean.author_exists = df_factuality_author_clean.author_exists.astype(int)

    group_cols = [main_col_group,'model','temperature','date','time','task_name','task_param','task_attempt'] 

    per_attempt, per_group = _aggregate(df_factuality_author_clean, 
                                        group_cols, 
                                        main_col_group,
                                        ("author_exists", "mean"),
                                        alpha=alpha,
                                        )

    return per_attempt, per_group


def aggregate_validity(df_summary, main_col_group='model_access', alpha=0.05):
    if main_col_group not in df_summary.columns:
        raise ValueError(f"main_col_group must be a column in df_summary")

    group_cols = [main_col_group, "model",'temperature', "date", "time", "task_name", "task_param"]

    per_attempt, per_group = _aggregate(df_summary, 
                                        group_cols, 
                                        main_col_group,
                                        ("valid_attempt", "any"),
                                        alpha=alpha,
                                        is_bernoulli=True,
                                        )
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt, per_group


def aggregate_duplicates(df_factuality_author, main_col_group='model_access', alpha=0.05):
    if main_col_group not in df_factuality_author.columns:
        raise ValueError(f"main_col_group must be a column in df_factuality_author")

    group_cols = [main_col_group, 'model', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']

    per_attempt, per_group = _aggregate(df_factuality_author, 
                                        group_cols, 
                                        main_col_group,
                                        {'ntotal':("clean_name", "size"), 'nunique':("clean_name", "nunique")},
                                        alpha=alpha,
                                        )
    
    return per_attempt, per_group



    
def aggregate_diversity(df_factuality_author, main_col_group='model_access', attribute=None, alpha=0.05):
    
    attrs = ['gender', 'ethnicity', 'prominence_pub', 'prominence_cit']
    if attribute not in attrs:
        raise ValueError(f"attribute must be one of {attrs}")

    def shannon_entropy(s):
        p = s.value_counts(normalize=True)
        return -(p * np.log(p)).sum()
    
    def normalized_entropy(s):
        p = s.value_counts(normalize=True)
        h = -(p * np.log(p)).sum()
        return h / np.log(len(p)) if len(p) > 1 else 0.0

    if main_col_group not in df_factuality_author.columns:
        raise ValueError(f"main_col_group must be a column in df_factuality_author")

    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=[main_col_group, 'model','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(id_author_oa) and @pd.notnull(id_author_oa) and @pd.notna(@attribute) and @pd.notnull(@attribute)")

    group_cols = [main_col_group, 'model', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']

    per_attempt = (
        df_factuality_author_clean
        .groupby(group_cols)[attribute]
        .apply(normalized_entropy)
        .reset_index(name="metric")
    )
    
    per_group = aggregate_per_group(per_attempt, main_col_group, alpha)

    return per_attempt, per_group

def aggregate_diversity_gender(df_factuality_author, main_col_group='model_access', alpha=0.05):
    return aggregate_diversity(df_factuality_author, main_col_group, attribute='gender', alpha=alpha)

def aggregate_diversity_ethnicity(df_factuality_author, main_col_group='model_access', alpha=0.05):
    return aggregate_diversity(df_factuality_author, main_col_group, attribute='ethnicity', alpha=alpha)

def aggregate_diversity_prominence_pub(df_factuality_author, main_col_group='model_access', alpha=0.05):
    return aggregate_diversity(df_factuality_author, main_col_group, attribute='prominence_pub', alpha=alpha)

def aggregate_diversity_prominence_cit(df_factuality_author, main_col_group='model_access', alpha=0.05):
    return aggregate_diversity(df_factuality_author, main_col_group, attribute='prominence_cit', alpha=alpha)


def aggregate_refusal(df_summary, main_col_group='model_access', alpha=0.05):
    if main_col_group not in df_summary.columns:
        raise ValueError(f"main_col_group must be a column in df_summary")

    group_cols = [main_col_group, 'model', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']

    per_attempt, per_group = _aggregate(df_summary, 
                                        group_cols, 
                                        main_col_group,
                                        {'ntotal':("is_refusal", "size"), 'npositive':("is_refusal", lambda s: (s == constants.REFUSAL_TRUE).sum())},
                                        alpha=alpha,
                                        is_bernoulli=True,
                                        )
    
    return per_attempt, per_group


def aggregate_consistency(df_factuality_author, main_col_group='model_access', alpha=0.05):
    def jaccard(a: set, b: set) -> float:
        u = a | b
        return (len(a & b) / len(u)) if len(u) else 1.0

    def set_transition_stability(name_sets: pd.Series) -> float:
        sets = list(name_sets)
        if len(sets) <= 1:
            return np.nan
        vals = [jaccard(sets[i], sets[i-1]) for i in range(1, len(sets))]
        return float(np.mean(vals))
        
    if main_col_group not in df_factuality_author.columns:
        raise ValueError(f"main_col_group must be a column in df_factuality_author")

    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=[main_col_group, 'model','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(clean_name) and @pd.notnull(clean_name) and clean_name != ''")

    # sort by date and time
    group_cols = [main_col_group, "model", 'temperature', "task_name", "task_param", "date", "time"]
    df_factuality_author_clean = df_factuality_author_clean.sort_values(group_cols)

    # name_Set per timestamp within each group
    per_time = (
        df_factuality_author_clean.groupby(group_cols, sort=True)["name"]
        .agg(lambda s: set(s.dropna().astype(str)))
        .reset_index(name="name_set")
    )

    # final
    group_cols = [main_col_group, "model", 'temperature', "task_name", "task_param"]
    per_attempt, per_group = _aggregate(per_time, 
                                        group_cols, 
                                        main_col_group,
                                        ("name_set",set_transition_stability),
                                        alpha=alpha,
                                        )
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt, per_group


def aggregate_parity(df_factuality_author, main_col_group='model_access', attribute=None, alpha=0.05, **kwargs):
    
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
    
    if main_col_group not in df_factuality_author.columns:
        raise ValueError(f"main_col_group must be a column in df_factuality_author")

    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=[main_col_group, 'model','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    # df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(@attribute) and @pd.notnull(@attribute)")

    group_cols = [main_col_group, 'model', 'temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']

    per_attempt, per_group = _aggregate(df_factuality_author_clean, 
                                        group_cols, 
                                        main_col_group,
                                        (attribute, parity),
                                        alpha=alpha,
                                        )

    return per_attempt, per_group

def aggregate_parity_gender(df_factuality_author, main_col_group='model_access', alpha=0.05, **kwargs):
    return aggregate_parity(df_factuality_author, main_col_group, attribute='gender', alpha=alpha, **kwargs)

def aggregate_parity_ethnicity(df_factuality_author, main_col_group='model_access', alpha=0.05, **kwargs):
    return aggregate_parity(df_factuality_author, main_col_group, attribute='ethnicity', alpha=alpha, **kwargs)

def aggregate_parity_prominence_pub(df_factuality_author, main_col_group='model_access', alpha=0.05, **kwargs):
    return aggregate_parity(df_factuality_author, main_col_group, attribute='prominence_pub', alpha=alpha, **kwargs)

def aggregate_parity_prominence_cit(df_factuality_author, main_col_group='model_access', alpha=0.05, **kwargs):
    return aggregate_parity(df_factuality_author, main_col_group, attribute='prominence_cit', alpha=alpha, **kwargs) 