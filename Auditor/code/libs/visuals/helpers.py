"""
helpers.py

Shared LOGIC for the intervention/trajectory figures. No figure drawing happens
here; the plotting lives in `interventions.py` (the per-metric dot-plot) and
`trajectories.py` (the technical-vs-social scatter).

Contents
--------
- Experiment data prep ...... filter_experiment_df, baseline_first
- Aggregated builders ....... aggregate_models_for_experiment,
                              build_model_trajectory, build_model_trajectory_ci
                              -> collapse metrics into one `technical` and one
                                 `social` coordinate (feeds the scatter plot)
- Per-metric builders ....... build_metric_trajectory, build_metric_trajectory_ci
                              -> keep every individual metric as its own column
                                 (feeds the dot-plot)
- Baseline deltas ........... BaselineDeltaComputer
- Label / geometry / colour . build_axis_label, resolve_axis_labels,
                              padded_limits, edge_aware_offset, make_shades

The aggregated and per-metric builders share the same primitives
(`filter_experiment_df`, `baseline_first`) rather than duplicating them.
"""

from __future__ import annotations

import colorsys
from collections import OrderedDict

import matplotlib.colors as mcolors
import pandas as pd


# =========================================================================== #
# Axis-label construction
# =========================================================================== #
DEFAULT_XLABEL = "Technical"
DEFAULT_YLABEL = "Social"


def build_axis_label(title, cols, *, display=None, sum_symbol=r"\sum", sep=" + "):
    """Build an axis label from a list of column names.

    Grouping rule (prefix = text before the first ``_``):
      - prefixes shared by >= 2 columns collapse to ``$\\sum$ <prefix>`` (placed last)
      - columns with a unique prefix are shown by name (placed first)
    e.g. ['validity', 'duplicates', 'parity_author', 'parity_field'] ->
        "Title: validity + duplicates + $\\sum$ parity".

    `display` maps a raw name/prefix to a prettier string (to re-attach footnote
    markers), and `sum_symbol` can carry an index, e.g. r"\\sum_{a}".
    """
    display = display or {}

    def disp(name):
        return display.get(name, name)

    groups = OrderedDict()
    for c in cols:
        groups.setdefault(c.split("_", 1)[0], []).append(c)

    singles, sums = [], []
    for prefix, members in groups.items():
        if len(members) >= 2:
            sums.append(f"${sum_symbol}$ {disp(prefix)}")
        else:
            singles.append(disp(members[0]))

    body = sep.join(singles + sums)
    return f"{title.title()}: {body}" if title else body


def resolve_axis_labels(xlabel, ylabel, technical_cols, social_cols,
                        axis_titles, label_display):
    """Pick explicit labels, else build them from the column lists, else defaults."""
    x_title, y_title = axis_titles
    if xlabel is None:
        xlabel = (build_axis_label(x_title, technical_cols, display=label_display)
                  if technical_cols is not None else DEFAULT_XLABEL)
    if ylabel is None:
        ylabel = (build_axis_label(y_title, social_cols, display=label_display)
                  if social_cols is not None else DEFAULT_YLABEL)
    return xlabel, ylabel


# =========================================================================== #
# Experiment data prep (shared by every builder)
# =========================================================================== #
def filter_experiment_df(df_per_attempt_all_mod, query, *, query_env=None):
    """Apply the shared per-experiment filtering: run the experiment query, keep
    top_100 / biased_top_k, and strip the '-grounded' suffix from model names."""
    df = df_per_attempt_all_mod.query(query, local_dict=query_env or {}).copy()
    # df = df.query("task_param == 'top_100' or task_name == 'biased_top_k'")
    df.loc[:, "model"] = df.model.apply(lambda x: x.replace("-grounded", ""))
    return df


def baseline_first(traj, baseline_experiment):
    """Reorder a trajectory df so `baseline_experiment` is the first row."""
    if traj.empty:
        return traj
    ordered = [baseline_experiment] + [
        e for e in traj.experiment if e != baseline_experiment
    ]
    rank = {e: i for i, e in enumerate(ordered)}
    return (
        traj.assign(_order=traj.experiment.map(rank))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


# =========================================================================== #
# Aggregated builders: one (technical, social) point per experiment
# =========================================================================== #
def aggregate_models_for_experiment(
    df_per_attempt_all_mod: pd.DataFrame,
    query: str,
    *,
    aggregate_fn,
    groupby,
    alpha_ci,
    metric_value_col,
    metric_name_col,
    cols_order,
    technical_cols,
    social_cols,
    query_env=None,
) -> pd.DataFrame:
    """Run the per-experiment pipeline and return one row per model with summed
    `technical` and `social` columns (plus family / model_size metadata)."""
    df = filter_experiment_df(df_per_attempt_all_mod, query, query_env=query_env)

    df_models = aggregate_fn(
        df, groupby, alpha=alpha_ci,
        metric_value_col=metric_value_col, metric_name_col=metric_name_col,
    )
    df_models = df_models[cols_order]

    df_models.loc[:, "metric_kind_plot"] = df_models.metric_name.apply(
        lambda x: "technical" if x in technical_cols
        else "social" if x in social_cols
        else "other"
    )
    df_models = (
        df_models.groupby(["model", "metric_kind_plot"])["mean"].sum()
        .reset_index()
        .pivot(index="model", columns="metric_kind_plot", values="mean")
        .reset_index()
    )
    df_models.loc[:, "family"] = df_models.model.apply(lambda x: x.split("-")[0])
    df_models = df_models.merge(
        df[["model", "model_size"]].drop_duplicates(), on="model", how="left"
    )
    return df_models


def build_model_trajectory(
    df_per_attempt_all_mod: pd.DataFrame,
    model: str,
    experiment_query_map: dict,
    *,
    baseline_experiment: str = "baseline",
    **agg_kwargs,
) -> pd.DataFrame:
    """One row per experiment (baseline first) with the aggregated technical/social
    coordinates for a single `model`. Columns: ['experiment','technical','social',
    'is_baseline']."""
    rows = []
    for experiment, query in experiment_query_map.items():
        df_models = aggregate_models_for_experiment(
            df_per_attempt_all_mod, query, **agg_kwargs
        )
        match = df_models[df_models.model == model]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append({
            "experiment": experiment,
            "technical": r.get("technical", float("nan")),
            "social": r.get("social", float("nan")),
            "is_baseline": experiment == baseline_experiment,
        })
    return baseline_first(pd.DataFrame(rows), baseline_experiment)


def build_model_trajectory_ci(
    df_per_attempt_all_mod: pd.DataFrame,
    model: str,
    experiment_query_map: dict,
    *,
    aggregate_fn,
    alpha_ci,
    metric_value_col,
    metric_name_col,
    technical_high_cols,
    social_cols,
    sample_id_cols,
    axis_groupby=None,
    mean_col="mean",
    ci_lo_col="ci_low",
    ci_hi_col="ci_high",
    baseline_experiment="baseline",
    query_env=None,
) -> pd.DataFrame:
    """Like `build_model_trajectory`, but with a CI per axis computed the correct
    way for a *sum of metrics*: sum the technical (and social) metrics per attempt,
    then feed those per-attempt sums to `aggregate_fn` so the CI uses the same
    estimator and captures correlation between metrics.

    Columns: ['experiment','technical','technical_lo','technical_hi',
              'social','social_lo','social_hi'].
    """
    axis_groupby = list(axis_groupby) if axis_groupby is not None else ["model", metric_name_col]

    def _summed_axis_long(df, axis_name, metric_cols):
        sub = df[df[metric_name_col].isin(metric_cols)]
        summed = sub.groupby(sample_id_cols, as_index=False)[metric_value_col].sum()
        summed[metric_name_col] = axis_name
        return summed

    rows = []
    for experiment, query in experiment_query_map.items():
        df = filter_experiment_df(df_per_attempt_all_mod, query, query_env=query_env)
        axis_long = pd.concat(
            [_summed_axis_long(df, "technical", technical_high_cols),
             _summed_axis_long(df, "social", social_cols)],
            ignore_index=True,
        )
        agg = aggregate_fn(
            axis_long, axis_groupby, alpha=alpha_ci,
            metric_value_col=metric_value_col, metric_name_col=metric_name_col,
        )
        agg = agg[agg.model == model]
        if agg.empty:
            continue

        def _pick(axis):
            r = agg[agg[metric_name_col] == axis]
            if r.empty:
                return (float("nan"),) * 3
            r = r.iloc[0]
            return r[mean_col], r[ci_lo_col], r[ci_hi_col]

        t_mean, t_lo, t_hi = _pick("technical")
        s_mean, s_lo, s_hi = _pick("social")
        rows.append({
            "experiment": experiment,
            "technical": t_mean, "technical_lo": t_lo, "technical_hi": t_hi,
            "social": s_mean, "social_lo": s_lo, "social_hi": s_hi,
        })

    return baseline_first(pd.DataFrame(rows), baseline_experiment)


# =========================================================================== #
# Per-metric builders: one column per individual metric (no aggregation)
# =========================================================================== #
def _metric_selection(metrics, technical_cols, social_cols):
    """Which metrics to keep and in what order: explicit `metrics`, else
    technical_cols + social_cols (technical first), else None (keep all)."""
    if metrics is not None:
        return list(metrics)
    selection = list(technical_cols or []) + list(social_cols or [])
    return selection or None


def _per_metric_estimates(
    df_per_attempt_all_mod, model, query, *,
    aggregate_fn, metric_value_col, metric_name_col,
    metrics=None, alpha_ci=None, groupby=None,
    mean_col="mean", ci_lo_col="ci_low", ci_hi_col="ci_high",
    with_ci=False, query_env=None,
):
    """One experiment -> long per-metric estimates for one model.

    Returns ['metric','value'] (+ ['value_lo','value_hi'] when `with_ci`), or
    None if the model is absent. Groups by [model, metric_name] so every metric
    stays on its own row (no technical/social sum-and-pivot)."""
    df = filter_experiment_df(df_per_attempt_all_mod, query, query_env=query_env)
    gb = list(groupby) if groupby is not None else ["model", metric_name_col]

    agg = aggregate_fn(
        df, gb, alpha=alpha_ci,
        metric_value_col=metric_value_col, metric_name_col=metric_name_col,
    )
    agg = agg[agg["model"] == model]
    
    if metrics is not None:
        agg = agg[agg[metric_name_col].isin(metrics)]
    if agg.empty:
        return None

    out = pd.DataFrame({
        "metric": agg[metric_name_col].to_numpy(),
        "value": agg[mean_col].to_numpy(),
    })
    if with_ci:
        out["value_lo"] = agg[ci_lo_col].to_numpy()
        out["value_hi"] = agg[ci_hi_col].to_numpy()
    return out


def _to_wide(long: pd.DataFrame, order, with_ci, baseline_experiment) -> pd.DataFrame:
    """Pivot tidy long estimates to one row per experiment, one column per metric
    (with adjacent `m_lo`/`m_hi` when `with_ci`); baseline row moved first."""
    def pivot(value_col):
        wide = long.pivot_table(index="experiment", columns="metric",
                                values=value_col, sort=False)
        wide.columns.name = None
        return wide

    mean_wide = pivot("value")
    blocks = [mean_wide]
    if with_ci:
        blocks += [pivot("value_lo").add_suffix("_lo"),
                   pivot("value_hi").add_suffix("_hi")]
    wide = pd.concat(blocks, axis=1).reset_index()

    present = list(mean_wide.columns)
    metric_order = [m for m in (order or present) if m in present]
    cols = ["experiment"]
    for m in metric_order:
        cols.append(m)
        if with_ci:
            cols += [f"{m}_lo", f"{m}_hi"]
    wide = wide[cols]
    wide["is_baseline"] = wide["experiment"] == baseline_experiment

    for group in ['diversity', 'parity', 'technical_high']:
        if group == 'technical_high':
            group_cols = [c for c in ['validity_pct', 'compliant_pct', 'uniqueness'] if c in wide.columns]
        else:
            group_cols = [c for c in wide.columns if c.startswith(group)]
        if group_cols:
            wide[group] = wide[group_cols].sum(axis=1)

    return baseline_first(wide, baseline_experiment)


def _build_metric_trajectory(
    df_per_attempt_all_mod, model, experiment_query_map, *,
    aggregate_fn, metric_value_col, metric_name_col,
    technical_cols=None, social_cols=None, metrics=None,
    alpha_ci=None, groupby=None,
    mean_col="mean", ci_lo_col="ci_low", ci_hi_col="ci_high",
    with_ci=False, baseline_experiment="baseline", query_env=None,
) -> pd.DataFrame:
    """Shared core for the per-metric builders (the only difference is `with_ci`)."""
    order = _metric_selection(metrics, technical_cols, social_cols)

    frames = []
    for experiment, query in experiment_query_map.items():
        est = _per_metric_estimates(
            df_per_attempt_all_mod, model, query,
            aggregate_fn=aggregate_fn, metric_value_col=metric_value_col,
            metric_name_col=metric_name_col, metrics=order, alpha_ci=alpha_ci,
            groupby=groupby, mean_col=mean_col, ci_lo_col=ci_lo_col,
            ci_hi_col=ci_hi_col, with_ci=with_ci, query_env=query_env,
        )
        if est is None:
            continue
        frames.append(est.assign(experiment=experiment))

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    return _to_wide(long, order, with_ci, baseline_experiment)


def build_metric_trajectory(*args, **kwargs) -> pd.DataFrame:
    """Per-metric trajectory, point estimates only.
    Wide: ['experiment', <metric...>, 'is_baseline'] (baseline first)."""
    return _build_metric_trajectory(*args, with_ci=False, **kwargs)


def build_metric_trajectory_ci(*args, **kwargs) -> pd.DataFrame:
    """Per-metric trajectory with a CI per metric.
    Wide: ['experiment', <m>, <m>_lo, <m>_hi, ..., 'is_baseline']."""
    return _build_metric_trajectory(*args, with_ci=True, **kwargs)


# =========================================================================== #
# Baseline deltas (for the dot-plot)
# =========================================================================== #
class BaselineDeltaComputer:
    """Compute each intervention's change relative to the baseline row.

    Accepts any config object exposing `experiment_col`, `baseline_col`,
    `metrics` (each with a `.column`), `label_map`, and `intervention_order`.
    """

    def __init__(self, config) -> None:
        self._cfg = config

    def _baseline_row(self, df):
        cfg = self._cfg
        if cfg.baseline_col in df.columns and df[cfg.baseline_col].astype(bool).any():
            return df.loc[df[cfg.baseline_col].astype(bool)].iloc[0]
        if cfg.experiment_col in df.columns:
            hit = df[cfg.experiment_col].astype(str).str.contains("baseline", case=False, na=False)
            if hit.any():
                return df.loc[hit].iloc[0]
        raise ValueError(
            f"No baseline row: set a truthy '{cfg.baseline_col}' column or name "
            f"an experiment containing 'baseline'."
        )

    def _order(self, present):
        desired = self._cfg.intervention_order
        if not desired:
            return list(present)
        present_set = set(present)
        ordered = [x for x in desired if x in present_set]
        return ordered + [x for x in present if x not in set(ordered)]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return tidy long deltas ['intervention','metric','delta'] (baseline excluded)."""
        cfg = self._cfg
        baseline = self._baseline_row(df)
        records = []
        for _, row in df.iterrows():
            if row[cfg.experiment_col] == baseline[cfg.experiment_col]:
                continue
            label = cfg.label_map.get(row[cfg.experiment_col], row[cfg.experiment_col])
            for m in cfg.metrics:
                records.append({
                    "intervention": label, "metric": m.column,
                    "delta": float(row[m.column]) - float(baseline[m.column]),
                })

        tidy = pd.DataFrame.from_records(records, columns=["intervention", "metric", "delta"])
        ordered = self._order(list(dict.fromkeys(tidy["intervention"])))
        tidy["intervention"] = pd.Categorical(tidy["intervention"], categories=ordered, ordered=True)
        return tidy.sort_values(["metric", "intervention"]).reset_index(drop=True)

    def compute_all(self, trajectories: dict) -> dict:
        return {model: self.compute(df) for model, df in trajectories.items()}


# =========================================================================== #
# Plotting geometry / colour utilities (pure functions, no drawing)
# =========================================================================== #
def padded_limits(vals, frac):
    """(lo, hi) expanded by `frac` of the data span on each side."""
    lo, hi = float(min(vals)), float(max(vals))
    span = (hi - lo) or (abs(hi) or 1.0)
    return lo - frac * span, hi + frac * span


def edge_aware_offset(x_norm, y_norm, dist):
    """Label offset (in points) + alignment that pushes text toward the interior,
    from the point's normalised [0,1] position."""
    if x_norm > 0.70:
        dx, ha = -dist, "right"
    elif x_norm < 0.30:
        dx, ha = dist, "left"
    else:
        dx, ha = 0.0, "center"
    if y_norm > 0.70:
        dy, va = -dist, "top"
    else:
        dy, va = dist, "bottom"
    return dx, dy, ha, va


def make_shades(base_color, n, light_range=(0.40, 0.72)):
    """`n` shades of `base_color` (light -> dark), e.g. for related-but-distinct
    colours within a model family."""
    r, g, b = mcolors.to_rgb(base_color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    if n == 1:
        lights = [sum(light_range) / 2]
    else:
        lo, hi = light_range
        lights = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [mcolors.to_hex(colorsys.hls_to_rgb(h, l, s)) for l in lights]