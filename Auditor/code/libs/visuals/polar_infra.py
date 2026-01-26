import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ----------------------------
# Expected inputs
#   df_metrics: columns = [metric_group, metric_name, model, task_name, task_param, mean, std]
#   df_meta:    columns = [model, access, size, model_class]  # names can be adapted below
# ----------------------------

# ---------- 1) Aggregate across all task_name/task_param per model x metric ----------
def aggregate_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    # If your df already has one row per model x metric, this still works.
    agg = (
        df_metrics
        .groupby(["model", "metric_group", "metric_name"], as_index=False)
        .agg(
            mean=("mean", "mean"),
            std=("std", "mean"),   # uncertainty across tasks/params
        )
    )
    # If a model-metric has only one observation, std becomes NaN; set to 0
    agg["std"] = agg["std"].fillna(0.0)
    return agg

# ---------- 2) Plot one "soccer-style" polar bar panel ----------
def plot_polar_panel(ax, df_one_model: pd.DataFrame, title: str, meta_text: str = "",
                     group_colors: dict | None = None, order: list[str] | None = None):
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)

    # Determine metric order around the circle
    if order is None:
        # stable order: metric_group then metric_name
        df_one_model = df_one_model.sort_values(["metric_group", "metric_name"])
        order = df_one_model["metric_group"].tolist()
    else:
        # keep only metrics present, in the requested order
        present = set(df_one_model["metric_group"])
        order = [m for m in order if m in present]

    # Align data to the order
    d = (
        df_one_model.set_index("metric_group")
        .loc[order, ["metric_name", "mean", "std"]]
        .reset_index()
    )

    n = len(d)
    if n == 0:
        ax.set_axis_off()
        return

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n

    if group_colors is None:
        group_colors = {
            "validity": "#4C72B0",
            "factuality": "#55A868",
            "consistency": "#C44E52",
            "duplicates": "#8172B3",
            "diversity": "#CCB974",
            "parity": "#64B5CD",
            "refusal": "#8C8C8C",
        }

    means = d["mean"].to_numpy()
    stds = d["std"].to_numpy()
    groups = d["metric_group"].astype(str).to_numpy()

    colors = [group_colors.get(g, "#999999") for g in groups]

    # polar bars
    ax.bar(theta, means, width=width * 0.95, bottom=0.0,
           color=colors, edgecolor="white", linewidth=1.0, align="edge")

    # radial errorbars (clipped to [0, 1])
    lo = np.clip(means - stds, 0, 1)
    hi = np.clip(means + stds, 0, 1)
    # place errors at bar centers
    centers = theta + width * 0.475
    ax.vlines(centers, lo, hi, color="black", linewidth=1.0, alpha=0.8)

    # labels around the circle
    ax.set_xticks(centers)
    ax.set_xticklabels(d["metric_name"].tolist(), fontsize=9)
    ax.tick_params(axis="x", pad=10)

    # radial scale and grid
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # title + metadata (second line)
    if meta_text:
        ax.set_title(f"{title}\n{meta_text}", fontsize=12, pad=18, y=1.05)
    else:
        ax.set_title(title, fontsize=12, pad=18, y=1.05)

    # center "hole" for the same visual impression
    center = plt.Circle((0, 0), 0.18, transform=ax.transData._b, color="white", zorder=10)
    ax.add_artist(center)


# ---------- 3) Build a grid: one panel per model ----------
def plot_model_grid(df_metrics: pd.DataFrame,
                    df_meta: pd.DataFrame | None = None,
                    meta_cols: dict | None = None,
                    ncols: int = 4,
                    order: list[str] | None = None,
                    group_colors: dict | None = None,
                    figsize_per_cell=(4.0, 4.0)):
    """
    meta_cols: mapping of canonical keys -> your df_meta column names, e.g.
      {"model": "model", "access": "access", "size": "size", "class": "model_class"}
    """
    agg = aggregate_metrics(df_metrics)

    models = sorted(agg["model"].unique().tolist())
    n = len(models)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.35, hspace=0.55)

    # metadata lookup
    meta_lookup = {}
    if df_meta is not None:
        if meta_cols is None:
            meta_cols = {"model": "model_name", "access": "model_access", "size": "model_size", "class": "model_class"}
        m = df_meta.copy()

        # keep only needed columns (ignore if missing)
        keep = [c for c in meta_cols.values() if c in m.columns]
        m = m[keep].drop_duplicates()

        # index by model
        if meta_cols["model"] in m.columns:
            m = m.set_index(meta_cols["model"])
            for model_name in m.index.unique():
                row = m.loc[model_name]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                meta_lookup[model_name] = row.to_dict()

    for i, model_name in enumerate(models):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c], projection="polar")

        df_one = agg[agg["model"] == model_name].copy()

        # metadata text under title (simple and robust)
        meta_text = ""
        if model_name in meta_lookup and meta_cols is not None:
            md = meta_lookup[model_name]
            access = md.get(meta_cols.get("access", ""), "")
            size = md.get(meta_cols.get("size", ""), "")
            mclass = md.get(meta_cols.get("class", ""), "")

            parts = []
            if isinstance(access, str) and access:
                # optional icon: open vs proprietary
                # s = f"Access: {access}"
                s = access.title()[0]
                parts.append(s)
            if isinstance(size, str) and size:
                # s = f"Size: {size}"
                parts.append(size)
            if isinstance(mclass, str) and mclass:
                # s = f"Class: {mclass}"
                parts.append(mclass)

            meta_text = " | ".join(parts)

        plot_polar_panel(
            ax=ax,
            df_one_model=df_one,
            title=str(model_name),
            meta_text=meta_text,
            group_colors=group_colors,
            order=order,
        )

    # create proxy handles
    handles = [
        mpatches.Patch(color=color, label=group)
        for group, color in group_colors.items()
    ]

    # add a single legend to the entire figure
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )
    
    # turn off unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    return fig
