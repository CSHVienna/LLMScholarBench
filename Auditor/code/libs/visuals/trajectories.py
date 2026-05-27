"""
trajectories.py

VISUALIZATION: each model's path through (technical, social) space across the
`experiment` interventions. Drawing only - the data builders, label/colour/limit
utilities, and aggregation live in `helpers.py`.

Use `helpers.build_model_trajectory` (or `_ci`) to assemble the per-model
trajectory frames, then `plot_model_trajectory` (single) or
`plot_models_trajectories` (overlay) here to draw them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from libs.visuals.helpers import (
    edge_aware_offset,
    make_shades,
    padded_limits,
    resolve_axis_labels,
)


# --------------------------------------------------------------------------- #
# Single model
# --------------------------------------------------------------------------- #
def plot_model_trajectory(
    traj,
    model,
    *,
    color=None,
    label_map=None,
    connect="sequential",                 # 'sequential' | 'from_baseline'
    figsize=(10, 7),
    pad_frac=0.14,
    label_offset_pts=12.0,
    use_adjust_text=False,
    technical_cols=None,
    social_cols=None,
    axis_titles=("Technical", "Social"),
    label_display=None,
    xlabel=None,
    ylabel=None,
    ax=None,
):
    """Plot one model's path through (technical, social) space.

    Baseline is a star (the start); each point is annotated with its experiment
    name (renamed via `label_map`) and placed edge-aware so labels avoid the
    axes/frame. `use_adjust_text=True` delegates de-overlap to adjustText if
    installed. `connect='sequential'` links baseline -> ... -> last;
    `connect='from_baseline'` draws an arrow from baseline to each point.
    """
    if traj.empty:
        raise ValueError(f"No trajectory data for model={model!r}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = traj.technical.to_numpy()
    y = traj.social.to_numpy()

    segments = _segments(connect, len(traj))

    for i, j in segments:
        ax.annotate(
            "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5,
                            shrinkA=10, shrinkB=10, alpha=0.85),
            zorder=2,
        )
        
    ax.scatter(x[1:], y[1:], s=90, color=color, edgecolor="white", zorder=3)
    ax.scatter(x[:1], y[:1], s=200, color=color, marker="*",
               edgecolor="black", linewidth=1.2, zorder=4, label="baseline (start)")

    xlim = padded_limits(x, pad_frac)
    ylim = padded_limits(y, pad_frac)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    adjust_text = _maybe_adjust_text(use_adjust_text)
    bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7)
    texts = []
    for xi, yi, exp in zip(x, y, traj.experiment):
        label = label_map.get(exp, exp) if label_map else exp
        if adjust_text is not None:
            texts.append(ax.text(xi, yi, label, fontsize=9, zorder=5, bbox=bbox))
        else:
            x_norm = (xi - xlim[0]) / (xlim[1] - xlim[0])
            y_norm = (yi - ylim[0]) / (ylim[1] - ylim[0])
            dx, dy, ha, va = edge_aware_offset(x_norm, y_norm, label_offset_pts)
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=9, zorder=5, bbox=bbox)

    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.3, 1.3),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    xlabel, ylabel = resolve_axis_labels(
        xlabel, ylabel, technical_cols, social_cols, axis_titles, label_display
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    return fig, ax


# --------------------------------------------------------------------------- #
# Multiple models overlaid
# --------------------------------------------------------------------------- #
def plot_models_trajectories(
    trajectories,
    *,
    model_colors=None,
    intervention_markers=None,
    label_map=None,
    shade_range=(0.62, 0.34),
    label_mode="none",                    # 'none' | 'per_intervention' | 'per_point'
    connect="sequential",
    figsize=(11, 7),
    pad_frac=0.16,
    marker_size=110.0,
    baseline_marker_size=300.0,
    show_ci=True,
    technical_cols=None,
    social_cols=None,
    axis_titles=("Technical", "Social"),
    label_display=None,
    xlabel=None,
    ylabel=None,
    label_offset_pts=12.0,
    legend_anchor=(0.01, 0.99),
    legend_gap=0.03,
    ax=None,
):
    """Overlay several models' trajectories on shared axes.

    Encoding: hue -> model, shade -> intervention-within-model (baseline lightest),
    shape -> intervention. Two legends (model colours, intervention markers).
    CI whiskers drawn when *_lo/*_hi columns are present and `show_ci`.
    """
    if not trajectories:
        raise ValueError("`trajectories` is empty.")

    models = list(trajectories)
    if model_colors is None:
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        model_colors = {m: cycle_colors[i % len(cycle_colors)] for i, m in enumerate(models)}

    # global intervention order (first-seen, baseline first)
    interv_order = []
    for t in trajectories.values():
        for exp in t.experiment:
            if exp not in interv_order:
                interv_order.append(exp)

    if intervention_markers is None:
        pool = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
        intervention_markers = {
            exp: ("*" if i == 0 else pool[(i - 1) % len(pool)])
            for i, exp in enumerate(interv_order)
        }

    shades = {
        m: dict(zip(interv_order,
                    make_shades(model_colors[m], len(interv_order), light_range=shade_range)))
        for m in models
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    def col(t, name, fallback):
        return t[name] if name in t.columns else t[fallback]

    all_x = [v for t in trajectories.values()
             for v in list(col(t, "technical_lo", "technical")) + list(col(t, "technical_hi", "technical"))]
    all_y = [v for t in trajectories.values()
             for v in list(col(t, "social_lo", "social")) + list(col(t, "social_hi", "social"))]
    xlim = padded_limits(all_x, pad_frac)
    ylim = padded_limits(all_y, pad_frac)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for model, traj in trajectories.items():
        base = model_colors[model]
        x = traj.technical.to_numpy()
        y = traj.social.to_numpy()

        for i, j in _segments(connect, len(traj)):
            ax.annotate(
                "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="->", color=base, lw=1.6,
                                shrinkA=11, shrinkB=11, alpha=0.6),
                zorder=2,
            )

        has_ci = show_ci and {"technical_lo", "technical_hi",
                              "social_lo", "social_hi"}.issubset(traj.columns)
        for _, row in traj.iterrows():
            xi, yi, exp = row.technical, row.social, row.experiment
            if has_ci:
                xerr = [[xi - row.technical_lo], [row.technical_hi - xi]]
                yerr = [[yi - row.social_lo], [row.social_hi - yi]]
                ax.errorbar(xi, yi, xerr=xerr, yerr=yerr, fmt="none",
                            ecolor=base, elinewidth=1.1, capsize=2.5, alpha=0.45, zorder=3)
            mk = intervention_markers.get(exp, "o")
            sz = baseline_marker_size if mk == "*" else marker_size
            ax.scatter(xi, yi, s=sz, color=shades[model][exp], marker=mk,
                       edgecolor="black", linewidth=0.8, zorder=4)

    _draw_overlay_labels(ax, trajectories, xlim, ylim, label_mode, label_map, label_offset_pts)
    _draw_two_legends(fig, ax, models, model_colors, interv_order,
                      intervention_markers, label_map, legend_anchor, legend_gap)

    xlabel, ylabel = resolve_axis_labels(
        xlabel, ylabel, technical_cols, social_cols, axis_titles, label_display
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


# --------------------------------------------------------------------------- #
# Small drawing helpers (private to this module)
# --------------------------------------------------------------------------- #
def _segments(connect, n):
    if connect == "sequential":
        return list(zip(range(n - 1), range(1, n)))
    if connect == "from_baseline":
        return [(0, j) for j in range(1, n)]
    raise ValueError("connect must be 'sequential' or 'from_baseline'.")


def _maybe_adjust_text(use_adjust_text):
    if not use_adjust_text:
        return None
    try:
        from adjustText import adjust_text
        return adjust_text
    except ImportError:
        import warnings
        warnings.warn("adjustText not installed; using edge-aware offsets instead. "
                      "Run `pip install adjustText` to enable automatic de-overlap.")
        return None


def _draw_overlay_labels(ax, trajectories, xlim, ylim, label_mode, label_map, label_offset_pts):
    if label_mode == "none":
        return
    bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75)

    def disp(exp):
        return label_map.get(exp, exp) if label_map else exp

    def place(xi, yi, text):
        x_norm = (xi - xlim[0]) / (xlim[1] - xlim[0])
        y_norm = (yi - ylim[0]) / (ylim[1] - ylim[0])
        dx, dy, ha, va = edge_aware_offset(x_norm, y_norm, label_offset_pts)
        ax.annotate(text, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=9, zorder=5, bbox=bbox)

    if label_mode == "per_point":
        for traj in trajectories.values():
            for xi, yi, exp in zip(traj.technical, traj.social, traj.experiment):
                place(xi, yi, disp(exp))
    elif label_mode == "per_intervention":
        agg = {}
        for traj in trajectories.values():
            for xi, yi, exp in zip(traj.technical, traj.social, traj.experiment):
                agg.setdefault(exp, []).append((xi, yi))
        for exp, pts in agg.items():
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            place(cx, cy, disp(exp))
    else:
        raise ValueError("label_mode must be 'per_intervention', 'per_point', or 'none'.")


def _draw_two_legends(fig, ax, models, model_colors, interv_order,
                      intervention_markers, label_map, legend_anchor, legend_gap):
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=model_colors[m], markeredgecolor="black",
               markeredgewidth=0.6, label=m)
        for m in models
    ]
    interv_handles = [
        Line2D([0], [0], marker=intervention_markers[exp], linestyle="none",
               markersize=12 if intervention_markers[exp] == "*" else 9,
               markerfacecolor="0.45", markeredgecolor="black", markeredgewidth=0.6,
               label=(label_map.get(exp, exp) if label_map else exp))
        for exp in interv_order
    ]
    leg_models = ax.legend(handles=model_handles, title="Model", loc="upper left",
                           bbox_to_anchor=legend_anchor, frameon=True, framealpha=0.95,
                           fontsize=9, title_fontsize=9)
    ax.add_artist(leg_models)

    fig.canvas.draw()
    box = leg_models.get_window_extent(renderer=fig.canvas.get_renderer())
    bottom_left = ax.transAxes.inverted().transform((box.x0, box.y0))
    second_anchor = (legend_anchor[0], bottom_left[1] - legend_gap)
    ax.legend(handles=interv_handles, title="Intervention", loc="upper left",
              bbox_to_anchor=second_anchor, frameon=True, framealpha=0.95,
              fontsize=9, title_fontsize=9)