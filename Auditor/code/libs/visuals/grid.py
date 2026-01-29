# metric_grid_plots.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

Number = Union[int, float]


# -----------------------------
# Specs
# -----------------------------
@dataclass(frozen=True)
class PanelSpec:
    """One metric column."""
    metric_name: str
    label: str

    # Per-metric axis config (can be overridden by globals)
    xlim: Optional[Tuple[Number, Number]] = None
    xticks: Optional[Sequence[Number]] = None

    # For line plots: treat ylim/yticks explicitly
    ylim: Optional[Tuple[Number, Number]] = None
    yticks: Optional[Sequence[Number]] = None

    # CI rendering per metric
    draw_ci: bool = True
    ci_style: str = "horizontal"  # {"horizontal", "vertical", "none"}

    # Value label format (bar plot)
    value_fmt: str = "{:.2f}"


@dataclass(frozen=True)
class LayoutSpec:
    """Grid layout controls."""
    figsize: Tuple[float, float] = (18.0, 4.6)
    wspace: float = 0.28
    hspace: float = 0.30

    # Width ratios: labels area vs each metric panel
    # If you want ~80% for metrics and 20% for labels, set with helper below.
    label_ratio: float = 1.75
    panel_ratio: float = 1.0

    # Figure margins
    left: float = 0.04
    right: float = 0.99
    top: float = 0.88
    bottom: float = 0.14

    # Group separator
    draw_group_separators: bool = True
    separator_lw: float = 1.2


@dataclass(frozen=True)
class StyleSpec:
    title_fontsize: int = 11
    label_fontsize: int = 11
    ylabel_fontsize: int = 11
    tick_fontsize: int = 10
    value_fontsize: int = 9

    grid_x: bool = True
    grid_y: bool = True
    grid_alpha: float = 0.35
    keep_bottom_spine: bool = True

    # Bars
    bar_height: float = 0.72

    # Lines
    line_width: float = 2.0
    point_size: float = 28.0

    # CI errorbar style (bars)
    ci_capsize: float = 2.5
    ci_lw: float = 1.0

    # Slope plot
    slope_linewidth: float = 2.0
    slope_point_size: float = 40.0

    # legend
    legend_fontsize: int = 10
    legend_frameon: bool = True

    # axis cosmetics
    hide_spines_left: bool = True
    hide_spines_bottom: bool = True
    hide_spines_top: bool = True
    hide_spines_right: bool = True

    # x reference lines at x=0 and x=1
    draw_x_vlines: bool = True
    x_vlines_lw: float = 1.8
    x_vlines_alpha: float = 0.95

    # horizontal reference lines at y ticks
    draw_y_hlines: bool = True
    hline_lw: float = 1.0
    hline_alpha: float = 0.25
    hline_style: str = '--'

    # labels
    show_xticklabels_only_bottom: bool = True
    hide_y_ticks_nonfirstcol: bool = True

    # row-label placement
    row_label_x: float = -0.22
    row_label_pad: float = 28

    # numeric annotation near endpoints
    annotate_points: bool = True
    annotation_fontsize: int = 10
    annotation_dx: float = 0.03  # in x-axis units (because x is [0,1])
    annotation_dy: float = 0.0   # in y-axis units
    annotation_color: str = "black"

    # internal y labels
    show_internal_y_labels: bool = True
    internal_y_label_x: float = 0.5     # axes fraction (middle)
    internal_y_label_fontsize: int = 10
    internal_y_label_alpha: float = 0.6
    internal_y_label_zorder: int = 2
    internal_y_label_bbox_pad: float = 0.35  # white background padding
    internal_y_label_color: float ='lightgray',

    # legend frame (white background)
    legend_frame: bool = True
    legend_frame_alpha: float = 0.85


@dataclass(frozen=True)
class CellDrawSpec:
    draw_hlines: bool = True
    draw_internal_y_labels: bool = True



CellSpecFn = Callable[[Any, int, int, PanelSpec], Optional[CellDrawSpec]]



def width_ratios_for_split(k_panels: int, bars_share: float = 0.8, panel_ratio: float = 1.0) -> List[float]:
    """Return width_ratios list for GridSpec: [label_ratio] + [panel_ratio]*k with target share for panels."""
    if not (0.0 < bars_share < 1.0):
        raise ValueError("bars_share must be in (0,1)")
    label_share = 1.0 - bars_share
    label_ratio = (label_share / bars_share) * k_panels * panel_ratio
    return [label_ratio] + [panel_ratio] * k_panels


# -----------------------------
# Shared helpers
# -----------------------------
def _resolve_axis_setting(
    *,
    global_val: Optional[Any],
    local_val: Optional[Any],
) -> Optional[Any]:
    """
    Global overrides local if provided.
    If global_val is None, use local_val.
    """
    return global_val if global_val is not None else local_val


def _build_row_index(
    df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    group_order: Sequence[str],
    model_order_within_group: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.MultiIndex:
    rows: List[Tuple[str, str]] = []
    for g in group_order:
        if model_order_within_group and g in model_order_within_group:
            models = list(model_order_within_group[g])
        else:
            models = list(pd.unique(df.loc[df[group_col] == g, model_col]))
        rows.extend([(g, m) for m in models])
    return pd.MultiIndex.from_tuples(rows, names=[group_col, model_col])


def _compute_group_bounds(row_index: pd.MultiIndex, group_order: Sequence[str]) -> List[Tuple[str, int, int]]:
    rows = list(row_index)
    bounds: List[Tuple[str, int, int]] = []
    start = 0
    for g in group_order:
        cnt = sum(1 for gg, _ in rows if gg == g)
        if cnt:
            bounds.append((g, start, start + cnt - 1))
            start += cnt
    return bounds


def _row_colors_from_group_palette(
    row_index: pd.MultiIndex,
    *,
    group_colors: Optional[Mapping[str, Sequence]] = None,
) -> List[Any]:
    colors: List[Any] = []
    rows = list(row_index)
    for g, m in rows:
        if group_colors and g in group_colors:
            palette = list(group_colors[g])
            models_in_group = [mm for gg, mm in rows if gg == g]
            idx = models_in_group.index(m)
            colors.append(palette[idx % len(palette)])
        else:
            colors.append(None)
    return colors


def _build_grid(
    df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    metric_col: str,
    panels: Sequence[PanelSpec],
    group_order: Sequence[str],
    model_order_within_group: Optional[Mapping[str, Sequence[str]]] = None,
    group_label_map: Optional[Mapping[str, str]] = None,
    model_label_map: Optional[Mapping[str, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    width_ratios: Optional[Sequence[float]] = None,
) -> Tuple[plt.Figure, plt.Axes, List[plt.Axes], pd.MultiIndex, np.ndarray, List[Tuple[str, int, int]]]:

    # Build row index and y positions
    row_index = _build_row_index(
        df, group_col=group_col, model_col=model_col,
        group_order=group_order, model_order_within_group=model_order_within_group
    )
    nrows = len(row_index)
    y = np.arange(nrows)[::-1]
    group_bounds = _compute_group_bounds(row_index, group_order)

    # Figure + GridSpec
    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)

    if width_ratios is None:
        width_ratios = [layout.label_ratio] + [layout.panel_ratio] * len(panels)

    gs = GridSpec(
        1, 1 + len(panels),
        figure=fig,
        width_ratios=width_ratios,
        wspace=layout.wspace
    )

    fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

    ax_lbl = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, j]) for j in range(1, 1 + len(panels))]

    # Label axis
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(-0.5, nrows - 0.5)
    ax_lbl.axis("off")

    # ax_lbl.text(0.02, nrows - 0.15, "Group", ha="left", va="top", fontsize=style.label_fontsize)
    # ax_lbl.text(0.42, nrows - 0.15, "Model", ha="left", va="top", fontsize=style.label_fontsize)

    # Model labels
    for i, (g, m) in enumerate(row_index):
        m_disp = model_label_map.get(m, m) if model_label_map else m
        ax_lbl.text(0.42, y[i], str(m_disp), ha="left", va="center", fontsize=style.label_fontsize)

    # Group labels centered
    for g, s, e in group_bounds:
        glab = group_label_map.get(g, g) if group_label_map else g
        yc = 0.5 * (y[s] + y[e])
        ax_lbl.text(0.02, yc, str(glab), ha="left", va="center", fontsize=style.label_fontsize)

    # Titles per metric
    for ax, p in zip(axes, panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        ax.set_title(title, fontsize=style.title_fontsize, pad=10)

    # Continuous separators
    if layout.draw_group_separators and len(group_bounds) > 1:
        ref_ax = ax_lbl
        for _, _, e in group_bounds[:-1]:
            y_data = 0.5 * (y[e] + y[e + 1])
            y_disp = ref_ax.transData.transform((0, y_data))[1]
            y_fig = fig.transFigure.inverted().transform((0, y_disp))[1]
            fig.add_artist(
                Line2D(
                    [layout.left, layout.right],
                    [y_fig, y_fig],
                    transform=fig.transFigure,
                    color="grey",
                    linewidth=layout.separator_lw,
                    zorder=10,
                )
            )

    return fig, ax_lbl, axes, row_index, y, group_bounds


# -----------------------------
# Bar plot grid
# -----------------------------
def plot_metric_grid_bars(
    df: pd.DataFrame,
    *,
    group_col: str = "model_group",
    model_col: str = "model_kind",
    metric_col: str = "metric_name",
    value_col: str = "mean",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    panels: Sequence[PanelSpec],
    group_order: Sequence[str],
    model_order_within_group: Optional[Mapping[str, Sequence[str]]] = None,
    group_label_map: Optional[Mapping[str, str]] = None,
    model_label_map: Optional[Mapping[str, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    group_colors: Optional[Mapping[str, Sequence]] = None,
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    # Global axis overrides
    xlim: Optional[Tuple[Number, Number]] = None,
    xticks: Optional[Sequence[Number]] = None,
    # Highlight winners: metric_name -> "max"/"min"
    highlight_rules: Optional[Mapping[str, str]] = None,
    width_ratios: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """
    df long format: one row per (group, model, metric) with mean and optional ci_low/ci_high.
    Global xlim/xticks override panel-level settings if provided.
    """

    # Filter to needed metrics early (avoids accidental extra rows)
    metric_names = [p.metric_name for p in panels]
    d = df[df[metric_col].isin(metric_names)].copy()

    fig, ax_lbl, axes, row_index, y, group_bounds = _build_grid(
        d,
        group_col=group_col, model_col=model_col, metric_col=metric_col,
        panels=panels, group_order=group_order,
        model_order_within_group=model_order_within_group,
        group_label_map=group_label_map, model_label_map=model_label_map, metric_label_map=metric_label_map,
        layout=layout, style=style,
        width_ratios=width_ratios
    )

    row_colors = _row_colors_from_group_palette(row_index, group_colors=group_colors)

    # Pivot values and CI to align with row_index
    mat = (
        d.pivot_table(index=[group_col, model_col], columns=metric_col, values=value_col, aggfunc="first")
        .reindex(row_index)
    )

    have_ci_cols = (ci_low_col in d.columns) and (ci_high_col in d.columns)
    low = (
        d.pivot_table(index=[group_col, model_col], columns=metric_col, values=ci_low_col, aggfunc="first")
        .reindex(row_index)
        if have_ci_cols else None
    )
    high = (
        d.pivot_table(index=[group_col, model_col], columns=metric_col, values=ci_high_col, aggfunc="first")
        .reindex(row_index)
        if have_ci_cols else None
    )

    # Winners: (group, metric) -> row position i
    winners: Optional[Dict[Tuple[str, str], int]] = None
    if highlight_rules:
        winners = {}
        for g, s, e in group_bounds:
            rows_g = list(range(s, e + 1))
            for p in panels:
                rule = highlight_rules.get(p.metric_name, None)
                if rule is None or p.metric_name not in mat.columns:
                    continue

                vals = mat.iloc[rows_g][p.metric_name].to_numpy(dtype=float)
                mask = np.isfinite(vals)
                if not mask.any():
                    continue

                rows_f = np.array(rows_g)[mask]
                vals_f = vals[mask]

                if rule == "max":
                    winners[(g, p.metric_name)] = int(rows_f[np.argmax(vals_f)])
                elif rule == "min":
                    winners[(g, p.metric_name)] = int(rows_f[np.argmin(vals_f)])
                else:
                    raise ValueError(f"Invalid highlight rule for {p.metric_name}: {rule}")

        # Sanity check
        (g0, m0), i0 = next(iter(winners.items()))
        assert 0 <= i0 < len(row_index)
        assert row_index[i0][0] == g0

    # Render each metric panel
    for ax, p in zip(axes, panels):
        x = mat[p.metric_name].to_numpy(dtype=float)

        ci_l = None
        ci_h = None
        if p.draw_ci and have_ci_cols and low is not None and high is not None and p.metric_name in low.columns:
            ci_l = low[p.metric_name].to_numpy(dtype=float)
            ci_h = high[p.metric_name].to_numpy(dtype=float)

        # Axis config (global overrides panel)
        ax_xlim = _resolve_axis_setting(global_val=xlim, local_val=p.xlim)
        ax_xticks = _resolve_axis_setting(global_val=xticks, local_val=p.xticks)

        ax.set_ylim(-0.5, len(y) - 0.5)
        ax.set_yticks([])

        if ax_xlim is not None:
            ax.set_xlim(*ax_xlim)
        if ax_xticks is not None:
            ax.set_xticks(list(ax_xticks))

        ax.tick_params(axis="x", labelsize=style.tick_fontsize)

        if style.grid_x:
            ax.grid(True, axis="x", linewidth=0.6, alpha=style.grid_alpha)

        # Spines: keep bottom only
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(style.keep_bottom_spine)

        # Bars
        ax.barh(y, x, height=style.bar_height, color=row_colors)

        # CI error bars
        if p.draw_ci and (ci_l is not None) and (ci_h is not None):
            err = np.vstack([x - ci_l, ci_h - x])
            err = np.where(np.isfinite(err), np.maximum(err, 0.0), np.nan)
            ax.errorbar(x, y, xerr=err, fmt="none", capsize=style.ci_capsize, linewidth=style.ci_lw)

        # Value labels: always black, to the right of ci_high if present else mean
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin

        for i, (yi, xm) in enumerate(zip(y, x)):
            if not np.isfinite(xm):
                continue

            anchor = xm
            if (ci_h is not None) and np.isfinite(ci_h[i]):
                anchor = ci_h[i]

            g_i = row_index[i][0]
            is_winner = (winners is not None) and (winners.get((g_i, p.metric_name)) == i)

            ax.text(
                anchor + 0.012 * span,
                yi,
                p.value_fmt.format(float(xm)),
                ha="left",
                va="center",
                fontsize=style.value_fontsize,
                color="black",
                fontweight="bold" if is_winner else "normal",
                clip_on=False,
            )

    return fig




def plot_metric_grid_temperature_lines(
    df: pd.DataFrame,
    *,
    group_col: str = "model_group",
    model_col: str = "model_kind",
    metric_col: str = "metric_name",
    temp_col: str = "temperature",
    value_col: str = "mean",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    panels: Sequence[PanelSpec],
    group_order: Sequence[str],
    model_order_within_group: Optional[Mapping[str, Sequence[str]]] = None,
    group_label_map: Optional[Mapping[str, str]] = None,
    model_label_map: Optional[Mapping[str, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    group_colors: Optional[Mapping[str, Sequence]] = None,
    layout: LayoutSpec = LayoutSpec(figsize=(18.0, 6.2)),
    style: StyleSpec = StyleSpec(),
    # Global overrides (x = temperature, y = metric)
    xlim: Optional[Tuple[Number, Number]] = None,
    xticks: Optional[Sequence[Number]] = None,
    ylim: Optional[Tuple[Number, Number]] = None,
    yticks: Optional[Sequence[Number]] = None,
    # Legend control per row (optional)
    row_legend_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    # Example:
    # row_legend_specs = {
    #   "model_access": {"col": 2, "ncol": 1, "y": 0.92, "loc": "upper center"},
    #   "model_size":   {"col": 0, "ncol": 2, "y": 0.88},
    #   "model_class":  {"col": 1, "ncol": 1, "y": 0.90},
    # }
    width_ratios: Optional[Sequence[float]] = None,
    # Line styling
    line_width: float = 2.0,
    marker: str = "o",
    marker_size: float = 4.0,
    ci_alpha: float = 0.15,
) -> plt.Figure:
    """
    df long format: one row per (group, model, metric, temperature) with mean and optional ci_low/ci_high.
    Layout: 3 rows (groups) x K columns (metrics) in a single grid, like the bar plot figure.
    Each cell: lines across temperature for the models within that group.
    """

    metric_names = [p.metric_name for p in panels]
    d = df[df[metric_col].isin(metric_names)].copy()
    if d.empty:
        raise ValueError("No rows left after filtering to requested metrics.")

    # Build grid once (same layout/labels as bar plot)
    fig, ax_lbl, axes, row_index, y_rows, group_bounds = _build_grid(
        d,
        group_col=group_col,
        model_col=model_col,
        metric_col=metric_col,
        panels=panels,
        group_order=group_order,
        model_order_within_group=model_order_within_group,
        group_label_map=group_label_map,
        model_label_map=model_label_map,
        metric_label_map=metric_label_map,
        layout=layout,
        style=style,
        width_ratios=width_ratios,
    )

    plt.close(fig)  # discard the 1xK grid from _build_grid

    # --- Build a pure 3xK grid (no label column)
    k = len(panels)
    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)

    gs = GridSpec(
        3, k,
        figure=fig,
        wspace=layout.wspace,
        hspace=0.30,
    )

    fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

    ax_grid = [[fig.add_subplot(gs[r, c]) for c in range(k)] for r in range(3)]

    # Column titles (top row only)
    for j, p in enumerate(panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        ax_grid[0][j].set_title(title, fontsize=style.title_fontsize, pad=8)

    # Row labels: put group names as y-label on first column only
    row_labels = [
        group_label_map.get(g, g) if group_label_map else g
        for g in group_order
    ]
    for r, lab in enumerate(row_labels):
        ax_grid[r][0].set_ylabel(lab, fontsize=style.label_fontsize, rotation=90, labelpad=16)
        # ax_grid[r][0].yaxis.set_label_coords(-0.22, 0.5)
        ax_grid[r][0].yaxis.set_label_coords(style.row_label_x, 0.5)

    # Draw row labels in label axis (centered per row)
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(0, 3)
    ax_lbl.text(0.02, 2.98, "Group", ha="left", va="top", fontsize=style.label_fontsize)

    # model list per group for left side
    for r, g in enumerate(group_order):
        y_center = 2.5 - r
        ax_lbl.text(
            0.02,
            y_center,
            str(row_labels[r]),
            ha="left",
            va="center",
            fontsize=style.label_fontsize,
        )

    # Determine CI availability
    have_ci_cols = (ci_low_col in d.columns) and (ci_high_col in d.columns)

    # Plot each row-group
    for r, g in enumerate(group_order):
        if model_order_within_group and g in model_order_within_group:
            models = list(model_order_within_group[g])
        else:
            models = list(pd.unique(d.loc[d[group_col] == g, model_col]))

        palette = list(group_colors[g]) if (group_colors and g in group_colors) else [None] * len(models)

        for j, p in enumerate(panels):
            ax = ax_grid[r][j]

            # axis overrides: global overrides local
            ax_xlim = _resolve_axis_setting(global_val=xlim, local_val=None)
            ax_xticks = _resolve_axis_setting(global_val=xticks, local_val=None)
            ax_ylim = _resolve_axis_setting(global_val=ylim, local_val=(p.ylim if p.ylim is not None else p.xlim))
            ax_yticks = _resolve_axis_setting(global_val=yticks, local_val=(p.yticks if p.yticks is not None else p.xticks))

            if ax_xlim is not None:
                ax.set_xlim(*ax_xlim)
            if ax_xticks is not None:
                ax.set_xticks(list(ax_xticks))
            if ax_ylim is not None:
                ax.set_ylim(*ax_ylim)
            if ax_yticks is not None:
                ax.set_yticks(list(ax_yticks))

            ax.tick_params(axis="both", labelsize=style.tick_fontsize)

            if style.grid_x:
                ax.grid(True, axis="x", linewidth=0.6, alpha=style.grid_alpha)
            ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(style.keep_bottom_spine)
            ax.spines["left"].set_visible(True)

            # x label only on bottom row
            ax.set_xlabel("Temperature" if r == 2 else "", fontsize=style.label_fontsize)

            # plot each model line
            for k_idx, m in enumerate(models):
                sub = d[
                    (d[group_col] == g)
                    & (d[model_col] == m)
                    & (d[metric_col] == p.metric_name)
                ].copy()
                if sub.empty:
                    continue

                sub[temp_col] = sub[temp_col].astype(float)
                sub = sub.sort_values(temp_col)

                x = sub[temp_col].to_numpy(dtype=float)
                yv = sub[value_col].to_numpy(dtype=float)

                c = palette[k_idx % len(palette)] if palette else None

                if p.draw_ci and have_ci_cols:
                    lo = sub[ci_low_col].to_numpy(dtype=float)
                    hi = sub[ci_high_col].to_numpy(dtype=float)
                    ax.fill_between(x, lo, hi, alpha=ci_alpha, linewidth=0, color=c)

                ax.plot(
                    x, yv,
                    linewidth=line_width,
                    marker=marker,
                    markersize=marker_size,
                    color=c,
                    label=str(model_label_map.get(m, m) if model_label_map else m),
                )

            # Legend per-row (optional), anchored per-row and per chosen column
            if row_legend_specs is not None:
                spec = row_legend_specs.get(g, None)
                if spec is not None and spec.get("col", j) == j:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ncol = int(spec.get("ncol", min(len(labels), 6)))
                        loc = str(spec.get("loc", "upper center"))
                        y_legend = float(spec.get("y", 0.92))
                        ax.legend(
                            handles, labels,
                            loc=loc,
                            ncol=ncol,
                            fontsize=style.legend_fontsize,
                            frameon=style.legend_frameon,
                            bbox_to_anchor=(0.5, y_legend),
                            bbox_transform=ax.transAxes,
                        )

    return fig














def plot_before_after_task_grid(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    panels: Sequence[PanelSpec],
    row_cols: Sequence[str] = ("task_name",),   # ("task_name",) or ("task_name","task_param") or ("model_kind",)
    hue_col: str = "model_kind",
    metric_col: str = "metric_name",
    value_col: str = "mean",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    # ordering / mapping
    row_order: Optional[Sequence[Any]] = None,              # keys: str if 1 row_col; tuple if 2 row_cols
    hue_order: Optional[Sequence[Any]] = None,
    row_label_map: Optional[Mapping[Any, str]] = None,      # row_key -> label
    hue_label_map: Optional[Mapping[Any, str]] = None,      # hue -> label
    hue_colors: Optional[Mapping[Any, Any]] = None,         # hue -> color
    metric_label_map: Optional[Mapping[str, str]] = None,   # metric_name -> label override
    # axis labels
    x_labels: Tuple[str, str] = ("Base", "RAG"),
    # global overrides (override per-panel)
    global_ylim: Optional[Tuple[Number, Number]] = None,
    global_yticks: Optional[Sequence[Number]] = None,
    # layout / style
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    # per-row legend placement (same as your timeseries plot)
    # key: row_key or "__default__"
    row_legend_specs: Optional[Dict[Any, Dict[str, Any]]] = None,
    # per-cell background control (row_key, col_idx) -> overrides
    cell_specs: Optional[Dict[Tuple[Any, int], CellDrawSpec]] = None,
    cell_spec_fnc: Optional[CellSpecFn] = None,
) -> plt.Figure:
    """
    Before/After (x=0/1) slope-grid.

    Rows:
      - determined by row_cols (1 col => scalar row_key; 2 cols => tuple row_key)
    Columns:
      - panels (metrics)
    Within each cell:
      - one slope per hue value (hue_col), from df_before to df_after

    Style changes implemented:
      - hide spines (left/bottom/top/right) by default
      - vertical black lines at x=0 and x=1
      - no y-axis ticks; instead subtle horizontal reference lines
      - internal y labels printed over white boxes (in front of hlines, behind data)
      - per-cell controls via cell_specs/cell_spec_fn (hlines/labels)
      - per-row legends via row_legend_specs, with optional white legend frame
    """

    if len(row_cols) not in (1, 2):
        raise ValueError("row_cols must have length 1 or 2.")

    metric_names = [p.metric_name for p in panels]

    # ---- Row keys extraction: scalar for 1 col, tuple for 2 cols
    def _extract_rows(df: pd.DataFrame) -> List[Any]:
        if len(row_cols) == 1:
            return list(pd.unique(df[row_cols[0]].dropna()))
        return list(map(tuple, df[list(row_cols)].drop_duplicates().to_numpy()))

    rows_b = _extract_rows(df_before)
    rows_a = _extract_rows(df_after)
    rows = list(row_order) if row_order is not None else sorted(set(rows_b) | set(rows_a))
    if not rows:
        raise ValueError("No rows found from row_cols in before/after dataframes.")

    # ---- Hue order
    if hue_order is None:
        hues = sorted(set(df_before[hue_col].dropna().unique()) | set(df_after[hue_col].dropna().unique()))
    else:
        hues = list(hue_order)

    # ---- Pivot helper: index=_rowkey, columns=(metric, hue)
    def _pivot(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
        tmp = df[df[metric_col].isin(metric_names)].copy()
        if tmp.empty:
            return pd.DataFrame(index=rows)

        if len(row_cols) == 1:
            tmp["_rowkey"] = tmp[row_cols[0]].astype(object)
        else:
            tmp["_rowkey"] = list(map(tuple, tmp[list(row_cols)].to_numpy()))

        out = tmp.pivot_table(index="_rowkey", columns=[metric_col, hue_col], values=val_col, aggfunc="first")
        return out.reindex(rows)

    mat_b = _pivot(df_before, value_col)
    mat_a = _pivot(df_after, value_col)

    have_ci_b = (ci_low_col in df_before.columns) and (ci_high_col in df_before.columns)
    have_ci_a = (ci_low_col in df_after.columns) and (ci_high_col in df_after.columns)
    low_b = _pivot(df_before, ci_low_col) if have_ci_b else None
    high_b = _pivot(df_before, ci_high_col) if have_ci_b else None
    low_a = _pivot(df_after, ci_low_col) if have_ci_a else None
    high_a = _pivot(df_after, ci_high_col) if have_ci_a else None

    # ---- Figure grid
    nrows = len(rows)
    ncols = len(panels)

    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)
    fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=layout.wspace, hspace=layout.hspace)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]

    # ---- Column titles
    for c, p in enumerate(panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        axes[0][c].set_title(title, fontsize=style.title_fontsize, pad=8)

    # ---- Row label helper
    def _row_label(row_key: Any) -> str:
        if row_label_map is not None and row_key in row_label_map:
            return str(row_label_map[row_key])
        if isinstance(row_key, tuple):
            return " / ".join(map(str, row_key))
        return str(row_key)

    # ---- Per-cell draw spec resolver
    def _resolve_cell_draw(row_key: Any, r: int, c: int, panel: PanelSpec) -> CellDrawSpec:
        base = CellDrawSpec(
            draw_hlines=style.draw_y_hlines,
            draw_internal_y_labels=style.show_internal_y_labels,
        )
        if cell_specs is not None:
            base = cell_specs.get((row_key, c), base)
        if cell_spec_fnc is not None:
            tmp = cell_spec_fnc(row_key, r, c, panel)
            if tmp is not None:
                base = tmp
        return base

    # ---- Draw all cells
    for r, row_key in enumerate(rows):
        for c, p in enumerate(panels):
            ax = axes[r][c]

            # axis limits/ticks
            ax.set_xlim(-0.08, 1.08)
            ax.set_xticks([0, 1])

            if style.show_xticklabels_only_bottom:
                if r == nrows - 1:
                    ax.set_xticklabels(list(x_labels))
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)
            else:
                ax.set_xticklabels(list(x_labels))

            ax.tick_params(axis="both", labelsize=style.tick_fontsize)

            # spines
            ax.spines["left"].set_visible(not style.hide_spines_left)
            ax.spines["bottom"].set_visible(not style.hide_spines_bottom)
            ax.spines["top"].set_visible(not style.hide_spines_top)
            ax.spines["right"].set_visible(not style.hide_spines_right)

            # remove y-axis ticks/labels (we use internal labels)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

            # y axis ranges (global overrides panel)
            ylim = global_ylim if global_ylim is not None else p.ylim
            if ylim is not None:
                ax.set_ylim(*ylim)

            # resolve per-cell drawing
            cell_draw = _resolve_cell_draw(row_key, r, c, p)

            # background: vertical lines once
            if style.draw_x_vlines:
                ax.axvline(0, color="black", linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0)
                ax.axvline(1, color="black", linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0)

            # background: horizontal lines and internal labels once
            yticks = global_yticks if global_yticks is not None else p.yticks
            if yticks is not None:
                # avoid accidental duplicates
                try:
                    yticks_u = sorted(set(float(y) for y in yticks))
                except Exception:
                    yticks_u = list(yticks)

                if cell_draw.draw_hlines:
                    for yy in yticks_u:
                        ax.axhline(
                            yy,
                            color="black",
                            linewidth=style.hline_lw,
                            alpha=style.hline_alpha,
                            linestyle=style.hline_style,
                            zorder=0,
                        )

                if cell_draw.draw_internal_y_labels:
                    for yy in yticks_u:
                        ax.text(
                            style.internal_y_label_x,
                            yy,
                            f"{yy:g}",
                            transform=ax.get_yaxis_transform(),  # x in axes, y in data
                            ha="center",
                            va="center",
                            fontsize=style.internal_y_label_fontsize,
                            color=style.internal_y_label_color,
                            alpha=style.internal_y_label_alpha,
                            zorder=style.internal_y_label_zorder,
                            bbox=dict(
                                facecolor="white",
                                edgecolor="none",
                                pad=style.internal_y_label_bbox_pad,
                            ),
                        )

            # data: slopes per hue (draw AFTER background)
            for h in hues:
                col = (p.metric_name, h)
                if (col not in mat_b.columns) or (col not in mat_a.columns):
                    continue

                v0 = mat_b.loc[row_key, col]
                v1 = mat_a.loc[row_key, col]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue

                color = hue_colors.get(h, None) if hue_colors else None
                label = hue_label_map.get(h, h) if hue_label_map else h

                ax.plot([0, 1], [v0, v1], linewidth=style.line_width, color=color, label=str(label), zorder=3)
                ax.scatter([0, 1], [v0, v1], s=style.point_size, color=color, zorder=4)

                # optional CI whiskers at endpoints
                if p.draw_ci:


                    if low_b is not None and high_b is not None and col in low_b.columns:
                        lo0 = low_b.loc[row_key, col]
                        hi0 = high_b.loc[row_key, col]


                        if np.isfinite(lo0) and np.isfinite(hi0):
                            ax.vlines(0, lo0, hi0, color=color, linewidth=style.ci_lw, alpha=0.9, zorder=10)
                            
                    if low_a is not None and high_a is not None and col in low_a.columns:
                        lo1 = low_a.loc[row_key, col]
                        hi1 = high_a.loc[row_key, col]

                        if np.isfinite(lo1) and np.isfinite(hi1):
                            ax.vlines(1, lo1, hi1, color=color, linewidth=style.ci_lw, alpha=0.9, zorder=10)

        # row label only on first column
        ax0 = axes[r][0]
        ax0.set_ylabel(
            _row_label(row_key),
            fontsize=style.ylabel_fontsize,
            # rotation=0,
            # ha="right",
            # va="center",
            # labelpad=style.row_label_pad,
        )
        ax0.yaxis.set_label_coords(style.row_label_x, 0.5)

    # ---- Per-row legends
    if row_legend_specs:
        default_spec = row_legend_specs.get("__default__", None)

        for r, row_key in enumerate(rows):
            spec = row_legend_specs.get(row_key, default_spec)
            if not spec:
                continue

            col_idx = int(spec.get("col", 0))
            col_idx = int(np.clip(col_idx, 0, ncols - 1))
            axL = axes[r][col_idx]

            handles, labels = axL.get_legend_handles_labels()
            if not handles:
                continue

            ncol = int(spec.get("ncol", 1))
            loc = str(spec.get("loc", "upper center"))
            y_anchor = float(spec.get("y", 0.90))
            fontsize = int(spec.get("fontsize", style.legend_fontsize))

            leg = axL.legend(
                handles,
                labels,
                frameon=style.legend_frame,
                ncol=ncol,
                fontsize=fontsize,
                loc=loc,
                bbox_to_anchor=(0.5, y_anchor),
                bbox_transform=axL.transAxes,
            )
            if style.legend_frame:
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("none")
                leg.get_frame().set_alpha(style.legend_frame_alpha)

    return fig
