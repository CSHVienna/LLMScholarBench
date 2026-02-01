"""
gridbars.py

Pivot-native grid plotting for bar panels with optional confidence intervals.

Key idea:
- You pass a pivot table `values` with index (Index or MultiIndex) and columns = metric names.
- Optionally pass matching `ci_low` and `ci_high` pivot tables.

Index convention:
- If MultiIndex with >=2 levels: level 0 is treated as the "group" that forms blocks.
- Remaining levels define the within-group rows.
- If single Index: rows are models; no group column is drawn. Row labels appear on the first panel (optional).

This module is intentionally strict:
- It validates alignment of values and CI tables.
- It validates panel metric presence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

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
    x_vlines_color: str = "grey"

    # horizontal reference lines at y ticks
    draw_y_hlines: bool = True
    hline_lw: float = 1.0
    hline_alpha: float = 0.25
    hline_style: str = "--"

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
    internal_y_label_color: str = "lightgray"

    # legend frame (white background)
    legend_frame: bool = True
    legend_frame_alpha: float = 0.85


# -----------------------------
# Validation and small helpers
# -----------------------------
def _resolve_axis_setting(global_val, local_val):
    return global_val if global_val is not None else local_val


def _ensure_metrics_present(values: pd.DataFrame, panels: Sequence[PanelSpec]) -> List[str]:
    metric_names = [p.metric_name for p in panels]
    missing = [m for m in metric_names if m not in values.columns]
    if missing:
        raise ValueError(f"Missing metrics in values.columns: {missing}")
    return metric_names


def _ensure_same_index_and_columns(
    base: pd.DataFrame, other: pd.DataFrame, *, base_name: str, other_name: str
) -> None:
    if (not base.index.equals(other.index)) or (not base.columns.equals(other.columns)):
        raise ValueError(f"{other_name} must match {base_name} in both index and columns.")


def _compute_group_bounds_from_index(idx: pd.Index) -> List[Tuple[Hashable, int, int]]:
    """
    For MultiIndex (>=2 levels), treat level 0 as group blocks, assuming rows are already ordered.
    Returns list of (group_value, start_row, end_row), inclusive.
    """
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        return []

    gvals = idx.get_level_values(0).to_numpy()
    bounds: List[Tuple[Hashable, int, int]] = []
    if len(gvals) == 0:
        return bounds

    start = 0
    for i in range(1, len(gvals) + 1):
        if i == len(gvals) or gvals[i] != gvals[i - 1]:
            bounds.append((gvals[i - 1], start, i - 1))
            start = i
    return bounds


def _apply_index_label_maps(
    idx: pd.Index,
    maps: Optional[Mapping[int, Mapping[Hashable, str]]],
    *,
    sep: str = " | ",
) -> List[str]:
    """
    maps[level][raw] -> display
    If maps is None, uses str(raw) per component.
    """
    if maps is None:
        if isinstance(idx, pd.MultiIndex):
            return [sep.join(map(str, t)) for t in idx.to_list()]
        return [str(x) for x in idx.to_list()]

    if isinstance(idx, pd.MultiIndex):
        out: List[str] = []
        for t in idx.to_list():
            parts: List[str] = []
            for level, raw in enumerate(t):
                m = maps.get(level, None)
                parts.append(m.get(raw, str(raw)) if m else str(raw))
            out.append(sep.join(parts))
        return out

    m0 = maps.get(0, None)
    return [m0.get(x, str(x)) if m0 else str(x) for x in idx.to_list()]


def _default_row_colors(n: int) -> List[str]:
    return ["#CCCCCC"] * n


def _row_colors_from_group_palette(
    idx: pd.Index,
    group_bounds: List[Tuple[Hashable, int, int]],
    group_palette: Optional[Mapping[Hashable, Sequence]],
) -> List:
    if group_palette is None or len(group_bounds) == 0:
        return _default_row_colors(len(idx))

    colors: List = []
    for g, s, e in group_bounds:
        pal = list(group_palette.get(g, []))
        if len(pal) == 0:
            pal = ["#CCCCCC"]
        for j in range(s, e + 1):
            colors.append(pal[(j - s) % len(pal)])
    if len(colors) != len(idx):
        return _default_row_colors(len(idx))
    return colors


from typing import Callable, Any

Color = Any  # matplotlib accepts many types (hex, rgba tuple, etc.)

def _row_colors_from_spec(
    idx: pd.Index,
    *,
    row_colors: Optional[Sequence[Color]] = None,
    row_color_map: Optional[Mapping[Hashable, Color]] = None,
    row_color_func: Optional[Callable[[Hashable], Color]] = None,
    group_bounds: Optional[List[Tuple[Hashable, int, int]]] = None,
    group_palette: Optional[Mapping[Hashable, Sequence[Color]]] = None,
    default_color: Color = "#CCCCCC",
) -> List[Color]:
    """
    Generic row color resolution.

    Priority:
      1) row_colors list aligned with idx
      2) row_color_func(key)
      3) row_color_map[key]
      4) group_palette (by group bounds)
      5) default_color
    """
    n = len(idx)

    if row_colors is not None:
        if len(row_colors) != n:
            raise ValueError("row_colors must have length equal to number of rows.")
        return list(row_colors)

    if row_color_func is not None:
        out = [row_color_func(k) for k in idx.to_list()]
        return [default_color if c is None else c for c in out]

    if row_color_map is not None:
        out = [row_color_map.get(k, default_color) for k in idx.to_list()]
        return out

    if group_palette is not None and group_bounds is not None and len(group_bounds) > 0:
        return _row_colors_from_group_palette(idx, group_bounds, group_palette)

    return [default_color] * n



def _apply_axis_cosmetics(ax: plt.Axes, style: StyleSpec) -> None:
    if style.hide_spines_top:
        ax.spines["top"].set_visible(False)
    if style.hide_spines_right:
        ax.spines["right"].set_visible(False)
    if style.hide_spines_left:
        ax.spines["left"].set_visible(False)
    if style.hide_spines_bottom:
        ax.spines["bottom"].set_visible(style.keep_bottom_spine)


# -----------------------------
# Grid builder (pivot-native)
# -----------------------------
def _build_grid_from_index(
    *,
    nrows: int,
    panels: Sequence[PanelSpec],
    group_bounds: Optional[List[Tuple[Hashable, int, int]]] = None,
    row_labels: Optional[Sequence[str]] = None,
    group_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    width_ratios: Optional[Sequence[float]] = None,
    row_labels_for_first_panel: Optional[Sequence[str]] = None,
) -> Tuple[plt.Figure, Optional[plt.Axes], List[plt.Axes], np.ndarray]:
    """
    Returns
    -------
    fig, ax_lbl (or None), axes, y
    """
    if nrows <= 0:
        raise ValueError("nrows must be positive.")
    if group_bounds is None:
        group_bounds = []

    y = np.arange(nrows)[::-1]

    use_label_axis = (len(group_bounds) > 0) or (row_labels is not None)

    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)

    if use_label_axis:
        ncols = 1 + len(panels)
        if width_ratios is None:
            width_ratios = [layout.label_ratio] + [layout.panel_ratio] * len(panels)
        if len(width_ratios) != ncols:
            raise ValueError(f"width_ratios must have length {ncols}.")

        gs = GridSpec(
            1, ncols,
            figure=fig,
            width_ratios=width_ratios,
            wspace=layout.wspace
        )

        fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

        ax_lbl = fig.add_subplot(gs[0, 0])
        axes = [fig.add_subplot(gs[0, j]) for j in range(1, ncols)]

        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(-0.5, nrows - 0.5)
        ax_lbl.axis("off")

        if row_labels is not None:
            if len(row_labels) != nrows:
                raise ValueError("row_labels must have length nrows.")
            for i, lab in enumerate(row_labels):
                ax_lbl.text(
                    0.42, y[i], str(lab),
                    ha="left", va="center",
                    fontsize=style.label_fontsize,
                )

        for g, s, e in group_bounds:
            glab = group_label_map.get(g, g) if group_label_map else g
            yc = 0.5 * (y[s] + y[e])
            ax_lbl.text(
                0.02, yc, str(glab),
                ha="left", va="center",
                fontsize=style.label_fontsize,
            )

        ref_ax = ax_lbl
    else:
        ncols = len(panels)
        if width_ratios is None:
            width_ratios = [layout.panel_ratio] * len(panels)
        if len(width_ratios) != ncols:
            raise ValueError(f"width_ratios must have length {ncols}.")

        gs = GridSpec(
            1, ncols,
            figure=fig,
            width_ratios=width_ratios,
            wspace=layout.wspace
        )
        fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

        ax_lbl = None
        axes = [fig.add_subplot(gs[0, j]) for j in range(ncols)]
        ref_ax = axes[0]

        if row_labels_for_first_panel is not None:
            if len(row_labels_for_first_panel) != nrows:
                raise ValueError("row_labels_for_first_panel must have length nrows.")
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(row_labels_for_first_panel, fontsize=style.tick_fontsize)

    for ax, p in zip(axes, panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        ax.set_title(title, fontsize=style.title_fontsize, pad=10)

    if layout.draw_group_separators and len(group_bounds) > 1:
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

    return fig, ax_lbl, axes, y


# -----------------------------
# Main plotter (bar grid)
# -----------------------------
def plot_metric_grid_from_pivot(
    values: pd.DataFrame,
    *,
    panels: Sequence[PanelSpec],
    ci_low: Optional[pd.DataFrame] = None,
    ci_high: Optional[pd.DataFrame] = None,
    # ordering
    index_order: Optional[Sequence[Hashable]] = None,
    group_order: Optional[Sequence[Hashable]] = None,
    within_group_order: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    # labeling
    index_label_maps: Optional[Mapping[int, Mapping[Hashable, str]]] = None,
    group_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    # colors
    row_colors: Optional[Sequence] = None,
    group_palette: Optional[Mapping[Hashable, Sequence]] = None,
    row_color_map: Optional[Mapping[Hashable, object]] = None,
    row_color_func: Optional[Callable[[Hashable], object]] = None,
    default_row_color: object = "#CCCCCC",

    # layout/style
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    width_ratios: Optional[Sequence[float]] = None,
    # axis overrides
    xlim: Optional[Tuple[Number, Number]] = None,
    xticks: Optional[Sequence[Number]] = None,
    # highlighting
    highlight_rules: Optional[Mapping[str, str]] = None,
    # label placement for single Index
    single_index_as_ylabel: bool = True,
) -> plt.Figure:
    metric_names = _ensure_metrics_present(values, panels)

    v = values.loc[:, metric_names].copy()

    have_ci = (ci_low is not None) and (ci_high is not None)
    if have_ci:
        lo = ci_low.loc[:, metric_names].copy()
        hi = ci_high.loc[:, metric_names].copy()
        _ensure_same_index_and_columns(v, lo, base_name="values", other_name="ci_low")
        _ensure_same_index_and_columns(v, hi, base_name="values", other_name="ci_high")
    else:
        lo = hi = None

    # Ordering
    if index_order is not None:
        v = v.reindex(index_order)
        if have_ci:
            lo = lo.reindex(v.index)
            hi = hi.reindex(v.index)
    else:
        if isinstance(v.index, pd.MultiIndex) and v.index.nlevels >= 2 and group_order is not None:
            blocks = []
            for g in group_order:
                try:
                    block = v.xs(g, level=0, drop_level=False)
                except KeyError:
                    continue
                if within_group_order and (g in within_group_order):
                    block = block.reindex(within_group_order[g])
                blocks.append(block)
            if len(blocks) > 0:
                v = pd.concat(blocks, axis=0)
                if have_ci:
                    lo = lo.reindex(v.index)
                    hi = hi.reindex(v.index)

    idx = v.index
    nlevels = idx.nlevels if isinstance(idx, pd.MultiIndex) else 1

    group_bounds = _compute_group_bounds_from_index(idx)

    # Labels
    full_labels = _apply_index_label_maps(idx, index_label_maps)

    if nlevels >= 2:
        # within-group label: levels 1..K-1 (mapped)
        row_labels = []
        for s in full_labels:
            parts = s.split(" | ")
            row_labels.append(" | ".join(parts[1:]) if len(parts) > 1 else s)
        # group labels come from level 0
        group_lbls = idx.get_level_values(0).to_list()
    else:
        row_labels = full_labels
        group_lbls = None

    # Colors
    row_colors = _row_colors_from_spec(
        idx,
        row_colors=row_colors,
        row_color_map=row_color_map,
        row_color_func=row_color_func,
        group_bounds=group_bounds,
        group_palette=group_palette,
        default_color=default_row_color,
    )

    # Build figure grid
    fig, ax_lbl, axes, y = _build_grid_from_index(
        nrows=len(v),
        panels=panels,
        group_bounds=group_bounds,
        row_labels=row_labels if nlevels >= 2 else None,
        group_label_map=group_label_map,
        metric_label_map=metric_label_map,
        layout=layout,
        style=style,
        width_ratios=width_ratios,
        row_labels_for_first_panel=row_labels if (nlevels == 1 and single_index_as_ylabel) else None,
    )

    # Winner highlighting
    # Winner highlighting (ties included): (group, metric) -> set of row indices
    winners: Optional[Dict[Tuple[Hashable, str], set[int]]] = None
    if highlight_rules:
        winners = {}

        def _mark_ties(rows: Sequence[int], vals: np.ndarray, rule: str, tol: float = 1e-3) -> set[int]:
            mask = np.isfinite(vals)
            if not mask.any():
                return set()

            rows_f = np.asarray(rows)[mask]
            vals_f = vals[mask]

            best = np.nanmax(vals_f) if rule == "max" else np.nanmin(vals_f)

            tied = np.isclose(vals_f, best, rtol=0.0, atol=tol)
            return set(map(int, rows_f[tied]))

        if nlevels >= 2:
            for g, s, e in group_bounds:
                rows_g = list(range(s, e + 1))
                for p in panels:
                    rule = highlight_rules.get(p.metric_name)
                    if rule is None:
                        continue

                    vals = v[p.metric_name].iloc[rows_g].to_numpy(dtype=float)
                    tied_rows = _mark_ties(rows_g, vals, rule)

                    if tied_rows:
                        winners[(g, p.metric_name)] = tied_rows
        else:
            rows_all = list(range(len(v)))
            for p in panels:
                rule = highlight_rules.get(p.metric_name)
                if rule is None:
                    continue

                vals = v[p.metric_name].to_numpy(dtype=float)
                tied_rows = _mark_ties(rows_all, vals, rule)

                if tied_rows:
                    winners[("ALL", p.metric_name)] = tied_rows

    # Render panels
    for col_i, (ax, p) in enumerate(zip(axes, panels)):
        x = v[p.metric_name].to_numpy(dtype=float)

        # Axis config
        ax_xlim = _resolve_axis_setting(xlim, p.xlim)
        ax_xticks = _resolve_axis_setting(xticks, p.xticks)

        ax.set_ylim(-0.5, len(y) - 0.5)
        if style.hide_y_ticks_nonfirstcol and col_i > 0:
            ax.set_yticks([])
        elif nlevels >= 2:
            ax.set_yticks([])

        if ax_xlim is not None:
            ax.set_xlim(*ax_xlim)
        if ax_xticks is not None:
            ax.set_xticks(list(ax_xticks))

        ax.tick_params(axis="x", labelsize=style.tick_fontsize)

        if style.grid_x:
            ax.grid(True, axis="x", linewidth=0.6, alpha=style.grid_alpha)
        if style.grid_y:
            ax.grid(True, axis="y", linewidth=0.6, alpha=style.grid_alpha)

        _apply_axis_cosmetics(ax, style)

        if style.draw_x_vlines:
            # only if xlim exists or axis is numeric, draw at 0 and 1
            ax.axvline(0, linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0, color=style.x_vlines_color)
            ax.axvline(1, linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0, color=style.x_vlines_color)

        if style.draw_y_hlines and (not (style.hide_y_ticks_nonfirstcol and col_i > 0)):
            yt = ax.get_yticks()
            for yy in yt:
                ax.axhline(yy, linewidth=style.hline_lw, alpha=style.hline_alpha, linestyle=style.hline_style, zorder=0)

        # Bars
        ax.barh(y, x, height=style.bar_height, color=row_colors)

        # CI
        if p.draw_ci and p.ci_style != "none" and (lo is not None) and (hi is not None):
            ci_l = lo[p.metric_name].to_numpy(dtype=float)
            ci_h = hi[p.metric_name].to_numpy(dtype=float)

            if p.ci_style == "horizontal":
                err = np.vstack([x - ci_l, ci_h - x])
                err = np.where(np.isfinite(err), np.maximum(err, 0.0), np.nan)
                ax.errorbar(
                    x, y, xerr=err,
                    fmt="none",
                    capsize=style.ci_capsize,
                    linewidth=style.ci_lw,
                )
            elif p.ci_style == "vertical":
                # Not typical for barh, but supported if you use y as a numeric scale
                err = np.vstack([y - ci_l, ci_h - y])
                err = np.where(np.isfinite(err), np.maximum(err, 0.0), np.nan)
                ax.errorbar(
                    x, y, yerr=err,
                    fmt="none",
                    capsize=style.ci_capsize,
                    linewidth=style.ci_lw,
                )
            else:
                raise ValueError(f"Invalid ci_style: {p.ci_style}")

        # Value labels (anchored at CI high if available, else at mean)
        xmin, xmax = ax.get_xlim()
        span = float(xmax) - float(xmin)

        if (lo is not None) and (hi is not None) and p.draw_ci and p.ci_style == "horizontal":
            ci_h_for_anchor = hi[p.metric_name].to_numpy(dtype=float)
        else:
            ci_h_for_anchor = None

        for i, (yi, xm) in enumerate(zip(y, x)):
            if not np.isfinite(xm):
                continue

            anchor = xm
            if ci_h_for_anchor is not None and np.isfinite(ci_h_for_anchor[i]):
                anchor = float(ci_h_for_anchor[i])

            if nlevels >= 2:
                g_i = idx[i][0]
                is_winner = (winners is not None) and (i in winners.get((g_i, p.metric_name), set()))
            else:
                is_winner = (winners is not None) and (i in winners.get(("ALL", p.metric_name), set()))

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

        # x tick labels only on bottom row of panels (you have 1 row, so this is always bottom)
        # Kept for compatibility with your multi-row layout plans.
        if style.show_xticklabels_only_bottom:
            ax.tick_params(axis="x", labelbottom=True)

    return fig
