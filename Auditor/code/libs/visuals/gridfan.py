
"""
gridfan.py

Grid renderer for a two-position "fan" plot (before vs after) with one axis per
(subgroup row × metric panel) cell.

This corrects the failure mode where all subgroup rows were overlaid inside a single
axis per metric. Here, each subgroup row gets its own small axis, matching the user's
"subgroups × metrics" grid expectation.

Per-cell plot
-------------
- X axis is categorical and fixed: before at x=0, after at x=1.
- Y axis is the metric value.
- One baseline point at x=0, one or more intervention points at x=1 (hue lines).
- Connectors from before to each after.
- Confidence intervals rendered as vertical error bars for both before and after.

Index alignment rule (critical)
-------------------------------
Let before have B index levels, after have A index levels.

Shared key used to match before rows to after rows:
1) If A == B + 1:
   shared levels = all B levels of before, hue = last level of after.

2) If A == B:
   - If B == 1: shared = the single level, no hue (single after series).
   - If B >= 2: shared = all levels except the last, hue = last level of after.
     This supports the common case:
        before index: (group, kind, fixed_slice) e.g. (..., ..., "top_100")
        after  index: (group, kind, intervention) e.g. (..., ..., "rag")

Left label area
---------------
- If MultiIndex with >=2 levels: level 0 is the group label shown once per block.
  Remaining varying levels (excluding constant levels) are shown per row.
- If single Index: row label is placed as ylabel-like text aligned to each row.

This module uses PanelSpec / LayoutSpec / StyleSpec from libs.visuals.grid.
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from libs.visuals.grid import PanelSpec, LayoutSpec, StyleSpec, Number  # type: ignore
from libs.visuals import grid

Color = Any


# -----------------------------
# Index helpers
# -----------------------------
def _nlevels(idx: pd.Index) -> int:
    return idx.nlevels if isinstance(idx, pd.MultiIndex) else 1


def _infer_shared_and_hue(before_index: pd.Index, after_index: pd.Index) -> Tuple[int, Optional[int], bool]:
    b = _nlevels(before_index)
    a = _nlevels(after_index)

    # Special case: "no row index" fan plot.
    # before has a single row keyed by a fixed slice (e.g., 'top_100') while after index is only hue values.
    # We match on an empty shared key and treat the sole after index level as hue.
    if b == 1 and a == 1:
        if len(before_index) == 1 and len(after_index) >= 1:
            b_key = before_index[0]
            # If keys do not overlap, it is almost certainly (fixed slice) vs (hue-only).
            if b_key not in set(after_index):
                return 0, 0, False
        # Otherwise: same-depth, single series (no hue)
        return 1, None, False

    if a == b + 1:
        return b, a - 1, False

    if a == b:
        if b == 1:
            return 1, None, False
        return b - 1, a - 1, True

    raise ValueError(
        "Unsupported index shapes: after must have the same number of levels as before, "
        "or exactly one more level."
    )


def _shared_key_from_before(row_key: Hashable, *, k_shared_levels: int, drop_before_last: bool) -> Tuple:
    if isinstance(row_key, tuple):
        t = row_key
    else:
        t = (row_key,)
    if k_shared_levels == 0:
        return tuple()
    if drop_before_last and len(t) >= 2:
        return t[:k_shared_levels]
    return t[:k_shared_levels]


def _select_after_block(
    after_df: pd.DataFrame,
    shared_key: Tuple,
    *,
    k_shared_levels: int,
    hue_level_pos: Optional[int],
) -> pd.DataFrame:
    # If there are no shared levels, the entire after_df corresponds to this (singleton) row.
    if k_shared_levels == 0:
        return after_df
    if hue_level_pos is None:
        if isinstance(after_df.index, pd.MultiIndex):
            return after_df.loc[[shared_key], :]
        return after_df.loc[[shared_key[0]], :]

    if not isinstance(after_df.index, pd.MultiIndex):
        raise ValueError("after_df must be MultiIndex when hue is present.")

    names = list(after_df.index.names)
    shared_names = names[:k_shared_levels]

    block = after_df.xs(shared_key, level=shared_names, drop_level=True)
    if isinstance(block.index, pd.MultiIndex):
        raise ValueError(
            "after index has more than one remaining level after dropping shared levels; "
            f"expected only hue, got {block.index.names}."
        )
    return block


# -----------------------------
# Label drawing
# -----------------------------
def _map_level_value(level: int, v: Hashable, index_label_maps: Optional[Mapping[int, Mapping[Hashable, str]]]) -> str:
    if index_label_maps and (level in index_label_maps) and (v in index_label_maps[level]):
        return index_label_maps[level][v]
    return str(v)


def _compute_group_bounds(index: pd.Index) -> List[Tuple[Hashable, int, int]]:
    """(group_value, start_row_inclusive, end_row_exclusive) for level 0 groups."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        return []
    lvl0 = pd.Index(index.get_level_values(0))
    bounds: List[Tuple[Hashable, int, int]] = []
    start = 0
    cur = lvl0[0]
    for i in range(1, len(lvl0)):
        if lvl0[i] != cur:
            bounds.append((cur, start, i))
            cur = lvl0[i]
            start = i
    bounds.append((cur, start, len(lvl0)))
    return bounds


def _within_row_label(
    row_key: Hashable,
    *,
    index: pd.Index,
    index_label_maps: Optional[Mapping[int, Mapping[Hashable, str]]],
) -> str:
    """Per-row label excluding group level 0 and dropping constant levels."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        return _map_level_value(0, row_key, index_label_maps)

    nlv = index.nlevels
    if not isinstance(row_key, tuple):
        row_key = (row_key,)

    # keep only varying levels among 1..n-1
    varying = []
    for lv in range(1, nlv):
        try:
            nunique = pd.Index(index.get_level_values(lv)).nunique()
        except Exception:
            nunique = 2
        if nunique > 1:
            varying.append(lv)

    parts = [_map_level_value(lv, row_key[lv], index_label_maps) for lv in varying]
    return " | ".join(parts) if parts else ""


# -----------------------------
# Main plotter
# -----------------------------
def plot_metric_grid_fan_from_pivot(
    before: Mapping[str, pd.DataFrame],
    after: Mapping[str, pd.DataFrame],
    *,
    panels: Sequence[PanelSpec],

    # ordering
    index_order: Optional[Sequence[Hashable]] = None,
    group_order: Optional[Sequence[Hashable]] = None,
    within_group_order: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    hue_order: Optional[Sequence[Hashable]] = None,

    # labeling
    index_label_maps: Optional[Mapping[int, Mapping[Hashable, str]]] = None,
    group_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    hue_label_map: Optional[Mapping[Hashable, str]] = None,

    # colors
    hue_color_map: Optional[Mapping[Hashable, Color]] = None,
    default_hue_color: Color = "#1f77b4",
    before_color: Color = "black",

    # style / layout
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),

    # x labels
    x_labels: Tuple[str, str] = ("before", "after"),

    # CI / line styling
    connector_kwargs: Optional[Mapping[str, Any]] = None,
    before_errorbar_kwargs: Optional[Mapping[str, Any]] = None,
    after_errorbar_kwargs: Optional[Mapping[str, Any]] = None,

    # legend
    add_legend: bool = True,
    legend_ncol: Optional[int] = None,
    legend_kwargs: Optional[Mapping[str, Any]] = None,

    # for single index
    single_index_as_ylabel: bool = True,
) -> plt.Figure:
    if "values" not in before or "values" not in after:
        raise ValueError("before and after must each contain a 'values' DataFrame.")

    b_vals = before["values"]
    a_vals = after["values"]

    metric_names = grid._ensure_metrics_present(b_vals, panels)
    _ = grid._ensure_metrics_present(a_vals, panels)

    b = b_vals.loc[:, metric_names].copy()
    a = a_vals.loc[:, metric_names].copy()

    b_lo = before.get("ci_low", None)
    b_hi = before.get("ci_high", None)
    a_lo = after.get("ci_low", None)
    a_hi = after.get("ci_high", None)

    have_b_ci = (b_lo is not None) and (b_hi is not None)
    have_a_ci = (a_lo is not None) and (a_hi is not None)

    if have_b_ci:
        b_lo = b_lo.loc[:, metric_names].copy()
        b_hi = b_hi.loc[:, metric_names].copy()
        grid._ensure_same_index_and_columns(b, b_lo, base_name="before.values", other_name="before.ci_low")
        grid._ensure_same_index_and_columns(b, b_hi, base_name="before.values", other_name="before.ci_high")

    if have_a_ci:
        a_lo = a_lo.loc[:, metric_names].copy()
        a_hi = a_hi.loc[:, metric_names].copy()
        grid._ensure_same_index_and_columns(a, a_lo, base_name="after.values", other_name="after.ci_low")
        grid._ensure_same_index_and_columns(a, a_hi, base_name="after.values", other_name="after.ci_high")

    # matching logic
    k_shared_levels, hue_level_pos, drop_before_last = _infer_shared_and_hue(b.index, a.index)

    # reorder rows based on before

    def _normalize_index_order_for_reindex(
        idx_obj: pd.Index,
        order: Sequence[Hashable],
    ) -> List[Hashable]:
        """
        Allow passing "partial" MultiIndex keys when trailing levels are constant.

        Example:
            before index levels = (group, kind, task_param)
            task_param is constant ('top_100')
            user supplies order with 2-tuples (group, kind)
            -> we append the constant tail to build full 3-tuples.
        """
        if not isinstance(idx_obj, pd.MultiIndex):
            return list(order)

        nlv = idx_obj.nlevels
        # Identify constant trailing levels
        const_tail: List[Hashable] = []
        for lv in range(nlv):
            vals = pd.Index(idx_obj.get_level_values(lv))
            if vals.nunique() == 1:
                const_tail.append(vals[0])
            else:
                const_tail.append(None)

        out: List[Hashable] = []
        for k in order:
            if isinstance(k, tuple):
                t = k
            else:
                t = (k,)
            if len(t) == nlv:
                out.append(k)
                continue
            if len(t) > nlv:
                raise AssertionError(
                    f"index_order key {k} has length {len(t)} but index has {nlv} levels."
                )
            # Only support shortening by dropping a constant suffix.
            # The missing levels must be constant; otherwise ordering is ambiguous.
            missing = nlv - len(t)
            tail_levels = list(range(nlv - missing, nlv))
            tail_vals = []
            ok = True
            for lv in tail_levels:
                if const_tail[lv] is None:
                    ok = False
                    break
                tail_vals.append(const_tail[lv])
            if not ok:
                raise AssertionError(
                    "index_order provides partial MultiIndex keys, but the omitted index levels "
                    "are not constant. Provide full keys."
                )
            out.append(tuple(t) + tuple(tail_vals))
        return out

    if index_order is not None:
        b = b.reindex(_normalize_index_order_for_reindex(b.index, index_order))
        if have_b_ci:
            b_lo = b_lo.reindex(b.index)
            b_hi = b_hi.reindex(b.index)
    else:
        if isinstance(b.index, pd.MultiIndex) and b.index.nlevels >= 2 and group_order is not None:
            blocks: List[pd.DataFrame] = []
            for g in group_order:
                try:
                    block = b.xs(g, level=0, drop_level=False)
                except KeyError:
                    continue
                if within_group_order and (g in within_group_order):
                    try:
                        block = block.reindex(within_group_order[g])
                    except Exception:
                        pass
                blocks.append(block)
            if blocks:
                b = pd.concat(blocks, axis=0)
                if have_b_ci:
                    b_lo = b_lo.reindex(b.index)
                    b_hi = b_hi.reindex(b.index)

    row_index = b.index
    nrows = len(row_index)
    nmetrics = len(panels)

    # hue sets
    hue_color_map = dict(hue_color_map or {})

    def hue_color(h):
        return hue_color_map.get(h, default_hue_color)

    if hue_level_pos is None:
        hue_values: List[Hashable] = []
    else:
        hue_name = a.index.names[hue_level_pos]
        hue_values = list(pd.Index(a.index.get_level_values(hue_name)).unique())
        if hue_order is not None:
            hue_values = [h for h in hue_order if h in set(hue_values)]

    # style defaults
    connector_kwargs = dict(connector_kwargs or {})
    connector_kwargs.setdefault("linewidth", style.slope_linewidth)
    connector_kwargs.setdefault("alpha", 0.95)

    before_errorbar_kwargs = dict(before_errorbar_kwargs or {})
    before_errorbar_kwargs.setdefault("capsize", style.ci_capsize)
    before_errorbar_kwargs.setdefault("elinewidth", style.ci_lw)
    before_errorbar_kwargs.setdefault("alpha", 0.95)

    after_errorbar_kwargs = dict(after_errorbar_kwargs or {})
    after_errorbar_kwargs.setdefault("capsize", style.ci_capsize)
    after_errorbar_kwargs.setdefault("elinewidth", style.ci_lw)
    after_errorbar_kwargs.setdefault("alpha", 0.95)


    if layout.width_ratios is None:
        layout.width_ratios = [layout.label_ratio] + [layout.panel_ratio] * nmetrics

    # figure + gridspec: (rows x (label + metrics))
    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)
    gs = GridSpec(
        nrows,
        1 + nmetrics,
        figure=fig,
        width_ratios=layout.width_ratios,
        wspace=layout.wspace,
        hspace=layout.hspace,
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
    )

    # label axis spanning all rows
    ax_lbl = fig.add_subplot(gs[:, 0])
    ax_lbl.set_axis_off()

    # group separators + labels
    group_bounds = _compute_group_bounds(row_index)
    if group_bounds and layout.draw_group_separators:
        for g, s, e in group_bounds:
            # separator line at boundary between groups (except at top)
            if s > 0:
                # y in axes coords for boundary between rows
                y_sep = 1.0 - (s / nrows)
                ax_lbl.plot([0, 1], [y_sep, y_sep], transform=ax_lbl.transAxes,
                            color="grey", lw=layout.separator_lw, alpha=0.9, clip_on=False)

    # group titles and row labels
    if isinstance(row_index, pd.MultiIndex) and row_index.nlevels >= 2:
        # group label per block
        for g, s, e in group_bounds:
            y_mid = 1.0 - ((s + e) / 2) / nrows
            g_txt = (group_label_map.get(g, str(g)) if group_label_map else _map_level_value(0, g, index_label_maps))
            ax_lbl.text(0.03, y_mid, g_txt, transform=ax_lbl.transAxes,
                        ha="left", va="center", fontsize=style.label_fontsize, color="black")

        # within-row labels
        for i, key in enumerate(row_index.to_list()):
            y_mid = 1.0 - (i + 0.5) / nrows
            txt = _within_row_label(key, index=row_index, index_label_maps=index_label_maps)
            ax_lbl.text(0.55, y_mid, txt, transform=ax_lbl.transAxes,
                        ha="left", va="center", fontsize=style.label_fontsize, color="black")
    else:
        # single index labels
        if single_index_as_ylabel:
            for i, key in enumerate(row_index.to_list()):
                y_mid = 1.0 - (i + 0.5) / nrows
                txt = _map_level_value(0, key, index_label_maps)
                ax_lbl.text(0.03, y_mid, txt, transform=ax_lbl.transAxes,
                            ha="left", va="center", fontsize=style.label_fontsize, color="black")

    # metric column titles
    for j, p in enumerate(panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        ax_t = fig.add_subplot(gs[0, 1 + j])
        ax_t.set_title(title, fontsize=style.title_fontsize, pad=style.title_pad, zorder=5)
        ax_t.set_axis_off()

    # plotting in each cell
    x_before, x_after = 0.0, 1.0
    for i, row_key in enumerate(row_index.to_list()):
        shared_key = _shared_key_from_before(row_key, k_shared_levels=k_shared_levels, drop_before_last=drop_before_last)

        # after block for this shared key
        try:
            a_block = _select_after_block(a, shared_key, k_shared_levels=k_shared_levels, hue_level_pos=hue_level_pos)
        except KeyError:
            # no after data for this row; skip after
            a_block = None

        a_lo_block = a_hi_block = None
        if have_a_ci and a_block is not None:
            try:
                a_lo_block = _select_after_block(a_lo, shared_key, k_shared_levels=k_shared_levels, hue_level_pos=hue_level_pos)
                a_hi_block = _select_after_block(a_hi, shared_key, k_shared_levels=k_shared_levels, hue_level_pos=hue_level_pos)
            except KeyError:
                a_lo_block = a_hi_block = None

        # determine row hues
        if hue_level_pos is None:
            row_hues = [None]
        else:
            present = set(a_block.index.to_list()) if a_block is not None else set()
            row_hues = [h for h in hue_values if h in present] if hue_values else list(present)

        for j, p in enumerate(panels):
            ax = fig.add_subplot(gs[i, 1 + j])

            # x fixed
            ax.set_xlim(-0.15, 1.15)
            ax.set_xticks([])

            # y scale per panel
            if p.ylim is not None:
                ax.set_ylim(*p.ylim)
            if p.yticks is not None:
                #ax.set_yticks(list(p.yticks))
                ax.set_yticks([])
            else:
                ax.set_yticks([])

            # cosmetics
            grid._apply_axis_cosmetics(ax, style)

            # internal y labels + horizontal guide lines
            if p.yticks is not None and style.draw_y_hlines:
                for yy in p.yticks:
                    ax.hlines(yy, 0.0, 1.0, color="grey", alpha=style.hline_alpha, linestyle=style.hline_style,
                               linewidth=style.hline_lw, zorder=0)
                    if style.show_internal_y_labels:
                        ax.text(
                            style.internal_y_label_x, yy, f"{yy:.2f}",
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="center",
                            fontsize=style.internal_y_label_fontsize,
                            alpha=style.internal_y_label_alpha,
                            color=style.internal_y_label_color,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=style.internal_y_label_bbox_pad),
                            zorder=1,
                        )

            # vertical guide lines at before/after
            if style.draw_x_vlines:
                ax.axvline(x_before, color=style.x_vlines_color, alpha=style.x_vlines_alpha,
                           linewidth=style.x_vlines_lw, zorder=0)
                ax.axvline(x_after, color=style.x_vlines_color, alpha=style.x_vlines_alpha,
                           linewidth=style.x_vlines_lw, zorder=0)

            # baseline
            b_mean = b.iloc[i, j]
            if pd.notna(b_mean):
                if have_b_ci and p.draw_ci:
                    lo = b_lo.iloc[i, j]
                    hi = b_hi.iloc[i, j]
                    if pd.notna(lo) and pd.notna(hi):
                        yerr = np.array([[b_mean - lo], [hi - b_mean]])
                        yerr = np.where(yerr < 0, 0, yerr) #TODO: check if this is correct
                        ax.errorbar([x_before], [b_mean], yerr=yerr, fmt="o", color=before_color,
                                    markersize=np.sqrt(style.slope_point_size), zorder=4, **before_errorbar_kwargs)
                    else:
                        ax.scatter([x_before], [b_mean], s=style.slope_point_size, color=before_color, zorder=4)
                else:
                    ax.scatter([x_before], [b_mean], s=style.slope_point_size, color=before_color, zorder=4)

            # after fan
            if a_block is not None:
                for h in row_hues:
                    if hue_level_pos is None:
                        a_mean = a_block.iloc[0][p.metric_name]
                        c = default_hue_color
                    else:
                        a_mean = a_block.loc[h, p.metric_name]
                        c = hue_color(h)

                    if pd.isna(a_mean):
                        continue

                    if pd.notna(b_mean):
                        ax.plot([x_before, x_after], [b_mean, a_mean], color=c, zorder=3, **connector_kwargs)

                    if have_a_ci and p.draw_ci and (a_lo_block is not None) and (a_hi_block is not None):
                        if hue_level_pos is None:
                            lo = a_lo_block.iloc[0][p.metric_name]
                            hi = a_hi_block.iloc[0][p.metric_name]
                        else:
                            lo = a_lo_block.loc[h, p.metric_name]
                            hi = a_hi_block.loc[h, p.metric_name]
                        if pd.notna(lo) and pd.notna(hi):
                            yerr = np.array([[a_mean - lo], [hi - a_mean]])
                            yerr = np.where(yerr < 0, 0, yerr) #TODO: check if this is correct
                            ax.errorbar([x_after], [a_mean], yerr=yerr, fmt="o", color=c,
                                        markersize=np.sqrt(style.slope_point_size), zorder=5, **after_errorbar_kwargs)
                        else:
                            ax.scatter([x_after], [a_mean], s=style.slope_point_size, color=c, zorder=5)
                    else:
                        ax.scatter([x_after], [a_mean], s=style.slope_point_size, color=c, zorder=5)

            # show category labels only on bottom row
            if i == nrows - 1 and x_labels != ("", ""):
                ax.text(x_before, -0.18+style.tick_pad, x_labels[0], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=style.tick_fontsize)
                ax.text(x_after, -0.18+style.tick_pad, x_labels[1], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=style.tick_fontsize)

            # hide y ticks except first metric column to reduce clutter
            if j > 0:
                ax.set_yticklabels([])

    # Legend
    if add_legend and hue_level_pos is not None and len(hue_values) > 0:
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=hue_color(h)) for h in hue_values]
        labels = [hue_label_map.get(h, str(h)) if hue_label_map else str(h) for h in hue_values]
        legend_ncol_use = legend_ncol if legend_ncol is not None else max(1, len(hue_values))
        legend_kwargs_use = dict(legend_kwargs or {})
        legend_kwargs_use.setdefault("loc", "upper center")
        legend_kwargs_use.setdefault("bbox_to_anchor", (0.5, layout.top + 0.06))
        legend_kwargs_use.setdefault("frameon", style.legend_frameon)
        legend_kwargs_use.setdefault("fontsize", style.legend_fontsize)
        legend_kwargs_use.setdefault("ncol", legend_ncol_use)
        fig.legend(handles, labels, **legend_kwargs_use)

    return fig
