
"""
gridslope.py

Pivot-native grid plotting for slope panels (before vs after), consistent with the
layout conventions used by gridfan.py.

Input format
------------
You pass two mappings:

before = {"values": df_vals, "ci_low": df_lo, "ci_high": df_hi}
after  = {"values": df_vals, "ci_low": df_lo, "ci_high": df_hi}

Each "*vals" DataFrame must be pivoted:
- index: Index or MultiIndex
- columns: metric names (one column per metric / panel)

Index convention
----------------
Let the index have K levels. The last level is interpreted as the hue (line id).
The first K-1 levels define the row facets shown on the left of the grid.

- If K == 1:
  There are no row facets; a single row is drawn and all hue categories are plotted.

- If K >= 2:
  Facets are defined by levels [0 .. K-2], hue by level K-1.

Per-cell plot
-------------
Within each (facet row × metric) cell:
- x is fixed: "before" at x=0, "after" at x=1
- for each hue, draw the two endpoints, optional y-error bars, and a connecting line
- remove spines and ticks
- draw horizontal reference lines at yticks, restricted to the interval [0, 1]
  (this is controlled by StyleSpec and per-call overrides)

This module aims to be strict about alignment:
- the before/after value tables must contain the same metrics as panels
- CI tables, if provided, must match their corresponding values table

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


def _nlevels(idx: pd.Index) -> int:
    return idx.nlevels if isinstance(idx, pd.MultiIndex) else 1


def _as_multiindex(idx: pd.Index) -> pd.MultiIndex:
    if isinstance(idx, pd.MultiIndex):
        return idx
    return pd.MultiIndex.from_arrays([idx.to_numpy()], names=[idx.name])


def _unique_facets(idx: pd.Index, shared_levels: int) -> List[Tuple]:
    """
    Return unique facet keys (levels [0..shared_levels-1]) in first-occurrence order.

    Always returns tuples of length `shared_levels`.
    If shared_levels == 0, returns [()].
    """
    if shared_levels == 0:
        return [()]

    mi = _as_multiindex(idx)

    # keep only the facet levels
    facet_part = mi.droplevel(list(range(shared_levels, mi.nlevels)))
    seen = set()
    out: List[Tuple] = []

    # droplevel may return an Index when only one level remains
    vals = facet_part.to_list() if hasattr(facet_part, "to_list") else list(facet_part)

    for v in vals:
        t = v if isinstance(v, tuple) else (v,)
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out


def _select_block(df: pd.DataFrame, facet: Tuple, shared_levels: int) -> pd.DataFrame:
    """Select all hue rows for a given facet."""
    if shared_levels == 0:
        return df
    mi = _as_multiindex(df.index)
    # pandas xs supports multi-level keys with list of levels
    levels = list(range(shared_levels))
    return df.xs(facet, level=levels, drop_level=False)


def _reindex_by_facet_and_hue(
    df: pd.DataFrame,
    *,
    facets: Sequence[Tuple],
    shared_levels: int,
    hue_level: int,
    hue_order: Optional[Sequence[Hashable]] = None,
) -> pd.DataFrame:
    """
    Reorder a DataFrame so that rows appear in facet order, and within each facet in hue order.
    Missing (facet, hue) combinations are skipped.
    """
    mi = _as_multiindex(df.index)
    hues_all = list(dict.fromkeys(mi.get_level_values(hue_level).to_list()))
    hues = list(hue_order) if hue_order is not None else hues_all

    want: List[Tuple] = []
    for f in facets:
        if shared_levels == 0:
            for h in hues:
                t = (h,)
                if t in mi:
                    want.append(t)
        else:
            for h in hues:
                f_key = f if isinstance(f, tuple) else (f,)
                t = f_key + (h,)
                if t in mi:
                    want.append(t)

    if len(want) == 0:
        return df.iloc[0:0].copy()
    return df.reindex(pd.MultiIndex.from_tuples(want, names=mi.names))


def _hue_colors(
    hues: Sequence[Hashable],
    *,
    hue_color_map: Optional[Mapping[Hashable, Color]],
    default_color: Color,
    cmap: Optional[str],
) -> Mapping[Hashable, Color]:
    if hue_color_map is not None:
        return {h: hue_color_map.get(h, default_color) for h in hues}
    if cmap:
        cm = plt.get_cmap(cmap)
        n = max(1, len(hues))
        return {h: cm(i / (n - 1) if n > 1 else 0.5) for i, h in enumerate(hues)}
    return {h: default_color for h in hues}


def plot_metric_grid_slope_from_pivot(
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

    # style / layout
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),

    # x labels
    x_labels: Tuple[str, str] = ("before", "after"),

    # CI / point / line styling
    line_kwargs: Optional[Mapping[str, Any]] = None,
    point_kwargs: Optional[Mapping[str, Any]] = None,
    before_errorbar_kwargs: Optional[Mapping[str, Any]] = None,
    after_errorbar_kwargs: Optional[Mapping[str, Any]] = None,

    # reference lines / internal y labels
    ytick_line_kwargs: Optional[Mapping[str, Any]] = None,
    yticks_override: Optional[Sequence[Number]] = None,

    # legend
    show_legend: bool = True,
    legend_ncol: Optional[int] = None,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, 1.04),
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

    # index structure
    b_k = _nlevels(b.index)
    a_k = _nlevels(a.index)
    if b_k != a_k:
        raise ValueError("before and after must have the same number of index levels for slope plots.")
    k = b_k
    if k < 1:
        raise ValueError("Index must have at least one level.")

    hue_level = k - 1
    shared_levels = max(0, k - 1)

    # facets (row keys)
    row_index = _as_multiindex(b.index) if k > 1 else b.index
    facets = _unique_facets(b.index, shared_levels)

    # ordering facets
    if shared_levels == 0:
        ordered_facets = [()]
    else:
        if index_order is not None:
            # index_order entries may be scalars (shared_levels==1) or tuples
            tmp: List[Tuple] = []
            for key in index_order:
                t = (key,) if shared_levels == 1 and not isinstance(key, tuple) else tuple(key)
                if t in facets:
                    tmp.append(t)
            ordered_facets = tmp if len(tmp) > 0 else list(facets)
        elif group_order is not None and shared_levels >= 2:
            # group = level0, within-group = remaining shared levels
            grouped: List[Tuple] = []
            for g in group_order:
                g_keys = [t for t in facets if t[0] == g]
                if within_group_order and (g in within_group_order):
                    want_within = within_group_order[g]
                    ordered_g: List[Tuple] = []
                    for w in want_within:
                        w_t = (w,) if (shared_levels == 2 and not isinstance(w, tuple)) else tuple(w)
                        # rebuild full tuple
                        full = (g,) + w_t
                        if full in g_keys:
                            ordered_g.append(full)
                    # append any remaining keys in original order
                    for t in g_keys:
                        if t not in ordered_g:
                            ordered_g.append(t)
                    grouped.extend(ordered_g)
                else:
                    grouped.extend(g_keys)
            ordered_facets = grouped if len(grouped) > 0 else list(facets)
        else:
            ordered_facets = list(facets)

    # reindex tables to facet/hue order (keeps group separators consistent)
    if k == 1:
        b = _reindex_by_facet_and_hue(b, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
        a = _reindex_by_facet_and_hue(a, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
        if have_b_ci:
            b_lo = _reindex_by_facet_and_hue(b_lo, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
            b_hi = _reindex_by_facet_and_hue(b_hi, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
        if have_a_ci:
            a_lo = _reindex_by_facet_and_hue(a_lo, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
            a_hi = _reindex_by_facet_and_hue(a_hi, facets=[()], shared_levels=0, hue_level=0, hue_order=hue_order)
        facets = [()]
    else:
        b = _reindex_by_facet_and_hue(b, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)
        a = _reindex_by_facet_and_hue(a, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)
        if have_b_ci:
            b_lo = _reindex_by_facet_and_hue(b_lo, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)
            b_hi = _reindex_by_facet_and_hue(b_hi, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)
        if have_a_ci:
            a_lo = _reindex_by_facet_and_hue(a_lo, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)
            a_hi = _reindex_by_facet_and_hue(a_hi, facets=ordered_facets, shared_levels=shared_levels, hue_level=hue_level, hue_order=hue_order)

    # rebuild facets after reindex
    facets = _unique_facets(b.index, shared_levels)

    # labels
    if shared_levels == 0:
        row_labels = [""]  # single row
        group_vals = None
        group_bounds: List[Tuple[Hashable, int, int]] = []
    else:
        # create a pseudo-index for facet rows (so label rendering matches other grids)
        facet_names = list(_as_multiindex(b.index).names[:shared_levels])
        if shared_levels == 1:
            facet_mi = pd.Index([t[0] for t in facets], name=facet_names[0])
        else:
            facet_mi = pd.MultiIndex.from_tuples(facets, names=facet_names)
        full_labels = grid._apply_index_label_maps(facet_mi, index_label_maps)

        if shared_levels >= 2:
            row_labels = []
            for s in full_labels:
                parts = s.split(" | ")
                row_labels.append(" | ".join(parts[1:]) if len(parts) > 1 else s)
            group_vals = facet_mi.get_level_values(0).to_list()
            group_bounds = grid._compute_group_bounds_from_index(facet_mi)
        else:
            row_labels = full_labels
            group_vals = None
            group_bounds = []

    # hues
    b_hues = list(dict.fromkeys(_as_multiindex(b.index).get_level_values(hue_level).to_list()))
    a_hues = list(dict.fromkeys(_as_multiindex(a.index).get_level_values(hue_level).to_list()))
    hues = list(dict.fromkeys(list(b_hues) + list(a_hues)))
    if hue_order is not None:
        hues = [h for h in hue_order if h in set(hues)]

    colors = _hue_colors(hues, hue_color_map=hue_color_map, default_color=default_hue_color, cmap=style.cmap)

    # defaults for artists
    lk = {"lw": style.slope_linewidth, "alpha": 1.0, "zorder": 3}
    if line_kwargs:
        lk.update(dict(line_kwargs))

    pk = {"s": style.slope_point_size, "zorder": 4}
    if point_kwargs:
        pk.update(dict(point_kwargs))

    eb_before = {"fmt": "none", "elinewidth": style.ci_lw, "capsize": style.ci_capsize, "alpha": 1.0, "zorder": 3}
    if before_errorbar_kwargs:
        eb_before.update(dict(before_errorbar_kwargs))

    eb_after = {"fmt": "none", "elinewidth": style.ci_lw, "capsize": style.ci_capsize, "alpha": 1.0, "zorder": 3}
    if after_errorbar_kwargs:
        eb_after.update(dict(after_errorbar_kwargs))

    hline_k = {"lw": style.hline_lw, "alpha": style.hline_alpha, "linestyle": style.hline_style, "color": "grey", "zorder": 1}
    if ytick_line_kwargs:
        hline_k.update(dict(ytick_line_kwargs))
    # figure + gridspec
    nrows = len(facets)
    nmetrics = len(panels)

    # If there is exactly one facet level (excluding hue), mimic the other grids:
    # do NOT allocate a dedicated label column. Instead, place the facet label as the
    # y-label of the first metric column.
    show_label_col = shared_levels >= 2

    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)

    if show_label_col:
        width_ratios = [layout.label_ratio] + [layout.panel_ratio] * nmetrics if layout.width_ratios is None else layout.width_ratios
        gs = GridSpec(
            nrows,
            1 + nmetrics,
            figure=fig,
            width_ratios=width_ratios,
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
        if group_bounds and layout.draw_group_separators:
            for g, s, e in group_bounds:
                if s > 0:
                    y_boundary = 1.0 - (s / nrows)
                    ax_lbl.plot([0, 1], [y_boundary, y_boundary], transform=ax_lbl.transAxes, color="black", lw=layout.separator_lw)

        # group labels (level0) and row labels (remaining)
        facet_names = list(_as_multiindex(b.index).names[:shared_levels])
        facet_mi = pd.MultiIndex.from_tuples(facets, names=facet_names)
        gvals = facet_mi.get_level_values(0).to_list()
        for i, (g, rl) in enumerate(zip(gvals, row_labels)):
            y_mid = 1.0 - ((i + 0.5) / nrows)
            # group label only at center of each block
            if i == 0 or g != gvals[i - 1]:
                for gg, s, e in group_bounds:
                    if gg == g and s <= i <= e:
                        y_mid_g = 1.0 - (((s + e + 1) / 2) / nrows)
                        g_txt = group_label_map.get(g, str(g)) if group_label_map else str(g)
                        ax_lbl.text(0.03, y_mid_g, g_txt, transform=ax_lbl.transAxes, ha="left", va="center",
                                    fontsize=style.label_fontsize)
                        break
            ax_lbl.text(0.55, y_mid, rl, transform=ax_lbl.transAxes, ha="left", va="center",
                        fontsize=style.label_fontsize)

        # column titles: place in a dedicated, invisible axis on row 0
        for j, p in enumerate(panels):
            title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
            ax_t = fig.add_subplot(gs[0, 1 + j])
            ax_t.set_title(title, fontsize=style.title_fontsize, pad=6)
            ax_t.set_axis_off()

    else:
        # no label column
        width_ratios = [layout.panel_ratio] * nmetrics if layout.width_ratios is None else layout.width_ratios
        gs = GridSpec(
            nrows,
            nmetrics,
            figure=fig,
            width_ratios=width_ratios,
            wspace=layout.wspace,
            hspace=layout.hspace,
            left=layout.left,
            right=layout.right,
            top=layout.top,
            bottom=layout.bottom,
        )
    # cell drawing
    for i, facet in enumerate(facets):
        for j, p in enumerate(panels):
            ax = fig.add_subplot(gs[i, (1 + j) if show_label_col else j])

                                    # facet label as ylabel for single-level facet grids
            if (not show_label_col) and shared_levels == 1 and j == 0:
                # facets are stored as 1-tuples
                raw = facet[0] if isinstance(facet, tuple) else facet
                txt = index_label_maps.get(0, {}).get(raw, str(raw)) if index_label_maps else str(raw)
                ax.set_ylabel(txt, fontsize=style.ylabel_fontsize, labelpad=style.ylabel_pad)

            # titles
            if i == 0:
                title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
                ax.set_title(title, fontsize=style.title_fontsize, pad=6)

            # cosmetics
            ax.set_xlim(-0.15, 1.15)
            ax.set_xticks([])
            ax.set_yticks([])
            if style.hide_spines_left:
                ax.spines["left"].set_visible(False)
            if style.hide_spines_bottom:
                ax.spines["bottom"].set_visible(False)
            if style.hide_spines_top:
                ax.spines["top"].set_visible(False)
            if style.hide_spines_right:
                ax.spines["right"].set_visible(False)

            # y limits / ticks per metric
            ylim = p.ylim
            yticks = p.yticks
            if yticks_override is not None:
                yticks = yticks_override
            if ylim is not None:
                ax.set_ylim(*ylim)

            # draw horizontal reference lines at y ticks, between x=0 and x=1
            if style.draw_y_hlines and yticks is not None:
                for yv in yticks:
                    ax.hlines(yv, 0.0, 1.0, **hline_k)
                # internal labels at the center, if requested
                if style.show_internal_y_labels:
                    for yv in yticks:
                        ax.text(
                            style.internal_y_label_x,
                            yv,
                            f"{yv:g}",
                            ha="center",
                            va="center",
                            fontsize=style.internal_y_label_fontsize,
                            alpha=style.internal_y_label_alpha,
                            color=style.internal_y_label_color,
                            zorder=style.internal_y_label_zorder,
                            bbox=dict(facecolor="white", edgecolor="none", pad=style.internal_y_label_bbox_pad),
                        )

            # optional x reference vlines
            if style.draw_x_vlines:
                ax.vlines([0, 1], *ax.get_ylim(), lw=style.x_vlines_lw, alpha=style.x_vlines_alpha, color=style.x_vlines_color, zorder=0)

            # x labels at bottom row
            if (not style.show_xticklabels_only_bottom) or (i == nrows - 1):
                ax.text(0, ax.get_ylim()[0]+style.tick_pad, x_labels[0], ha="center", va="top", fontsize=style.tick_fontsize, alpha=0.9)
                ax.text(1, ax.get_ylim()[0]+style.tick_pad, x_labels[1], ha="center", va="top", fontsize=style.tick_fontsize, alpha=0.9)

            # select hue rows
            b_block = _select_block(b, facet, shared_levels)
            a_block = _select_block(a, facet, shared_levels)

            # value series (metric column)
            b_series = b_block[p.metric_name]
            a_series = a_block[p.metric_name]

            b_mi = _as_multiindex(b_series.index)
            a_mi = _as_multiindex(a_series.index)

            def _by_hue(series: pd.Series, hues_seq):
                """Return mapping hue -> selected value(s) for that hue.

                Assumes hue is the last index level when `series` has a MultiIndex.
                """
                out = {}
                mi = _as_multiindex(series.index)
                if mi.nlevels == 1:
                    # simple Index of hues
                    for hh in hues_seq:
                        if hh in series.index:
                            out[hh] = series.loc[hh]
                    return out
                # MultiIndex: take cross-section on last level
                for hh in hues_seq:
                    try:
                        out[hh] = series.xs(hh, level=-1, drop_level=True)
                    except KeyError:
                        continue
                return out

            b_by_hue = _by_hue(b_series, hues)
            a_by_hue = _by_hue(a_series, hues)

            # scalar extraction
            def _scalar(x):
                if isinstance(x, (pd.Series, np.ndarray, list)):
                    try:
                        return float(np.asarray(x).ravel()[0])
                    except Exception:
                        return np.nan
                try:
                    return float(x)
                except Exception:
                    return np.nan

            # ci series for this metric
            if have_b_ci:
                b_lo_s = _select_block(b_lo, facet, shared_levels)[p.metric_name]
                b_hi_s = _select_block(b_hi, facet, shared_levels)[p.metric_name]
            else:
                b_lo_s = b_hi_s = None
            if have_a_ci:
                a_lo_s = _select_block(a_lo, facet, shared_levels)[p.metric_name]
                a_hi_s = _select_block(a_hi, facet, shared_levels)[p.metric_name]
            else:
                a_lo_s = a_hi_s = None

            for h in hues:
                if (h not in b_by_hue) or (h not in a_by_hue):
                    continue

                y0 = _scalar(b_by_hue[h])
                y1 = _scalar(a_by_hue[h])

                c = colors.get(h, default_hue_color)

                # connecting line
                ax.plot([0, 1], [y0, y1], color=c, **lk)

                # points
                ax.scatter([0, 1], [y0, y1], color=c, **pk)

                # error bars
                def _hue_loc(series: pd.Series, hh):
                    mi2 = _as_multiindex(series.index)
                    if mi2.nlevels == 1:
                        return series.loc[hh]
                    return series.xs(hh, level=-1, drop_level=True)

                if have_b_ci and b_lo_s is not None and b_hi_s is not None:
                    lo0 = _scalar(_hue_loc(b_lo_s, h))
                    hi0 = _scalar(_hue_loc(b_hi_s, h))
                    yerr = np.array([[y0 - lo0], [hi0 - y0]])
                    yerr = np.where(yerr < 0, 0, yerr) #TODO: check if this is correct
                    ax.errorbar([0], [y0], yerr=yerr, color=c, **eb_before)

                if have_a_ci and a_lo_s is not None and a_hi_s is not None:
                    lo1 = _scalar(_hue_loc(a_lo_s, h))
                    hi1 = _scalar(_hue_loc(a_hi_s, h))
                    yerr = np.array([[y1 - lo1], [hi1 - y1]])
                    yerr = np.where(yerr < 0, 0, yerr) #TODO: check if this is correct
                    ax.errorbar([1], [y1], yerr=yerr, color=c, **eb_after)

                # numeric annotations
                if style.annotate_points:
                    ax.text(0 + style.annotation_dx, y0 + style.annotation_dy, p.value_fmt.format(y0),
                            fontsize=style.annotation_fontsize, color=style.annotation_color, ha="left", va="center", zorder=5)
                    ax.text(1 + style.annotation_dx, y1 + style.annotation_dy, p.value_fmt.format(y1),
                            fontsize=style.annotation_fontsize, color=style.annotation_color, ha="left", va="center", zorder=5)

    # legend
    if show_legend and len(hues) > 0:
        handles = []
        labels = []
        for h in hues:
            lab = hue_label_map.get(h, str(h)) if hue_label_map else str(h)
            handles.append(Line2D([0], [0], color=colors.get(h, default_hue_color), lw=style.slope_linewidth))
            labels.append(lab)
        ncol = legend_ncol if legend_ncol is not None else len(hues)
        leg = fig.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=ncol,
            frameon=style.legend_frame,
            fontsize=style.legend_fontsize,
        )
        if style.legend_frame:
            leg.get_frame().set_alpha(style.legend_frame_alpha)

    return fig
