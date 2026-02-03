from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


# ----------------------------
# Configuration dataclasses
# ----------------------------
@dataclass
class PanelSpec:
    row_levels: Tuple[str, str] = ("model_group", "model_kind")
    row_order: Optional[List[Tuple[str, str]]] = None
    group_label_map: Optional[Mapping[str, str]] = None
    kind_label_map: Optional[Mapping[str, str]] = None
    figsize: Optional[Tuple[float, float]] = None  # moved here
    wspace: float = 0.35
    hspace: float = 0.35
    group_separators: bool = True
    group_sep_lw: float = 0.8
    group_sep_alpha: float = 0.35
    group_sep_color: str = "black"
    group_sep_pad: float = 0.00  # + moves line slightly up, - slightly down
    

@dataclass
class HueSpec:
    hue_level: str = "task_param"
    hue_order: Optional[List[str]] = None
    legend_label_map: Optional[Mapping[str, str]] = None
    color_map: Optional[Mapping[str, str]] = None
    line_alpha: float = 0.9
    line_width: float = 1.6
    endpoint_size: float = 18
    baseline_size: float = 26


@dataclass
class StyleSpec:
    # shared x for all cells
    x_before: float = 0.0
    x_after: float = 1.0
    xlabel_before: str = "Before"
    xlabel_after: str = "After"
    rails_lw: float = 1.2

    # global shared y for all cells (set if you want to override)
    global_ylim: Optional[Tuple[float, float]] = None

    # middle "tick-like" labels
    mid_ticks: Optional[List[float]] = None
    mid_tick_fmt: str = "{:.1f}"
    mid_tick_fontsize: float = 8.5
    mid_tick_alpha: float = 1.0
    mid_tick_mark: bool = True
    mid_tick_mark_frac: float = 0.10  # tick length fraction of rail distance


@dataclass
class LegendSpec:
    enabled: bool = True
    title: Optional[str] = None
    loc: str = "upper center"
    bbox_to_anchor: Tuple[float, float] = (0.5, 0.985)  # closer to the plot
    ncol: Optional[int] = None  # default: len(hues)
    frameon: bool = False
    fontsize: Optional[float] = None
    handlelength: float = 2.2
    columnspacing: float = 1.6
    # Reserve space at top for legend (fraction of figure height)
    top: float = 0.92



# ----------------------------
# Helpers
# ----------------------------
def _ensure_index_levels(df: pd.DataFrame, required: Sequence[str]) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected df.index to be a MultiIndex.")
    missing = [lvl for lvl in required if lvl not in df.index.names]
    if missing:
        raise ValueError(f"Missing index levels: {missing}. Found: {df.index.names}")


def _get_rows(
    pivot_before: pd.DataFrame,
    pivot_after: pd.DataFrame,
    panel: PanelSpec,
) -> List[Tuple[str, str]]:
    lvl_g, lvl_k = panel.row_levels

    def rows_from(df: pd.DataFrame) -> set:
        keep = (lvl_g, lvl_k)
        drop = [n for n in df.index.names if n not in keep]
        idx = df.index.droplevel(drop) if drop else df.index
        return set(idx.unique().tolist())

    rows = list(rows_from(pivot_before).union(rows_from(pivot_after)))

    out: List[Tuple[str, str]] = []
    for r in rows:
        if not isinstance(r, tuple) or len(r) != 2:
            raise ValueError("Row tuples must be (model_group, model_kind).")
        out.append((str(r[0]), str(r[1])))

    if panel.row_order is not None:
        present = set(out)
        ordered = [t for t in panel.row_order if t in present]
        leftovers = sorted(list(present.difference(ordered)))
        return ordered + leftovers

    return sorted(out)


def _get_hues(pivot_after: pd.DataFrame, hue: HueSpec) -> List[str]:
    vals = pivot_after.index.get_level_values(hue.hue_level).unique().tolist()
    vals = [str(v) for v in vals]
    if hue.hue_order is not None:
        ordered = [h for h in hue.hue_order if h in vals]
        extras = [h for h in vals if h not in ordered]
        return ordered + sorted(extras)
    return sorted(vals)


def _compute_global_ylim(
    pivot_before: pd.DataFrame,
    pivot_after: pd.DataFrame,
    metrics: List[str],
) -> Tuple[float, float]:
    vals = []
    for m in metrics:
        if m in pivot_before.columns:
            vals.append(pivot_before[m].to_numpy(dtype=float))
        if m in pivot_after.columns:
            vals.append(pivot_after[m].to_numpy(dtype=float))

    allv = np.concatenate(vals) if vals else np.array([0.0])
    finite = allv[np.isfinite(allv)]
    if finite.size == 0:
        return (0.0, 1.0)

    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    pad = 0.05 * (ymax - ymin)
    return (ymin - pad, ymax + pad)


def _default_mid_ticks(global_ylim: Tuple[float, float]) -> List[float]:
    ymin, ymax = global_ylim
    if ymin <= 0.0 and ymax >= 1.0 and (ymax - ymin) <= 2.0:
        return [round(x, 1) for x in np.arange(0.0, 1.0001, 0.1)]
    ticks = np.linspace(ymin, ymax, 5)
    return [float(t) for t in ticks]


def _draw_fan_cell(
    ax: plt.Axes,
    y0: float,
    after_values: Dict[str, float],
    colors: Dict[str, str],
    global_ylim: Tuple[float, float],
    style: StyleSpec,
    hue: HueSpec,
):
    ymin, ymax = global_ylim
    rng = (ymax - ymin) if (ymax != ymin) else 1.0

    def norm(y: float) -> float:
        return (y - ymin) / rng

    # blank canvas
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # rails (shared x)
    ax.plot([style.x_before, style.x_before], [0, 1], lw=style.rails_lw, color="black")
    ax.plot([style.x_after, style.x_after], [0, 1], lw=style.rails_lw, color="black")

    # middle ticks (shared y)
    tick_vals = style.mid_ticks if style.mid_ticks is not None else _default_mid_ticks(global_ylim)
    xm = 0.5
    tick_len = style.mid_tick_mark_frac * (style.x_after - style.x_before)

    for tv in tick_vals:
        if not np.isfinite(tv):
            continue
        yn = norm(float(tv))
        if yn < 0.0 or yn > 1.0:
            continue
        if style.mid_tick_mark:
            ax.plot(
                [0,1],
                [yn, yn],
                lw=0.9,
                color="grey",
                alpha=0.3,
                zorder=0,
                ls="--",
            )
        ax.text(
            xm,
            yn,
            style.mid_tick_fmt.format(float(tv)),
            ha="center",
            va="center",
            fontsize=style.mid_tick_fontsize,
            alpha=style.mid_tick_alpha,
            color="grey",
            zorder=1,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        )

    # baseline
    if not np.isfinite(y0):
        return
    y0n = norm(float(y0))
    ax.scatter([style.x_before], [y0n], s=hue.baseline_size, color="black", zorder=3)

    # fan lines
    for h, y1 in after_values.items():
        if not np.isfinite(y1):
            continue
        y1n = norm(float(y1))
        ax.plot(
            [style.x_before, style.x_after],
            [y0n, y1n],
            color=colors[h],
            lw=hue.line_width,
            alpha=hue.line_alpha,
            zorder=2,
        )
        ax.scatter([style.x_after], [y1n], s=hue.endpoint_size, color=colors[h], zorder=3)


# ----------------------------
# Main function
# ----------------------------
def fanplot(
    pivot_before: pd.DataFrame,
    pivot_after: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    panel: PanelSpec = PanelSpec(),
    hue: HueSpec = HueSpec(),
    style: StyleSpec = StyleSpec(),
    metric_order: Optional[List[str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    metric_suffix_map: Optional[Mapping[str, str]] = None,
    legendspec: LegendSpec = LegendSpec(),
    savepath: Optional[str] = None,
    dpi: int = 200,
):
    lvl_g, lvl_k = panel.row_levels

    _ensure_index_levels(pivot_after, [lvl_g, lvl_k, hue.hue_level])
    _ensure_index_levels(pivot_before, [lvl_g, lvl_k])

    # metrics
    if metrics is None:
        metrics = [c for c in pivot_before.columns if c in pivot_after.columns]
    metrics = list(metrics)

    if metric_order is not None:
        metrics = [m for m in metric_order if m in metrics] + [m for m in metrics if m not in metric_order]
    if not metrics:
        raise ValueError("No metrics to plot.")

    rows = _get_rows(pivot_before, pivot_after, panel)
    hues = _get_hues(pivot_after, hue)

    # colors
    if hue.color_map is not None:
        color_for = {str(k): v for k, v in hue.color_map.items()}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        color_for = {h: cycle[i % max(1, len(cycle))] for i, h in enumerate(hues)}

    # shared global ylim (all grid cells)
    global_ylim = style.global_ylim if style.global_ylim is not None else _compute_global_ylim(
        pivot_before, pivot_after, metrics
    )

    n_rows = len(rows)
    n_cols = len(metrics)

    # figsize comes from PanelSpec
    if panel.figsize is None:
        figsize = (2.2 * n_cols + 3.2, 1.2 * n_rows + 1.6)
    else:
        figsize = panel.figsize

    fig = plt.figure(figsize=figsize)

    # header row for column titles
    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        figure=fig,
        height_ratios=[0.6] + [1.0] * n_rows,
        width_ratios=[1.35] + [1.0] * n_cols,
        wspace=panel.wspace,
        hspace=panel.hspace,
    )

    # column titles
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")
    for c_i, m in enumerate(metrics):
        ax_t = fig.add_subplot(gs[0, c_i + 1])
        ax_t.axis("off")
        m_disp = metric_label_map.get(m, m) if metric_label_map else m
        suf = metric_suffix_map.get(m, "") if metric_suffix_map else ""
        title = f"{m_disp}{(' ' + suf) if suf else ''}"
        ax_t.text(0.5, 0.6, title, ha="center", va="center", fontsize=11)

    row_label_axes: List[plt.Axes] = []
    row_groups: List[str] = []

    # rows
    for r_i, (g, k) in enumerate(rows, start=1):
        # left panel: two lines only
        ax_lab = fig.add_subplot(gs[r_i, 0])
        row_label_axes.append(ax_lab)
        row_groups.append(g)
        ax_lab.axis("off")
        g_disp = panel.group_label_map.get(g, g) if panel.group_label_map else g
        k_disp = panel.kind_label_map.get(k, k) if panel.kind_label_map else k
        ax_lab.text(0.02, 0.65, str(g_disp), ha="left", va="center", fontsize=12)
        ax_lab.text(0.05, 0.30, str(k_disp), ha="left", va="center", fontsize=11)

        # baseline row
        sel_b = pivot_before.xs((g, k), level=(lvl_g, lvl_k), drop_level=False)
        baseline_row = sel_b.iloc[0] if sel_b.shape[0] else None

        # after rows
        try:
            sel_a = pivot_after.xs((g, k), level=(lvl_g, lvl_k), drop_level=False)
        except Exception:
            sel_a = pivot_after.iloc[0:0]

        for c_i, m in enumerate(metrics):
            ax = fig.add_subplot(gs[r_i, c_i + 1])

            y0 = np.nan
            if baseline_row is not None and m in baseline_row.index:
                y0 = float(baseline_row[m])

            after_vals: Dict[str, float] = {}
            for h in hues:
                try:
                    v = sel_a.xs(h, level=hue.hue_level)[m]
                    if isinstance(v, pd.Series):
                        v = v.iloc[0]
                    after_vals[h] = float(v)
                except Exception:
                    continue

            _draw_fan_cell(
                ax=ax,
                y0=y0,
                after_values=after_vals,
                colors=color_for,
                global_ylim=global_ylim,
                style=style,
                hue=hue,
            )

            # Base/RAG only on last row
            if r_i == n_rows:
                ax.text(style.x_before, -0.12, style.xlabel_before, ha="center", va="top", fontsize=10)
                ax.text(style.x_after, -0.12, style.xlabel_after, ha="center", va="top", fontsize=10)

    

    if panel.group_separators and n_rows > 1:
        # x span: from left edge of first metric cell to right edge of last metric cell
        first_cell_ax = fig.axes[2]  # not reliable; compute robustly below

        # robust: get any axis from first data row, first metric col
        ax_left = fig.add_subplot(gs[1, 1])  # temporary handle to retrieve position
        ax_right = fig.add_subplot(gs[1, n_cols])  # temporary handle
        # immediately remove them (do not keep extra axes)
        fig.delaxes(ax_left)
        fig.delaxes(ax_right)

        # Use positions from existing axes in the grid:
        # Find first metric axis and last metric axis by scanning fig.axes for those in gs[1,1] etc is messy.
        # Easier: use the label axes list plus the last metric axis in that same row:
        # We get x0 from the first metric axis in row 1, and x1 from the last metric axis in row 1.
        # To do that, reconstruct their indices: each data row has (1 label + n_cols metric) axes, plus header axes.
        # Header axes count = 1 (corner) + n_cols (titles) = n_cols + 1
        header_axes = n_cols + 1

        # In each data row, axes were created in this order: label, then metrics left->right
        # So for first data row (r=0), first metric axis index is header_axes + 1
        first_metric_ax = fig.axes[header_axes + 1]
        last_metric_ax = fig.axes[header_axes + n_cols]

        x0 = first_metric_ax.get_position().x0
        x1 = last_metric_ax.get_position().x1

        # draw a line between rows whenever model_group changes
        for i in range(n_rows - 1):
            if row_groups[i] != row_groups[i + 1]:
                bb_top = row_label_axes[i].get_position()
                bb_bot = row_label_axes[i + 1].get_position()
                y = (bb_top.y0 + bb_bot.y1) / 2.0 + panel.group_sep_pad
                fig.add_artist(
                    Line2D(
                        [x0, x1],
                        [y, y],
                        transform=fig.transFigure,
                        lw=panel.group_sep_lw,
                        alpha=panel.group_sep_alpha,
                        color=panel.group_sep_color,
                        zorder=0,
                    )
                )


    # legend
    if legendspec.enabled:
        handles: List[Line2D] = []
        for h in hues:
            lab = hue.legend_label_map.get(h, h) if hue.legend_label_map else h
            handles.append(Line2D([0], [0], color=color_for[h], lw=hue.line_width, label=lab))

        fig.legend(
            handles=handles,
            loc=legendspec.loc,
            bbox_to_anchor=legendspec.bbox_to_anchor,
            ncol=(legendspec.ncol if legendspec.ncol is not None else max(1, len(hues))),
            frameon=legendspec.frameon,
            title=legendspec.title,
            fontsize=legendspec.fontsize,
            handlelength=legendspec.handlelength,
            columnspacing=legendspec.columnspacing,
        )


    fig.subplots_adjust(top=legendspec.top, wspace=panel.wspace, hspace=panel.hspace)
    # fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig


# ----------------------------
# Example usage
# ----------------------------
# fig = fanplot(
#     pivot_before=pivot_before,
#     pivot_after=pivot_after,
#     metrics=["refusal_pct", "validity_pct", "consistency", "duplicates", "factuality_author"],
#     panel=PanelSpec(
#         figsize=(18, 12),
#         group_label_map={"model_access": "Access", "model_class": "Reasoning", "model_size": "model_size"},
#         kind_label_map={"non-reasoning": "Disabled", "reasoning": "Enabled"},
#     ),
#     hue=HueSpec(
#         hue_order=[
#             "top_100_bias_gender_equal",
#             "top_100_bias_gender_female",
#             "top_100_bias_gender_male",
#             "top_100_bias_gender_neutral",
#         ],
#         legend_label_map={
#             "top_100_bias_gender_equal": "Equal",
#             "top_100_bias_gender_female": "Female",
#             "top_100_bias_gender_male": "Male",
#             "top_100_bias_gender_neutral": "Neutral",
#         },
#     ),
#     style=StyleSpec(
#         mid_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#         global_ylim=(0.0, 1.0),  # set explicitly if you want strict [0,1]
#     ),
# )
# plt.show()
