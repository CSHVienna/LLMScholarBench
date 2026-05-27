"""
interventions.py

VISUALIZATION: the intervention dot-plot. For each model it draws one stacked
block - a row of panels, one per metric - showing each intervention's change
from baseline. All number-crunching (building per-metric trajectories, computing
deltas) lives in `helpers.py`; this module only configures and draws.

Model labelling is selectable via `model_grouping_kind`:
  - "separator" : a full-width coloured rule with the model name in the gap
                  (drawn above the model's row of panels).
  - "ylabel"    : a vertical coloured rectangle on the left with the model name
                  rotated 90 degrees in white (acts like a per-model y-label).

Quick start
-----------
>>> from helpers import build_metric_trajectory
>>> from interventions import MetricSpec, plot_intervention_effects
>>> trajectories = {m: build_metric_trajectory(...) for m in models}
>>> fig = plot_intervention_effects(
...     trajectories,
...     metrics=[MetricSpec("validity", "\u0394 Validity"),
...              MetricSpec("parity_gender", "\u0394 Gender")],
...     label_map={"rag_top_100": "RAG", "constrained_top_100": "CP"},
...     intervention_order=["RAG", "CP"],
...     model_grouping_kind="ylabel",
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from libs.visuals.helpers import BaselineDeltaComputer

__all__ = ["MetricSpec", "DotPlotConfig", "InterventionDotPlot",
           "plot_intervention_effects"]


# --------------------------------------------------------------------------- #
# Config (declarative spec for the figure)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricSpec:
    """One panel. `column` is the data column; `title` is the heading shown
    (decoupled so e.g. "parity_gender" can display as "\u0394 Gender")."""
    column: str
    title: str
    higher_is_better: Optional[bool] = None


@dataclass
class DotPlotConfig:
    """Everything needed to render the dot-plot. Only `metrics` is required."""
    metrics: Sequence[MetricSpec]

    experiment_col: str = "experiment"
    baseline_col: str = "is_baseline"

    label_map: Dict[str, str] = field(default_factory=dict)
    intervention_order: Optional[Sequence[str]] = None

    # display names for models (raw key -> label shown on the plot)
    model_map: Dict[str, str] = field(default_factory=dict)

    model_colors: Dict[str, str] = field(default_factory=dict)
    default_palette: Sequence[str] = ("#1a4f9c", "#c0392b", "#2e7d32",
                                      "#8e44ad", "#d35400")

    # how each model's block is labelled: "separator" | "ylabel"
    model_grouping_kind: str = "separator"
    # "ylabel" mode: the left colour band
    ylabel_band_width: float = 0.9        # inches (the column reserved for the band)
    ylabel_band_fill: float = 0.25        # fraction of that column the bar fills
    ylabel_band_radius: float = 0.12      # rounded-corner size
    ylabel_band_gap: float = 0.3         # GridSpec wspace before the panels
    ylabel_text_color: str = "white"
    # share the x-axis down each column: titles only on the top row, x-axis
    # (label + tick labels) only on the bottom row
    ylabel_share_x: bool = True

    xlabel: str = "Change (relative to baseline)"
    value_fmt: str = "{:+.2f}"
    xlim: Optional[Tuple[float, float]] = None
    xlim_pad_frac: float = 0.12
    xlim_min_positive: float = 0.2

    panel_width: float = 4.2
    panel_height: float = 3.0
    header_height: float = 0.55
    dot_size: float = 95.0
    zero_line_color: str = "#666666"
    grid_line_color: str = "#c9c9c9"
    title_fontsize: int = 13
    model_title_fontsize: int = 16
    label_fontsize: int = 11
    tick_fontsize: int = 11
    ytick_pad: float = 30.0                # nudge intervention labels left (avoid overlap)
    value_label_offset_frac: float = 0.02
    facecolor: str = "white"


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
class InterventionDotPlot:
    """Render one stacked dot-plot block per model, panels sharing x-limits."""

    def __init__(self, config: DotPlotConfig) -> None:
        self._cfg = config

    def _colors(self, models: Sequence[str]) -> Dict[str, str]:
        palette = cycle(self._cfg.default_palette)
        return {m: self._cfg.model_colors.get(m, next(palette)) for m in models}

    def _xlim(self, deltas: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        cfg = self._cfg
        if cfg.xlim is not None:
            return cfg.xlim
        vals = [d for f in deltas.values() for d in f["delta"].tolist()] + [0.0]
        lo, hi = min(vals), max(vals)
        pad = max(hi - lo, 1e-9) * cfg.xlim_pad_frac
        return lo - pad, max(hi + pad, cfg.xlim_min_positive)

    def _draw_panel(self, ax, frame, metric, color, xlim, *, show_ylabels,
                    show_title=True, show_xlabel=True, show_xticklabels=True) -> None:
        cfg = self._cfg
        sub = frame[frame["metric"] == metric.column]
        labels = list(sub["intervention"].cat.categories)
        y_pos = list(range(len(labels) - 1, -1, -1))     # first label on top
        lookup = dict(zip(sub["intervention"].astype(str), sub["delta"]))
        width = xlim[1] - xlim[0]

        ax.axvline(0.0, color=cfg.zero_line_color, lw=1.2, zorder=1)
        for label, y in zip(labels, y_pos):
            ax.axhline(y, color=cfg.grid_line_color, lw=0.8, ls=(0, (4, 4)), zorder=0)
            value = lookup.get(label, float("nan"))
            if value != value:                           # NaN guard
                continue
            value = 0.0 if abs(value) < 1e-9 else value
            ax.scatter([value], [y], s=cfg.dot_size, color=color, zorder=3,
                       edgecolors="white", linewidths=0.6)
            offset = width * cfg.value_label_offset_frac
            x_text, ha = (value - offset, "right") if value < 0 else (value + offset, "left")
            fvalue = cfg.value_fmt.format(value)
            fvalue = f" {fvalue}" if ha == "left" else f"{fvalue} "
            ax.text(x_text, y, fvalue, color=color, ha=ha,
                    va="center", fontsize=cfg.label_fontsize, fontweight="medium")

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if show_ylabels else [""] * len(labels), fontsize=cfg.tick_fontsize)
        # push the intervention labels a bit further left so they don't overlap
        # the leftmost panel's value labels / the colour band
        ax.tick_params(axis="y", length=0, pad=cfg.ytick_pad)
        ax.tick_params(axis="x", labelsize=cfg.tick_fontsize, labelbottom=show_xticklabels)
        if show_title:
            ax.set_title(metric.title, fontsize=cfg.title_fontsize, fontweight="bold", pad=8)
        if show_xlabel:
            ax.set_xlabel(cfg.xlabel, fontsize=cfg.tick_fontsize)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#999999")

    # -- model labels -------------------------------------------------------- #
    def _draw_header(self, ax, model, color) -> None:
        """'separator' mode: full-width coloured rule with the name in the gap."""
        cfg = self._cfg
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0.0, 1.0], [0.5, 0.5], color=color, lw=2.5, zorder=1)
        # white bbox masks the rule behind the title -> "line with a gap" look
        ax.text(0.5, 0.5, model, ha="center", va="center", color=color,
                fontsize=cfg.model_title_fontsize, fontweight="bold", zorder=2,
                bbox=dict(facecolor=cfg.facecolor, edgecolor="none", pad=6.0))

    def _draw_side_band(self, ax, model, color) -> None:
        """'ylabel' mode: a large vertical coloured bar with white rotated text.

        The bar is left-flush within its (column-0) axis and occupies only the
        left part of the column, leaving the rest of the column as clear space
        before the leftmost panel's intervention labels.
        """
        cfg = self._cfg
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        bar = FancyBboxPatch(
            (0.0, 0.03), cfg.ylabel_band_fill, 0.94,
            boxstyle=f"round,pad=0,rounding_size={cfg.ylabel_band_radius}",
            mutation_aspect=cfg.ylabel_band_width / cfg.panel_height,
            transform=ax.transAxes, facecolor=color, edgecolor="none",
            clip_on=False, zorder=1,
        )
        ax.add_patch(bar)
        ax.text(cfg.ylabel_band_fill / 2, 0.5, model, rotation=90,
                rotation_mode="anchor", ha="center", va="center",
                color=cfg.ylabel_text_color, fontsize=cfg.model_title_fontsize,
                fontweight="bold", zorder=2)

    # -- layout -------------------------------------------------------------- #
    def _build_grid(self, models, n_metrics, colors):
        """Create the figure + axes for the chosen grouping kind.

        Returns (fig, {model: [panel_ax per metric]}).
        """
        cfg = self._cfg
        n_models = len(models)
        panel_axes: Dict[str, list] = {}

        if cfg.model_grouping_kind == "separator":
            fig = plt.figure(
                figsize=(cfg.panel_width * n_metrics,
                         n_models * (cfg.header_height + cfg.panel_height)),
                facecolor=cfg.facecolor,
            )
            gs = GridSpec(n_models * 2, n_metrics, figure=fig,
                          height_ratios=[cfg.header_height, cfg.panel_height] * n_models,
                          hspace=0.55, wspace=0.18)
            for i, model in enumerate(models):
                self._draw_header(fig.add_subplot(gs[i * 2, :]),
                                  cfg.model_map.get(model, model), colors[model])
                panel_axes[model] = [fig.add_subplot(gs[i * 2 + 1, j]) for j in range(n_metrics)]
            fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.06)

        elif cfg.model_grouping_kind == "ylabel":
            fig = plt.figure(
                figsize=(cfg.ylabel_band_width + cfg.panel_width * n_metrics,
                         n_models * cfg.panel_height + cfg.header_height),
                facecolor=cfg.facecolor,
            )
            gs = GridSpec(n_models, n_metrics + 1, figure=fig,
                          width_ratios=[cfg.ylabel_band_width] + [cfg.panel_width] * n_metrics,
                          hspace=(0.18 if cfg.ylabel_share_x else 0.6),
                          wspace=cfg.ylabel_band_gap)
            for i, model in enumerate(models):
                self._draw_side_band(fig.add_subplot(gs[i, 0]),
                                     cfg.model_map.get(model, model), colors[model])
                panel_axes[model] = [fig.add_subplot(gs[i, j + 1]) for j in range(n_metrics)]
            # band flush to the far left; the wide column + gap clear the labels
            fig.subplots_adjust(left=-0.5, right=0.97, top=0.93, bottom=0.07)

        else:
            raise ValueError(
                f"model_grouping_kind must be 'separator' or 'ylabel', "
                f"got {cfg.model_grouping_kind!r}."
            )

        return fig, panel_axes

    def plot(self, deltas: Dict[str, pd.DataFrame]) -> Figure:
        cfg = self._cfg
        models = list(deltas.keys())
        n_models = len(models)
        n_metrics = len(cfg.metrics)
        xlim = self._xlim(deltas)
        colors = self._colors(models)

        fig, panel_axes = self._build_grid(models, n_metrics, colors)
        share_x = cfg.model_grouping_kind == "ylabel" and cfg.ylabel_share_x
        for i, model in enumerate(models):
            # when sharing x down the column: titles on the top row only,
            # x-axis (label + tick labels) on the bottom row only
            show_title = (i == 0) if share_x else True
            show_x = (i == n_models - 1) if share_x else True
            for j, metric in enumerate(cfg.metrics):
                self._draw_panel(panel_axes[model][j], deltas[model], metric,
                                 colors[model], xlim, show_ylabels=(j == 0),
                                 show_title=show_title, show_xlabel=show_x,
                                 show_xticklabels=show_x)
        return fig


# --------------------------------------------------------------------------- #
# Facade (deltas via helpers + figure in one call)
# --------------------------------------------------------------------------- #
def _auto_metrics(df: pd.DataFrame, experiment_col: str, baseline_col: str) -> List[MetricSpec]:
    """Infer metrics from numeric (non-bool) columns, excluding role columns."""
    skip = {experiment_col, baseline_col}
    return [MetricSpec(c, f"\u0394 {c.capitalize()}")
            for c in df.columns
            if c not in skip
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def plot_intervention_effects(
    trajectories: Dict[str, pd.DataFrame],
    metrics: Optional[Sequence[MetricSpec]] = None,
    *,
    save_path: Optional[str] = None,
    dpi: int = 200,
    config: Optional[DotPlotConfig] = None,
    **config_kwargs,
) -> Figure:
    """Build deltas (via `helpers.BaselineDeltaComputer`) and render in one call.
    If `metrics` is omitted it is inferred from the first trajectory's numeric
    columns."""
    if config is None:
        if metrics is None:
            first = next(iter(trajectories.values()))
            metrics = _auto_metrics(
                first,
                config_kwargs.get("experiment_col", "experiment"),
                config_kwargs.get("baseline_col", "is_baseline"),
            )
        config = DotPlotConfig(metrics=metrics, **config_kwargs)

    deltas = BaselineDeltaComputer(config).compute_all(trajectories)
    fig = InterventionDotPlot(config).plot(deltas)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=config.facecolor)
    return fig