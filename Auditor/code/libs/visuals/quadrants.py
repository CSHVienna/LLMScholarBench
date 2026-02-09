# quality_social_space.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Mapping, Tuple, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ThresholdMode = Literal["zero", "median", "mean", "quantile", "value"]


@dataclass(frozen=True)
class SpaceSpec:
    x: str = "technical_score"
    y: str = "social_score"
    id_col: str = "model"               # for labels (optional)
    hue: Optional[str] = None           # optional grouping column
    size: Optional[str] = None          # optional size column
    hue_colors: Optional[Dict[str, str]] = None

    # thresholds
    x_thr_mode: ThresholdMode = "median"
    y_thr_mode: ThresholdMode = "median"
    x_thr_value: Optional[float] = None
    y_thr_value: Optional[float] = None
    x_quantile: float = 0.5             # used if mode="quantile"
    y_quantile: float = 0.5

    # axes, normalization, and clipping
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    clip: bool = False                  # clip points outside xlim/ylim (visual only)

    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (0.5, 1.10)
    figsize: Optional[Tuple[float, float]] = (7.5, 6.0)
    annotation_pad: float = 0.01
    legend_kwargs: Optional[Mapping] = None

def _compute_threshold(s: pd.Series, mode: ThresholdMode, *, value: Optional[float], q: float) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    if mode == "zero":
        return 0.0
    if mode == "median":
        return float(s.median())
    if mode == "mean":
        return float(s.mean())
    if mode == "quantile":
        if not (0.0 < q < 1.0):
            raise ValueError("quantile must be in (0,1)")
        return float(s.quantile(q))
    if mode == "value":
        if value is None:
            raise ValueError("value threshold mode requires *_thr_value")
        return float(value)
    raise ValueError(f"Unknown threshold mode: {mode}")


def assign_quadrant(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_thr: float,
    y_thr: float,
    labels: Tuple[str, str, str, str] = ("high-high", "low-high", "low-low", "high-low"),
) -> pd.Series:
    """
    Returns quadrant labels for each row using:
      Q1 (top-right):  x >= x_thr and y >= y_thr
      Q2 (top-left):   x <  x_thr and y >= y_thr
      Q3 (bottom-left):x <  x_thr and y <  y_thr
      Q4 (bottom-right):x >= x_thr and y <  y_thr
    """
    xvals = df[x].to_numpy()
    yvals = df[y].to_numpy()

    q = np.empty(len(df), dtype=object)
    q[:] = None

    tr = (xvals >= x_thr) & (yvals >= y_thr)
    tl = (xvals <  x_thr) & (yvals >= y_thr)
    bl = (xvals <  x_thr) & (yvals <  y_thr)
    br = (xvals >= x_thr) & (yvals <  y_thr)

    q[tr] = labels[0]
    q[tl] = labels[1]
    q[bl] = labels[2]
    q[br] = labels[3]

    return pd.Series(q, index=df.index, name="quadrant")


def plot_quality_social_space(
    df: pd.DataFrame,
    *,
    spec: SpaceSpec = SpaceSpec(),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    label_points: bool = False,
    label_kwargs: Optional[Mapping] = None,
    jitter: float = 0.0,
    alpha: float = 0.85,
    marker_size: float = 45.0,
    density: Literal["none", "hexbin"] = "none",
    hexbin_gridsize: int = 25,
    show_quadrant_counts: bool = True,
    count_box_kwargs: Optional[Mapping] = None,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plots models in (technical, social) space with quadrant thresholds and counts.
    Returns (fig, ax, quadrant_counts_df).
    """

    if spec.x not in df.columns or spec.y not in df.columns:
        raise KeyError(f"df must contain columns {spec.x!r} and {spec.y!r}")

    d = df.copy()

    # thresholds
    x_thr = _compute_threshold(d[spec.x], spec.x_thr_mode, value=spec.x_thr_value, q=spec.x_quantile)
    y_thr = _compute_threshold(d[spec.y], spec.y_thr_mode, value=spec.y_thr_value, q=spec.y_quantile)

    # jitter (for overplotting)
    if jitter > 0:
        rng = np.random.default_rng(0)
        d[spec.x] = d[spec.x] + rng.normal(0.0, jitter, size=len(d))
        d[spec.y] = d[spec.y] + rng.normal(0.0, jitter, size=len(d))

    # quadrant assignment and counts
    d["quadrant"] = assign_quadrant(d, x=spec.x, y=spec.y, x_thr=x_thr, y_thr=y_thr)
    quad_counts = (
        d.dropna(subset=["quadrant"])
         .groupby("quadrant", dropna=False)
         .size()
         .rename("n")
         .reset_index()
         .sort_values("quadrant")
         .reset_index(drop=True)
    )

    # axes
    if ax is None:
        fig, ax = plt.subplots(figsize=spec.figsize)
    else:
        fig = ax.figure

    # optional clip
    if spec.xlim is not None:
        ax.set_xlim(*spec.xlim)
        if spec.clip:
            d = d[(d[spec.x] >= spec.xlim[0]) & (d[spec.x] <= spec.xlim[1])]
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
        if spec.clip:
            d = d[(d[spec.y] >= spec.ylim[0]) & (d[spec.y] <= spec.ylim[1])]

    # density background
    if density == "hexbin":
        ax.hexbin(d[spec.x], d[spec.y], gridsize=hexbin_gridsize, mincnt=1)

    # scatter, with optional hue groups
    if spec.hue is None:
        sizes = marker_size if spec.size is None else _size_from_col(d[spec.size], base=marker_size)
        # sizes = marker_size if spec.size is None else size_from_raw(d[spec.size], default=marker_size, scale=1.0)
        ax.scatter(d[spec.x], d[spec.y], s=sizes, alpha=alpha)
    else:
        if spec.hue not in d.columns:
            raise KeyError(f"hue column {spec.hue!r} not in df")
        for key, sub in d.groupby(spec.hue, dropna=False):
            sizes = marker_size if spec.size is None else _size_from_col(sub[spec.size], base=marker_size)
            # sizes = marker_size if spec.size is None else size_from_raw(sub[spec.size], default=marker_size, scale=1.0)
            ax.scatter(sub[spec.x], sub[spec.y], s=sizes, alpha=alpha, label=str(key), color=spec.hue_colors[key])

    # quadrant lines
    ax.axvline(x_thr, linewidth=1.2, color='grey', lw=1.0, ls='--')
    ax.axhline(y_thr, linewidth=1.2, color='grey', lw=1.0, ls='--')

    # labels
    ax.set_title(title)
    ax.set_xlabel(xlabel or spec.x)
    ax.set_ylabel(ylabel or spec.y)

    # point labels
    if label_points:
        if spec.id_col not in d.columns:
            raise KeyError(f"id_col {spec.id_col!r} not in df")
        kw = dict(fontsize=9)
        if label_kwargs:
            kw.update(dict(label_kwargs))
        for _, r in d.iterrows():
            if pd.notna(r[spec.x]) and pd.notna(r[spec.y]):
                ax.text(float(r[spec.x]) + spec.annotation_pad, float(r[spec.y]), str(r[spec.id_col]), **kw)

    # quadrant counts overlay
    if show_quadrant_counts:
        kw = dict(boxstyle="round", alpha=0.9)
        if count_box_kwargs:
            kw.update(dict(count_box_kwargs))

        # place counts near corners of current axis limits
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xm = x_thr
        ym = y_thr
        pad_x = 0.02 * (x1 - x0)
        pad_y = 0.02 * (y1 - y0)

        counts: Dict[str, int] = {row["quadrant"]: int(row["n"]) for _, row in quad_counts.iterrows()}
        # match default labels used in assign_quadrant
        tr = counts.get("high-high", 0)
        tl = counts.get("low-high", 0)
        bl = counts.get("low-low", 0)
        br = counts.get("high-low", 0)

        ax.text(x1 - pad_x, y1 - pad_y, f"high tech, high social: {tr}",
                ha="right", va="top", bbox=kw)
        ax.text(x0 + pad_x, y1 - pad_y, f"low tech, high social: {tl}",
                ha="left", va="top", bbox=kw)
        ax.text(x0 + pad_x, y0 + pad_y, f"low tech, low social: {bl}",
                ha="left", va="bottom", bbox=kw)
        ax.text(x1 - pad_x, y0 + pad_y, f"high tech, low social: {br}",
                ha="right", va="bottom", bbox=kw)

    if spec.hue is not None and legend:
        if spec.legend_kwargs is None:
            ax.legend(loc="upper center", bbox_to_anchor=spec.legend_bbox_to_anchor, ncol=_best_ncol(d[spec.hue]))
        else:
            ax.legend(**spec.legend_kwargs)

    return fig, ax, quad_counts


def _best_ncol(series: pd.Series) -> int:
    k = int(series.nunique(dropna=False))
    if k <= 1:
        return 1
    if k <= 3:
        return k
    if k <= 6:
        return 3
    return 4


def _size_from_col(s: pd.Series, *, base: float) -> np.ndarray:
    """Maps a numeric column to marker sizes in a stable way."""
    x = pd.to_numeric(s, errors="coerce").to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(len(s), base, dtype=float)

    lo, hi = np.nanpercentile(x[finite], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(len(s), base, dtype=float)

    z = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return base * (0.6 + 1.4 * z)

def size_from_raw(s: pd.Series, *, default: float = 45.0, scale: float = 1.0) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = default
    return scale * x