from libs.visuals.grid import *
from libs.visuals import grid


def plot_metric_grid_temperature_from_pivot(
    values: pd.DataFrame,
    *,
    panels: Sequence[PanelSpec],
    ci_low: Optional[pd.DataFrame] = None,
    ci_high: Optional[pd.DataFrame] = None,
    left_group_cols: Sequence[str],
    series_col: str,
    x_col: str,
    x_order: Optional[Sequence[Hashable]] = None,
    group_order: Optional[Sequence[Hashable]] = None,
    # labeling
    row_label_maps: Optional[Mapping[str, Mapping[Hashable, str]]] = None,
    series_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    # colors
    series_colors: Optional[Mapping[Hashable, object]] = None,
    default_series_color: object = "#1f77b4",
    series_order: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    default_series_order: Optional[Sequence[Hashable]] = None,
    layout: LayoutSpec = LayoutSpec(figsize=(18.0, 6.0), top=0.92, bottom=0.10),
    style: StyleSpec = StyleSpec(),
    # legend
    add_legend: bool = True,
    legend_panel: Optional[Sequence[Tuple[int, int]]] = None,
    legend_kwargs: Optional[Sequence[dict]] = None,
) -> plt.Figure:
    """
    Figure 3 style renderer using pivoted inputs.

    Contract:
      - values is a DataFrame whose index is a MultiIndex including levels:
          left_group_cols + [series_col, x_col]
        and whose columns contain all panel metric names.
      - ci_low / ci_high (optional) have the same shape/index/columns as values.
      - Within each facet row (left_group_cols), a line is drawn per series_col over x_col.
    """
    if not isinstance(values.index, pd.MultiIndex):
        raise ValueError("values must use a MultiIndex index.")

    need_levels = list(left_group_cols) + [series_col, x_col]
    if any(lv not in values.index.names for lv in need_levels):
        raise ValueError(
            "values.index must include levels "
            f"{need_levels}; got {values.index.names}"
        )

    if ci_low is not None and (ci_low.index.names != values.index.names):
        raise ValueError("ci_low must have the same MultiIndex level names as values.")
    if ci_high is not None and (ci_high.index.names != values.index.names):
        raise ValueError("ci_high must have the same MultiIndex level names as values.")

    metric_names = [p.metric_name for p in panels]
    missing_cols = [m for m in metric_names if m not in values.columns]
    if missing_cols:
        raise ValueError(f"values is missing required metric columns: {missing_cols}")

    # facet index in first-occurrence order
    idx_df = values.reset_index()[need_levels]
    facet_index = grid._unique_facet_index(idx_df, left_group_cols)

    if group_order is not None:
        # reorder by first left column only
        lvl0 = facet_index.names[0]
        order_map = {v: i for i, v in enumerate(group_order)}

        def _key(t):
            return order_map.get(t[0], float("inf"))

        facet_index = pd.MultiIndex.from_tuples(
            sorted(facet_index.to_list(), key=_key),
            names=facet_index.names,
        )
        
    nrows = len(facet_index)

    
    use_label_col = (len(left_group_cols) >= 2)
    n_metric_cols = len(panels)

    if use_label_col:
        ncols = n_metric_cols + 1
        width_ratios = [layout.label_ratio] + [layout.panel_ratio] * n_metric_cols
        label_col_offset = 1
    else:
        ncols = n_metric_cols
        width_ratios = [layout.panel_ratio] * n_metric_cols
        label_col_offset = 0

    fig = plt.figure(figsize=layout.figsize)
    gs = GridSpec(
        nrows=nrows, ncols=ncols, figure=fig,
        width_ratios=width_ratios,
        wspace=layout.wspace, hspace=layout.hspace
    )

    row_maps_level = grid._apply_row_label_maps_from_cols(facet_index, row_label_maps=row_label_maps)
    row_labels = [grid._facet_row_label(t, maps=row_maps_level) for t in facet_index.to_list()]

    # series order in appearance
    if default_series_order is not None:
        global_series = list(default_series_order)
    else:
        global_series = list(dict.fromkeys(values.index.get_level_values(series_col).tolist()))

    # x order
    x_vals_unique = list(dict.fromkeys(values.index.get_level_values(x_col).tolist()))
    if x_order is None:
        try:
            x_order = sorted(x_vals_unique)
        except Exception:
            x_order = x_vals_unique

    x_is_numeric = np.issubdtype(np.asarray(list(x_order)).dtype, np.number)
    x_plot = np.asarray(list(x_order), dtype=float) if x_is_numeric else np.arange(len(x_order))

    
    axes = [[None for _ in range(ncols)] for _ in range(nrows)]
    
    for i, facet in enumerate(facet_index.to_list()):

        row_key = facet[0]  # first left column value

        if series_order is not None and row_key in series_order:
            row_series = list(series_order[row_key])
        else:
            # discover series present in this row (and keep global order)
            present = set(
                values.xs(tuple(facet), level=list(left_group_cols), drop_level=False)
                    .index.get_level_values(series_col)
                    .unique()
                    .tolist()
            )
            row_series = [s for s in global_series if s in present]
            
        if use_label_col:
            ax_label = fig.add_subplot(gs[i, 0])
            axes[i][0] = ax_label
            ax_label.axis("off")
            ax_label.text(
                1.0, 0.5, row_labels[i],
                ha="right", va="center",
                fontsize=style.ylabel_fontsize,
                transform=ax_label.transAxes,
            )

        facet_key = {col: facet[k] for k, col in enumerate(left_group_cols)}

        for pj, p in enumerate(panels):
            col = pj + label_col_offset
            ax = fig.add_subplot(gs[i, col])
            axes[i][col] = ax

            if (not use_label_col) and (pj == 0):
                ax.set_ylabel(
                    row_labels[i],
                    fontsize=style.ylabel_fontsize,
                    # rotation=0,
                    labelpad=style.ylabel_pad,
                    va="center",
                )
                
            if i == 0:
                title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
                ax.set_title(title, fontsize=style.title_fontsize)

            for s in row_series:
                ys, lo, hi = [], [], []
                for xv in x_order:
                    idx = tuple([facet_key[c] for c in left_group_cols] + [s, xv])
                    try:
                        ys.append(values.loc[idx, p.metric_name])
                    except KeyError:
                        ys.append(np.nan)

                    if ci_low is not None and ci_high is not None:
                        try:
                            lo.append(ci_low.loc[idx, p.metric_name])
                            hi.append(ci_high.loc[idx, p.metric_name])
                        except KeyError:
                            lo.append(np.nan)
                            hi.append(np.nan)

                ys_arr = np.asarray(ys, dtype=float)
                if not np.any(np.isfinite(ys_arr)):
                    continue

                color = (series_colors.get(s, default_series_color) if series_colors else default_series_color)
                label = (series_label_map.get(s, str(s)) if series_label_map else str(s))

                ax.plot(x_plot, ys_arr, linewidth=style.line_width, color=color, label=label)
                ax.scatter(x_plot, ys_arr, s=style.point_size, color=color)

                if p.draw_ci and ci_low is not None and ci_high is not None:
                    lo_arr = np.asarray(lo, dtype=float)
                    hi_arr = np.asarray(hi, dtype=float)
                    if np.any(np.isfinite(lo_arr)) and np.any(np.isfinite(hi_arr)):
                        ax.fill_between(x_plot, lo_arr, hi_arr, color=color, alpha=0.18, linewidth=0)

            if p.ylim is not None:
                ax.set_ylim(p.ylim)
            if p.yticks is not None:
                ax.set_yticks(list(p.yticks))

            if x_is_numeric:
                if p.xticks is not None:
                    ax.set_xticks(list(p.xticks))
                if p.xlim is not None:
                    ax.set_xlim(p.xlim)
            else:
                ax.set_xticks(x_plot)
                ax.set_xticklabels([str(v) for v in x_order], fontsize=style.tick_fontsize)

            if style.grid_x:
                ax.grid(True, axis="x", linewidth=0.6, alpha=style.grid_alpha)
            if style.grid_y:
                ax.grid(True, axis="y", linewidth=0.6, alpha=style.grid_alpha)

            if style.show_xticklabels_only_bottom and (i != nrows - 1):
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel(x_col, fontsize=style.label_fontsize)

            is_first_metric_col = (pj == 0)
            if style.hide_y_ticks_nonfirstcol and (not is_first_metric_col):
                ax.set_yticklabels([])
                ax.set_ylabel("")

            ax.tick_params(labelsize=style.tick_fontsize)


    if add_legend and legend_panel is not None:
        if legend_kwargs is None:
            legend_kwargs = [dict() for _ in range(len(legend_panel))]
        if len(legend_kwargs) != len(legend_panel):
            raise ValueError("legend_kwargs must have same length as legend_panel.")

        # grab a single consistent set of handles/labels (when the legend is on the same panel for all rows)
        # ref_handles, ref_labels = None, None
        # for rr in range(nrows):
        #     for pj in range(len(panels)):
        #         ax_ref = axes[rr][pj + label_col_offset]
        #         h, l = ax_ref.get_legend_handles_labels()
        #         if h:
        #             ref_handles, ref_labels = h, l
        #             break
        #     if ref_handles:
        #         break

        for (r, pj), kw in zip(legend_panel, legend_kwargs):
            r = max(0, min(nrows - 1, r))
            pj = max(0, min(len(panels) - 1, pj))
            ax_leg = axes[r][pj + label_col_offset]
            handles, labels = ax_leg.get_legend_handles_labels()
            ax_leg.legend(
                handles,
                labels,
                fontsize=style.legend_fontsize,
                frameon=style.legend_frameon,
                **kw,
            )


    fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)
    return fig