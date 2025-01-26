from operator import le
import attr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter

import seaborn as sns
import matplotlib.patches as mpatches
from libs import constants

DPI = 300

def sns_reset():
    sns.reset_orig()

def sns_paper_style():
    sns.set_context("paper", font_scale=1.51) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

def _finish_plot(fig, fn = None):
    plt.tight_layout()
    
    if fn is not None:
        fig.savefig(fn, dpi=DPI, bbox_inches='tight')

    plt.show()
    plt.close()

def _polish_plot(fig, ax, x_col=None, y_col=None, group_col=None, x_order=None, **kwargs):
    xlabel_rotation = kwargs.get('xlabel_rotation', None)
    xlabel = kwargs.get('xlabel', x_col)
    ylabel = kwargs.get('ylabel', y_col) 
    yscale = kwargs.get('yscale', 'linear')
    xscale = kwargs.get('xscale', 'linear')
    xticks = kwargs.get('xticks', False)
    xticklabels_rename = kwargs.pop('xticklabels_rename', None)
    ylim = kwargs.get('ylim', None)

    legend_kwargs = kwargs.get('legend_kwargs', {})
    # legend_title = legend_kwargs.pop('title', group_col)
    legend = kwargs.get('legend', True)

    ytickmax = kwargs.get('ytickval', None)
    if ytickmax:
        # Customize radial ticks
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # Ensure ticks are updated
        if ytickmax < 0.5:
            yticklabels = [f"{tick:.2f}"  for i, tick in enumerate(yticks)]
        else:
            yticklabels = [f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(yticks)]
        ax.set_yticklabels(yticklabels)  # Show every second tick label

    # Add labels and legend
    if x_col:
        ax.set_xlabel(xlabel)
        
    if y_col:
        ax.set_ylabel(ylabel)
    
    if xlabel or xlabel is None:
        ax.set_xlabel(xlabel)

    if ylabel or ylabel is None:
        ax.set_ylabel(ylabel)
    
    if xscale:
        ax.set_xscale(xscale)

    if yscale:
        ax.set_yscale(yscale)

    if (group_col and legend) or legend:
        # plt.legend(title=legend_title, **legend_kwargs)
        fig.legend(**legend_kwargs)

    # Adjust x-axis order if provided
    if x_order is not None:
        plt.xticks(ticks=np.arange(len(x_order)), labels=x_order)

    # Apply rotation to x-axis labels if specified
    if xlabel_rotation is not None:
        plt.xticks(rotation=xlabel_rotation)

    if xticks is None:
        ax.set_xticklabels([])

    if xticklabels_rename:
        xticks = ax.get_xticklabels()
        newxticks = [xticklabels_rename[t.get_text()] for t in xticks]
        ax.set_xticklabels(newxticks)
    
    if ylim:
        ax.set_ylim(ylim)

def plot_barplot(data, x_col, y_col, group_col=None, x_order=None, group_order=None, fn=None, **kwargs):
    """
    Plots a barplot using matplotlib.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - x_col (str): Column name for x-axis.
    - y_col (str): Column name for y-axis.
    - group_col (str, optional): Column name for grouping (to stack bars). Default is None.
    - x_order (list, optional): Order of categories for the x-axis. Default is None (no specific order).
    - group_order (list, optional): Order of group categories. Default is None (no specific order).
    - **kwargs: Additional keyword arguments:
        - xlabel_rotation (int, optional): Rotation angle for x-axis labels. Default is None (no rotation).
        - xlabel (str, optional): Custom label for the x-axis. Default is the x_col name.
        - ylabel (str, optional): Custom label for the y-axis. Default is the y_col name.
        - legend_title (str, optional): Custom title for the legend. Default is the group_col name.
        - colors (dict, optional): Dictionary mapping group names to colors. Default is None.
        - std_col (str, optional): Column name for the standard deviation values. Default is None.

    Returns:
    - None
    """
    
    df = data.copy()
    figsize = kwargs.get('figsize', (10,5))
    colors = kwargs.get('colors', None)
    err_col = kwargs.get('err_col', None)
    bar_width = kwargs.get('bar_width', 0.5)

    # Set x-axis order
    if x_order is not None:
        df[x_col] = pd.Categorical(df[x_col], categories=x_order, ordered=True)
        df = df.sort_values(by=x_col)

    # Set group order
    if group_col and group_order is not None:
        df[group_col] = pd.Categorical(df[group_col], categories=group_order, ordered=True)

    fig, ax = plt.subplots(1,1,figsize=figsize)

    if group_col:
        groups = group_order if group_order is not None else df[group_col].unique()
        x_positions = np.arange(len(df[x_col].unique()))

        bottom_values = np.zeros(len(x_positions))
        
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]
            if group_data.empty:
                continue
            group_data = group_data if x_order is None else group_data.sort_values(by=x_col).reset_index(drop=True)
            x_data = group_data.groupby(x_col, observed=False)[y_col].sum()
            err_data = group_data.groupby(x_col, observed=False)[err_col].sum() if err_col else None
            
            ax.bar(
                x=x_data.index,
                height=x_data,
                bottom=bottom_values,
                width=bar_width,
                label=str(group),
                color=colors[group] if colors and group in colors else None
            )

            if err_col:
                ax.errorbar(
                    x=x_data.index,
                    y=x_data + bottom_values,
                    yerr=err_data,
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

            bottom_values += x_data.reindex(x_order if x_order else x_data.index, fill_value=0).values
    else:
        # No grouping
        df = df if x_order is None else df.sort_values(by=x_col).reset_index(drop=True)
        y_data = df[y_col]
        err_data = df[err_col] if err_col else None

        ax.bar(
            x=y_data.index,
            height=y_data,
            width=bar_width,
        )

        if err_col:
            ax.errorbar(
                x=y_data.index,
                y=y_data,
                yerr=err_data,
                fmt='none',
                ecolor='black',
                capsize=5
            )

    # Display the plot
    _polish_plot(fig, ax, x_col, y_col, group_col, x_order, **kwargs)
    _finish_plot(fig, fn)



def _plot_spiderweb(categories, values, label=None, color='blue', fig=None, ax=None, **kwargs):
    """
    Plots a spiderweb (radar) plot.

    Parameters:
    - categories (list): List of category names.
    - values (list): List of values corresponding to the categories.
    - title (str, optional): Title of the plot. Default is "".
    - color (str, optional): Color for the plot. Default is 'blue'.

    Returns:
    - None
    """
    
    figsize = kwargs.get('figsize', (10,5))
    title = kwargs.get('title', '')
    ylim = kwargs.get('ylim', None)

    # Ensure the data is circular (start and end points match)
    categories = list(categories)
    values = list(values)
    values += values[:1]

    # Calculate angles for the axes
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Create the plot
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(polar=True))
        
    # Draw the spiderweb
    ax.fill(angles, values, color=color, alpha=0.1)
    ax.plot(angles, values, label=label, color=color, linewidth=2)

    # Style grid
    ax.set_rscale('linear')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, ha='center')
    
    # Remove polar labels
    # ax.yaxis.grid(True)

    # Add category labels
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories)

    # Set radial limit
    ax.set_ylim(ylim)

    # Customize circular gridlines
    #ax.yaxis.grid(color='gray', linestyle='solid', linewidth=0.4)  # Lighter gridlines
    
    # Remove the outermost border (spine)
    ax.spines['polar'].set_visible(False)
    
    return fig, ax

def plot_spiderweb(data, x_col, x_order, y_col, hue, hue_order, hue_colors, fn=None, **kwargs):
    
    figsize = kwargs.get('figsize', (10,5))
    ylim = kwargs.get('ylim', (-0.05,1.05))
    df_data = data.copy()

    fig, ax = None, None
    for group in hue_order:
        df = df_data.query(f'{hue}==@group').copy()
        try:
            values = [df.query(f"{x_col}==@x").iloc[0][y_col] for x in x_order]
            fig, ax = _plot_spiderweb(x_order, values, label=group, color=hue_colors[group], fig=fig, ax=ax, 
                                        figsize=figsize, 
                                        ylim=ylim)
            
            # ax.set_xticklabels(x_order, fontsize=10, ha='center')
        except Exception as e:
            print(group, df.shape, e)

    # _polish_plot(fig, ax, x_col=x_col, group_col=hue, group_order=hue_order, x_order=x_order, **kwargs)
    _finish_plot(fig, fn)


def plot_parallel_coords(df, hue, hue_order, hue_colors, df_err=None, fn=None, **kwargs):

    figsize = kwargs.get('figsize', (10,5))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    baselines = kwargs.pop('baselines', None)
    
    
    parallel_coordinates(df, hue, color=[hue_colors[m] for m in hue_order], axvlines=False, ax=ax, lw=2,)
    plt.gca().legend_.remove()

    # Add error shading
    if df_err is not None and hue:
        for group, tmp in df_err.groupby(hue):

            if tmp.empty:
                continue
            
            v_err = tmp.drop(columns=hue).T
            index = v_err.index.to_list()
            v_val = df.set_index(hue).loc[group].loc[index].values.astype(float)
            v_err = v_err.values.reshape(1,-1)[0].astype(float)

            ax.fill_between(
                index,
                v_val - v_err,  # Lower bound
                v_val + v_err,  # Upper bound
                color=hue_colors[group],
                alpha=0.1,  # Transparency
            )


    if baselines is not None:
        for bname, obj in baselines.items():
            baseline = obj['values']
            color = obj.get('color', 'black')
            ls = obj.get('ls', '-')
            x = baseline.index.categories.to_list()
            y = [baseline[i] if i in baseline else None for i in x ]
            ax.plot(x, y, color=color, linestyle=ls, lw=1.5, label=bname)

    
    ax.grid(axis='y')

    _polish_plot(fig, ax, x_col=None, group_col=hue, group_order=hue_order, x_order=df.columns[1:], **kwargs)
    _finish_plot(fig, fn)


def plot_components_by_model_including_population(results, colors, col_order, fn=None, **kwargs):
    
    colors.update({'APS': constants.COMPONENT_POPULATION_COLOR})

    ncols = len(col_order)
    nrows = 1
    w = 2
    h = 2
    figsize = (ncols * w, nrows * h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    for i, model in enumerate(col_order):
        obj = results.get(model, None)

        if obj is None:
            continue

        r = i//ncols
        c = i%ncols
        ax = axes[r,c] if nrows > 1 else axes[c]
        
        combined_data = obj['reduction']

        y = 1.0
        for label, data in combined_data.groupby('label'):    
            col = colors[label]    
            ax.scatter(data['dim1'], data['dim2'], label=model if label==model else None, 
                    alpha=1 if label==model else 0.5, 
                    color=col, rasterized=True)
            
            pop = data.id_author_oa.nunique()
            ax.text(s=f"{pop} people", x=0.6, y=y, color=col,
                    fontsize=8, ha='right', va='top', transform=ax.transAxes) 
            
            y -= 0.08

        ax.set_title(model)

        for artist in ax.artists:
            artist.set_rasterized(True)
        #ax.set_rasterized(True)

        if kwargs.get('xlabel', True):
            ax.set_xlabel('PC 1')

        y_col = 'PC2' if c==0 else ''

        _polish_plot(fig, ax, x_col=None, y_col=y_col, group_col=None, group_order=None, **kwargs)

    plt.subplots_adjust(wspace=0.0, hspace=0.01)
    _finish_plot(fig, fn)
 


def plot_components_by_model_and_param_value_including_population(results, col_order, hue_order, fn=None, **kwargs):
    
    colors = constants.COMPPONENT_TASK_PARAM_COLORS
    color_map = {task_param: colors[i] for i, task_param in enumerate(hue_order)}

    markers = constants.COMPPONENT_TASK_PARAM_MARKERS
    marker_map = {task_param: markers[i] for i, task_param in enumerate(hue_order)}

    annotated_flag = {model:{} for model in col_order}

    for mode in col_order:
        annotated_flag[mode] = {task_param: False for task_param in hue_order}

    xticks = kwargs.get('xticks', False)
    title = kwargs.get('title', None)

    ncols = len(col_order)
    nrows = 1
    w = 2
    h = 1.7 if xticks is None else 2.2
    figsize = (ncols * w, nrows * h)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    for (model, task_name, task_param), obj in results.items():
        
        c = col_order.index(model)

        ax = axes[c]

        if obj is not None:

            combined_data = obj['reduction']

            sample_size = combined_data.query('label != "APS"').groupby('task_param', observed=False).size()
            
            for label, data in combined_data.groupby('label', observed=True): 
                if label == 'APS':
                    ax.scatter(data['dim1'], data['dim2'], alpha=0.1, color=constants.COMPONENT_POPULATION_COLOR, zorder=-1, rasterized=True) 
                    
                else:
                    
                    for i, (task_param, df) in enumerate(data.groupby('task_param', observed=False)):
                        
                        color = color_map[task_param]
                        marker = marker_map[task_param]

                        ax.scatter(df['dim1'], df['dim2'], 
                                label=task_param, 
                                color=color,
                                alpha=0.5,
                                marker=marker,
                                zorder=1/sample_size[task_param], # the smallest group on top
                                rasterized=True
                                )

                        y = 0.01 + hue_order.index(task_param) * 0.08
                        ax.text(s=f"{df.shape[0]} people", x=0.6, y=y, color=color,fontsize=8, ha='right', va='bottom', transform=ax.transAxes) 
                        annotated_flag[model][task_param] = True

        for artist in ax.artists:
            artist.set_rasterized(True)
        #ax.set_rasterized(True)

    
    # if param was not found in the data
    for model, obj_a in annotated_flag.items():
        c = col_order.index(model)
        ax = axes[c]

        if title:
            ax.set_title(model)

        y_col = 'PC2' if c==0 else False
        kwargs['legend'] = False    
        _polish_plot(fig, ax, x_col='PC1', y_col=y_col, group_col=None, group_order=None, **kwargs)

        for task_param, flag in obj_a.items():
            if not flag:
                color = color_map[task_param]
                y = 0.01 + hue_order.index(task_param) * 0.08
                ax.text(s=f"0 people", x=0.6, y=y, color=color,fontsize=8, ha='right', va='bottom', transform=ax.transAxes) 
   

    # add legend
    c = 2
    custom_legend = [Line2D([0], [0], 
                            color=color, marker=marker_map[label], 
                            markersize=6, 
                            label=label.split('_')[-1].replace('career','early_career') if not label.startswith('top_') else label, 
                            linestyle="") for label, color in color_map.items()] 
    axes[c].legend(
        handles=custom_legend,  # Add custom legend handles
        loc="upper right",      # Position at the top-right corner
        frameon=True,           # Add a frame around the legend
        borderpad=0.2,          # Adjust padding inside the legend box
        handlelength=0.5,       # Adjust the length of the colored marker
        fontsize="small"        # Adjust text size
    )

    plt.subplots_adjust(wspace=0.0, hspace=0.01)
    _finish_plot(fig, fn)



def plot_error_box_plot_comparison(df_sample, df_full, metric, fn=None, **kwargs):

    figsize = kwargs.get('figsize', (5, 4))
    group_col = 'model'
    categories_yaxis = constants.LLMS
    colors_categories_yaxis = constants.LLMS_COLORS   
    yticks = kwargs.get('yticks', True)

    #########################################
    # Filter out NaN values
    data_llm = df_sample.copy()
    data_llm.dropna(subset=[metric], inplace=True)

    data_gt = df_full.copy()
    data_gt.dropna(subset=[metric], inplace=True)

    #########################################
    # Create the figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.0)

    #########################################
    # Top Panel: Population Distribution
    ax_top = fig.add_subplot(gs[0])
    sns.boxplot(data=data_gt,  x=metric, ax=ax_top, color='lightgray')
    if yticks:
        ax_top.set_ylabel('APS')
    for artist in ax_top.artists:  # Rasterize individual boxes
        artist.set_rasterized(True)
    ax_top.set_rasterized(True)
    ax_top.set_xlabel('')
    ax_top.set_yticks([])
    ax_top.axes.get_xaxis().set_visible(False)

    #########################################
    # Main Panel: Violin Plot or Box Plot
    ax_main = fig.add_subplot(gs[1], sharex=ax_top)
    sns.boxplot(data=data_llm, y=group_col, x=metric, ax=ax_main, hue=group_col, palette=constants.LLMS_COLORS)
    ax_main.set_xlabel(metric)
    ax_main.set_ylabel('')
    if yticks:
        ax_main.set_yticklabels(categories_yaxis)
    else:
        ax_main.set_yticklabels([])
    
    #########################################
    # legend
    legend = kwargs.get('legend', False)
    colors_categories_yaxis.update({'APS': 'lightgray'})
    if legend:
        legend_patches = [mpatches.Patch(color=colors_categories_yaxis[g], label=g) for g in  ['APS'] + categories_yaxis]
        # ax_main.legend(
        #         handles=legend_patches,  # Add custom legend handles
        #         loc="upper right",      # Position at the top-right corner
        #         frameon=True,           # Add a frame around the legend
        #         borderpad=0.4,          # Adjust padding inside the legend box
        #         handlelength=0.6,       # Adjust the length of the colored marker
        #         fontsize="small"        # Adjust text size
        # )
        legend_kwargs = kwargs.get('legend_kwargs', {'loc': 'upper right', 'frameon': True, 'borderpad': 0.4, 'handlelength': 0.6, 'fontsize': 'small'})    
        legend_kwargs['handles'] = legend_patches
        fig.legend(**legend_kwargs)

    # Adjust aesthetics
    ax_top.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.set_xscale('log')
    for artist in ax_main.artists:  # Rasterize individual boxes
        artist.set_rasterized(True)
    ax_main.set_rasterized(True)
    ax_main.set_xlabel('')
    ax_main.set_yticks([])

    #########################################
    _finish_plot(fig, fn)




def plot_task_param_comparison_bars(df_result, metric, all_labels, model, color_map_attribute, fn=None, **kwargs):
    

    # Step 1: Group and Aggregate Counts
    grouped = df_result.query("model==@model").groupby(["task_name", "task_param", "attribute_label"]).sum().reset_index()

    # Step 2: Prepare Data for Plotting
    # Pivot data to organize counts by task_name and task_param/gender
    pivoted = grouped.pivot_table(
        index="task_name",
        columns=["ax", "attribute_label"],
        values=metric,
        fill_value=0,
    )

    # Define the unique task_names and task_params
    task_names = df_result.task_name.unique()
    task_params = [0,1]

    # sort index
    pivoted = pivoted.reindex(task_names)

    #################################################
    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5), gridspec_kw={'width_ratios': [2.5, 2.5, 2.5]})
    #################################################

    #################################################
    # Left plot: First task_param
    #################################################
    task_param1 = task_params[0]
    bottom = np.zeros(len(task_names))  # Initialize stacking
    for label in all_labels:
        if (task_param1, label) in pivoted.columns:
            axes[0].barh(
                task_names,
                pivoted[(task_param1, label)].apply(lambda x: -x),  # Invert to stack from top to bottom
                # label=gender,
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param1, label)].apply(lambda x: -x)  # Invert to stack from top to bottom
    axes[0].invert_yaxis()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)


    #################################################
    # Middle subplot: Names
    #################################################
    axes[1].set_yticks([])
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].invert_yaxis()  # Match the left subplot
    axes[1].set_xlim(-1, 1)

    y = 0.1035
    for i, name in enumerate(task_names):
        axes[1].text(s=name, x=0, y=0.09 + (i*y), ha='center', va='center', color='black')


    #################################################
    # Right plot: Second task_param
    #################################################
    task_param2 = task_params[1]
    bottom = np.zeros(len(task_names))  # Reset stacking for the second subplot
    for label in all_labels:
        if (task_param2, label) in pivoted.columns:
            axes[2].barh(
                task_names,
                pivoted[(task_param2, label)],
                label=label.replace('Unisex','Non-binary').replace(constants.ETHNICITY_BLACK,'Black').replace(constants.ETHNICITY_LATINO,'Latino'),
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param2, label)]
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)


    #################################################
    # Y-ticks
    #################################################
    # Synchronize x-axis limits
    if metric == 'counts':
        max_x = max(axes[0].get_xlim()[0]*-1, axes[2].get_xlim()[1])
    else:
        max_x = 1
    axes[0].set_xlim(-max_x,0)
    axes[2].set_xlim(0, max_x)
     

    for i in [0,1]:
        j = i * 2
        ax = axes[j]

        # add task_param values
        y_ticks = [df_result.groupby(['ax','task_name']).task_param.unique()[i,task_name][0].replace(f"{task_name.split('_')[-1]}_",'') for task_name in task_names]
        y_ticks = [t.replace('male','(male)').replace('fe(male)','(female)') for t in y_ticks]
        ax = axes[j]
        if i == 1:
            ax2 = ax.twinx()
            ax2.set_yticks(range(task_names.shape[0]))
            ax2.set_yticklabels(y_ticks, color='grey')
            ax2.invert_yaxis()
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.set_ylim(ax.get_ylim())
        else:
            ax.set_yticks(range(task_names.shape[0]))
            ax.set_yticklabels(y_ticks, color='grey')
        
        # remove sign
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs((tick))) if max_x==1 else str(abs(int(tick))) for tick in ticks])

    # Adjust subplot appearance
    xlabel = "Fraction" if metric=='percentage' else "Unique counts"
    axes[0].set_xlabel(xlabel)
    axes[2].set_xlabel(xlabel)

    bbox = (0.06, 0.93, 0.5, 0.5) if constants.GENDER_FEMALE in all_labels else [0.0, 0.93, 0.5, 0.5]
    ncols = 5  
    fig.legend(loc="lower left", ncols=ncols, bbox_to_anchor=bbox)

    # Overall layout
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots
    _finish_plot(fig, fn)



def plot_task_param_comparison_line(df_result, col_val, col_err, model, fn=None, **kwargs):
    

    # Step 1: Group and Aggregate Counts
    grouped = df_result.query("model==@model").groupby(["task_name", "task_param", "attribute_label"]).sum().reset_index()

    # Step 2: Prepare Data for Plotting
    # Pivot data to organize counts by task_name and task_param/gender
    pivoted_vals = grouped.pivot_table(
        index="task_name",
        columns=["ax", "attribute_label"],
        values=col_val,
        fill_value=0,
    )

    pivoted_errs = [None,None]
    if col_err:
        pivoted_errs = grouped.pivot_table(
            index="task_name",
            columns=["ax", "attribute_label"],
            values=col_err,
            fill_value=0,
        )

    # Define the unique task_names and task_params
    task_names = df_result.task_name.unique()

    # sort index
    pivoted_vals = pivoted_vals.reindex(task_names)
    pivoted_errs = pivoted_errs.reindex(task_names) if pivoted_errs is not None else None
    
    #################################################
    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5), gridspec_kw={'width_ratios': [1, 5, 5]})

    #################################################

    #################################################
    # Left plot: First task_param
    #################################################

    index = 0
    axes[1].errorbar(np.squeeze(pivoted_vals[index].values), task_names, xerr=np.squeeze(pivoted_errs[index].values), fmt='o', color='red', label="Mean ± SD")
    axes[1].invert_yaxis()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_yticks([])

    #################################################
    # Middle subplot: Names
    #################################################
    axes[0].set_yticks([])
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].invert_yaxis()  # Match the left subplot
    axes[0].set_xlim(-1, 1)

    y = 0.1035
    for i, name in enumerate(task_names):
        axes[0].text(s=name, x=0, y=0.09 + (i*y), ha='right', va='center', color='black')


    #################################################
    # Right plot: Second task_param
    #################################################
    index = 1
    axes[2].errorbar(np.squeeze(pivoted_vals[index].values), task_names, xerr=np.squeeze(pivoted_errs[index].values), fmt='o', color='red', label="Mean ± SD")
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)


    #################################################
    # Y-ticks
    #################################################
    # Synchronize x-axis limits
    max_x = max(axes[1].get_xlim()[1], axes[2].get_xlim()[1])
    min_x = min(axes[1].get_xlim()[0], axes[2].get_xlim()[0])
    smooth = 0
    axes[1].set_xlim(min_x-smooth, max_x+smooth)
    axes[2].set_xlim(min_x-smooth, max_x+smooth)
     

    for i in [0,1]:
        j = i + 1
        ax = axes[j]

        # add task_param values
        y_ticks = [df_result.groupby(['ax','task_name']).task_param.unique()[i,task_name][0].replace(f"{task_name.split('_')[-1]}_",'') for task_name in task_names]
        y_ticks = [t.replace('male','(male)').replace('fe(male)','(female)') for t in y_ticks]
        
        ax2 = ax.twinx()
        ax2.set_yticks(range(task_names.shape[0]))
        ax2.set_yticklabels(y_ticks, color='grey')
        ax2.invert_yaxis()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_ylim(ax.get_ylim())
    

    # Adjust subplot appearance
    xlabel = "Mean ± SD"
    axes[1].set_xlabel(xlabel)
    axes[2].set_xlabel(xlabel)

    # Overall layout
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots
    _finish_plot(fig, fn)




def plot_gt_demographics(df_gt_stats, attribute, fn=None, **kwargs):
    from postprocessing import bias

    figsize = kwargs.get('figsize', (10, 5))

    df_c = bias.get_data_by_attribute_and_metric(df_gt_stats, attribute, 'counts')
    df_f = bias.get_data_by_attribute_and_metric(df_gt_stats, attribute, 'fractions')

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=False)
    colors = constants.GENDER_COLOR_DICT.values() if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER else constants.ETHNICITY_COLOR_DICT.values()
    
    # counts
    ax = axes[0]
    ax = df_c.plot.barh(stacked=True,  color=colors, ax=ax, legend=False)   
    ax.set_ylabel('')

    # Customize specific labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER:
        custom_labels = {
            "Female": "Female",
            "Male": "Male",
            "Unisex": "Non-binary",
            "Unknown": "Unknown",
        }
    else:
        custom_labels = {
        "Black or African American": "Black",
        "Asian": "Asian",
        "White": "White",
        "Hispanic or Latino": "Latino",
        "Unknown": "Unknown",
    }
    new_labels = [custom_labels.get(label, label) for label in labels]
    ax.legend(handles, new_labels)


    # fractions
    ax = axes[1]
    ax = df_f.plot.barh(stacked=True,  color=colors, ax=ax, legend=False)
    ax.set_ylabel('')

    # Make specific ytick labels bold
    bold_labels = ["Physics Education Research", "Condensed Matter & Materials Physics"]  # Specify labels to make bold
    for label in axes[0].get_yticklabels():
        if label.get_text() in bold_labels:
            label.set_fontweight("bold")  # Make the label bold


    # Layout adjustments
    _finish_plot(fig, fn)