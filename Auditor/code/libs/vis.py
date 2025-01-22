from operator import le
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import numpy as np

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
    
    legend_kwargs = kwargs.get('legend_kwargs', {})
    legend_title = legend_kwargs.pop('title', group_col)
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
    if x_col and y_col:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if group_col and legend:
        plt.legend(title=legend_title, **legend_kwargs)

    # Adjust x-axis order if provided
    if x_order is not None:
        plt.xticks(ticks=np.arange(len(x_order)), labels=x_order)

    # Apply rotation to x-axis labels if specified
    if xlabel_rotation is not None:
        plt.xticks(rotation=xlabel_rotation)

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
    # ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)

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

    fig, ax = None, None
    for group in hue_order:
        df = data.query(f'{hue}==@group').copy()
        try:
            values = [df.query(f"{x_col}==@x").iloc[0][y_col] for x in x_order]
            fig, ax = _plot_spiderweb(x_order, values, label=group, color=hue_colors[group], fig=fig, ax=ax, 
                                        figsize=figsize, 
                                        ylim=ylim)
        except Exception as e:
            print(group, df.shape, e)

    _polish_plot(fig, ax, x_col=x_col, group_col=hue, group_order=hue_order, **kwargs)
    _finish_plot(fig, fn)