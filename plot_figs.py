# Plot figures for thesis project - SR cores

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from grainsize import GrainSize
from grainsize import XRF
from grainsize import Stratigraphy
from matplotlib.gridspec import GridSpec


def plot_combined_subfigures(stratigraphy_obj, grain_size_obj, strat_kwargs={}, stats_kwargs={}, figsize=(25, 18),
                             savefig=False, savepath="combined_plot.png", dpi=350):
    # Create the main figure
    main_fig = plt.figure(layout="constrained", figsize=figsize)

    # Create subfigures with adjusted width ratios
    subfigs = main_fig.subfigures(1, 2, width_ratios=[1, 7], wspace=0.01)


    # Plot stratigraphy in the first subfigure
    strat_fig, _ = stratigraphy_obj.plot_stratigraphy(**strat_kwargs)
    strat_ax = subfigs[0].subplots(1, 1)
    strat_ax.remove()
    strat_ax = subfigs[0].add_subplot(1, 1, 1)

    for params in stratigraphy_obj.fill_params:
        strat_ax.fill_betweenx([params['top'], params['bottom']], 0, 1, color=params['color'])

    for ax in strat_fig.get_axes():
        for line in ax.get_lines():
            strat_ax.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())
        for text in ax.texts:
            strat_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(), va=text.get_va(), ha=text.get_ha(), fontsize=text.get_fontsize(), color=text.get_color())
        strat_ax.set_xlim(ax.get_xlim())
        strat_ax.set_ylim(ax.get_ylim())
        strat_ax.set_title("Units")
        # strat_ax.set_title(ax.get_title())
        strat_ax.set_xlabel(ax.get_xlabel())
        strat_ax.set_ylabel(ax.get_ylabel())
        strat_ax.set_xticks(ax.get_xticks())
        strat_ax.set_yticks(ax.get_yticks())
        # eliminate visibility of x-axis ticks and label
        strat_ax.xaxis.label.set_color("white")
        strat_ax.tick_params(axis="x", colors="white")

    # Plot grain size statistics in the second subfigure
    num_stats_axes = len(grain_size_obj.dataframe.columns) - 1  # Number of columns to plot
    stats_axes = subfigs[1].subplots(1, num_stats_axes, sharey=strat_ax)
    stats_fig, _ = grain_size_obj.plot_stats_fines(**stats_kwargs)

    for i, ax in enumerate(stats_fig.get_axes()[:num_stats_axes]):
        for line in ax.get_lines():
            stats_axes[i].plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())
        stats_axes[i].set_xlim(ax.get_xlim())
        stats_axes[i].set_ylim(ax.get_ylim())
        # set y axis limits explicitly
        # stats_axes[i].set_ylim(y_min, y_max)
        stats_axes[i].set_title(ax.get_title())
        stats_axes[i].set_xlabel(ax.get_xlabel())
        stats_axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        stats_axes[i].grid(True)

    # set suptitle for main figure
    main_fig.suptitle(f"{strat_kwargs['core_name']} - Grain Size", fontsize=24)

    # Adjust layout
    plt.tight_layout()

    if savefig:
        main_fig.savefig(savepath, dpi=dpi)

    plt.show()





# Usage for SR19-P4 stratigraphy and grain size data
path_strat = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\stratigraphy\sr19_strat.csv"
sr19_strat = Stratigraphy(dataframe=pd.read_csv(path_strat, header=0))


#
# path_stats = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\full_stats\sr19_stats_fines.csv"
# sr19_stats = GrainSize(dataframe=pd.read_csv(path_stats))
#
# plot_combined_subfigures(sr19_strat, sr19_stats,
#                          strat_kwargs={"core_name": "SR19-P4", "figsize": (3, 18)},
#                          stats_kwargs={"core_name": "SR19-P4", "marker": ".", "linestyle": "-"},
#                          savefig=False,
#                          savepath="sr19_gs_stats_combined.png"
#                          )

# Usage for SR21-P7 stratigraphy and grain size data
path_strat = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\stratigraphy\sr21_strat.csv"
sr21_strat = Stratigraphy(dataframe=pd.read_csv(path_strat, header=0))

path_stats = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\full_stats\sr21_stats_fines.csv"
sr21_stats = GrainSize(dataframe=pd.read_csv(path_stats))

plot_combined_subfigures(sr21_strat, sr21_stats,
                         strat_kwargs={"core_name": "SR21-P7", "figsize": (3, 18)},
                         stats_kwargs={"core_name": "SR21-P7", "marker": ".", "linestyle": "-"},
                         savefig=False,
                         savepath="sr21_gs_stats_combined.png"
                         )

def plot_combine_xrf(stratigraphy_obj, xrf_obj, strat_kwargs={}, xrf_kwargs={}, figsize=(25, 18),
                             savefig=False, savepath="combined_plot.png", dpi=350):
    # Create the main figure
    main_fig = plt.figure(layout="constrained", figsize=figsize)

    # Create subfigures with adjusted width ratios
    subfigs = main_fig.subfigures(1, 2, width_ratios=[1, 7], wspace=0.01)


    # Plot stratigraphy in the first subfigure
    strat_fig, _ = stratigraphy_obj.plot_stratigraphy(**strat_kwargs)
    strat_ax = subfigs[0].subplots(1, 1)
    strat_ax.remove()
    strat_ax = subfigs[0].add_subplot(1, 1, 1)

    for params in stratigraphy_obj.fill_params:
        strat_ax.fill_betweenx([params['top'], params['bottom']], 0, 1, color=params['color'])

    for ax in strat_fig.get_axes():
        for line in ax.get_lines():
            strat_ax.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())
        for text in ax.texts:
            strat_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(), va=text.get_va(), ha=text.get_ha(), fontsize=text.get_fontsize(), color=text.get_color())
        strat_ax.set_xlim(ax.get_xlim())
        strat_ax.set_ylim(ax.get_ylim())
        strat_ax.set_title("Units", fontsize=14)
        # strat_ax.set_title(ax.get_title())
        # strat_ax.set_xlabel(ax.get_xlabel())
        strat_ax.set_ylabel(ax.get_ylabel())
        strat_ax.set_xticks(ax.get_xticks())
        strat_ax.set_yticks(ax.get_yticks())
        # eliminate visibility of x-axis ticks and label
        strat_ax.xaxis.label.set_color("white")
        strat_ax.tick_params(axis="x", colors="white")

    # Plot XRF ratios in the second subfigure
    ratio_list = xrf_kwargs.pop("ratio_list", [])
    num_xrf_axes = len(ratio_list)  # Number of columns to plot
    xrf_axes = subfigs[1].subplots(1, num_xrf_axes, sharey=strat_ax)
    xrf_fig, _ = xrf_obj.plot_ratios(ratio_list=ratio_list, **xrf_kwargs)

    for i, ax in enumerate(xrf_fig.get_axes()[:num_xrf_axes]):
        for line in ax.get_lines():
            xrf_axes[i].plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())
        xrf_axes[i].set_xlim(ax.get_xlim())
        xrf_axes[i].set_ylim(ax.get_ylim())
        xrf_axes[i].set_title(ax.get_title())
        xrf_axes[i].set_xlabel(ax.get_xlabel())
        xrf_axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        xrf_axes[i].grid(True)

    # set suptitle for main figure
    main_fig.suptitle(f"{xrf_kwargs['core_name']} - XRF Elemental Ratios", fontsize=24)

    # Adjust layout
    plt.tight_layout()

    if savefig:
        main_fig.savefig(savepath, dpi=dpi)

    plt.show()


sr21_strat_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\stratigraphy\sr21_strat.csv"
sr21_strat = Stratigraphy(dataframe=pd.read_csv(sr21_strat_path, header=0))

sr21_ppm = XRF(dataframe=pd.read_csv("sr21_ppm.csv", header=0))
ratios = [("Ti", "Ca"), ("Ti", "Al"),
          ("Ba", "Al"), ("Cr", "Al"),
          ("K", "Al"), ("Ca", "Al"),
          ("Si", "Al"), ("Fe", "Si")]

# plot combined

# plot_combine_xrf(sr21_strat,
#                  sr21_ppm,
#                  strat_kwargs={"core_name": "SR21-P7", "figsize": (3, 18)},
#                  xrf_kwargs={"core_name": "SR21-P7", "ratio_list": ratios},
#                  savefig=False,
#                  savepath="sr21_xrf_combined.png")

sr19_ppm = XRF(dataframe=pd.read_csv("sr19_ppm.csv", header=0))

# plot_combine_xrf(sr19_strat,
#                  sr19_ppm,
#                  strat_kwargs={"core_name": "SR19-P4", "figsize": (3, 18)},
#                  xrf_kwargs={"core_name": "SR19-P4", "ratio_list": ratios},
#                  savefig=False,
#                  savepath="sr21_xrf_combined.png")

