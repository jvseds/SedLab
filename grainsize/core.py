import re
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
import copy


class GrainSize(object):

    def __init__(self, fname=None, sheet_name="raw_data", skiprows=1, dataframe=None):
        self.dataframe = None
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.load_data(fname, sheet_name=sheet_name, skiprows=skiprows)

    def __repr__(self):
        if self.dataframe is not None:
            return repr(self.dataframe)
        return "GrainSize object with no data loaded"

    def __getattr__(self, item):
        """Delegate to the DataFrame class for methods that are
        not defined in this class."""
        if not hasattr(self, 'dataframe'):
            raise AttributeError(
                f"'GrainSize' object has no attribute '{item}'")
        try:
            return getattr(self.dataframe, item)
        except AttributeError:
            raise AttributeError(
                f"'GrainSize' object or its 'dataframe' has no attribute '{item}'")

    def load_data(self, fname, sheet_name="raw_data", skiprows=1, **kwargs):
        """Loads core data from a MasterSizer 3000 Excel file into a DataFrame.

                Parameters:
                - fname: Filename or path of the Excel file.
                - sheet_name: Name of the sheet in the Excel file to read.
                - skiprows: Number of rows to skip at the beginning of the sheet.

                Returns:
                - A pandas DataFrame containing the imported data.
                """
        df = pd.read_excel(fname, sheet_name=sheet_name, skiprows=skiprows)
        self.dataframe = df
        return df

    def set_gs_index(self):
        """Sets the DataFrame's index to the 'grain_size' column."""
        if self.dataframe is not None:
            self.dataframe = self.dataframe.set_index(
                "grain_size").rename_axis("grain_size")

        return self.dataframe

    def rename_range_col(self):
        """Finds the DataFrame column with minimum amount of null values
        and renames it 'grain_size'."""

        # find column with minimum amount of nulls
        min_null_col = self.dataframe.isnull().sum().idxmin()
        self.dataframe.rename(
            mapper={min_null_col: "grain_size"}, axis=1, inplace=True)

        return self.dataframe

    def get_depths(self):
        """Extracts depths from the raw column labels."""

        import re
        # pattern of depth to look for, appears as "-xxx"
        pattern = r"(?<=-)\d{3}"

        depths = []
        for col in self.dataframe.columns:
            match = re.search(pattern, col)
            if match:
                depths.append(match.group())
            # append "None" if depth not found
            else:
                depths.append(None)

        return depths

    def drop_redun(self, to_drop="Unnamed"):
        """Drop empty and redundant columns from the DataFrame.
        to_drop: redundant column headers, default value: 'Unnamed'.
        """
        # drop empty columns
        self.dataframe = self.dataframe.dropna(axis=1, how="all")
        # drop redundant columns which contain to_drop phrase in header
        cols_to_drop = [
            col for col in self.dataframe.columns if to_drop in col]
        self.dataframe = self.dataframe.drop(labels=cols_to_drop, axis=1)

        return self.dataframe

    def rename_headers(self):
        """Renames column headers as respective depth.
        *** Use after transposing the matrix, so the depth is represented as columns."""

        # rename column headers with "depths" instead of original file headers
        depth_list = self.get_depths()
        mapper = {self.dataframe.columns[i]: float(
            depth_list[i]) for i in range(len(self.dataframe.columns))}
        self.dataframe.rename(mapper=mapper, axis=1, inplace=True)

        return self.dataframe

    def sort_by_depth(self):
        """Sorts DataFrame's rows by depth."""
        return self.dataframe.sort_index(axis=0, ascending=True)

    def clean_data(self):
        """Clean the initial DataFrame created by importing the MasterSizer 3000
        original Excel file."""

        import copy

        # create a copy of the original data frame
        self.dataframe = copy.deepcopy(self.dataframe)
        # find column with entire GS range
        self.rename_range_col()
        # drop empty columns
        self.drop_redun()
        # set index to grain_size column
        self.set_gs_index()

        # rename column headers
        self.rename_headers()
        # transpose the DataFrame
        self.dataframe = self.dataframe.transpose()
        # drop the column which is labels "Size Classes"
        self.dataframe = self.dataframe.drop(
            labels="Size Classes (μm)", axis=1)
        # rename index
        self.dataframe.rename_axis("depth", inplace=True)
        # replace remaining NaN with 0
        self.dataframe.fillna(0, inplace=True)
        # sort index by depth
        self.dataframe = self.sort_by_depth()

        # make sure all values are of type "float"
        self.dataframe = self.dataframe.astype(float)

        # remove duplicates from index, keeping the last sample
        self.dataframe = self.dataframe.loc[~self.dataframe.index.duplicated(
            keep="last")]

        return self.dataframe

    def normalize_gs(self):
        """Create a normalized DataFrame of grain size distribution to represent percents
        (out of 100%)."""

        # ensure the DataFrame is not empty
        if self.dataframe is not None:
            # create a copy of the data frame
            data = self.dataframe.copy()
            # sum each row
            data["sum"] = data.apply(np.sum, axis=1)
            # normalize each value in the dataframe
            normalized_df = data.apply(lambda x: x * 100 / data["sum"])

            # drop the "sum" column since it's unnecessary now
            normalized_df.drop(labels="sum", axis=1, inplace=True)
            # fill NaN with 0
            normalized_df.fillna(0, inplace=True)

            # drop columns that contain only zeros (no representation for any grains)
            normalized_df = normalized_df.loc[:,
                                              (normalized_df != 0).any(axis=0)]

            # return as a new GrainSize object with the normalized DataFrame
            return GrainSize(dataframe=normalized_df)
        else:
            # return an empty GrainSize object with an empty DataFrame
            return GrainSize()

    def create_categories(self, save=False, fpath="core_cats.csv"):
        """Create a DataFrame of grain size categories:
        Clay, silt, sand, and gravel."""

        if self.dataframe is not None:
            # assert that column labels are of numeric type
            self.dataframe.columns = self.dataframe.columns.astype(float)

            # create gs categories masks
            clay_mask = self.dataframe.columns < 4
            silt_mask = (self.dataframe.columns >= 4) & (
                self.dataframe.columns < 63.5)
            sand_mask = (self.dataframe.columns >= 63.5) & (
                self.dataframe.columns < 2001)
            gravel_mask = self.dataframe.columns >= 2001

            # create pandas Series objects for each gs category
            clay_col = self.dataframe.loc[:, clay_mask].sum(axis=1)
            silt_col = self.dataframe.loc[:, silt_mask].sum(axis=1)
            sand_col = self.dataframe.loc[:, sand_mask].sum(axis=1)
            gravel_col = self.dataframe.loc[:, gravel_mask].sum(axis=1)

            # build a pd.DataFrame out of the Series objects
            categories = pd.DataFrame({"clay": clay_col,
                                       "silt": silt_col,
                                       "sand": sand_col,
                                       "gravel": gravel_col},
                                      index=self.dataframe.index)

            # if save=True, save the data frame as a csv file
            if save:
                categories.to_csv(fpath)

            # return the categories DataFrame into an instantiated GrainSize object
            return GrainSize(dataframe=categories)

        else:
            # return an empty GrainSize object with an empty DataFrame
            return GrainSize()

    def find_median_mode(self):
        """
        Create a pd.DataFrame of median and mode of grain size
        distribution.

        :return:
        pd.DataFrame
        """

        if self.dataframe is not None:
            # create a cumulative sum DataFrame across each row
            data_cumsum = self.dataframe.cumsum(axis=1)
            # create a dictionary that will hold median and mode labels
            cumsum_dict = dict()
            # iterate over data_cumsum and populate cumsum_dict
            for i, s in data_cumsum.iterrows():
                cumsum_dict[i] = {"median": s[s >= 50].index[0]}

            for i, s in self.dataframe.iterrows():
                cumsum_dict[i]["mode"] = self.dataframe.columns[s.to_numpy().argmax()]

            # build a pd.DataFrame out of cumsum_dict
            med_mode_df = pd.DataFrame.from_dict(cumsum_dict, orient="index")

            # return a GrainSize object with med_mode_df as dataframe
            return GrainSize(dataframe=med_mode_df)

        else:
            return GrainSize()

    def find_mean(self):
        """
        Find the mean values of the grain size distribution.
        :return: pd.Series
        """

        mean_values = self.dataframe.apply(
            lambda row: row[row > 0].mean(), axis=1)

        return mean_values

    def find_mean_labels(self, row, mean_vals):
        """
        Find grain size column label which represent the mean
        frequency in a gs distribution dataset.
        :return:
        """
        # filter row to only include values greater than 0
        filtered_row = row[row > 0]
        if filtered_row.empty:
            return np.nan
        # find the difference between the row values and the mean
        differences = (filtered_row - mean_vals).abs()
        # find the label (grain size) of the minimal difference in the first occurrence
        return differences.idxmin()

    def add_mean_gs(self):
        """
        Add labels of mean grain size to the med_mode_df.
        :return:
        """
        # find mean values
        mean_values = self.find_mean()
        # find labels of mean values
        mean_labels = self.dataframe.apply(lambda row: self.find_mean_labels(row, mean_values[row.name]),
                                           axis=1)

        return mean_labels

    def create_stats_df(self, save=False, fpath=r"full_core_stats.csv"):
        """
        Create a statistical data frame that holds median,
        mode, and mean grain size classes for a given normalized
        grain size dataset.
        :return:
        """

        # normalize the data
        normalized_df = self.normalize_gs()

        # calculate median and mode, return a GrainSize object
        stats = normalized_df.find_median_mode()

        # add mean grain size labels
        stats.dataframe["mean"] = self.add_mean_gs()
        # add a standard deviation column
        stats.dataframe["std"] = self.dataframe.apply(
            lambda row: row[row != 0].std(), axis=1)
        # add skewness column
        stats.dataframe["skewness"] = self.dataframe.apply(
            lambda row: row[row != 0].skew(), axis=1)
        # create kurtosis column
        stats.dataframe["kurtosis"] = self.dataframe.apply(
            lambda row: row[row != 0].kurtosis(), axis=1)

        stats.dataframe.rename_axis("depth")

        # if save=True, save as a csv file
        if save:
            stats.dataframe.to_csv(fpath)

        return stats

    def plot_stats(self,
                   core_name="core",
                   figsize=(22, 18),
                   marker="o",
                   linestyle="dashed",
                   save_fig=False,
                   fpath="gs_stat_plot.png",
                   dpi=350):
        """
        Plots line graphs of core's statistics.

        :param core_name: core's name
        :param figsize: figure size
        :param marker: marker style
        :param linestyle: line style
        :param save_fig: whether to save returned figure
        :param fpath: saving path
        :param dpi: dpi for returned figure
        :return: .png figure of GrainSize object statistics
        """

        stats_labels = ["median", "mode", "mean",
                        "std", "skewness", "kurtosis"]
        colors = ["#e0bb34", "#913800", "#521101",
                  "#03265e", "#8a32b3", "#bd2a82"]

        # define all 6 axes objects with shared x and y axes
        fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=len(stats_labels),
                                 sharey=True, sharex=False)

        for ax, stat, color in zip(axes, stats_labels, colors):
            ax.plot(self.dataframe[stat], self.dataframe.index, marker=marker,
                    linestyle=linestyle, linewidth=0.85, color=color)
            # set the corresponding label
            ax.set_xlabel("Grain size (µm)")
            ax.set_title(f"{stat.capitalize()}")
            # show grid
            ax.grid(True)

        # set for all axes
        axes[0].set_ylabel("Depth (cm)")
        # invert the y-axis
        plt.gca().invert_yaxis()
        # add a suptitle with core's name
        plt.suptitle(f"{core_name} Grain Size Distribution Statistics")

        plt.tight_layout()

        # if save_fig=True, save to fpath:
        if save_fig:
            plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

        return fig, axes

    def fine_fraction(self,
                      save=False,
                      save_path="fine_fraction.csv"):

        if self.dataframe is not None:
            fine_categories = ["clay", "silt"]
            fine_percents = self.create_categories().loc[:, fine_categories]
            fine_percents["total"] = fine_percents.loc[:,
                                                       "clay"] + fine_percents.loc[:, "silt"]

            if save:
                fine_percents.to_csv(save_path)

            return fine_percents
        else:
            return GrainSize()

    def plot_stats_fines(self,
                         fine_col="total",
                         core_name="core",
                         figsize=(22, 18),
                         marker="o",
                         linestyle="dashed",
                         save_fig=False,
                         fpath="gs_stat_plot.png",
                         dpi=350):
        """
        Plots line graphs of core's statistics and also the fraction of
        fine grains (< 63 um, silt + clay).

        :param core_name: core's name
        :param figsize: figure size
        :param marker: marker style
        :param linestyle: line style
        :param save_fig: whether to save returned figure
        :param fpath: saving path
        :param dpi: dpi for returned figure
        :return: .png figure of GrainSize object statistics
        """

        stats_labels = ["median", "mode", "mean",
                        "std", "skewness", "kurtosis"]
        colors = ["#e0bb34", "#913800", "#521101",
                  "#03265e", "#8a32b3", "#bd2a82"]

        # define all 6 axes objects with shared x and y axes
        fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=len(stats_labels) + 1,
                                 sharey=True, sharex=False)

        for ax, stat, color in zip(axes[:-3], stats_labels[:-2], colors[:-2]):
            ax.plot(self.dataframe[stat], self.dataframe.index, marker=marker,
                    linestyle=linestyle, linewidth=0.85, color=color)
            # set the corresponding label
            ax.set_xlabel("Grain size (µm)")
            ax.set_title(f"{stat.capitalize()}")
            # show grid
            ax.grid(True)

        # plot skewness and kurtosis
        for ax, stat, color in zip(axes[-3:-1], stats_labels[-2:], colors[-2:]):
            ax.plot(self.dataframe[stat], self.dataframe.index, marker=marker,
                    linestyle=linestyle, linewidth=0.85, color=color)
            ax.set_title(f"{stat.capitalize()}")
            ax.set_ylim(0, self.dataframe.index[-1])
            # show grid
            ax.grid(True)

        # plot fine grains fraction at the end of the row
        ax_percentage = axes[-1]
        fine_grains = self.dataframe.loc[:, fine_col]
        ax_percentage.plot(fine_grains, self.dataframe.index, marker=marker,
                           linestyle=linestyle, linewidth=0.85, color="#0da818")
        ax_percentage.set_xlabel("Percentage (%)")
        ax_percentage.set_title("< 63 µm")
        ax_percentage.set_xlim(0, 100)

        ax_percentage.set_ylim(0, self.dataframe.index[-1])
        ax_percentage.grid(True)

        # set for all axes
        axes[0].set_ylabel("Depth (cm)")
        axes[0].set_ylim(0, self.dataframe.index[-1])
        plt.gca().invert_yaxis()
        plt.suptitle(f"{core_name} Grain Size Distribution Statistics")

        plt.tight_layout()

        if save_fig:
            plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

        return fig, axes

    @staticmethod
    def compare_gs(cores: list,
                   core_names: list = None,
                   fine_col="total",
                   figsize_per_plot=(1.5, 8),
                   marker=".",
                   linestyle="dashed",
                   save_fig=False,
                   fpath="compare_gs_grid.png",
                   dpi=350):
        """
        Compare cores in a subplot grid with 1 row per core, and 4 columns (median, mode, mean, <63 µm).

        :param cores: list of GrainSize objects
        :param core_names: optional list of names for the cores
        :param fine_col: column name for < 63 µm
        :param figsize_per_plot: size of each subplot (width, height)
        :param marker: marker style
        :param linestyle: line style
        :param save_fig: whether to save the plot
        :param fpath: path to save
        :param dpi: resolution
        :return: fig, axes
        """

        for i, core in enumerate(cores):
            if not isinstance(core, GrainSize):
                raise TypeError(
                    f"Item at index {i} is not a GrainSize object.")

        if core_names is None:
            core_names = [f"Core {i+1}" for i in range(len(cores))]

        n_cores = len(cores)
        stats_labels = ["median", "mode", "mean", "< 63 µm"]
        colors = ["#e0bb34", "#913800", "#521101", "#0da818"]

        # find maximal depth between the cores
        max_depth = max(core.dataframe.index.max() for core in cores)

        # total figure size
        figsize = (figsize_per_plot[0] * 4, figsize_per_plot[1] * n_cores)
        fig, axes = plt.subplots(
            nrows=n_cores, ncols=4,
            figsize=figsize,
            sharey=True,
            squeeze=False
        )

        for row_idx, (core, name) in enumerate(zip(cores, core_names)):
            for col_idx, (label, color) in enumerate(zip(stats_labels, colors)):
                ax = axes[row_idx, col_idx]
                data_col = fine_col if label == "< 63 µm" else label

                if data_col in core.dataframe.columns:
                    ax.plot(core.dataframe[data_col], core.dataframe.index,
                            marker=marker, linestyle=linestyle, color=color, linewidth=0.85)

                if row_idx == 0:
                    ax.set_title(label)

                if label == "< 63 µm":
                    ax.set_xlim(0, 100)
                else:
                    ax.set_xlim(-0.5, core.dataframe[data_col].max() + 1)

                ax.set_ylim(0, max_depth)
                ax.invert_yaxis()

                if col_idx == 0:
                    ax.set_ylabel(f"{name}\nDepth (cm)")
                    yticks = np.arange(0, max_depth + 1, 50)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([str(int(y)) for y in yticks])
                    ax.tick_params(axis='y', labelsize=8, labelleft=True)
                else:
                    ax.tick_params(axis='y', labelleft=False)

                if row_idx == n_cores - 1:
                    if label == "< 63 µm":
                        ax.set_xlabel("Percentage (%)")
                    else:
                        ax.set_xlabel("Grain size (µm)")

                # plot a horizontal line if core is shorter than max_depth
                if max_depth > core.dataframe.index[-1]:
                    ax.axhline(
                        y=core.dataframe.index[-1], color='red', linestyle='--', linewidth=0.3, alpha=0.5)

                ax.grid(True)

        fig.suptitle("Grain Size Comparison by Core")
        fig.tight_layout()

        if save_fig:
            plt.savefig(fpath, dpi=dpi)

        return fig, axes


class XRF(object):

    def __init__(self,
                 fname=None,
                 sheet_name="calibrated_results",
                 header=0, usecols="A:B",
                 index_col=0,
                 nrows=27,
                 tp=True,
                 dataframe=None):
        """
        Initializes the XRF object. Loads data from an Excel file if 'fname' is provided.

        Parameters:
        - fname: Filename or path of the XRF Excel file.
        - sheet_name: Name of the sheet in the Excel file to read.
        - header: Row number where headers are located (0-indexed).
        - usecols: Columns to parse from the Excel file.
        - index_col: Column to set as index.
        - dataframe: Existing pandas DataFrame to initialize the object with.
        """
        self.dataframe = dataframe
        if dataframe is None and fname:
            self.load_data(fname,
                           sheet_name=sheet_name,
                           header=header,
                           usecols=usecols,
                           index_col=index_col,
                           nrows=nrows,
                           tp=tp)

    def __repr__(self):
        if self.dataframe is not None:
            return repr(self.dataframe)
        return "XRF object with no data loaded"

    def __getattr__(self, item):
        """Delegate to the DataFrame class for methods that are
        not defined in this class."""
        if not hasattr(self, 'dataframe') or self.dataframe is None:
            # Prevent recursion if 'dataframe' doesn't exist yet
            raise AttributeError(f"'XRF' object has no attribute '{item}'")
        try:
            return getattr(self.dataframe, item)
        except AttributeError:
            raise AttributeError(
                f"'XRF' object or its 'dataframe' has no attribute '{item}'")

    def load_data(self,
                  fname,
                  sheet_name="calibrated_data",
                  header=0,
                  nrows=27,
                  usecols="A:B",
                  index_col=0,
                  tp=True):
        """
        Loads XRF data from an Excel file into a pandas DataFrame.

        Parameters:
        - fname: Filename or path of the Excel file.
        - sheet_name: Name of the sheet in the Excel file to read.
        - header: Row number where headers are located (0-indexed).
        - usecols: Columns to parse from the Excel file.
        - index_col: Column to set as index.

        Returns:
        - A pandas DataFrame containing the imported data.
        """
        self.dataframe = pd.read_excel(io=fname,
                                       sheet_name=sheet_name,
                                       header=header,
                                       nrows=nrows,
                                       usecols=usecols,
                                       index_col=index_col)

        if tp:
            self.dataframe = self.dataframe.transpose()

        return self.dataframe

    def to_ppm(self):
        """

        :return:
        """

        # get the elements that are presented in percents
        percents_row = self.dataframe.iloc[0,
                                           :][self.dataframe.iloc[0, :] == "%"]
        percents_df = pd.DataFrame(percents_row)

        # list elements that are measured in percents
        elements_in_pc = percents_df.index.tolist()
        # create new XRF object with ppm dataframe
        ppm_df = XRF(dataframe=copy.deepcopy(self.dataframe))
        # drop the second raw
        ppm_df.dataframe.drop(labels="FileName", axis=0, inplace=True)
        # define conversion factor from percents to ppm
        conversion_factor = 10_000
        # convert percents to ppm
        ppm_df.dataframe[elements_in_pc] *= conversion_factor

        # return new XRF object with elements measured in ppm
        return ppm_df

    def clean_data(self):
        """
        Cleans the data by updating the index to be the larger number in the interval as a float.
        """
        if self.dataframe is not None:
            cleaned = copy.deepcopy(self.dataframe)

            def extract_larger_number(index):
                # Match patterns like '001-002' or '(2-3)'
                match = re.search(r'(\d+)-(\d+)', index)
                if match:
                    return float(match.group(2))
                match = re.search(r'\((\d+)-(\d+)\)', index)
                if match:
                    return float(match.group(2))
                # Return None for unexpected index values
                return None

            # Apply extraction function to index
            cleaned_index = self.dataframe.index.map(extract_larger_number)
            cleaned = self.dataframe[cleaned_index.notnull()]
            cleaned.index = cleaned_index.dropna().astype(float)
        else:
            raise ValueError("No dataframe loaded to clean")

        return XRF(dataframe=cleaned)

    def plot_elements(self, core_name="Core", figsize=(10, 8), ylimit=None, xlimit=None, rows=2, lw=0.75, marker=".", unit="percent",
                      savefig=False, dpi=350, savepath="elements.png"):

        elements = list(self.dataframe.columns)

        num_elements = len(elements)
        num_cols = (num_elements + rows - 1) // rows

        fig, axs = plt.subplots(
            nrows=rows, ncols=num_cols, figsize=figsize, sharey=True)

        if rows > 1 and num_cols > 1:
            axs = axs.flatten()

        # plot every element on the elements list
        for i, element in enumerate(elements):
            row = i // num_cols
            col = i % num_cols
            ax = axs[i] if num_elements > 1 else axs
            ax.plot(self.dataframe[element],
                    self.dataframe.index, marker=marker, lw=lw)
            ax.grid()
            ax.set_title(f"{element}", fontsize=12)

            if ylimit:
                ax.set_ylim(0, (max(ylimit, self.dataframe.index[-1])))
            else:
                ax.set_ylim(self.dataframe.index[0], self.dataframe.index[-1])
            if xlimit:
                ax.set_xlim(0, (max(xlimit[i], self.dataframe[element].max())))
            else:
                ax.set_xlim(0, self.dataframe[element].max())

            ax.yaxis.set_inverted(True)
            if col == 0:
                ax.set_ylabel("Depth (cm)")

        plt.suptitle(f"{core_name} XRF Results (in {unit})",
                     fontsize=16, y=0.99)

        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axs

    def plot_ratios(self, core_name="Core", ratio_list=[("Si", "Al")], lw=0.75, figsize=(6, 8), ylimit=None, xlimit=None, marker=".",
                    savefig=False, dpi=350, savepath="element_ratios.png"):

        num_ratios = len(ratio_list)

        fig, axs = plt.subplots(nrows=1, ncols=num_ratios, figsize=figsize,
                                sharey=True)
        if num_ratios == 1:
            axs = [axs]

        for i, (num, denom) in enumerate(ratio_list):
            # calculate elemental ratio
            ratio = self.dataframe[num] / self.dataframe[denom]
            axs[i].plot(ratio, self.dataframe.index,
                        marker=marker, lw=lw, ls="-")
            axs[i].grid()

            if ylimit:
                axs[i].set_ylim(0, max(ylimit, self.dataframe.index[-1]))
            else:
                axs[i].set_ylim(self.dataframe.index[0],
                                self.dataframe.index[-1])
            if xlimit:
                axs[i].set_xlim(0, max(xlimit[i], max(ratio)))
            else:
                axs[i].set_xlim(0, max(ratio))

            axs[i].yaxis.set_inverted(True)
            axs[i].set_title(f"{num}/{denom}", fontsize=14)
            if i == 0:
                axs[i].set_ylabel("Depth (cm)", fontsize=15)

        plt.suptitle(f"{core_name} Elemental Ratios", fontsize=16, y=0.99)
        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axs

    @staticmethod
    def compare_ratios(cores,
                       core_names=None,
                       ratio_list=[("Si", "Al")],
                       colors=None,
                       lw=0.75,
                       marker=".",
                       figsize=(6, 8),
                       ylimit=None,
                       xlimit=None,
                       savefig=False,
                       dpi=350,
                       savepath="compare_element_ratios.png"):
        """
        Compare one or more elemental ratios across several XRF cores.

        Parameters
        ----------
        cores : list[XRF]
            List of XRF objects to compare.
        core_names : list[str] | None
            Optional names that will be used in the legend. Defaults to Core 1, Core 2, …
        ratio_list : list[tuple[str, str]]
            List of (numerator, denominator) pairs to plot. One subplot per pair.
        colors : list[str] | None
            Matplotlib‑style colors, one per core. Defaults to the current rcParams cycle.
        lw, marker, figsize, ylimit, xlimit, savefig, dpi, savepath
            Same meaning as in `.plot_ratios()`.
        """

        import itertools

        n_cores = len(cores)
        n_ratios = len(ratio_list)
        if core_names is None:
            core_names = [f"Core {i+1}" for i in range(n_cores)]
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = list(itertools.islice(itertools.cycle(colors), n_cores))

        fig, axs = plt.subplots(
            nrows=1,
            ncols=n_ratios,
            figsize=figsize,
            sharey=True
        )
        if n_ratios == 1:
            axs = [axs]

        for j, (num, denom) in enumerate(ratio_list):
            ax = axs[j]

            for i, core in enumerate(cores):
                ratio_series = core.dataframe[num] / core.dataframe[denom]
                ax.plot(ratio_series,
                        core.dataframe.index,
                        marker=marker,
                        lw=lw,
                        color=colors[i],
                        label=core_names[i])

            if ylimit:
                ax.set_ylim(0, max(ylimit, core.dataframe.index[-1]))
            else:
                ax.set_ylim(cores[0].dataframe.index[0],
                            cores[0].dataframe.index[-1])

            if xlimit:
                ax.set_xlim(0, max(xlimit[j],
                                   max(ratio_series.max() for core in cores)))
            else:
                ax.set_xlim(0, max(ratio_series.max() for core in cores))

            ax.set_title(f"{num}/{denom}")
            ax.yaxis.set_inverted(True)
            ax.grid(True)

            if j == 0:
                ax.set_ylabel("Depth (cm)")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.97),
                   ncol=len(handles),
                   frameon=False,
                   fontsize=9)
        plt.suptitle("Elemental Ratios Comparison")
        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axs


class Stratigraphy(object):

    def __init__(self, fname=None, header=0, dataframe=None):
        # complete more attributes as needed
        self.dataframe = None
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.load_data(fname, header=header)

        self.fill_params = []

    def __repr__(self):
        if self.dataframe is not None:
            return repr(self.dataframe)
        return "Stratigraphy object with no data loaded"

    def __getattr__(self, item):
        """Delegate to the DataFrame class for methods that are
        not defined in this class."""
        if not hasattr(self, 'dataframe'):
            # Prevent recursion if 'dataframe' doesn't exist yet
            raise AttributeError(
                f"'Stratigraphy' object has no attribute '{item}'")
        try:
            return getattr(self.dataframe, item)
        except AttributeError:
            raise AttributeError(
                f"'Stratigraphy' object or its 'dataframe' has no attribute '{item}'")

    def load_data(self, fname, header=0):
        """Loads stratigraphic units data from a csv file into a DataFrame.

                Parameters:
                - fname: Filename or path of the csv file.
                - header: int, row number (0-indexed) to use as df headers.

                Returns:
                - A pandas DataFrame containing the imported data.
                """
        df = pd.read_csv(fname, header=header)
        self.dataframe = df
        return df

    def plot_stratigraphy(self, figsize=(2.75, 18), core_name="Core", savefig=False,
                          savepath="core_strat.png", dpi=350):
        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            'Silty Sand': '#d2b48c',  # Tan
            'Silty Mud': '#8b4513',  # SaddleBrown
            'Clay': '#a52a2a',  # Brown
            'Sand': '#ffd700',  # Gold
            'Gravel': '#808080'  # Gray
        }

        for _, row in self.dataframe.iterrows():
            color = colors[row["unit"]]
            fill = ax.fill_betweenx(
                [row["top"], row["bottom"]], 0, 1, color=color)
            self.fill_params.append(
                {'top': row["top"], 'bottom': row["bottom"], 'color': color})

            mid_depth = (row["top"] + row["bottom"]) / 2
            if row["symbol"] is not np.nan:
                ax.text(0.5, mid_depth, row["symbol"],
                        va="center", ha="left", fontsize=10)
            ax.text(0.5, mid_depth, row["unit"],
                    va="center", ha="right", fontsize=10)

        for bottom in self.dataframe["bottom"]:
            ax.axhline(y=bottom, color="k", linewidth=0.5)

        ax.set_ylim(self.dataframe["bottom"].iloc[-1], 0)
        ax.set_ylabel("Depth (cm)")
        ax.set_title(f"{core_name} Units")
        # ax.set_xticks([])
        ax.set_xlabel("Units")
        ax.xaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.yaxis.set_ticks(np.arange(
            self.dataframe["top"].iloc[0], self.dataframe["bottom"].iloc[-1], 20))

        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, ax


class CoreAnalysis:
    # all methods here are static.
    @staticmethod
    def plot_combined_stats(gs_obj, xrf_obj, core_name="Core", ylimit=263, xrf_els=["Mg", "Fe"], gs_stats=["mean", "mode", "median"], figsize=(10, 8), marker="o", linestyle="dashed"):
        """
        Plot mode, median, and mean from GrainSize alongside selected XRF elements in separate subplots.

        :param gs_obj: GrainSize object
        :param xrf_obj: XRF object
        :param xrf_els: List of selected XRF elements to plot (default: ["Mg", "Fe"])
        :param figsize: Tuple defining figure size
        :param marker: Marker style for lines
        :param linestyle: Line style
        """

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # extract depth index
        depth = gs_obj.dataframe.index

        # define grain size statistics to plot
        gs_stats = gs_stats
        gs_colors = ["#913800", "#e0bb34", "#521101"]

        # Generate colors dynamically for XRF elements
        cmap = cm.get_cmap("viridis", len(xrf_els))
        xrf_colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(xrf_els))]

        # total number of subplots (GS stats + selected XRF elements)
        num_plots = len(gs_stats) + len(xrf_els)

        # fig and axes
        fig, axes = plt.subplots(
            nrows=1, ncols=num_plots, figsize=figsize, sharey=True)

        # make `axes` iterable in any case
        if num_plots == 1:
            axes = [axes]

        # plot GrainSize statistics
        for ax, stat, color in zip(axes[:len(gs_stats)], gs_stats, gs_colors):
            if stat in gs_obj.dataframe.columns:
                ax.plot(gs_obj.dataframe[stat], depth, label=stat.capitalize(),
                        marker=marker, linestyle=linestyle, color=color)
                ax.set_title(stat.capitalize())
                ax.grid(True)
                ax.set_xlabel("Grain Size (µm)")

        # plot XRF elements
        for ax, (el, color) in zip(axes[len(gs_stats):], zip(xrf_els, xrf_colors)):
            el = el.capitalize()
            if el in xrf_obj.dataframe.columns:
                ax.plot(xrf_obj.dataframe[el], xrf_obj.dataframe.index, label=el,
                        marker=marker, linestyle=linestyle, color=color)
                max_concentration = xrf_obj.dataframe[el].max()
                ax.set_title(el)
                ax.set_xlim(0, max_concentration)
                ax.grid(True)
                ax.set_xlabel("Concentration (%)")

        # labels and foramtting
        for ax in axes:
            ax.set_ylim(0, max(depth[-1], ylimit))

        axes[0].set_ylabel("Depth (cm)")
        plt.gca().invert_yaxis()
        plt.suptitle(f"Summary plot for {core_name}")
        plt.tight_layout()

        return fig, ax
