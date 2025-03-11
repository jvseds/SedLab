import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
import warnings


class Forams:
    def __init__(self, dataframe=None, fname=None, size_fraction=125, volume=1.25, index_col=0, header=0):
        if dataframe is not None:
            self.dataframe = dataframe.copy()
        elif fname:
            self.dataframe = pd.read_csv(
                fname, index_col=index_col, header=header)
        else:
            self.dataframe = None
            warnings.warn("'Forams' object created with no dataframe.")

        self.size_fraction = size_fraction
        self.volume = volume

        if self.dataframe is not None:
            self.validate_df()
            self.calc_totals()
            self.normalize_per_1cc()
            self.calc_pb_ratio()
            self.calc_planktic_percents()

    def __repr__(self):
        return repr(self.dataframe) if self.dataframe is not None else "Forams object with no dataframe."

    def validate_df(self):
        required_cols = {"planktic", "benthic", "num_of_splits"}
        missing_cols = required_cols - set(self.dataframe.columns)

        if missing_cols:
            warnings.warn(
                f"Missing required column(s): {', '.join(missing_cols)}")

    def calc_totals(self):
        if self.dataframe is not None:
            self.dataframe["total"] = self.dataframe.get(
                "planktic", 0) + self.dataframe.get("benthic", 0)

    def normalize_per_1cc(self):
        if self.dataframe is not None and "total" in self.dataframe.columns and "num_of_splits" in self.dataframe.columns:
            self.dataframe["normalized_per_1cc"] = (
                self.dataframe["total"] * self.dataframe["num_of_splits"]) / self.volume

    def calc_pb_ratio(self):
        if self.dataframe is not None:
            self.dataframe["p/b_ratio"] = self.dataframe["planktic"] / \
                self.dataframe["benthic"].replace(0, np.nan)

    def calc_planktic_percents(self):
        if self.dataframe is not None and "total" in self.dataframe.columns:
            self.dataframe["planktic_percent"] = (
                self.dataframe["planktic"] / self.dataframe["total"].replace(0, np.nan)) * 100
            self.dataframe["benthic_percent"] = (
                self.dataframe["benthic"] / self.dataframe["total"].replace(0, np.nan)) * 100

    def plot_forams(self, core_name="Core", figsize=(6, 8), cmap="winter", ylim=270, savefig=False, savepath="forams.png", dpi=350, limit_sm=False, sm_limit=20):
        if self.dataframe is None:
            raise ValueError("No dataframe available for plotting.")

        vmin = self.dataframe["planktic_percent"].min()
        vmax = min(self.dataframe["planktic_percent"].max(
        ), sm_limit) if limit_sm else self.dataframe["planktic_percent"].max()

        norm_planktic = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_planktic)
        fig, ax = plt.subplots(figsize=figsize)

        q3 = np.nanpercentile(self.dataframe["planktic_percent"], 75)

        for depth, total, planktic in zip(self.dataframe.index, self.dataframe["total"], self.dataframe["planktic_percent"]):
            ax.barh(depth, total, color=sm.to_rgba(planktic))

            if planktic == 100 or planktic >= q3:
                ax.text(total, depth, f"{planktic:.1f}%",
                        va='center', ha='left', fontsize=6.5, color='black')

        ax.set_xlim(0)
        # TODO: set ylim to either max samples of the df or 90 quantile - ask Bev
        ax.set_ylim(0, max(ylim, self.dataframe.index[-1]))
        ax.yaxis.set_inverted(True)
        plt.colorbar(sm, ax=ax, label="Planktic %")
        ax.set_title(f"Foraminifera Abundance in {core_name}")
        ax.set_ylabel("Depth (cm)")
        ax.set_xlabel("Individuals / 1 cc")
        plt.grid(True)
        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, ax


class Bryozoans:
    def __init__(self, dataframe=None, fname=None, sheet_name=0, index_col=0, header=0):
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.dataframe = pd.read_excel(
                io=fname, sheet_name=sheet_name, index_col=index_col, header=header)
        else:
            self.dataframe = None
            warnings.warn("'Bryozoans' object created with no dataframe.")

    def __repr__(self):
        return repr(self.dataframe) if self.dataframe is not None else "Bryozoans object with no dataframe."

    def validate_df(self):
        if self.dataframe is None:
            raise ValueError(
                "The dataframe for this 'Bryozoans' object is empty.")

    def ensure_numeric_values(self):
        if self.dataframe is not None:
            self.dataframe = self.dataframe.apply(
                pd.to_numeric, errors='coerce')

    def create_contingency_table(self, other, column, core1="Core1", core2="Core2"):

        return pd.DataFrame({
            f"{core1}": [self.dataframe[column].sum(), len(self.dataframe) - self.dataframe[column].sum()],
            f"{core2}": [other.dataframe[column].sum(), len(other.dataframe) - other.dataframe[column].sum()]
        }, index=["Present", "Absent"])

    def calc_chi2(self, other):

        self.validate_df()
        other.validate_df()

        chi2_results = {}
        bryo_cols = ["net", "branch", "flat", "bryo>5mm"]
        for col in bryo_cols:
            table = self.create_contingency_table(other, col)
            chi2, p, dof, expected = chi2_contingency(table)
            if (expected < 5).any():
                warnings.warn(
                    f"Chi-square may be invalid due to low expected frequencies in {col}. Consider Fisher's Exact Test.")

            chi2_results[col] = {"Chi-Squared": chi2, "p-value": p}
            chi2_df = pd.DataFrame(chi2_results)

        return chi2_df

    def calc_mann_whitney(self, other, column="category"):
        """
        Compare the bryozoan abundance categories between two cores using the Mann-Whitney U test for two independent samples.
        """
        self.validate_df()
        other.validate_df()

        if len(self.dataframe[column].unique()) < 3:
            warnings.warn(
                "Mann-Whitney U test may not be meaningful due to few unique values in 'category'.")

        mw_stat, p = mannwhitneyu(
            self.dataframe[column], other.dataframe[column], alternative="two-sided")
        mw_results = pd.DataFrame({
            "Mann-Whitney U": [mw_stat],
            "p-value": [p]
        })

        return mw_results

    def calc_corr(self, method="spearman"):
        """
        Calculate the correlation matrix for the bryozoan abundance categories.
        """
        corr_matrix = self.dataframe[[
            "whole", ">2cm", "net", "branch", "flat", "bryo>5mm"]].corr(method=method)

        return corr_matrix

    def plot_corr_matrix(self, core_name="Core", method="spearman", figsize=(5, 5), cmap="coolwarm",
                         savefig=False, savepath="bryo_corr.png", dpi=350):

        corr_matrix = self.calc_corr(method=method)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data=corr_matrix, cmap=cmap, linewidths=0.5,
                    annot=True, fmt=".2f", cbar=True, ax=ax)
        ax.set_title(
            f"{method} Correlation of Bryozoan and Biomarkers in {core_name}")
        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, ax

    def plot_large_bryo(self, others=[], core_names=[], figsize=(5, 6), palette="Set2", n_colors=None,
                        savefig=False, savepath="large_bryos.png", dpi=350):

        num_cores = len(others) + 1
        fig, ax = plt.subplots(nrows=1, ncols=num_cores,
                               figsize=figsize, sharex=True, sharey=True)

        sns.set_palette(palette, n_colors=n_colors)
        if num_cores == 1:
            ax = [ax]

        # plot the current (`self`) core
        sns.countplot(data=self.dataframe, x="category", hue="category",
                      palette=palette, ax=ax[0], zorder=10)
        ax[0].set_title(f"{core_names[0]}" if core_names else "Core 1")
        ax[0].set_ylabel("Number of samples")
        # grid
        ax[0].grid(axis="y", alpha=0.5, zorder=0)
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax[0].set_xlabel("")

        for core in range(len(others)):
            sns.countplot(data=others[core].dataframe, x="category", hue="category",
                          palette=palette, ax=ax[core + 1], zorder=10)

            ax[core +
                1].set_title(f"{core_names[core]}" if core_names else f"Core {core + 2}")
            # grid
            ax[core + 1].grid(axis="y", alpha=0.5, zorder=0)
            ax[core + 1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax[core + 1].set_xlabel("")

        fig.supxlabel("Category")
        fig.suptitle("Large bryozoans category".title())

        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, ax
