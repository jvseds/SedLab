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
            # replace zero splits with 1 to avoid division errors
            safe_splits = np.where(
                self.dataframe["num_of_splits"] == 0, 1, self.dataframe["num_of_splits"])

            self.dataframe["normalized_per_1cc"] = (
                self.dataframe["total"] * safe_splits) / self.volume

            self.dataframe["norm_benthic"] = (
                self.dataframe["benthic"] * safe_splits) / self.volume

            self.dataframe["norm_planktic"] = (
                self.dataframe["planktic"] * safe_splits) / self.volume

    def calc_pb_ratio(self):
        if self.dataframe is not None:
            self.dataframe["p/b_ratio"] = np.where(
                self.dataframe["norm_benthic"] == 0,
                np.nan,
                self.dataframe["norm_planktic"] /
                self.dataframe["norm_benthic"]
            )

    def calc_planktic_percents(self):
        if self.dataframe is not None:
            self.dataframe["planktic_percent"] = np.where(
                self.dataframe["normalized_per_1cc"] == 0,
                0,
                (self.dataframe["norm_planktic"] /
                 self.dataframe["normalized_per_1cc"]) * 100
            )

            self.dataframe["benthic_percent"] = np.where(
                self.dataframe["normalized_per_1cc"] == 0,
                0,
                (self.dataframe["norm_benthic"] /
                 self.dataframe["normalized_per_1cc"]) * 100
            )

    def plot_forams(self, core_name="Core", figsize=(6, 8), cmap="winter", ylim=270, xlim=None, percentile=90,
                    savefig=False, savepath="forams.png", dpi=350, limit_sm=False, sm_limit=20):
        if self.dataframe is None:
            raise ValueError("No dataframe available for plotting.")

        vmin = self.dataframe["planktic_percent"].min()
        vmax = min(self.dataframe["planktic_percent"].max(
        ), sm_limit) if limit_sm else self.dataframe["planktic_percent"].max()

        norm_planktic = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_planktic)
        fig, ax = plt.subplots(figsize=figsize)

        q_percentile = np.nanpercentile(
            self.dataframe["planktic_percent"], percentile)

        for depth, total, planktic in zip(self.dataframe.index, self.dataframe["normalized_per_1cc"], self.dataframe["planktic_percent"]):
            ax.barh(depth, total, color=sm.to_rgba(planktic))

            # annotate the bar with planktic percentage if it is 100% or above a given percentile
            if planktic == 100 or planktic >= q_percentile:
                label = f"{planktic:.1f}%"
                if xlim:
                    ax.text(
                        min(total, xlim - 15),
                        depth,
                        label,
                        va="center",
                        ha="left",
                        fontsize=6.5,
                        color="black",
                    )
                else:
                    ax.text(
                        total,
                        depth,
                        label,
                        va="center",
                        ha="left",
                        fontsize=6.5,
                        color="black",
                    )

        # total abundance annotation
        if xlim:
            ax.set_xlim(0, xlim)

            for depth, total in zip(self.dataframe.index, self.dataframe["normalized_per_1cc"]):
                if total > xlim:
                    ax.annotate(
                        f"total: {total:.1f}",
                        xy=(xlim / 2, depth),
                        va="center",
                        fontsize=7.5,
                        color="#990000"
                    )
        else:
            ax.set_xlim(0)
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

    def compare_forams_plot(self, cores, core_names=None, figsize=(4, 8), cmap="winter",
                            ylim=270, xlim=None, percentile=75, savefig=False, savepath="compare_forams.png", dpi=350,
                            limit_sm=False, sm_limit=20):
        """
        Compare foram plots from multiple cores using side-by-side subplots.
        Args:
            cores (list): List of `Forams` objects to compare.
            core_names (list): Optional list of core names, must match the length of `cores`.
        """
        n = len(cores)
        if core_names is None:
            core_names = [f"Core {i+1}" for i in range(n)]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n,
            figsize=(figsize[0]*n, figsize[1]),
            sharex=True,
            sharey=True,
            layout="constrained")

        if n == 1:
            axes = [axes]

        # normalize for colormap
        all_planktic = pd.concat(
            [core.dataframe["planktic_percent"] for core in cores])
        vmin = all_planktic.min()
        vmax = min(all_planktic.max(),
                   sm_limit) if limit_sm else all_planktic.max()
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # maximal abundance of forams in all cores
        xmax = max([core.dataframe["normalized_per_1cc"].max()
                   for core in cores])

        for i, (core, ax) in enumerate(zip(cores, axes)):
            df = core.dataframe
            q = np.nanpercentile(df["planktic_percent"], percentile)

            for depth, total, planktic in zip(df.index, df["normalized_per_1cc"], df["planktic_percent"]):
                ax.barh(depth, total, color=sm.to_rgba(planktic))
                if planktic == 100 or planktic >= q:
                    label = f"{planktic:.1f}%"
                    if xlim:
                        ax.text(
                            min(total, xlim - 15),
                            depth,
                            label,
                            va="center",
                            ha="left",
                            fontsize=6.5,
                            color="black",
                        )
                    else:
                        ax.text(
                            total,
                            depth,
                            label,
                            va="center",
                            ha="left",
                            fontsize=6.5,
                            color="black",
                        )

            ax.set_title(core_names[i])

            if xlim:
                ax.set_xlim(0, xlim)

                for depth, total in zip(df.index, df["normalized_per_1cc"]):
                    if total > xlim:
                        ax.annotate(
                            f"total: {total:.1f}",
                            xy=(xlim / 2, depth),
                            va="center",
                            fontsize=7.5,
                            color="#990000"
                        )
            else:
                ax.set_xlim(0, xmax)

            ax.set_ylim(0, max(ylim, df.index[-1]))
            ax.yaxis.set_inverted(True)
            ax.set_xlabel("Individuals / 1 cc")
            ax.grid(True)
            if i == 0:
                ax.set_ylabel("Depth (cm)")
            else:
                ax.set_ylabel("")

        fig.colorbar(sm, ax=axes, location="right", label="Planktic %")
        plt.suptitle("Foraminifera Abundance Comparison")

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axes

    def plot_benthic(self, cores, core_names=None, figsize=(3, 6), ylim=270, color="#854442",
                     xlim=None, savefig=False, savepath="compare_benthic.png", dpi=350):
        """
        Plot normalized benthic foraminifera counts from multiple cores.

        Args:
            cores (list): List of `Forams` objects.
            core_names (list): Optional list of names corresponding to the cores.
            figsize (tuple): Size of each subplot (width, height).
            ylim (float): Maximum depth for y-axis.
            savefig (bool): Whether to save the figure.
            savepath (str): Path to save the figure.
            dpi (int): Resolution of the saved figure.
        """
        n = len(cores)
        if core_names is None:
            core_names = [f"Core {i+1}" for i in range(n)]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n,
            figsize=(figsize[0]*n, figsize[1]),
            sharex=True,
            sharey=True,
            layout="constrained"
        )

        if n == 1:
            axes = [axes]

        xmax = max([core.dataframe["norm_benthic"].max() for core in cores])

        for i, (core, ax) in enumerate(zip(cores, axes)):
            df = core.dataframe
            ax.barh(df.index, df["norm_benthic"], color=color)
            ax.set_title(core_names[i])
            # if forams abundance exceeds given xlim, annotate the bar
            if xlim:
                ax.set_xlim(0, xlim)

                for depth, benthic in zip(df.index, df["norm_benthic"]):
                    if benthic > xlim:
                        ax.annotate(
                            f"total: {benthic:.1f}",
                            xy=(xlim / 2, depth),
                            va="center",
                            fontsize=7.5,
                            color="#4b3832"
                        )
            else:
                ax.set_xlim(0, xmax)

            ax.set_ylim(0, max(ylim, df.index[-1]))
            ax.yaxis.set_inverted(True)
            ax.set_xlabel("Benthic individuals / 1 cc")
            ax.grid(True)
            if i == 0:
                ax.set_ylabel("Depth (cm)")
            else:
                ax.set_ylabel("")

        plt.suptitle("Benthic Foraminifera Abundance Comparison")

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axes


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
                1].set_title(f"{core_names[core + 1]}" if core_names else f"Core {core + 2}")
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

    def plot_depth_bars(self, core_name="Core", ax=None, figsize=(3, 8), bar_height=1.5,
                        bar_width=0.2, ylim=265, colors=None, savefig=False,
                        savepath="bryo_abundance.png", dpi=350, categories=None):

        self.validate_df()

        bryo_type = ["net", "branch", "flat"]

        if colors is None:
            colors = {"net": "#b88c8c", "branch": "#d6c7c7", "flat": "#9fb9bf"}
        else:
            colors = {k: v for k, v in zip(bryo_type, colors)}

        features = list(colors.keys())

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if categories is None:
            categories = sorted(self.dataframe["category"].dropna().unique())
        category_to_x = {cat: i for i, cat in enumerate(categories)}

        seen_labels = set()

        for idx, row in self.dataframe.iterrows():
            depth = idx
            cat = row["category"]
            if pd.isna(cat):
                continue

            x_center = category_to_x[cat]
            for i, feature in enumerate(features):
                if pd.notna(row[feature]) and row[feature] > 0:
                    offset = (i - 1) * bar_width
                    label = feature if feature not in seen_labels else None
                    ax.barh(y=depth,
                            width=bar_width,
                            left=x_center + offset,
                            height=bar_height,
                            color=colors[feature],
                            edgecolor="black",
                            linewidth=0.2,
                            label=label,
                            zorder=3)

                    seen_labels.add(feature)

        if ylim > self.dataframe.index[-1]:
            ax.set_ylim(0, ylim)
        else:
            ax.set_ylim(0, self.dataframe.index[-1] + 1)

        ax.invert_yaxis()
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)

        if ax is None:
            ax.set_title(f"Bryozoan Types by Depth in {core_name}")
            ax.set_xlabel("Category")
            ax.set_ylabel("Depth (cm)")
        else:
            ax.set_title(f"{core_name}")

        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        if ax is None:
            fig.legend(by_label.values(), by_label.keys(),
                       loc="outside lower center", ncol=2)

        plt.tight_layout()
        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, ax

    def compare_bryo_types(self, cores=None, core_names=[], figsize=(3, 8), bar_height=1.5, bar_width=0.2,
                           ylim=265, colors=None, savefig=False, savepath="bryo_type_comparison.png", dpi=350):
        """
        Compare bryozoan types across multiple cores using horizontal bar plots.
        """
        if cores is None:
            self.plot_depth_bars()
            return

        fig, axes = plt.subplots(nrows=1, ncols=len(cores),
                                 figsize=(figsize[0] * len(cores), figsize[1]),
                                 sharey=True, sharex=True)

        all_handles = []
        all_labels = []

        all_cats = set()
        for core in cores:
            all_cats.update(core.dataframe["category"].dropna().unique())
        all_categories = sorted(all_cats)

        for core, core_name, ax in zip(cores, core_names, axes):
            core.plot_depth_bars(core_name=core_name, ax=ax, bar_height=bar_height,
                                 bar_width=bar_width, ylim=ylim, colors=colors, categories=all_categories)
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        fig.supxlabel("Category")
        fig.supylabel("Depth (cm)")
        fig.suptitle("Bryozoan Types by Depth")

        # create legend
        by_label = {}
        label_map = {"flat": "F", "net": "N", "branch": "B"}
        for handle, label in zip(all_handles, all_labels):
            short_label = label_map.get(label, label)
            if short_label not in by_label:
                by_label[short_label] = handle

        fig.subplots_adjust(left=0.2)

        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            fontsize=8
        )

        plt.tight_layout()

        if savefig:
            plt.savefig(savepath, dpi=dpi)

        return fig, axes
