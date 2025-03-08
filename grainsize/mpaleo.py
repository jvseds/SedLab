import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    def plot_forams(self, core_name="Core", figsize=(6, 8), ylim=270, savefig=False, savepath="forams.png", dpi=350, limit_sm=False, sm_limit=20):
        if self.dataframe is None:
            raise ValueError("No dataframe available for plotting.")

        vmin = self.dataframe["planktic_percent"].min()
        vmax = min(self.dataframe["planktic_percent"].max(
        ), sm_limit) if limit_sm else self.dataframe["planktic_percent"].max()

        norm_planktic = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap="winter", norm=norm_planktic)
        fig, ax = plt.subplots(figsize=figsize)

        q3 = np.nanpercentile(self.dataframe["planktic_percent"], 75)

        for depth, total, planktic in zip(self.dataframe.index, self.dataframe["total"], self.dataframe["planktic_percent"]):
            ax.barh(depth, total, color=sm.to_rgba(planktic))

            if planktic == 100 or planktic >= q3:
                ax.text(total, depth, f"{planktic:.1f}%",
                        va='center', ha='left', fontsize=8, color='black')

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
