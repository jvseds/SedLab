import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from grainsize.core import GrainSize, XRF


class BCD(object):
    def __init__(self, dataframe=None, fname=None, index_col=0, header=0):
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.dataframe = pd.read_csv(
                fname, index_col=index_col, header=header)
        else:
            self.dataframe = None

        self.bc_dist = None
        self.bc_square = None

        # compute Bray Curtis distance when dataframe is initialized
        if self.dataframe is not None:
            self.compute_BCD()

    def __repr__(self):
        if self.dataframe is not None:
            return repr(self.dataframe)
        return "An empty BCD object with no dataframe"

    def __getattr__(self, item):
        """
        Delegate to the DataFrame class for methods that are not defined in this class.
        """
        if self.dataframe is None:
            raise AttributeError(
                f"'BCD' object has no attribute '{item}' because the dataframe is empty.")

        try:
            return getattr(self.dataframe, item)
        except AttributeError:
            raise AttributeError(
                f"'BCD' object or its 'dataframe' has no attribute '{item}'")

    def create_CB_matrix(self):
        """
        Create a matrix for Bray-Curtis distance computation.
        """
        if self.dataframe is None:
            raise ValueError(
                "Dataframe for this BCD object is None, cannot compute distance")

        # validate that all values are numeric while preserving index
        bc_matrix = self.dataframe.apply(
            pd.to_numeric, errors="coerce")
        # convert percentages to fractions between 0 and 1
        bc_matrix = bc_matrix.div(100)

        return bc_matrix.values

    def compute_BCD(self):
        """
        Compute Bray-Curtis distance between samples.
        """
        if self.dataframe is None:
            raise ValueError(
                "Dataframe for this BCD object is None, cannot compute distance")

        bc_matrix = self.create_CB_matrix()
        self.bc_dist = pdist(bc_matrix, metric="braycurtis")

        return self.bc_dist

    def compute_squareform(self):
        """
        Convert the computed Bray-Curtis distances into a squareform.
        """

        if self.bc_dist is None:
            self.bc_dist = self.compute_BCD()

        if self.bc_square is None:
            self.bc_square = squareform(self.bc_dist)

        return pd.DataFrame(self.bc_square)

    def plot_dendrogram(self, method="average", core_name="Core", figsize=(10, 8), savefig=False,
                        save_path="dendrogram.png", dpi=350):
        """
        Plot a dendrogram from the Bray-Curtis distances.
        """
        if self.bc_dist is None:
            self.bc_dist = self.compute_BCD()
        if self.bc_square is None:
            self.bc_square = squareform(self.bc_dist)

        la_matrix = linkage(self.bc_dist, method=method)

        fig, ax = plt.subplots(nrows=1, figsize=figsize)
        dendrogram(la_matrix, labels=self.dataframe.index,
                   orientation="right", ax=ax)
        ax.set_xlabel("Dissimilarity")
        ax.set_ylabel("Depth (cm)")
        ax.set_title(f"Bray-Curtis Dissimilarity for {core_name}")

        if savefig:
            plt.savefig(save_path, dpi=dpi)

        return fig, ax

    def interpolate_depth(self, new_depth=None, method="linear"):
        """
        Interpolate the grain size data to a new depth scale.
        """
        depth_col = self.dataframe.index

        if new_depth is None:
            new_depth = np.arange(depth_col.min(), depth_col.max() + 1, 1)

        interpolated = pd.DataFrame(index=new_depth)
        for col in self.dataframe.columns:
            interpolated[col] = np.interp(
                new_depth, depth_col, self.dataframe[col])

        return interpolated

    def merge_interp(self, other_BCD, new_depth=None, method="linear", core1_name="Core1", core2_name="Core2"):
        """
        Merge two BCD objects and interpolate the grain size data to a new depth scale.
        """
        if not isinstance(other_BCD, BCD):
            raise TypeError("other_BCD must be a BCD object")

        interp_self = self.interpolate_depth(
            new_depth, method).add_prefix(f"{core1_name}_")
        interp_other = other_BCD.interpolate_depth(
            new_depth, method).add_prefix(f"{core2_name}_")
        merged = interp_self.merge(
            interp_other, left_index=True, right_index=True)

        return BCD(dataframe=merged)

    @classmethod
    def from_grain_size(cls, gs_obj: GrainSize):
        if not isinstance(gs_obj, GrainSize):
            raise TypeError("Expected a GrainSize object")
        return cls(dataframe=gs_obj.dataframe)

    @classmethod
    def from_xrf(cls, xrf_obj: XRF):
        if not isinstance(xrf_obj, XRF):
            raise TypeError("Expected an XRF object")
        return cls(dataframe=xrf_obj.dataframe)
