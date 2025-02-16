import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


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
            # Delegate method to the DataFrame
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

        # validate that all values are numeric
        bc_matrix = self.dataframe.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce")
        # convert percentages to fractions between 0 and 1
        bc_matrix = bc_matrix / 100

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
        # self.bc_square = squareform(self.bc_dist)

        return self.bc_dist

    def compute_squareform(self):
        """
        Convert the computed Bray-Curtis distances into a squareform.
        """
        # bc_distances = self.compute_BCD()
        # bc_squareform = squareform(bc_distances)
        # # turn the squareform array into a dataframe
        # sf_df = BCD(dataframe=pd.DataFrame(bc_squareform))
        if self.bc_dist is None:
            self.bc_dist = self.compute_BCD()

        if self.bc_square is None:
            self.bc_square = squareform(self.bc_dist)

        return pd.DataFrame(self.bc_square)
