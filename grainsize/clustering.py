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

    def __repr__(self):
        if self.dataframe is not None:
            return repr(self.dataframe)
        return "An empty BCD object with no dataframe"

    # def __getattribute__(self, item):
    #     if item == "dataframe":
    #         return object.__getattribute__(self, "dataframe")
    #     dataframe = object.__getattribute__(self, "dataframe")
    #     if dataframe is not None and hasattr(dataframe, item):
    #         return getattr(dataframe, item)
    #     return object.__getattribute__(self, item)

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
        # validate that all values are numeric
        bc_matrix = self.dataframe.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce")
        # drop depth index and convert data to np array
        bc_matrix = bc_matrix.values
        # convert percentages to fractions between 0 and 1
        bc_matrix = bc_matrix / 100

        return bc_matrix

    def compute_BCD(self):
        """
        Compute Bray-Curtis distance between samples.
        """
        bc_matrix = self.create_CB_matrix()
        bc_distances = pdist(bc_matrix, metric='braycurtis')

        return bc_distances

    def compute_squareform(self):
        """
        Convert the computed Bray-Curtis distances into a squareform.
        """
        bc_distances = self.compute_BCD()
        bc_squareform = squareform(bc_distances)
        # turn the squareform array into a dataframe
        sf_df = BCD(dataframe=pd.DataFrame(bc_squareform))

        return sf_df
