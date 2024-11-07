# building a GrainSize class based on functions
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas.io.parsers import TextFileReader


# define a GrainSize class
class GrainSize(object):

    def __init__(self, fname, sheet_name="raw_data", skiprows=1):
        # complete more attributes as needed
        self.dataframe = None
        if fname:
            self.load_data(io=fname, sheet_name=sheet_name, skiprows=skiprows)

    def load_data(self, fname, sheet_name="raw_data", skiprows=1, **kwargs):
        """Loads core data from a MasterSizer 3000 Excel file into a DataFrame.

                Parameters:
                - fname: Filename or path of the Excel file.
                - sheet_name: Name of the sheet in the Excel file to read.
                - skiprows: Number of rows to skip at the beginning of the sheet.

                Returns:
                - A pandas DataFrame containing the imported data.
                """
        df = pd.read_excel(io=fname, sheet_name=sheet_name, skiprows=skiprows)
        self.dataframe = df
        return df

    def set_gs_index(self):
        """Sets the DataFrame's index to the 'grain_size' column."""
        if self.dataframe is not None:
            self.dataframe = self.dataframe.set_index("grain_size").rename_axis("grain_size")

        return self.dataframe

    def rename_range_col(self):
        """Finds the DataFrame column with minimum amount of null values
        and renames it 'grain_size'."""

        # find column with minimum amount of nulls
        min_null_col = self.dataframe.isnull().sum().idxmin()
        self.dataframe.rename(mapper={min_null_col: "grain_size"}, axis=1, inplace=True)

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
        cols_to_drop = [col for col in self.dataframe.columns if to_drop in col]
        self.dataframe = self.dataframe.drop(labels=cols_to_drop, axis=1)

        return self.dataframe

    def rename_headers(self):
        """Renames column headers as respective depth.
        *** Use after transposing the matrix, so the depth is represented as columns."""

        # rename column headers with "depths" instead of original file headers
        depth_list = self.get_depths()
        mapper = {self.dataframe.columns[i]: float(depth_list[i]) for i in range(len(self.dataframe.columns))}
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

        # get sample depths within the core - maybe unnecessary
        # depths = self.get_depths()

        # rename column headers
        self.rename_headers()
        # transpose the DataFrame
        self.dataframe = self.dataframe.transpose()
        # drop the column which is labels "Size Classes"
        self.dataframe = self.dataframe.drop(labels="Size Classes (μm)", axis=1)
        # rename index
        self.dataframe.rename_axis("depth", inplace=True)
        # replace remaining NaN with 0
        self.dataframe.fillna(0, inplace=True)
        # sort index by depth
        self.dataframe = self.sort_by_depth()

        return self.dataframe
