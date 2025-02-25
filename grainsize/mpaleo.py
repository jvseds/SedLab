import pandas as pd
import matplotlib.pyplot as plt


class Forams(object):

    def __init__(self, dataframe=None, fname=None, index_col=0, header=0):
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.dataframe = pd.read_csv(
                fname, index_col=index_col, header=header
            )
        else:
            self.dataframe = None
