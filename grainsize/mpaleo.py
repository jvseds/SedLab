import pandas as pd
import matplotlib.pyplot as plt
import warnings


class Forams(object):

    def __init__(self, dataframe=None, fname=None, size_fraction=125, volume=1.25, index_col=0, header=0):
        if dataframe is not None:
            self.dataframe = dataframe
        elif fname:
            self.dataframe = pd.read_csv(
                fname, index_col=index_col, header=header
            )
        else:
            self.dataframe = None

        self.size_fraction = size_fraction
        self.volume = volume

        # validate that the dataframe has the required columns containing planktic and benthic data
        if self.dataframe is not None:
            has_planktic, has_benthic, has_splits = self.validate_df(self)
            missing = []
            if not has_planktic:
                missing.append("planktic")
            if not has_benthic:
                missing.append("benthic")
            if not has_splits:
                missing.append("num_of_splits")
            warnings.warn(
                f"Pay Attention! Missing required column(s): {', '.join(missing)}")


def __repr__(self):
    if self.dataframe is not None:
        return repr(self.dataframe)
    return "An empty Forams object with no dataframe"


def __getattr__(self, item):
    if self.dataframe is None:
        raise AttributeError(
            f"`Forams` object has no attribute {item} because the dataframe is None"
        )

    try:
        return getattr(self.dataframe, item)
    except AttributeError:
        raise AttributeError(
            f"`Forams` object or its dataframe has no attribute {item}"
        )


def validate_df(self):
    if self.dataframe is None or self.dataframe.empty:
        return False, False

    if self.dataframe is not None:
        planktic_terms = ["planktic", "planktonic", "p", "P"]
        benthic_terms = ["benthic", "benthonic", "b", "B"]
        splits = ["split", "num_of_splits", "splits"]
        df_cols = [col.lower() for col in self.dataframe.columns]

        has_planktic = any(term.lower() in df_cols for term in planktic_terms)
        has_benthic = any(term.lower() in df_cols for term in benthic_terms)
        has_splits = any(term.lower() in df_cols for term in splits)

    return has_planktic, has_benthic, has_splits


def calc_totals(self):
    if self.dataframe is None or self.dataframe.empty:
        return None

    if "total" not in self.dataframe.columns:
        self.dataframe["total"] = self.dataframe["planktic"] + \
            self.dataframe["benthic"]


def normalize_per_1cc(self):

    total, splits = self.dataframe.columns["total"], splits = self.dataframe.columns["num_of_splits"]
    vol = self.volume

    self.dataframe["normalized_per_1cc"] = (total * splits) / vol


def calc_pb_ratio(self):
    if self.dataframe is None or self.dataframe.empty:
        return None

    self.dataframe["p/b_ratio"] = self.dataframe["planktic"] / \
        self.dataframe["benthic"]


def calc_planktic_percents(self):
    if self.dataframe is None or self.dataframe.empty:
        return None

    self.dataframe["planktic_percent"] = (
        self.dataframe["planktic"] / self.dataframe["total"] * 100
    )
