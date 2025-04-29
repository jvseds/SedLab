#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from grainsize import GrainSize
from grainsize import XRF

# import both parts of sr21 data
sr21_a_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\sedimentary-records\msc_github_code\SR21P7_GS_2023_11_08.xlsx"
sr21_b_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\sedimentary-records\msc_github_code\SR21P7_GS_2023_11_22.xlsx"

# instantiate 2 GrainSize objects
sr21_top = GrainSize(fname=sr21_a_path)
sr21_bottom = GrainSize(fname=sr21_b_path)

# clean the data
sr21_top.clean_data()
sr21_bottom.clean_data()

# concat into one object
sr21_df = pd.concat([sr21_top, sr21_bottom])
sr21 = GrainSize(dataframe=sr21_df)

# create a normalized GrainSize object for sr21 data
norm_sr21 = sr21.normalize_gs()
# print(norm_sr21)

# create a categorized df for sr21
sr21_cats_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\cats_sr21.csv"
sr21_cats = norm_sr21.create_categories()
# print(sr21_cats)

sr21_med_mode = norm_sr21.find_median_mode()
# print(sr21_med_mode)

# path for saving as a csv
sr21_stats_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\stats_sr21.csv"
# save as a csv file
# sr21_stats = sr21.create_stats_df(save=True, fpath=sr21_stats_path)

sr21_stats = sr21.create_stats_df()

print(sr21_stats)

# import and create statistics df for sr19 data

# yup, this is it! 2024-04-01
sr19_norm = pd.read_csv(
    r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\sr19_clean\SR19_norm_2024-03-11.csv"
).set_index("depth")

sr19_norm = GrainSize(dataframe=sr19_norm)
sr19_stats_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\stats_sr19.csv"
sr19_norm_stats = sr19_norm.create_stats_df(save=False, fpath=sr19_stats_path)
# print(sr19_norm_stats)


sr19_cats_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\cats_sr19.csv"
sr19_norm_cats = sr19_norm.create_categories(save=False, fpath=sr19_cats_path)
# print(sr19_norm_cats)

sr21_cats = norm_sr21.create_categories(save=False, fpath=sr21_cats_path)
# print(sr21_stats)

sr19_stats_plot_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\plot_stats_sr19.png"
sr19_norm_stats.plot_stats(
    core_name="SR19P4", save_fig=False, fpath=sr19_stats_plot_path)

sr21_stats_plot_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\plot_stats_sr21.png"
sr21_stats.plot_stats(core_name="SR21-P7", save_fig=False,
                      fpath=sr21_stats_plot_path)

# Now let's try to import Timor's data from 2019 and clean it
timor_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\SRP19_timor.xlsx"
timor_data = GrainSize(fname=timor_path, skiprows=0, sheet_name="P4_data")
# clean the data
timor_data = GrainSize(dataframe=timor_data.clean_data())
# print("*********** Timor's data ***********")
# print(timor_data)

# normalize timor_data
timor_norm = timor_data.normalize_gs()
# print(timor_norm)

# concatenate all of SR19-P4 data
unified_sr19 = pd.concat([sr19_norm, timor_norm]).sort_index(axis=0).fillna(0)
# print(unified_sr19)
sr19_full = GrainSize(dataframe=unified_sr19)
# print(sr19_full)
# sr19_full.dataframe.to_csv("sr19_complete.csv")

# create categories dataframe for sr19_full
sr19_full_cat_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\cats_sr19_full.csv"
sr19_full_cats = sr19_full.create_categories(
    save=False, fpath=sr19_full_cat_path)
# create statistics dataframe for sr19_full
sr19_full_stats_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\stats_sr19_full.csv"
sr19_full_stats = sr19_full.create_stats_df(
    save=False, fpath=sr19_full_stats_path)
# plot statistics and save
sr19_full_stats_plot_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\stats_sr19_full_02.png"
sr19_full_stats.plot_stats(marker=".", save_fig=False, core_name="SR19-P4",
                           fpath=sr19_full_stats_plot_path)

sr19_fines_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\sr19_fine_fraction.csv"
sr19_full_fines = sr19_full.fine_fraction(
    save=False, save_path=sr19_fines_path)
# print(sr19_full_fines)

# try to plot GS stats including the < 63 um fraction
sr19_stats_fines_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\sr19_stats_with_fines_lims_small.png"
# sr19_full_stats["fines"] = sr19_full_stats.concat(sr19_full_fines["total"])
sr19_with_fines = GrainSize(dataframe=pd.concat(
    [sr19_full_stats, sr19_full_fines["total"]], axis=1))
# save data to a csv file
# csv_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\full_stats\sr19_stats_fines.csv"
# sr19_with_fines.dataframe.to_csv(csv_path)
# print(sr19_with_fines)
sr19_with_fines.plot_stats_fines(figsize=(10, 8), marker=".", linestyle="--", core_name="SR19-P4", fine_col="total",
                                 save_fig=False, fpath=sr19_stats_fines_path)

sr21_fines_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\clean_gs_files\sr21_stats_with_fines_lims_small.png"
sr21_fines = norm_sr21.fine_fraction()
sr21_with_fines = GrainSize(dataframe=pd.concat(
    [sr21_stats, sr21_fines["total"]], axis=1))
# save data to a csv file
# csv_path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\msc_github_code\figures\full_stats\sr21_stats_fines.csv"
# sr21_with_fines.dataframe.to_csv(csv_path)

sr21_with_fines.plot_stats_fines(figsize=(10, 8), marker=".", core_name="SR21-P7", fine_col="total",
                                 save_fig=False, fpath=sr21_fines_path)

# # -------- XRF Data --------
# # SR21 XRF
# path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\results\XRF\xrf_calibrated\XRF\SR21_calibrated.xlsx"
# sr21_xrf = XRF(fname=path,
#                usecols="I:BF")
# # print(sr21_xrf)
# sr21_ppm = sr21_xrf.to_ppm()
# sr21_ppm = sr21_ppm.clean_data()
# # print(sr21_ppm)

# # print(sr19_with_fines.dataframe.index)

# sr21_ppm.plot_elements(core_name="SR21-P7", figsize=(16, 12), savefig=False, savepath="sr21_xrf_full_small.png")

# ratios = [("Ti", "Ca"), ("Ti", "Al"),
#           ("Ba", "Al"), ("Cr", "Al"),
#           ("K", "Al"), ("Ca", "Al"),
#           ("Si", "Al"), ("Fe", "Si")]

# sr21_ppm.plot_ratios(core_name="SR21-P7", ratio_list=ratios, savefig=False, savepath="sr21_xrf_ratios_small.png")

# # sr21_ppm.dataframe.to_csv("sr21_ppm.csv")

# # sr19 XRF
# path = r"C:\Users\jvaal\OneDrive\Documents\Academics\msc_thesis\results\XRF\xrf_calibrated\XRF\SR19_calibrated.xlsx"
# sr19_xrf = XRF(fname=path,
#                sheet_name="Sheet1",
#                header=0,
#                usecols="H:AZ",
#                nrows=27,
#                index_col=0)
# sr19_ppm = sr19_xrf.to_ppm()
# sr19_ppm = sr19_ppm.clean_data()
# sr19_ppm.dataframe.dropna(how="all", axis=1, inplace=True)

# sr19_ppm.plot_elements(core_name="SR19-P4", figsize=(16, 12), savefig=False, savepath="sr19_xrf_full_small.png")
# sr19_ppm.plot_ratios(core_name="SR19-P4", ratio_list=ratios, savefig=False, savepath="sr19_xrf_ratios_small.png")
# # print(sr19_ppm)
# # sr19_ppm.to_csv("sr19_ppm.csv")
