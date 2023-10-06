""" Comparisons between monopolar estimation methods """


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

import json
import os
import mne
import pickle

# internal Imports
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults

group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")
group_figures_path = find_folders.get_monopolar_project_path(folder="GroupFigures")

incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]
segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    

################## method weighted by euclidean coordinates ##################
# only directional contacts
monopolar_fooof_euclidean_segmental = loadResults.load_fooof_monopolar_weighted_psd(
    fooof_spectrum = "periodic_spectrum",
    segmental = "yes",
    similarity_calculation = "inverse_distance"
)

monopolar_fooof_euclidean_segmental = pd.concat([
    monopolar_fooof_euclidean_segmental["postop_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu3m_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu12m_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu18or24m_monopolar_Dataframe"],])

# add column with method name
monopolar_fooof_euclidean_segmental_copy = monopolar_fooof_euclidean_segmental.copy()
monopolar_fooof_euclidean_segmental_copy["method"] = "euclidean_directional"

# columns: coord_z, coord_xy, session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank


################## method by JLB ##################
# only directional contacts
monopolar_fooof_JLB = loadResults.load_fooof_monopolar_JLB()
# columns: session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

# add column with method name
monopolar_fooof_JLB_copy = monopolar_fooof_JLB.copy()
monopolar_fooof_JLB_copy["method"] = "JLB_directional"




def spearman_monopol_fooof_beta_methods(
        method_1:str,
        method_2:str
):
    
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - method_1: "JLB_directional", "euclidean_directional", "Strelow"
        - method_2: "JLB_directional", "euclidean_directional", "Strelow"
    """

    # results
    spearman_result = {}
    sample_size_dict = {}
    
    # get data from method 1
    if method_1 == "JLB_directional":
        method_1_data = monopolar_fooof_JLB_copy
            
    elif method_1 == "euclidean_directional":
        method_1_data = monopolar_fooof_euclidean_segmental_copy
    
    elif method_1 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # get data from method 2
    if method_2 == "JLB_directional":
        method_2_data = monopolar_fooof_JLB_copy
            
    elif method_2 == "euclidean_directional":
        method_2_data = monopolar_fooof_euclidean_segmental_copy
    
    elif method_2 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # Perform spearman correlation for every session separately and within each STN
    for ses in incl_sessions:

        method_1_session = method_1_data.loc[method_1_data.session == ses]
        method_2_session = method_2_data.loc[method_2_data.session == ses]

        # find STNs with data from both methods
        stn_unique_method_1 = list(method_1_session.subject_hemisphere.unique())
        stn_unique_method_2 = list(method_2_session.subject_hemisphere.unique())

        stn_comparison_list = list(set(stn_unique_method_1) & set(stn_unique_method_2))
        stn_comparison_list.sort()

        comparison_df_method_1 = method_1_session.loc[method_1_session["subject_hemisphere"].isin(stn_comparison_list)]
        comparison_df_method_2 = method_2_session.loc[method_2_session["subject_hemisphere"].isin(stn_comparison_list)]

        comparison_df = pd.concat([comparison_df_method_1, comparison_df_method_2], axis=0)

        for sub_hem in stn_comparison_list:

            # only run, if sub_hem STN exists in both session Dataframes
            if sub_hem not in comparison_df.subject_hemisphere.values:
                print(f"{sub_hem} is not in the comparison Dataframe.")
                continue

            # only take one electrode at both sessions and get spearman correlation
            stn_comparison = comparison_df.loc[comparison_df["subject_hemisphere"] == sub_hem]

            stn_method_1 = stn_comparison.loc[stn_comparison.method == method_1]
            stn_method_2 = stn_comparison.loc[stn_comparison.method == method_2]

            # Spearman correlation between beta average
            spearman_beta_stn = stats.spearmanr(stn_method_1.estimated_monopolar_beta_psd.values, stn_method_2.estimated_monopolar_beta_psd.values)

            # store values in a dictionary
            spearman_result[f"{ses}_{sub_hem}"] = [method_1, method_2, ses, sub_hem, spearman_beta_stn.statistic, spearman_beta_stn.pvalue]
        

    # save result
    results_DF = pd.DataFrame(spearman_result)
    results_DF.rename(index={0: "method_1", 1: "method_2", 2: "session", 3: "subject_hemisphere", 4: f"spearman_r", 5: f"pval"}, inplace=True)
    results_DF = results_DF.transpose()

    # save Dataframe to Excel
    results_DF_copy = results_DF.copy()

    # add new column: significant yes, no
    significant_correlation = results_DF_copy["pval"] < 0.05
    results_DF_copy["significant_correlation"] = ["yes" if cond else "no" for cond in significant_correlation]

    # save as Excel
    results_DF_copy.to_excel(os.path.join(group_results_path, f"fooof_monopol_beta_correlations_per_stn_{method_1}_{method_2}.xlsx"), 
                                    sheet_name="monopolar_beta_correlations",
                                    index=False)
    print("file: ", 
          f"fooof_monopol_beta_correlations_per_stn_{method_1}_{method_2}xlsx",
          "\nwritten in: ", group_results_path)


    # get sample size
    for ses in incl_sessions:

        ses_df = results_DF_copy.loc[results_DF_copy.session == ses]
        ses_count = ses_df["session"].count()

        spearman_mean = ses_df.spearman_r.mean()
        spearman_median = ses_df.spearman_r.median()
        spearman_std = np.std(ses_df.spearman_r)

        # calculate how often significant?
        significant_count = ses_df.loc[ses_df.significant_correlation == "yes"]
        significant_count = significant_count["session"].count()
        percentage_significant = significant_count / ses_count

        sample_size_dict[f"{ses}"] = [ses, ses_count, spearman_mean, spearman_median, spearman_std,
                                      significant_count, percentage_significant]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(index={
        0: "session", 
        1: "sample_size",
        2: "spearman_mean",
        3: "spearman_median",
        4: "spearman_std",
        5: "significant_count",
        6: "percentage_significant"}, inplace=True)
    sample_size_df = sample_size_df.transpose()


    
    return results_DF_copy, sample_size_df, stn_comparison




        












        





