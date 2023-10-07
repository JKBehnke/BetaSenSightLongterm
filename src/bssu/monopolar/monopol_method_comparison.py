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
from .. utils import load_data_files as load_data

group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")
group_figures_path = find_folders.get_monopolar_project_path(folder="GroupFigures")

incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]
segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    

################## beta ranks of directional contacts from externalized LFP ##################
# only directional contacts
externalized_fooof_beta_ranks = load_data.load_externalized_pickle(filename = "fooof_externalized_beta_ranks_directional_contacts")

# add column with method name
externalized_fooof_beta_ranks_copy = externalized_fooof_beta_ranks.copy()
externalized_fooof_beta_ranks_copy["method"] = "externalized"
externalized_fooof_beta_ranks_copy["session"] = "postop"
externalized_fooof_beta_ranks_copy["estimated_monopolar_beta_psd"] = externalized_fooof_beta_ranks_copy["beta_average"]

# drop columns 
externalized_fooof_beta_ranks_copy.drop(columns=[
    'subject', 
    'hemisphere',
    "fooof_error", 
    "fooof_r_sq", 
    "fooof_exponent",
    "fooof_offset",
    "fooof_power_spectrum",
    "periodic_plus_aperiodic_power_log",
    'fooof_periodic_flat',
    'fooof_number_peaks', 
    'alpha_peak_CF_power_bandWidth',
    'low_beta_peak_CF_power_bandWidth', 
    'high_beta_peak_CF_power_bandWidth',
    'beta_peak_CF_power_bandWidth', 
    'gamma_peak_CF_power_bandWidth',
    'beta_average', 
    ], inplace=True)

# drop rows of subject 052 Right, because directional contact 2C was used as common reference, so there is no data for contact 2C
externalized_fooof_beta_ranks_copy.reset_index(drop=True, inplace=True)
externalized_fooof_beta_ranks_copy.drop(externalized_fooof_beta_ranks_copy[externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "052_Right"].index, inplace=True)
externalized_fooof_beta_ranks_copy.drop(externalized_fooof_beta_ranks_copy[externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "048_Right"].index, inplace=True)

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
monopolar_fooof_euclidean_segmental_copy["beta_rank"] = monopolar_fooof_euclidean_segmental_copy["rank"]
monopolar_fooof_euclidean_segmental_copy.drop(columns=["rank"], inplace=True)

# columns: coord_z, coord_xy, session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank


################## method by JLB ##################
# only directional contacts
monopolar_fooof_JLB = loadResults.load_fooof_monopolar_JLB()
# columns: session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

# add column with method name
monopolar_fooof_JLB_copy = monopolar_fooof_JLB.copy()
monopolar_fooof_JLB_copy["method"] = "JLB_directional"
monopolar_fooof_JLB_copy["beta_rank"] = monopolar_fooof_JLB_copy["rank"]
monopolar_fooof_JLB_copy.drop(columns=["rank"], inplace=True)


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

            # contacts with beta rank 1 and 2
            # method 1:
            rank1_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 1.0]
            rank1_method_1 = rank1_method_1.contact.values[0]

            rank2_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 2.0]
            rank2_method_1 = rank2_method_1.contact.values[0]

            rank_1_and_2_method_1 = [rank1_method_1, rank2_method_1]
            
            # method 2:
            rank1_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 1.0]
            rank1_method_2 = rank1_method_2.contact.values[0]

            rank2_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 2.0]
            rank2_method_2 = rank2_method_2.contact.values[0]

            rank_1_and_2_method_2 = [rank1_method_2, rank2_method_2]

            # yes if contact with rank 1 is the same
            if rank1_method_1 == rank1_method_2:
                compare_rank_1_contact = "same"
            
            else:
                compare_rank_1_contact = "different"
            
            # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
            # if set(rank_1_and_2_method_1) == set(rank_1_and_2_method_2):
            #     compare_rank_1_and_2_contacts = "same"

            # check if at least one contact selected as beta rank 1 or 2 match for both methods
            if set(rank_1_and_2_method_1).intersection(set(rank_1_and_2_method_2)):
                compare_rank_1_and_2_contacts = "at_least_one_contact_match"
            
            else:
                compare_rank_1_and_2_contacts = "no_contacts_match"


            # store values in a dictionary
            spearman_result[f"{ses}_{sub_hem}"] = [method_1, method_2, ses, sub_hem, spearman_beta_stn.statistic, spearman_beta_stn.pvalue,
                                                   rank1_method_1, rank1_method_2, rank_1_and_2_method_1, rank_1_and_2_method_2, 
                                                   compare_rank_1_contact, compare_rank_1_and_2_contacts]
        

    # save result
    results_DF = pd.DataFrame(spearman_result)
    results_DF.rename(index={0: "method_1", 1: "method_2", 2: "session", 3: "subject_hemisphere", 4: f"spearman_r", 5: f"pval",
                             6: "contact_rank_1_method_1", 7: "contact_rank_1_method_2", 8: "contacts_rank_1_2_method_1", 9: "contacts_rank_1_2_method_2",
                             10: "compare_rank_1_contact", 11: "compare_rank_1_and_2_contacts"}, inplace=True)
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

        # count how often compare_rank_1_contact same
        same_rank_1 = ses_df.loc[ses_df.compare_rank_1_contact == "same"]
        same_rank_1 = same_rank_1["session"].count()
        precentage_same_rank_1 = same_rank_1 / ses_count

        # count how often there is at least one matching contact in compare_rank_1_and_2_contact 
        same_rank_1_and_2 = ses_df.loc[ses_df.compare_rank_1_and_2_contacts == "at_least_one_contact_match"]
        same_rank_1_and_2 = same_rank_1_and_2["session"].count()
        precentage_same_rank_1_and_2 = same_rank_1_and_2 / ses_count

        sample_size_dict[f"{ses}"] = [ses, ses_count, spearman_mean, spearman_median, spearman_std,
                                      significant_count, percentage_significant, 
                                      same_rank_1, precentage_same_rank_1,
                                      same_rank_1_and_2, precentage_same_rank_1_and_2]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(index={
        0: "session", 
        1: "sample_size",
        2: "spearman_mean",
        3: "spearman_median",
        4: "spearman_std",
        5: "significant_count",
        6: "percentage_significant",
        7: "same_rank_1",
        8: "precentage_same_rank_1",
        9: "at_least_one_same_contact_rank_1_and_2",
        10: "precentage_at_least_one_same_contact_rank_1_and_2"}, inplace=True)
    sample_size_df = sample_size_df.transpose()


    
    return results_DF_copy, sample_size_df, stn_comparison



# TODO: count how often beta rank 1 is equally selected in both methods
# TODO: count how often beta rank 1 and beta rank 2 are bothe equally selected in both methods


        

def spearman_validation_monopol_fooof(
        method:str
):
    
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - method: "JLB_directional", "euclidean_directional", "Strelow"
    """

    # results
    spearman_result = {}
    sample_size_dict = {}
    
    # get only postop data from method
    if method == "JLB_directional":
        method_data = monopolar_fooof_JLB_copy
        method_data = method_data.loc[method_data.session == "postop"]
            
    elif method == "euclidean_directional":
        method_data = monopolar_fooof_euclidean_segmental_copy
        method_data = method_data.loc[method_data.session == "postop"]
    
    elif method == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # get data from externalized LFP
    externalized_data = externalized_fooof_beta_ranks_copy
    

    # Perform spearman correlation for every session separately and within each STN

    # find STNs with data from both methods and externalized
    stn_unique_method = list(method_data.subject_hemisphere.unique())
    stn_unique_externalized = list(externalized_data.subject_hemisphere.unique())

    stn_comparison_list = list(set(stn_unique_method) & set(stn_unique_externalized))
    stn_comparison_list.sort()

    comparison_df_method = method_data.loc[method_data["subject_hemisphere"].isin(stn_comparison_list)]
    comparison_df_externalized = externalized_data.loc[externalized_data["subject_hemisphere"].isin(stn_comparison_list)]

    comparison_df = pd.concat([comparison_df_method, comparison_df_externalized], axis=0)

    for sub_hem in stn_comparison_list:

        # only run, if sub_hem STN exists in both session Dataframes
        if sub_hem not in comparison_df.subject_hemisphere.values:
            print(f"{sub_hem} is not in the comparison Dataframe.")
            continue

        # only take one electrode at both sessions and get spearman correlation
        stn_comparison = comparison_df.loc[comparison_df["subject_hemisphere"] == sub_hem]

        stn_method = stn_comparison.loc[stn_comparison.method == method]
        stn_externalized = stn_comparison.loc[stn_comparison.method == "externalized"]

        # Spearman correlation between beta average
        spearman_beta_stn = stats.spearmanr(stn_method.estimated_monopolar_beta_psd.values, stn_externalized.estimated_monopolar_beta_psd.values)

        # contacts with beta rank 1 and 2
        # method:
        rank1_method = stn_method.loc[stn_method.beta_rank == 1.0]
        rank1_method = rank1_method.contact.values[0]

        rank2_method = stn_method.loc[stn_method.beta_rank == 2.0]
        rank2_method = rank2_method.contact.values[0]

        rank_1_and_2_method = [rank1_method, rank2_method]
        
        # externalized:
        rank1_externalized = stn_externalized.loc[stn_externalized.beta_rank == 1.0]
        rank1_externalized = rank1_externalized.contact.values[0]

        rank2_externalized = stn_externalized.loc[stn_externalized.beta_rank == 2.0]
        # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank2_externalized.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 2 contact in the externalized recording.")
            continue

        rank2_externalized = rank2_externalized.contact.values[0]
        

        rank_1_and_2_externalized = [rank1_externalized, rank2_externalized]

        # yes if contact with rank 1 is the same
        if rank1_method == rank1_externalized:
            compare_rank_1_contact = "same"
        
        else:
            compare_rank_1_contact = "different"
        
        # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
        # if set(rank_1_and_2_method_1) == set(rank_1_and_2_method_2):
        #     compare_rank_1_and_2_contacts = "same"

        # check if at least one contact selected as beta rank 1 or 2 match for both methods
        if set(rank_1_and_2_method).intersection(set(rank_1_and_2_externalized)):
            compare_rank_1_and_2_contacts = "at_least_one_contact_match"
        
        else:
            compare_rank_1_and_2_contacts = "no_contacts_match"
    
        # store values in a dictionary
        spearman_result[f"{sub_hem}"] = [method, "externalized", "postop", sub_hem, spearman_beta_stn.statistic, spearman_beta_stn.pvalue,
                                         rank1_method, rank1_externalized, rank_1_and_2_method, rank_1_and_2_externalized, 
                                         compare_rank_1_contact, compare_rank_1_and_2_contacts]
    

    # save result
    results_DF = pd.DataFrame(spearman_result)
    results_DF.rename(index={0: "method", 1: "externalized", 2: "session", 3: "subject_hemisphere", 4: f"spearman_r", 5: f"pval",
                             6: "contact_rank_1_method", 7: "contact_rank_1_externalized", 8: "contacts_rank_1_2_method", 9: "contacts_rank_1_2_externalized",
                             10: "compare_rank_1_contact", 11: "compare_rank_1_and_2_contacts"}, inplace=True)
    results_DF = results_DF.transpose()

    # save Dataframe to Excel
    results_DF_copy = results_DF.copy()

    # add new column: significant yes, no
    significant_correlation = results_DF_copy["pval"] < 0.05
    results_DF_copy["significant_correlation"] = ["yes" if cond else "no" for cond in significant_correlation]

    # save as Excel
    results_DF_copy.to_excel(os.path.join(group_results_path, f"fooof_monopol_beta_correlations_per_stn_{method}_externalized.xlsx"), 
                                    sheet_name="monopolar_beta_correlations",
                                    index=False)
    print("file: ", 
          f"fooof_monopol_beta_correlations_per_stn_{method}_externalized.xlsx",
          "\nwritten in: ", group_results_path)


    # get sample size
    result_count = results_DF_copy["subject_hemisphere"].count()

    spearman_mean = results_DF_copy.spearman_r.mean()
    spearman_median = results_DF_copy.spearman_r.median()
    spearman_std = np.std(results_DF_copy.spearman_r)

    # calculate how often significant?
    significant_count = results_DF_copy.loc[results_DF_copy.significant_correlation == "yes"]
    significant_count = significant_count["subject_hemisphere"].count()
    percentage_significant = significant_count / result_count

    # count how often compare_rank_1_contact same
    same_rank_1 = results_DF_copy.loc[results_DF_copy.compare_rank_1_contact == "same"]
    same_rank_1 = same_rank_1["subject_hemisphere"].count()
    precentage_same_rank_1 = same_rank_1 / result_count

    # count how often there is at least one matching contact in compare_rank_1_and_2_contact 
    same_rank_1_and_2 = results_DF_copy.loc[results_DF_copy.compare_rank_1_and_2_contacts == "at_least_one_contact_match"]
    same_rank_1_and_2 = same_rank_1_and_2["subject_hemisphere"].count()
    precentage_same_rank_1_and_2 = same_rank_1_and_2 / result_count

    sample_size_dict = {
        "sample_size": [result_count],
        "spearman_mean": [spearman_mean],
        "spearman_median": [spearman_median],
        "spearman_std": [spearman_std],
        "significant_count": [significant_count],
        "percentage_significant": [percentage_significant],
        "same_rank_1_count": [same_rank_1],
        "percentage_same_rank_1": [precentage_same_rank_1],
        "same_rank_1_and_2_count": [same_rank_1_and_2],
        "percentage_same_rank_1_and_2": [precentage_same_rank_1_and_2]
        }

    sample_size_df = pd.DataFrame(sample_size_dict)

    return results_DF_copy, sample_size_df, stn_comparison


