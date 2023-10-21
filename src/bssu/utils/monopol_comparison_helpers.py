""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..tfr import feats_ssd as feats_ssd
from ..utils import find_folders as find_folders
from ..utils import load_data_files as load_data
from ..utils import loadResults as loadResults

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")


################## beta ranks of directional contacts from externalized LFP ##################
################## FOOOF ##################


def load_externalized_fooof_data(fooof_version: str, reference=None):
    """
    Input:
        - fooof_version: str "v1" or "v2"
        - reference: str "bipolar_to_lowermost" or "no"

    """
    # only directional contacts
    # FOOOF version: only 1 Hz high-pass filtered
    externalized_fooof_beta_ranks = load_data.load_externalized_pickle(
        filename="fooof_externalized_beta_ranks_directional_contacts_only_high_pass_filtered",
        fooof_version=fooof_version,
        reference=reference,
    )

    # add column with method name
    externalized_fooof_beta_ranks_copy = externalized_fooof_beta_ranks.copy()
    externalized_fooof_beta_ranks_copy["method"] = "externalized_fooof"
    externalized_fooof_beta_ranks_copy["session"] = "postop"
    externalized_fooof_beta_ranks_copy["estimated_monopolar_beta_psd"] = externalized_fooof_beta_ranks_copy[
        "beta_average"
    ]

    # drop columns
    externalized_fooof_beta_ranks_copy.drop(
        columns=[
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
        ],
        inplace=True,
    )

    # drop rows of subject 052 Right, because directional contact 2C was used as common reference, so there is no data for contact 2C
    externalized_fooof_beta_ranks_copy.reset_index(drop=True, inplace=True)
    externalized_fooof_beta_ranks_copy.drop(
        externalized_fooof_beta_ranks_copy[
            externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "052_Right"
        ].index,
        inplace=True,
    )
    externalized_fooof_beta_ranks_copy.drop(
        externalized_fooof_beta_ranks_copy[
            externalized_fooof_beta_ranks_copy["subject_hemisphere"] == "048_Right"
        ].index,
        inplace=True,
    )

    return externalized_fooof_beta_ranks_copy


def load_externalized_ssd_data(reference=None):
    """
    Input:
        - reference: str "bipolar_to_lowermost" or "no"

    """
    ################## SSD ##################
    externalized_SSD_beta_ranks = load_data.load_externalized_pickle(
        filename="SSD_directional_externalized_channels", reference=reference
    )

    # add column with method name
    externalized_SSD_beta_ranks_copy = externalized_SSD_beta_ranks.copy()
    externalized_SSD_beta_ranks_copy["method"] = "externalized_ssd"
    externalized_SSD_beta_ranks_copy["session"] = "postop"
    externalized_SSD_beta_ranks_copy["estimated_monopolar_beta_psd"] = externalized_SSD_beta_ranks_copy["ssd_pattern"]

    # drop columns
    externalized_SSD_beta_ranks_copy.drop(
        columns=[
            'ssd_filtered_timedomain',
        ],
        inplace=True,
    )

    return externalized_SSD_beta_ranks_copy


def load_euclidean_method(fooof_version: str):
    """ """
    ################## method weighted by euclidean coordinates ##################
    # only directional contacts
    monopolar_fooof_euclidean_segmental = loadResults.load_fooof_monopolar_weighted_psd(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        segmental="yes",
        similarity_calculation="inverse_distance",
    )

    monopolar_fooof_euclidean_segmental = pd.concat(
        [
            monopolar_fooof_euclidean_segmental["postop_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu3m_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu12m_monopolar_Dataframe"],
            monopolar_fooof_euclidean_segmental["fu18or24m_monopolar_Dataframe"],
        ]
    )

    # add column with method name
    monopolar_fooof_euclidean_segmental_copy = monopolar_fooof_euclidean_segmental.copy()
    monopolar_fooof_euclidean_segmental_copy["method"] = "euclidean_directional"
    monopolar_fooof_euclidean_segmental_copy["beta_rank"] = monopolar_fooof_euclidean_segmental_copy["rank"]
    monopolar_fooof_euclidean_segmental_copy.drop(columns=["rank"], inplace=True)

    # columns: coord_z, coord_xy, session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

    return monopolar_fooof_euclidean_segmental_copy


def load_JLB_method(fooof_version: str):
    """ """
    ################## method by JLB ##################
    # only directional contacts
    monopolar_fooof_JLB = loadResults.load_pickle_group_result(
        filename="MonoRef_JLB_fooof_beta", fooof_version=fooof_version
    )
    # columns: session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank

    # add column with method name
    monopolar_fooof_JLB_copy = monopolar_fooof_JLB.copy()
    monopolar_fooof_JLB_copy["method"] = "JLB_directional"
    monopolar_fooof_JLB_copy["beta_rank"] = monopolar_fooof_JLB_copy["rank"]
    monopolar_fooof_JLB_copy.drop(columns=["rank"], inplace=True)

    return monopolar_fooof_JLB_copy


def load_best_bssu_method(fooof_version: str):
    """ """
    ################## method by Binder et al. - best directional Survey contact pair ##################
    best_bssu_contacts = loadResults.load_pickle_group_result(
        filename="best_2_contacts_from_directional_bssu", fooof_version=fooof_version
    )

    # add column with method name
    best_bssu_contacts_copy = best_bssu_contacts.copy()
    best_bssu_contacts_copy["method"] = "best_bssu_contacts"

    return best_bssu_contacts_copy


def save_result_excel(result_df: pd.DataFrame, filename: str, sheet_name: str):
    """
    Saves dataframe as Excel file

    Input:
        - result_df
        - filename
        - sheet_name

    """

    xlsx_filename = f"{filename}.xlsx"

    result_df.to_excel(os.path.join(GROUP_RESULTS_PATH, xlsx_filename), sheet_name=sheet_name, index=False)

    print(
        "file: ",
        f"{xlsx_filename}",
        "\nwritten in: ",
        GROUP_RESULTS_PATH,
    )


def correlation_tests_percept_methods(
    method_1: str, method_2: str, method_1_df: pd.DataFrame, method_2_df: pd.DataFrame, ses: str
):
    """
    For each session:
    for each subject hemisphere:

    perform 3 correlation tests between both methods:
        - estimated_beta_spearman
        - normalized_beta_pearson
        - cluster_beta_spearman

    return a dataframe with results

    """
    results_DF = pd.DataFrame()

    # find STNs with data from both methods
    stn_unique_method_1 = list(method_1_df.subject_hemisphere.unique())
    stn_unique_method_2 = list(method_2_df.subject_hemisphere.unique())

    stn_comparison_list = list(set(stn_unique_method_1) & set(stn_unique_method_2))
    stn_comparison_list.sort()

    comparison_df_method_1 = method_1_df.loc[method_1_df["subject_hemisphere"].isin(stn_comparison_list)]
    comparison_df_method_2 = method_2_df.loc[method_2_df["subject_hemisphere"].isin(stn_comparison_list)]

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

        # Spearman correlation between estimated beta average
        spearman_beta_stn = stats.spearmanr(
            stn_method_1["estimated_monopolar_beta_psd"].values, stn_method_2["estimated_monopolar_beta_psd"].values
        )

        # Pearson correlation between normalized beta to maximum within each electrode
        pearson_normalized_beta_stn = stats.pearsonr(
            stn_method_1["beta_relative_to_max"].values, stn_method_2["beta_relative_to_max"].values
        )

        spearman_beta_cluster_stn = stats.spearmanr(
            stn_method_1["beta_cluster"].values, stn_method_2["beta_cluster"].values
        )

        # contacts with beta rank 1 and 2
        ############## method 1: ##############
        rank1_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 1.0]
        # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank1_method_1.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 1 contact in the recording {method_1}.")
            continue

        rank1_method_1 = rank1_method_1.contact.values[0]

        rank2_method_1 = stn_method_1.loc[stn_method_1.beta_rank == 2.0]
        # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank2_method_1.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 2 contact in the recording {method_1}.")
            continue

        rank2_method_1 = rank2_method_1.contact.values[0]

        rank_1_and_2_method_1 = [rank1_method_1, rank2_method_1]

        ############### method 2: ##############
        rank1_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 1.0]
        # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank1_method_2.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 1 contact in the recording {method_2}.")
            continue
        rank1_method_2 = rank1_method_2.contact.values[0]

        rank2_method_2 = stn_method_2.loc[stn_method_2.beta_rank == 2.0]
        # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
        if len(rank2_method_2.contact.values) == 0:
            print(f"Sub-{sub_hem} has no rank 2 contact in the recording {method_2}.")
            continue
        rank2_method_2 = rank2_method_2.contact.values[0]

        rank_1_and_2_method_2 = [rank1_method_2, rank2_method_2]

        # yes if contact with rank 1 is the same
        if rank1_method_1 == rank1_method_2:
            compare_rank_1_contact = "same"

        else:
            compare_rank_1_contact = "different"

        # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
        if set(rank_1_and_2_method_1) == set(rank_1_and_2_method_2):
            both_contacts_matching = "yes"

        else:
            both_contacts_matching = "no"

        # check if at least one contact selected as beta rank 1 or 2 match for both methods
        if set(rank_1_and_2_method_1).intersection(set(rank_1_and_2_method_2)):
            compare_rank_1_and_2_contacts = "at_least_one_contact_match"

        else:
            compare_rank_1_and_2_contacts = "no_contacts_match"

        # store values in a dictionary
        spearman_result = {
            "method_1": [method_1],
            "method_2": [method_2],
            "session": [ses],
            "subject_hemisphere": [sub_hem],
            "estimated_beta_spearman_r": [spearman_beta_stn.statistic],
            "estimated_beta_spearman_pval": [spearman_beta_stn.pvalue],
            "normalized_beta_pearson_r": [pearson_normalized_beta_stn.statistic],
            "normalized_beta_pearson_pval": [pearson_normalized_beta_stn.pvalue],
            "cluster_beta_spearman_r": [spearman_beta_cluster_stn.statistic],
            "cluster_beta_spearman_pval": [spearman_beta_cluster_stn.pvalue],
            "contact_rank_1_method_1": [rank1_method_1],
            "contact_rank_1_method_2": [rank1_method_2],
            "contacts_rank_1_2_method_1": [rank_1_and_2_method_1],
            "contacts_rank_1_2_method_2": [rank_1_and_2_method_2],
            "compare_rank_1_contact": [compare_rank_1_contact],
            "compare_rank_1_and_2_contacts": [compare_rank_1_and_2_contacts],
            "both_contacts_matching": [both_contacts_matching],
        }
        results_single_DF = pd.DataFrame(spearman_result)

        results_DF = pd.concat([results_DF, results_single_DF], ignore_index=True)

    return results_DF


def rank_comparison_percept_methods(
    method_1: str, method_2: str, method_1_df: pd.DataFrame, method_2_df: pd.DataFrame, ses: str
):
    """
    For each session:
    for each subject hemisphere:

    compare if rank 1 and 2 contacts both match or at least one contact match

    return a dataframe with results

    """
    comparison_result = pd.DataFrame()

    # find STNs with data from both methods and externalized
    stn_unique_method = list(method_1_df.subject_hemisphere.unique())
    stn_unique_best_bssu = list(method_2_df.subject_hemisphere.unique())

    stn_comparison_list = list(set(stn_unique_method) & set(stn_unique_best_bssu))
    stn_comparison_list.sort()

    comparison_df_method = method_1_df.loc[method_1_df["subject_hemisphere"].isin(stn_comparison_list)]
    comparison_df_best_bssu = method_2_df.loc[method_2_df["subject_hemisphere"].isin(stn_comparison_list)]

    comparison_df = pd.concat([comparison_df_method, comparison_df_best_bssu], axis=0)

    for sub_hem in stn_comparison_list:
        # only run, if sub_hem STN exists in both session Dataframes
        if sub_hem not in comparison_df.subject_hemisphere.values:
            print(f"{sub_hem} is not in the comparison Dataframe.")
            continue

        # only take one electrode at both sessions and get spearman correlation
        stn_comparison = comparison_df.loc[comparison_df["subject_hemisphere"] == sub_hem]

        stn_method = stn_comparison.loc[stn_comparison.method == method_1]
        stn_best_bssu = stn_comparison.loc[stn_comparison.method == method_2]

        # contacts with beta rank 1 and 2
        # method:
        rank1_method = stn_method.loc[stn_method.beta_rank == 1.0]
        rank1_method = rank1_method.contact.values[0]

        rank2_method = stn_method.loc[stn_method.beta_rank == 2.0]
        rank2_method = rank2_method.contact.values[0]

        rank_1_and_2_method = [rank1_method, rank2_method]

        # best BSSU contact pair:
        best_contact_pair = stn_best_bssu.selected_2_contacts.values[0]

        # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
        if set(rank_1_and_2_method) == set(best_contact_pair):
            both_contacts_matching = "both_contacts_same"

        else:
            both_contacts_matching = "one_at_least_different"

        # check if at least one contact selected as beta rank 1 or 2 match for both methods
        if set(rank_1_and_2_method).intersection(set(best_contact_pair)):
            at_least_1_contact_matching = "at_least_one_contact_match"

        else:
            at_least_1_contact_matching = "no_contacts_match"

        # store values in a dictionary
        comparison_result_dict = {
            "method": [method_1],
            "best_bssu_contacts": [method_2],
            "session": [ses],
            "subject_hemisphere": [sub_hem],
            "contact_rank_1_method": [rank1_method],
            "contact_rank_2_method": [rank2_method],
            "rank_1_and_2_method": [rank_1_and_2_method],
            "bssu_best_contact_pair": [best_contact_pair],
            "both_contacts_matching": [both_contacts_matching],
            "at_least_1_contact_matching": [at_least_1_contact_matching],
        }
        comparison_single_result = pd.DataFrame(comparison_result_dict)
        comparison_result = pd.concat([comparison_result, comparison_single_result], ignore_index=True)

    return comparison_result


def get_sample_size_percept_methods(ses: str, ses_df: pd.DataFrame, rank_1_exists: str, method_vs_best_bssu=None):
    """
    Input:
        - rank_1_exists: "yes" if you compare both monopolar estimation methods
                        "no" if you compare the best_bssu_method to the monopolar estimation methods

    from a comparison result dataframe
        - count how often rank 1 contacts are the same
        - count how often there is at least one matching contact in compare_rank_1_and_2_contact

    """
    # sample size
    ses_count = ses_df["session"].count()

    if rank_1_exists == "yes":
        # count how often compare_rank_1_contact same
        same_rank_1 = ses_df.loc[ses_df.compare_rank_1_contact == "same"]
        same_rank_1 = same_rank_1["session"].count()
        precentage_same_rank_1 = same_rank_1 / ses_count

        # count how often there is at least one matching contact in compare_rank_1_and_2_contact
        same_rank_1_and_2 = ses_df.loc[ses_df.compare_rank_1_and_2_contacts == "at_least_one_contact_match"]
        same_rank_1_and_2 = same_rank_1_and_2["session"].count()
        precentage_same_rank_1_and_2 = same_rank_1_and_2 / ses_count

        sample_size_dict = {
            "session": [ses],
            "sample_size": [ses_count],
            "same_rank_1": [same_rank_1],
            "precentage_same_rank_1": [precentage_same_rank_1],
            "at_least_one_same_contact_rank_1_and_2": [same_rank_1_and_2],
            "precentage_at_least_one_same_contact_rank_1_and_2": [precentage_same_rank_1_and_2],
        }
        sample_size_single_df = pd.DataFrame(sample_size_dict)

    elif rank_1_exists == "no":
        # count how often compare_rank_1_contact same
        same_rank_1_and_2 = ses_df.loc[ses_df.both_contacts_matching == "both_contacts_same"]
        same_rank_1_and_2 = same_rank_1_and_2["subject_hemisphere"].count()
        precentage_same = same_rank_1_and_2 / ses_count

        # count how often there is at least one matching contact in compare_rank_1_and_2_contact
        at_least_1_same = ses_df.loc[ses_df.at_least_1_contact_matching == "at_least_one_contact_match"]
        at_least_1_same = at_least_1_same["subject_hemisphere"].count()
        precentage_at_least_1_same = at_least_1_same / ses_count

        sample_size_dict = {
            "method": [method_vs_best_bssu],
            "session": [ses],
            "sample_size": [ses_count],
            "same_rank_1_and_2_count": [same_rank_1_and_2],
            "precentage_same": [precentage_same],
            "at_least_1_contact_same": [at_least_1_same],
            "precentage_at_least_1_same": [precentage_at_least_1_same],
        }

        sample_size_single_df = pd.DataFrame(sample_size_dict)

    return sample_size_single_df


def best_bssu_contact_pair_vs_monopol_beta():
    """ """
