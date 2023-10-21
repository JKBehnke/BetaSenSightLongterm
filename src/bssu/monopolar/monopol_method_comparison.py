""" Comparisons between monopolar estimation methods """


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import load_data_files as load_data
from ..utils import loadResults as loadResults
from ..utils import monopol_comparison_helpers as helpers

group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")
group_figures_path = find_folders.get_monopolar_project_path(folder="GroupFigures")

incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]
segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]


def correlation_monopol_fooof_beta_methods(method_1: str, method_2: str, fooof_version: str):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - method_1: "JLB_directional", "euclidean_directional", "Strelow"
        - method_2: "JLB_directional", "euclidean_directional", "Strelow"
        - fooof_version: "v1", "v2"
    """

    # results
    results_DF = pd.DataFrame()
    sample_size_df = pd.DataFrame()
    corr_ses_df = pd.DataFrame()

    # get data from method 1
    if method_1 == "JLB_directional":
        method_1_data = helpers.load_JLB_method(fooof_version=fooof_version)

    elif method_1 == "euclidean_directional":
        method_1_data = helpers.load_euclidean_method(fooof_version=fooof_version)

    elif method_1 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # get data from method 2
    if method_2 == "JLB_directional":
        method_2_data = helpers.load_JLB_method(fooof_version=fooof_version)

    elif method_2 == "euclidean_directional":
        method_2_data = helpers.load_euclidean_method(fooof_version=fooof_version)

    elif method_2 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # Perform 3 versions of correlation tests for every session separately and within each STN
    for ses in incl_sessions:
        method_1_session = method_1_data.loc[method_1_data.session == ses]
        method_2_session = method_2_data.loc[method_2_data.session == ses]

        ses_result_df = helpers.correlation_tests_percept_methods(
            method_1=method_1, method_2=method_2, method_1_df=method_1_session, method_2_df=method_2_session, ses=ses
        )
        results_DF = pd.concat([results_DF, ses_result_df], ignore_index=True)

    # save Dataframe to Excel
    results_DF_copy = results_DF.copy()

    correlation_results = ["estimated_beta_spearman", "normalized_beta_pearson", "cluster_beta_spearman"]

    for corr in correlation_results:
        # add new column: significant yes, no
        significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
        results_DF_copy[f"significant_{corr}"] = ["yes" if cond else "no" for cond in significant_correlation]

    # save as Excel
    helpers.save_result_excel(
        result_df=results_DF_copy,
        filename=f"fooof_monopol_beta_correlations_per_stn_{method_1}_{method_2}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    # get sample size
    for ses in incl_sessions:
        ses_df = results_DF_copy.loc[results_DF_copy.session == ses]
        ses_count = ses_df["session"].count()

        sample_size_single_df = helpers.get_sample_size_percept_methods(
            ses=ses, ses_df=ses_df, method_1=method_1, method_2=method_2, rank_1_exists="yes"
        )
        sample_size_df = pd.concat([sample_size_df, sample_size_single_df], ignore_index=True)

        # correlation results per correlation test
        for corr in correlation_results:
            corr_mean = ses_df[f"{corr}_r"].mean()
            corr_median = ses_df[f"{corr}_r"].median()
            corr_std = np.std(ses_df[f"{corr}_r"])

            # calculate how often significant?
            significant_count = ses_df.loc[ses_df[f"significant_{corr}"] == "yes"]
            significant_count = significant_count["session"].count()
            percentage_significant = significant_count / ses_count

            corr_ses_result = {
                "session": [ses],
                "method_1": [method_1],
                "method_2": [method_2],
                "sample_size": [ses_count],
                "correlation": [corr],
                "corr_mean": [corr_mean],
                "corr_median": [corr_median],
                "corr_std": [corr_std],
                "significant_count": [significant_count],
                "percentage_significant": [percentage_significant],
            }
            corr_ses_single_df = pd.DataFrame(corr_ses_result)

            corr_ses_df = pd.concat([corr_ses_df, corr_ses_single_df], ignore_index=True)

    # save session Dataframes as Excel files
    helpers.save_result_excel(
        result_df=sample_size_df,
        filename=f"fooof_monopol_beta_correlations_sample_size_df_{method_1}_{method_2}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    helpers.save_result_excel(
        result_df=corr_ses_df,
        filename=f"fooof_monopol_beta_correlations_corr_ses_df_{method_1}_{method_2}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    return results_DF_copy, sample_size_df, corr_ses_df


def compare_method_to_best_bssu_contact_pair(fooof_version: str):
    """
    Comparing the selected best BSSU contact pair to the two directional contacts selected with method Euclidean and JLB

    Input:
        - fooof_version: "v1", "v2"

    """

    sample_size_df = pd.DataFrame()
    comparison_result = pd.DataFrame()

    # get data
    JLB_method = helpers.load_JLB_method(fooof_version=fooof_version)
    Euclidean_method = helpers.load_euclidean_method(fooof_version=fooof_version)

    best_bssu_contact_data = helpers.load_best_bssu_method(fooof_version=fooof_version)

    two_methods = [1, 2]

    for run in two_methods:
        if run == 1:
            method_data = JLB_method
            method = "JLB_directional"

        elif run == 2:
            method_data = Euclidean_method
            method = "euclidean_directional"

        for ses in incl_sessions:
            ses_data_method = method_data.loc[method_data.session == ses]
            ses_data_best_bssu = best_bssu_contact_data.loc[best_bssu_contact_data.session == ses]

            comparison_single_result = helpers.rank_comparison_percept_methods(
                method_1=method,
                method_2="best_bssu_contacts",
                method_1_df=ses_data_method,
                method_2_df=ses_data_best_bssu,
                ses=ses,
            )

            comparison_result = pd.concat([comparison_result, comparison_single_result], ignore_index=True)

        # save as Excel
        helpers.save_result_excel(
            result_df=comparison_result,
            filename=f"fooof_monopol_best_contacts_per_stn_{method}_best_bssu_contacts_{fooof_version}",
            sheet_name="fooof_monopol_best_contacts",
        )

        results_DF_copy = comparison_result.copy()

        for ses in incl_sessions:
            ses_result = results_DF_copy.loc[results_DF_copy.session == ses]

            # get sample size
            sample_size_single_df = helpers.get_sample_size_percept_methods(
                ses=ses,
                ses_df=ses_result,
                rank_1_exists="no",
                method_1=method,
                method_2="best_bssu_contacts",
            )

            # sample_size_single_df = pd.DataFrame(sample_size_dict)
            sample_size_df = pd.concat([sample_size_df, sample_size_single_df])

        # save session Dataframes as Excel files
        helpers.save_result_excel(
            result_df=sample_size_df,
            filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_best_bssu_contacts_{fooof_version}",
            sheet_name="fooof_monopol_beta_correlations",
        )

    return sample_size_df, results_DF_copy


# def percept_vs_externalized(method: str, fooof_version: str, externalized_version: str, reference=None):
#     """
#     Spearman correlation between monopolar beta power estimations between 2 methods
#     only of directional contacts

#     Input: define methods to compare
#         - method: "JLB_directional", "euclidean_directional", "Strelow", "best_bssu_contacts"
#         - fooof_version: "v1", "v2"
#         - externalized_version: "externalized_fooof", "externalized_ssd"
#         - reference: str "bipolar_to_lowermost" or "no"
#     """
#     if reference == "bipolar_to_lowermost":
#         reference_name = "_bipolar_to_lowermost"

#     else:
#         reference_name = ""

#     # results
#     spearman_result = {}

#     sample_size_dict = {}

#     methods_for_spearman = ["JLB_directional", "euclidean_directional"]
#     methods_without_spearman = ["best_bssu_contacts"]

#     # get only postop data from method
#     if method == "JLB_directional":
#         method_data = helpers.load_JLB_method(fooof_version=fooof_version)
#         method_data = method_data.loc[method_data.session == "postop"]

#     elif method == "euclidean_directional":
#         method_data = helpers.load_euclidean_method(fooof_version=fooof_version)
#         method_data = method_data.loc[method_data.session == "postop"]

#     elif method == "Strelow":
#         print("Strelow method only gives optimal contacts with beta ranks 1-3")

#     elif method == "best_bssu_contacts":
#         method_data = helpers.load_best_bssu_method(fooof_version=fooof_version)
#         method_data = method_data.loc[method_data.session == "postop"]

#     # get data from externalized LFP
#     if externalized_version == "externalized_fooof":
#         externalized_data = helpers.load_externalized_fooof_data(fooof_version=fooof_version, reference=reference)

#     elif externalized_version == "externalized_ssd":
#         externalized_data = helpers.load_externalized_ssd_data(reference=reference)

#     # Perform spearman correlation for every session separately and within each STN

#     # find STNs with data from both methods and externalized
#     stn_unique_method = list(method_data.subject_hemisphere.unique())
#     stn_unique_externalized = list(externalized_data.subject_hemisphere.unique())

#     stn_comparison_list = list(set(stn_unique_method) & set(stn_unique_externalized))
#     stn_comparison_list.sort()

#     comparison_df_method = method_data.loc[method_data["subject_hemisphere"].isin(stn_comparison_list)]
#     comparison_df_externalized = externalized_data.loc[
#         externalized_data["subject_hemisphere"].isin(stn_comparison_list)
#     ]

#     comparison_df = pd.concat([comparison_df_method, comparison_df_externalized], axis=0)

#     for sub_hem in stn_comparison_list:
#         # only run, if sub_hem STN exists in both session Dataframes
#         if sub_hem not in comparison_df.subject_hemisphere.values:
#             print(f"{sub_hem} is not in the comparison Dataframe.")
#             continue

#         # only take one electrode at both sessions and get spearman correlation
#         stn_comparison = comparison_df.loc[comparison_df["subject_hemisphere"] == sub_hem]

#         stn_method = stn_comparison.loc[stn_comparison.method == method]
#         stn_externalized = stn_comparison.loc[stn_comparison.method == externalized_version]

#         ############## externalized rank contacts: ##############
#         rank1_externalized = stn_externalized.loc[stn_externalized.beta_rank == 1.0]

#         # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
#         if len(rank1_externalized.contact.values) == 0:
#             print(f"Sub-{sub_hem} has no rank 1 contact in the recording.")
#             continue

#         rank1_externalized = rank1_externalized.contact.values[0]

#         rank2_externalized = stn_externalized.loc[stn_externalized.beta_rank == 2.0]
#         # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
#         if len(rank2_externalized.contact.values) == 0:
#             print(f"Sub-{sub_hem} has no rank 2 contact in the externalized recording.")
#             continue

#         rank2_externalized = rank2_externalized.contact.values[0]

#         rank_1_and_2_externalized = [rank1_externalized, rank2_externalized]

#         ############## rank contacts from method ##############
#         # Spearman correlation between beta average only for the 2 monopolar methods
#         if method in methods_for_spearman:
#             spearman_beta_stn = stats.spearmanr(
#                 stn_method.estimated_monopolar_beta_psd.values, stn_externalized.estimated_monopolar_beta_psd.values
#             )
#             spearman_statistic = spearman_beta_stn.statistic
#             spearman_pval = spearman_beta_stn.pvalue

#             # contacts with beta rank 1 and 2
#             # method:
#             rank1_method = stn_method.loc[stn_method.beta_rank == 1.0]
#             rank1_method = rank1_method.contact.values[0]

#             rank2_method = stn_method.loc[stn_method.beta_rank == 2.0]
#             rank2_method = rank2_method.contact.values[0]

#             rank_1_and_2_method = [rank1_method, rank2_method]

#             # yes if contact with rank 1 is the same
#             if rank1_method == rank1_externalized:
#                 compare_rank_1_contact = "same"

#             else:
#                 compare_rank_1_contact = "different"

#         # for method "best_bssu_contacts" we only have a list of best 2 contacts
#         elif method in methods_without_spearman:
#             spearman_statistic = "no_spearman"
#             spearman_pval = "no_spearman"
#             rank1_method = "no_rank_1"
#             compare_rank_1_contact = "no_rank_1"

#             # get beta rank 1 and 2 contacts
#             rank_1_and_2_method = stn_method.selected_2_contacts.values[
#                 0
#             ]  # list of the 2 contacts (bipolar contact pair with highest beta)

#         # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
#         if set(rank_1_and_2_method) == set(rank_1_and_2_externalized):
#             both_contacts_matching = "yes"

#         else:
#             both_contacts_matching = "no"

#         # check if at least one contact selected as beta rank 1 or 2 match for both methods
#         if set(rank_1_and_2_method).intersection(set(rank_1_and_2_externalized)):
#             compare_rank_1_and_2_contacts = "at_least_one_contact_match"

#         else:
#             compare_rank_1_and_2_contacts = "no_contacts_match"

#         # store values in a dictionary
#         spearman_result[f"{sub_hem}"] = [
#             method,
#             externalized_version,
#             "postop",
#             sub_hem,
#             spearman_statistic,
#             spearman_pval,
#             rank1_method,
#             rank1_externalized,
#             rank_1_and_2_method,
#             rank_1_and_2_externalized,
#             compare_rank_1_contact,
#             compare_rank_1_and_2_contacts,
#             both_contacts_matching,
#             reference,
#         ]

#     # save result
#     results_DF = pd.DataFrame(spearman_result)
#     results_DF.rename(
#         index={
#             0: "method",
#             1: "externalized",
#             2: "session",
#             3: "subject_hemisphere",
#             4: f"spearman_r",
#             5: f"pval",
#             6: "contact_rank_1_method",
#             7: "contact_rank_1_externalized",
#             8: "contacts_rank_1_2_method",
#             9: "contacts_rank_1_2_externalized",
#             10: "compare_rank_1_contact",
#             11: "compare_rank_1_and_2_contacts",
#             12: "both_contacts_match",
#             13: "reference",
#         },
#         inplace=True,
#     )
#     results_DF = results_DF.transpose()

#     # save Dataframe to Excel
#     results_DF_copy = results_DF.copy()

#     # add new column: significant yes, no
#     if method in methods_for_spearman:
#         significant_correlation = results_DF_copy["pval"] < 0.05
#         results_DF_copy["significant_correlation"] = ["yes" if cond else "no" for cond in significant_correlation]

#     # save as Excel
#     results_DF_copy.to_excel(
#         os.path.join(
#             group_results_path,
#             f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{reference_name}_{fooof_version}.xlsx",
#         ),
#         sheet_name="monopolar_beta_correlations",
#         index=False,
#     )
#     print(
#         "file: ",
#         f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{reference_name}_{fooof_version}.xlsx",
#         "\nwritten in: ",
#         group_results_path,
#     )

#     # get sample size
#     result_count = results_DF_copy["subject_hemisphere"].count()

#     if method in methods_for_spearman:
#         spearman_mean = results_DF_copy.spearman_r.mean()
#         spearman_median = results_DF_copy.spearman_r.median()
#         spearman_std = np.std(results_DF_copy.spearman_r)

#         # calculate how often significant?
#         significant_count = results_DF_copy.loc[results_DF_copy.significant_correlation == "yes"]
#         significant_count = significant_count["subject_hemisphere"].count()
#         percentage_significant = significant_count / result_count

#         # count how often compare_rank_1_contact same
#         same_rank_1 = results_DF_copy.loc[results_DF_copy.compare_rank_1_contact == "same"]
#         same_rank_1 = same_rank_1["subject_hemisphere"].count()
#         precentage_same_rank_1 = same_rank_1 / result_count

#         # count how often there is at least one matching contact in compare_rank_1_and_2_contact
#         at_least_one_contact_match = results_DF_copy.loc[
#             results_DF_copy.compare_rank_1_and_2_contacts == "at_least_one_contact_match"
#         ]
#         at_least_one_contact_match = at_least_one_contact_match["subject_hemisphere"].count()
#         precentage_at_least_one_contact_match = at_least_one_contact_match / result_count

#         # count how often both contacts match in compare_rank_1_and_2_contact
#         both_contacts_matching_count = results_DF_copy.loc[results_DF_copy.both_contacts_match == "yes"]
#         both_contacts_matching_count = both_contacts_matching_count["subject_hemisphere"].count()
#         precentage_both_contacts_match = both_contacts_matching_count / result_count

#         sample_size_dict = {
#             "sample_size": [result_count],
#             "spearman_mean": [spearman_mean],
#             "spearman_median": [spearman_median],
#             "spearman_std": [spearman_std],
#             "significant_count": [significant_count],
#             "percentage_significant": [percentage_significant],
#             "same_rank_1_count": [same_rank_1],
#             "percentage_same_rank_1": [precentage_same_rank_1],
#             "at_least_one_contact_match": [at_least_one_contact_match],
#             "percentage_at_least_one_contact_match": [precentage_at_least_one_contact_match],
#             "both_contacts_matching_count": [both_contacts_matching_count],
#             "precentage_both_contacts_match": [precentage_both_contacts_match],
#         }

#         sample_size_df = pd.DataFrame(sample_size_dict)

#     elif method in methods_without_spearman:
#         # count how often there is at least one matching contact in compare_rank_1_and_2_contact
#         at_least_one_contact_match = results_DF_copy.loc[
#             results_DF_copy.compare_rank_1_and_2_contacts == "at_least_one_contact_match"
#         ]
#         at_least_one_contact_match = at_least_one_contact_match["subject_hemisphere"].count()
#         precentage_at_least_one_contact_match = at_least_one_contact_match / result_count

#         # count how often both contacts match in compare_rank_1_and_2_contact
#         both_contacts_matching_count = results_DF_copy.loc[results_DF_copy.both_contacts_match == "yes"]
#         both_contacts_matching_count = both_contacts_matching_count["subject_hemisphere"].count()
#         precentage_both_contacts_match = both_contacts_matching_count / result_count

#         sample_size_dict = {
#             "sample_size": [result_count],
#             "at_least_one_contact_match": [at_least_one_contact_match],
#             "precentage_at_least_one_contact_match": [precentage_at_least_one_contact_match],
#             "both_contacts_matching_count": [both_contacts_matching_count],
#             "precentage_both_contacts_match": [precentage_both_contacts_match],
#         }

#         sample_size_df = pd.DataFrame(sample_size_dict)

#     return results_DF_copy, sample_size_df, stn_comparison


def percept_vs_externalized(method: str, fooof_version: str, externalized_version: str, reference=None):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - method: "JLB_directional", "euclidean_directional", "Strelow", "best_bssu_contacts"
        - fooof_version: "v1", "v2"
        - externalized_version: "externalized_fooof", "externalized_ssd"
        - reference: str "bipolar_to_lowermost" or "no"
    """
    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    methods_for_spearman = ["JLB_directional", "euclidean_directional"]
    methods_without_spearman = ["best_bssu_contacts"]

    # get only postop data from method
    if method == "JLB_directional":
        method_data = helpers.load_JLB_method(fooof_version=fooof_version)
        method_data = method_data.loc[method_data.session == "postop"]

    elif method == "euclidean_directional":
        method_data = helpers.load_euclidean_method(fooof_version=fooof_version)
        method_data = method_data.loc[method_data.session == "postop"]

    elif method == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    elif method == "best_bssu_contacts":
        method_data = helpers.load_best_bssu_method(fooof_version=fooof_version)
        method_data = method_data.loc[method_data.session == "postop"]

    # get data from externalized LFP
    if externalized_version == "externalized_fooof":
        externalized_data = helpers.load_externalized_fooof_data(fooof_version=fooof_version, reference=reference)

    elif externalized_version == "externalized_ssd":
        externalized_data = helpers.load_externalized_ssd_data(reference=reference)

    # Perform comparison for every session separately and within each STN
    if method in methods_for_spearman:
        result_df = helpers.correlation_tests_percept_methods(
            method_1=method,
            method_2=externalized_version,
            method_1_df=method_data,
            method_2_df=externalized_data,
            ses="postop",
        )

        # save Dataframe to Excel
        results_DF_copy = result_df.copy()

        correlation_results = ["estimated_beta_spearman", "normalized_beta_pearson", "cluster_beta_spearman"]

        for corr in correlation_results:
            # add new column: significant yes, no
            significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
            results_DF_copy[f"significant_{corr}"] = ["yes" if cond else "no" for cond in significant_correlation]

        # save as Excel
        helpers.save_result_excel(
            result_df=results_DF_copy,
            filename=f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{reference_name}_{fooof_version}",
            sheet_name="fooof_monopol_beta_correlations",
        )

        # get sample size
        count = results_DF_copy["subject_hemisphere"].count()

        sample_size_df = helpers.get_sample_size_percept_methods(
            ses="postop", ses_df=results_DF_copy, method_1=method, method_2=externalized_version, rank_1_exists="yes"
        )

        # correlation results per correlation test
        corr_ses_df = pd.DataFrame()

        for corr in correlation_results:
            corr_mean = results_DF_copy[f"{corr}_r"].mean()
            corr_median = results_DF_copy[f"{corr}_r"].median()
            corr_std = np.std(results_DF_copy[f"{corr}_r"])

            # calculate how often significant?
            significant_count = results_DF_copy.loc[results_DF_copy[f"significant_{corr}"] == "yes"]
            significant_count = significant_count["session"].count()
            percentage_significant = significant_count / count

            corr_ses_result = {
                "session": ["postop"],
                "method_1": [method],
                "method_2": [externalized_version],
                "sample_size": [count],
                "correlation": [corr],
                "corr_mean": [corr_mean],
                "corr_median": [corr_median],
                "corr_std": [corr_std],
                "significant_count": [significant_count],
                "percentage_significant": [percentage_significant],
            }
            corr_ses_single_df = pd.DataFrame(corr_ses_result)

            corr_ses_df = pd.concat([corr_ses_df, corr_ses_single_df], ignore_index=True)

        # save session Dataframes as Excel files
        helpers.save_result_excel(
            result_df=sample_size_df,
            filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_{externalized_version}{reference_name}_{fooof_version}",
            sheet_name="fooof_monopol_beta_correlations",
        )

        helpers.save_result_excel(
            result_df=corr_ses_df,
            filename=f"fooof_monopol_beta_correlations_corr_ses_df_{method}_{externalized_version}{reference_name}_{fooof_version}",
            sheet_name="fooof_monopol_beta_correlations",
        )

    elif method in methods_without_spearman:
        corr_ses_df = pd.DataFrame()

        result_df = helpers.rank_comparison_percept_methods(
            method_1=externalized_version,
            method_2=method,
            method_1_df=externalized_data,
            method_2_df=method_data,
            ses="postop",
        )

        # save as Excel
        helpers.save_result_excel(
            result_df=result_df,
            filename=f"fooof_monopol_beta_correlations_per_stn_{method}_{externalized_version}{reference_name}_{fooof_version}",
            sheet_name="fooof_monopol_best_contacts",
        )

        results_DF_copy = result_df.copy()

        # get sample size
        sample_size_df = helpers.get_sample_size_percept_methods(
            ses="postop",
            ses_df=results_DF_copy,
            rank_1_exists="no",
            method_1=externalized_version,
            method_2=method,
        )

        # save session Dataframes as Excel files
        helpers.save_result_excel(
            result_df=sample_size_df,
            filename=f"fooof_monopol_beta_correlations_sample_size_df_{method}_{externalized_version}{reference_name}_{fooof_version}",
            sheet_name="fooof_monopol_beta_correlations",
        )

    return results_DF_copy, sample_size_df, corr_ses_df


def externalized_versions_comparison(
    externalized_version_1: str, externalized_version_2: str, fooof_version: str, reference=None
):
    """
    Spearman correlation between monopolar beta power estimations between 2 methods
    only of directional contacts

    Input: define methods to compare
        - externalized_version_1: "externalized_ssd", "externalized_fooof"
        - externalized_version_2: "externalized_ssd", "externalized_fooof"
        - fooof_version: "v1", "v2"
        - reference: "bipolar_to_lowermost" or "no"
    """

    if reference == "bipolar_to_lowermost":
        reference_name = "_bipolar_to_lowermost"

    else:
        reference_name = ""

    # results
    # spearman_result = {}

    # sample_size_dict = {}

    # get data from externalized LFP version 1
    if externalized_version_1 == "externalized_fooof":
        externalized_data_1 = helpers.load_externalized_fooof_data(fooof_version=fooof_version, reference=reference)

    elif externalized_version_1 == "externalized_ssd":
        externalized_data_1 = helpers.load_externalized_ssd_data(reference=reference)

    # get data from externalized LFP version 2
    if externalized_version_2 == "externalized_fooof":
        externalized_data_2 = helpers.load_externalized_fooof_data(fooof_version=fooof_version, reference=reference)

    elif externalized_version_2 == "externalized_ssd":
        externalized_data_2 = helpers.load_externalized_ssd_data(reference=reference)

    # Perform spearman correlation for every session separately and within each STN
    result_df = helpers.correlation_tests_percept_methods(
        method_1=externalized_version_1,
        method_2=externalized_version_2,
        method_1_df=externalized_data_1,
        method_2_df=externalized_data_2,
        ses="postop",
    )

    # save Dataframe to Excel
    results_DF_copy = result_df.copy()

    correlation_results = ["estimated_beta_spearman", "normalized_beta_pearson", "cluster_beta_spearman"]

    for corr in correlation_results:
        # add new column: significant yes, no
        significant_correlation = results_DF_copy[f"{corr}_pval"] < 0.05
        results_DF_copy[f"significant_{corr}"] = ["yes" if cond else "no" for cond in significant_correlation]

    # save as Excel
    helpers.save_result_excel(
        result_df=results_DF_copy,
        filename=f"fooof_monopol_beta_correlations_per_stn_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    # get sample size
    count = results_DF_copy["subject_hemisphere"].count()

    sample_size_df = helpers.get_sample_size_percept_methods(
        ses="postop",
        ses_df=results_DF_copy,
        rank_1_exists="yes",
        method_1=externalized_version_1,
        method_2=externalized_version_2,
    )

    # correlation results per correlation test
    corr_ses_df = pd.DataFrame()

    for corr in correlation_results:
        corr_mean = results_DF_copy[f"{corr}_r"].mean()
        corr_median = results_DF_copy[f"{corr}_r"].median()
        corr_std = np.std(results_DF_copy[f"{corr}_r"])

        # calculate how often significant?
        significant_count = results_DF_copy.loc[results_DF_copy[f"significant_{corr}"] == "yes"]
        significant_count = significant_count["session"].count()
        percentage_significant = significant_count / count

        corr_ses_result = {
            "session": ["postop"],
            "method_1": [externalized_version_1],
            "method_2": [externalized_version_2],
            "sample_size": [count],
            "correlation": [corr],
            "corr_mean": [corr_mean],
            "corr_median": [corr_median],
            "corr_std": [corr_std],
            "significant_count": [significant_count],
            "percentage_significant": [percentage_significant],
        }
        corr_ses_single_df = pd.DataFrame(corr_ses_result)

        corr_ses_df = pd.concat([corr_ses_df, corr_ses_single_df], ignore_index=True)

    # save session Dataframes as Excel files
    helpers.save_result_excel(
        result_df=sample_size_df,
        filename=f"fooof_monopol_beta_correlations_sample_size_df_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    helpers.save_result_excel(
        result_df=corr_ses_df,
        filename=f"fooof_monopol_beta_correlations_corr_ses_df_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}",
        sheet_name="fooof_monopol_beta_correlations",
    )

    return results_DF_copy, sample_size_df, corr_ses_df

    # # find STNs with data from both methods
    # stn_unique_method = list(externalized_data_1.subject_hemisphere.unique())
    # stn_unique_externalized = list(externalized_data_2.subject_hemisphere.unique())

    # stn_comparison_list = list(set(stn_unique_method) & set(stn_unique_externalized))
    # stn_comparison_list.sort()

    # comparison_df_externalized_1 = externalized_data_1.loc[
    #     externalized_data_1["subject_hemisphere"].isin(stn_comparison_list)
    # ]
    # comparison_df_externalized_2 = externalized_data_2.loc[
    #     externalized_data_2["subject_hemisphere"].isin(stn_comparison_list)
    # ]

    # comparison_df = pd.concat([comparison_df_externalized_1, comparison_df_externalized_2], axis=0)

    # for sub_hem in stn_comparison_list:
    #     # only run, if sub_hem STN exists in both session Dataframes
    #     if sub_hem not in comparison_df.subject_hemisphere.values:
    #         print(f"{sub_hem} is not in the comparison Dataframe.")
    #         continue

    #     # only take one electrode at both sessions and get spearman correlation
    #     stn_comparison = comparison_df.loc[comparison_df["subject_hemisphere"] == sub_hem]

    #     ############## externalized rank contacts: ##############
    #     two_versions = ["externalized_1", "externalized_2"]
    #     column_name_version = [externalized_version_1, externalized_version_2]

    #     externalized_sub_hem_dict = {}
    #     sub_hem_with_no_rank_1_or_2 = (
    #         []
    #     )  # capture subject hemispheres that don't have rank 1 or rank 2, take them out of the analysis

    #     for m, method in enumerate(two_versions):
    #         version = column_name_version[m]
    #         stn_data = stn_comparison.loc[stn_comparison.method == version]
    #         stn_estimated_monopolar_beta_psd = stn_data.estimated_monopolar_beta_psd.values

    #         rank_1 = stn_data.loc[stn_data.beta_rank == 1.0]

    #         # check if externalized has a rank 1 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
    #         if len(rank_1.contact.values) == 0:
    #             sub_hem_with_no_rank_1_or_2.append(sub_hem)
    #             print(f"Sub-{sub_hem} has no rank 1 contact in the {version}.")
    #             continue

    #         rank_1 = rank_1.contact.values[0]

    #         rank_2 = stn_data.loc[stn_data.beta_rank == 2.0]
    #         # check if externalized has a rank 2 contact (sometimes there is so little beta activity, so there is only rank 1 and 5x rank 4)
    #         if len(rank_2.contact.values) == 0:
    #             sub_hem_with_no_rank_1_or_2.append(sub_hem)
    #             print(f"Sub-{sub_hem} has no rank 2 contact in the {version}.")
    #             continue

    #         rank_2 = rank_2.contact.values[0]

    #         rank_1_and_2 = [rank_1, rank_2]

    #         # save for each method
    #         externalized_sub_hem_dict[m] = [
    #             method,
    #             version,
    #             stn_estimated_monopolar_beta_psd,
    #             rank_1,
    #             rank_2,
    #             rank_1_and_2,
    #         ]

    #     externalized_sub_hem_columns = [
    #         "externalized_1_or_2",
    #         "ssd_or_fooof",
    #         "estimated_monopolar_beta_psd",
    #         "rank_1",
    #         "rank_2",
    #         "rank_1_and_2",
    #     ]
    #     externalized_sub_hem_dataframe = pd.DataFrame.from_dict(
    #         externalized_sub_hem_dict, orient="index", columns=externalized_sub_hem_columns
    #     )

    #     # check if subject hemisphere does not have rank 1 or rank 2, take this one out!
    #     if sub_hem in sub_hem_with_no_rank_1_or_2:
    #         continue

    #     ############## rank contacts from method ##############
    #     # Spearman correlation between beta average only for the 2 monopolar methods
    #     data_from_externalized_1 = externalized_sub_hem_dataframe.loc[
    #         externalized_sub_hem_dataframe.externalized_1_or_2 == "externalized_1"
    #     ]
    #     data_from_externalized_2 = externalized_sub_hem_dataframe.loc[
    #         externalized_sub_hem_dataframe.externalized_1_or_2 == "externalized_2"
    #     ]

    #     spearman_beta_stn = stats.spearmanr(
    #         data_from_externalized_1.estimated_monopolar_beta_psd.values[0],
    #         data_from_externalized_2.estimated_monopolar_beta_psd.values[0],
    #     )
    #     spearman_statistic = spearman_beta_stn.statistic
    #     spearman_pval = spearman_beta_stn.pvalue

    #     # contacts with beta rank 1 and 2
    #     # yes if contact with rank 1 is the same
    #     if data_from_externalized_1.rank_1.values[0] == data_from_externalized_2.rank_1.values[0]:
    #         compare_rank_1_contact = "same"

    #     else:
    #         compare_rank_1_contact = "different"

    #     # yes if 2 contacts with rank 1 or 2 are the same (independent of which one is rank 1 or 2)
    #     if set(data_from_externalized_1.rank_1_and_2.values[0]) == set(data_from_externalized_2.rank_1_and_2.values[0]):
    #         both_contacts_matching = "yes"

    #     else:
    #         both_contacts_matching = "no"

    #     # check if at least one contact selected as beta rank 1 or 2 match for both methods
    #     if set(data_from_externalized_1.rank_1_and_2.values[0]).intersection(
    #         set(data_from_externalized_2.rank_1_and_2.values[0])
    #     ):
    #         compare_rank_1_and_2_contacts = "at_least_one_contact_match"

    #     else:
    #         compare_rank_1_and_2_contacts = "no_contacts_match"

    #     # store values in a dictionary
    #     spearman_result[f"{sub_hem}"] = [
    #         externalized_version_1,
    #         externalized_version_2,
    #         sub_hem,
    #         spearman_statistic,
    #         spearman_pval,
    #         data_from_externalized_1.rank_1.values[0],
    #         data_from_externalized_2.rank_1.values[0],
    #         data_from_externalized_1.rank_1_and_2.values[0],
    #         data_from_externalized_2.rank_1_and_2.values[0],
    #         compare_rank_1_contact,
    #         compare_rank_1_and_2_contacts,
    #         both_contacts_matching,
    #         reference,
    #     ]

    # # save result
    # results_DF_columns = [
    #     "externalized_version_1",
    #     "externalized_version_2",
    #     "subject_hemisphere",
    #     "spearman_r",
    #     "pval",
    #     "contact_rank_1_externalized_1",
    #     "contact_rank_1_externalized_2",
    #     "contacts_rank_1_2_externalized_1",
    #     "contacts_rank_1_2_externalized_2",
    #     "compare_rank_1_contact",
    #     "compare_rank_1_and_2_contacts",
    #     "both_contacts_match",
    #     "reference",
    # ]

    # results_DF = pd.DataFrame.from_dict(spearman_result, orient="index", columns=results_DF_columns)

    # # save Dataframe to Excel
    # results_DF_copy = results_DF.copy()

    # # add new column: significant yes, no
    # significant_correlation = results_DF_copy["pval"] < 0.05
    # results_DF_copy["significant_correlation"] = ["yes" if cond else "no" for cond in significant_correlation]

    # # save as Excel
    # results_DF_copy.to_excel(
    #     os.path.join(
    #         group_results_path,
    #         f"fooof_monopol_beta_correlations_per_stn_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}.xlsx",
    #     ),
    #     sheet_name="monopolar_beta_correlations",
    #     index=False,
    # )
    # print(
    #     "file: ",
    #     f"fooof_monopol_beta_correlations_per_stn_{externalized_version_1}_{externalized_version_2}{reference_name}_{fooof_version}.xlsx",
    #     "\nwritten in: ",
    #     group_results_path,
    # )

    # # get sample size
    # result_count = results_DF_copy["subject_hemisphere"].count()

    # spearman_mean = results_DF_copy.spearman_r.mean()
    # spearman_median = results_DF_copy.spearman_r.median()
    # spearman_std = np.std(results_DF_copy.spearman_r)

    # # calculate how often significant?
    # significant_count = results_DF_copy.loc[results_DF_copy.significant_correlation == "yes"]
    # significant_count = significant_count["subject_hemisphere"].count()
    # percentage_significant = significant_count / result_count

    # # count how often compare_rank_1_contact same
    # same_rank_1 = results_DF_copy.loc[results_DF_copy.compare_rank_1_contact == "same"]
    # same_rank_1 = same_rank_1["subject_hemisphere"].count()
    # precentage_same_rank_1 = same_rank_1 / result_count

    # # count how often there is at least one matching contact in compare_rank_1_and_2_contact
    # at_least_one_contact_match = results_DF_copy.loc[
    #     results_DF_copy.compare_rank_1_and_2_contacts == "at_least_one_contact_match"
    # ]
    # at_least_one_contact_match = at_least_one_contact_match["subject_hemisphere"].count()
    # precentage_at_least_one_contact_match = at_least_one_contact_match / result_count

    # # count how often both contacts match in compare_rank_1_and_2_contact
    # both_contacts_matching_count = results_DF_copy.loc[results_DF_copy.both_contacts_match == "yes"]
    # both_contacts_matching_count = both_contacts_matching_count["subject_hemisphere"].count()
    # precentage_both_contacts_match = both_contacts_matching_count / result_count

    # sample_size_dict = {
    #     "sample_size": [result_count],
    #     "spearman_mean": [spearman_mean],
    #     "spearman_median": [spearman_median],
    #     "spearman_std": [spearman_std],
    #     "significant_count": [significant_count],
    #     "percentage_significant": [percentage_significant],
    #     "same_rank_1_count": [same_rank_1],
    #     "percentage_same_rank_1": [precentage_same_rank_1],
    #     "at_least_one_contact_match": [at_least_one_contact_match],
    #     "percentage_at_least_one_contact_match": [precentage_at_least_one_contact_match],
    #     "both_contacts_matching_count": [both_contacts_matching_count],
    #     "precentage_both_contacts_match": [precentage_both_contacts_match],
    # }

    # sample_size_df = pd.DataFrame(sample_size_dict)


def heatmap_method_comparison(value_to_plot: str):
    """
    methods: "externalized_ssd", "externalized_fooof", "JLB_directional", "euclidean_directional", "best_bssu_contacts"

    Input:
        - value_to_plot: str e.g.
            "percentage_at_least_one_same_contact_rank_1_and_2" must be a column in the sample size result Excel file
            "percentage_both_contacts_matching"
            "estimated_beta_spearman",
            "normalized_beta_pearson"

    """

    # load the comparison matrix for the value to plot
    loaded_comparison_matrix = helpers.get_comparison_matrix_for_heatmap(value_to_plot=value_to_plot)

    comparison_matrix = loaded_comparison_matrix["comparison_matrix"]
    comparison_dict = loaded_comparison_matrix["comparison_dict"]
    sample_size = loaded_comparison_matrix["sample_size"]
    list_of_methods = loaded_comparison_matrix["list_of_methods"]

    # Create a heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(comparison_matrix, cmap='coolwarm', interpolation='nearest')

    cbar = fig.colorbar(heatmap)
    # cbar.set_label(f"{value_to_plot}")

    ax.set_xticks(range(len(list_of_methods)))
    ax.set_yticks(range(len(list_of_methods)))
    ax.set_xticklabels(list_of_methods, rotation=45)
    ax.set_yticklabels(list_of_methods)
    ax.grid(False)

    title_str = {
        "percentage_at_least_one_same_contact_rank_1_and_2": 'Selecting 2 contacts from 6 directional contacts:'
        + '\nhow many hemispheres with at least 1 matching contact [%]?',
        "percentage_both_contacts_matching": 'Selecting 2 contacts from 6 directional contacts:'
        + '\nhow many hemispheres with with both contacts matching [%]?',
        "estimated_beta_spearman": "Spearman correlation of 6 directional values per hemisphere"
        + '\nhow many hemispheres with significant correlation [%]?',
        "normalized_beta_pearson": "Pearson correlation of 6 directional normalized values per hemisphere"
        + '\nhow many hemispheres with significant correlation [%]?',
    }

    ax.set_title(title_str[value_to_plot])

    # Add the values to the heatmap cells
    for i in range(len(list_of_methods)):
        for j in range(len(list_of_methods)):
            ax.text(j, i, f"{comparison_matrix[i][j]:.2f}", ha='center', va='center', color='black', fontsize=10)

    helpers.save_fig_png_and_svg(filename=f"heatmap_method_comparison_{value_to_plot}", figure=fig)

    return comparison_matrix, comparison_dict, sample_size
