""" Spatial distribution of beta power: bipolar and monopolar"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.stats import ttest_ind
import statistics
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import plotly.express as px

import itertools
import seaborn as sns


######### PRIVATE PACKAGES #########
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import percept_helpers as helpers

RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
FIGURES_PATH = find_folders.get_local_path(folder="GroupFigures")


############ BIPOLAR SPEARMAN CORRELATION ############
def fooof_beta_write_session_comparison_df(cohort: str):
    """
    Write a dictionary consisting of dataframes for each session comparison

        - for all channel groups: Ring, SegmIntra, SegmInter
        - and all session comparisons


    Input:
        - cohort: str e.g. "all_included" "group_1", "group_2", "group_3"

    1) Load the fooof beta rank dataframes: e.g. beta_ranks_all_channels_fooof_periodic_spectrum.pickle

    2) for each comparison

        and for each group ["ring", "segm_inter", "segm_intra"]

        - per STN:  calculate the MEAN difference of ranks
        - get average of all STN MEAN differences of ranks



    """

    # load the beta rank dataframes
    beta_rank_DF = loadResults.select_fooof_data(dataset="bipolar_beta", cohort=cohort)
    beta_rank_DF = beta_rank_DF["fooof_data"]

    # new column with stn and channel info combined
    beta_rank_DF_copy = beta_rank_DF.copy()
    beta_rank_DF_copy["stn_channel"] = (
        beta_rank_DF_copy.subject_hemisphere.values + "_" + beta_rank_DF_copy.bipolar_channel.values
    )

    if cohort in ["group_3", "all_included"]:
        compare_sessions = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "postop_fu18or24m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_postop",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_2":
        compare_sessions = [
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_1":
        compare_sessions = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
        ]

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    ##########################    WRITE COMPARISON DATAFRAMES PER SESSION COMPARISON  ##########################
    # for each session comparison, get STNs that have recordings at both sessions
    # subtract rank_session1 from rank_session2 and take the absolute value

    comparisons_storage = {}
    sample_size_dict = {}

    for group in channel_groups:
        if group == "ring":
            channels = ['01', '12', '23']

        elif group == "segm_inter":
            channels = ["1A2A", "1B2B", "1C2C"]

        elif group == "segm_intra":
            channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

        # dataframe only of one channel group
        group_df = beta_rank_DF_copy.loc[beta_rank_DF_copy["bipolar_channel"].isin(channels)]

        for comparison in compare_sessions:
            two_sessions = comparison.split("_")
            session_1 = two_sessions[0]
            session_2 = two_sessions[1]

            # Dataframe per session
            session_1_df = group_df.loc[(group_df["session"] == session_1)]
            session_2_df = group_df.loc[(group_df["session"] == session_2)]

            # list of STNs per session
            session_1_stns = list(session_1_df.subject_hemisphere.unique())
            session_2_stns = list(session_2_df.subject_hemisphere.unique())

            # list of STNs included in both sessions
            STN_list = list(set(session_1_stns) & set(session_2_stns))
            STN_list.sort()

            # get the rows with STNs with both sessions
            comparison_df_1 = session_1_df.loc[session_1_df["subject_hemisphere"].isin(STN_list)]
            comparison_df_2 = session_2_df.loc[session_2_df["subject_hemisphere"].isin(STN_list)]

            # subtract ranks from each other row by row from session 1 to session 2
            # check if for both sessions all channels are available (sometimes FOOOF fitting didnt work and channels are missing)
            comparison_df_1 = comparison_df_1[comparison_df_1["stn_channel"].isin(comparison_df_2["stn_channel"])]
            comparison_df_2 = comparison_df_2[comparison_df_2["stn_channel"].isin(comparison_df_1["stn_channel"])]

            abs_difference_ranks = np.absolute(
                comparison_df_1.beta_rank.values - comparison_df_2.beta_rank.values
            )  # array with differences of ranks of one session comparison
            sample_size = len(abs_difference_ranks)

            comparison_df_merged = comparison_df_1.merge(comparison_df_2, left_on="stn_channel", right_on="stn_channel")
            comparison_df_merged_copy = comparison_df_merged.copy()
            comparison_df_merged_copy["abs_difference_ranks"] = abs_difference_ranks

            sample_size_dict[f"{group}_{comparison}"] = [group, comparison, sample_size]
            comparisons_storage[f"{group}_{comparison}"] = comparison_df_merged_copy

    return comparisons_storage


def fooof_bip_channel_groups_beta_spearman(cohort: str, all_groups_together: str):
    """

    Input:

        - cohort: str e.g. "all_included", "group_1", "group_2", "group_3"
        - all_groups_together: "yes" or "no"    -> "yes" will calculate the beta correlation of all LFPs of each subject
                                                -> "no" will calculate the beta correlation of each LFP group seperately of each subject

    From the above function load the comparison_storage dictionary
    fooof_beta_write_session_comparison_df()

        - keys of the dictionary: "ring_fu3m_fu12m" for each channel group and session comparison

    """

    if cohort in ["group_3", "all_included"]:
        compare_sessions = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "postop_fu18or24m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_postop",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_2":
        compare_sessions = [
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_1":
        compare_sessions = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
        ]

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    # load the data
    session_comp_df = fooof_beta_write_session_comparison_df(cohort=cohort)

    ##########################      GET MEAN OF SPEARMAN CORRELATION OF BETA POWER PER SESSION COMPARISON AND CHANNEL GROUP  ##########################
    # 1) get the spearman correlation between all channels within one STN and then across STNs
    # 2) calculate the mean of all spearman r and pvalues for each session comparison and channel group

    fooof_beta_spearman = {}
    single_hemisphere_fooof_beta_spearman = {}

    fontdict = {"size": 25}

    for comp in compare_sessions:
        # Figure Layout per comparison: 3 rows (Ring, SegmIntra, SegmInter), 1 column
        # fig, axes = plt.subplots(3,1,figsize=(10,15))

        if all_groups_together == "yes":
            comp_all_groups_DF = pd.DataFrame()

            group_name = "all_LFPs"

            # concatenate all groups together:
            for group in channel_groups:
                c_g_DF = session_comp_df[f"{group}_{comp}"]
                comp_all_groups_DF = pd.concat([comp_all_groups_DF, c_g_DF])

            # list of available STNs
            stn_list = list(comp_all_groups_DF["subject_hemisphere_x"].unique())

            # list with all spearman r values per comparison and channel group
            spearman_r_list = []
            spearman_pval_list = []

            for stn in stn_list:
                # Dataframe of one stn
                stn_data = comp_all_groups_DF.loc[comp_all_groups_DF.subject_hemisphere_x == stn]

                # correlate each stn channel group session comparison seperately
                spearman_beta = stats.spearmanr(stn_data.beta_average_x, stn_data.beta_average_y)
                spearman_beta_r = spearman_beta.statistic
                spearman_beta_pval = spearman_beta.pvalue

                # make sure to not store r values equal to 1 (this leads to inf value when transofrming to fisher z)
                if spearman_beta_r == 1.0:
                    # Define epsilon
                    epsilon = 1e-10
                    # Calculate 1 - epsilon
                    spearman_beta_r = 1 - epsilon

                # store all spearman values of all stns in a group
                spearman_r_list.append(spearman_beta_r)
                spearman_pval_list.append(spearman_beta_pval)

                # add single stn correlation values to a dictionary
                single_hemisphere_fooof_beta_spearman[f"{comp}_{group_name}_{stn}"] = [
                    comp,
                    group_name,
                    stn,
                    spearman_beta_r,
                    spearman_beta_pval,
                ]

            # for each session comparison - get description of data list
            # spearman r
            mean_spearman_comp_group = np.mean(spearman_r_list)
            median_spearman_comp_group = np.median(spearman_r_list)
            std_spearman_comp_group = np.std(spearman_r_list)

            # Fisher transformation of spearman r
            fisher_transformation_spearman_r = np.arctanh(spearman_r_list)

            # spearman pval
            mean_pval_comp_group = np.mean(spearman_pval_list)
            median_pval_comp_group = np.median(spearman_pval_list)
            std_pval_comp_group = np.std(spearman_pval_list)

            sample_size_spearman = len(spearman_r_list)  # number of STNs in one session comparison

            # store all values in dictionary
            fooof_beta_spearman[f"{comp}_{group_name}"] = [
                comp,
                group_name,
                sample_size_spearman,
                std_spearman_comp_group,
                mean_spearman_comp_group,
                median_spearman_comp_group,
                fisher_transformation_spearman_r,
                std_pval_comp_group,
                mean_pval_comp_group,
                median_pval_comp_group,
            ]

        elif all_groups_together == "no":
            for g, group in enumerate(channel_groups):
                # Dataframe of one comparison and one channel group
                comp_group_DF = session_comp_df[f"{group}_{comp}"]

                # list of available STNs
                stn_list = list(comp_group_DF["subject_hemisphere_x"].unique())

                # list with all spearman r values per comparison and channel group
                spearman_r_list = []
                spearman_pval_list = []

                for stn in stn_list:
                    # Dataframe of one stn
                    stn_data = comp_group_DF.loc[comp_group_DF.subject_hemisphere_x == stn]

                    # correlate each stn channel group session comparison seperately
                    spearman_beta = stats.spearmanr(stn_data.beta_average_x, stn_data.beta_average_y)
                    spearman_beta_r = spearman_beta.statistic
                    spearman_beta_pval = spearman_beta.pvalue

                    # make sure to not store r values equal to 1 (this leads to inf value when transofrming to fisher z)
                    if spearman_beta_r == 1.0:
                        # Define epsilon
                        epsilon = 1e-10
                        # Calculate 1 - epsilon
                        spearman_beta_r = 1 - epsilon

                    # store all spearman values of all stns in a group
                    spearman_r_list.append(spearman_beta_r)
                    spearman_pval_list.append(spearman_beta_pval)

                    # add single stn correlation values to a dictionary
                    single_hemisphere_fooof_beta_spearman[f"{comp}_{group}_{stn}"] = [
                        comp,
                        group,
                        stn,
                        spearman_beta_r,
                        spearman_beta_pval,
                    ]

                # for each channel group and session comparison - get description of data list
                # spearman r
                mean_spearman_comp_group = np.mean(spearman_r_list)
                median_spearman_comp_group = np.median(spearman_r_list)
                std_spearman_comp_group = np.std(spearman_r_list)

                # Fisher transformation of spearman r
                fisher_transformation_spearman_r = np.arctanh(spearman_r_list)

                # spearman pval
                mean_pval_comp_group = np.mean(spearman_pval_list)
                median_pval_comp_group = np.median(spearman_pval_list)
                std_pval_comp_group = np.std(spearman_pval_list)

                sample_size_spearman = len(spearman_r_list)  # number of STNs in one session comparison

                # store all values in dictionary
                fooof_beta_spearman[f"{comp}_{group}"] = [
                    comp,
                    group,
                    sample_size_spearman,
                    std_spearman_comp_group,
                    mean_spearman_comp_group,
                    median_spearman_comp_group,
                    fisher_transformation_spearman_r,
                    std_pval_comp_group,
                    mean_pval_comp_group,
                    median_pval_comp_group,
                ]

    # from dictionary to DF: single spearman values per STN, group and session comparison
    single_stn_spearman_DF = pd.DataFrame(single_hemisphere_fooof_beta_spearman)
    single_stn_spearman_DF.rename(
        index={0: "comparison", 1: "channel_group", 2: "subject_hemisphere", 3: "spearman_r", 4: "spearman_pval"},
        inplace=True,
    )
    single_stn_spearman_DF = single_stn_spearman_DF.transpose()
    single_stn_spearman_DF_copy = single_stn_spearman_DF.copy()

    # add new column: significant yes, no
    significant_correlation = single_stn_spearman_DF_copy["spearman_pval"] < 0.05
    single_stn_spearman_DF_copy["significant_correlation"] = [
        "yes" if cond else "no" for cond in significant_correlation
    ]

    # delete postop_postop, fu3m_fu3m etc
    filter_comparisons = [
        "postop_postop",
        "fu3m_fu3m",
        "fu12m_fu12m",
        "fu18or24m_fu18or24m",
        "fu3m_postop",
        "fu12m_postop",
        "fu18or24m_postop",
        "fu12m_fu3m",
        "fu18or24m_fu3m",
        "fu18or24m_fu12m",
    ]
    single_stn_spearman_DF_copy = single_stn_spearman_DF_copy[
        ~single_stn_spearman_DF_copy["comparison"].isin(filter_comparisons)
    ]  # filters out all rows with comparison values in the given list

    # save as Excel
    single_stn_spearman_DF_copy.to_excel(
        os.path.join(RESULTS_PATH, f"revision_{cohort}_bipolar_LFPs_beta_correlations_per_stn.xlsx"),
        sheet_name="bipolar_beta_correlations",
        index=False,
    )
    print(
        "file: ",
        f"revision_{cohort}_bipolar_LFPs_beta_correlations_per_stn.xlsx",
        "\nwritten in: ",
        RESULTS_PATH,
    )

    # Permutation_BIP transform from dictionary to Dataframe
    spearman_result_df = pd.DataFrame(fooof_beta_spearman)
    spearman_result_df.rename(
        index={
            0: "comparison",
            1: "channel_group",
            2: "sample_size_stn",
            3: "standard_deviation_spearman_values",
            4: "mean_spearman_values",
            5: "median_spearman_values",
            6: "fisher_transformation_spearman_r",
            7: "standard_deviation_pval",
            8: "mean_pval",
            9: "median_pval",
        },
        inplace=True,
    )
    spearman_result_df = spearman_result_df.transpose()

    spearman_result_df[["session_1", "session_2"]] = spearman_result_df["comparison"].str.split("_", expand=True)

    return {"spearman_result_df": spearman_result_df, "single_stn_spearman_DF_copy": single_stn_spearman_DF_copy}


def plot_boxplot_spearman_r_values(cohort: str, all_groups_together: str):
    """
    Plot a boxplot of the spearman r values of all LFPs per session comparison

    Input:
        - cohort: str e.g. "all_included", "group_1", "group_2", "group_3"
        - all_groups_together: "yes" or "no"    -> "yes" will calculate the beta correlation of all LFPs of each subject
                                                -> "no" function doesnt work for individual channel groups
    """

    # palette = sns.color_palette("Paired")

    spearman_result = fooof_bip_channel_groups_beta_spearman(cohort=cohort, all_groups_together=all_groups_together)
    spearman_result_df = spearman_result["single_stn_spearman_DF_copy"]
    spearman_group_result = spearman_result["spearman_result_df"]

    if all_groups_together == "yes":

        # plot a boxplot
        fig, ax = plt.subplots()

        sns.boxplot(
            x="comparison",
            y="spearman_r",
            data=spearman_result_df,
            color="white",
            width=0.6,
            showmeans=True,
            meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"},
        )
        # palette="Set2"

        # Overlay the dots (individual data points)
        # sns.stripplot(x="comparison", y="spearman_r", data=spearman_result_df,
        #             jitter=True, color="black", alpha=0.7, size=7)

        sns.stripplot(
            x="comparison",
            y="spearman_r",
            data=spearman_result_df,
            jitter=True,
            # hue="subject_hemisphere",  # Different color per sub_id
            color="grey",
            # palette=palette,  # Use a palette that supports multiple colors
            dodge=True,
            size=7,
            alpha=0.4,
        )

        # Customize the legend
        # plt.legend(title="STN", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Customize the plot
        plt.title(f"Spearman Correlation between sessions: {cohort}", fontsize=16)
        plt.xlabel("Comparison", fontsize=14)
        plt.ylabel("Spearman r", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()

    else:
        print("No boxplot for individual channel groups")

    fig.savefig(
        os.path.join(
            FIGURES_PATH,
            f"revision_{cohort}_bipolar_LFPs_beta_correlations_boxplot.png",
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(
            FIGURES_PATH,
            f"revision_{cohort}_bipolar_LFPs_beta_correlations_boxplot.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    return {"spearman_result_df": spearman_group_result, "single_stn_spearman_DF_copy": spearman_result_df}


def t_test_fisher_transformed_correlation_coeff(cohort: str, all_groups_together: str, correction: str = None):
    """
    Calculate the t-test of the fisher transformed correlation coefficients
    This test only works for cohorts with consistent recording sessions (e.g. group_1, group_2, group_3)

    Input:
        - cohort: str e.g. "group_1", "group_2", "group_3"
        - all_groups_together: "yes" or "no"    -> "yes" will calculate the beta correlation of all LFPs of each subject
                                                -> "no" will calculate the beta correlation of each LFP group seperately of each subject
        - correction: "bonferroni", "fdr_bh", "holm" or None


    """

    # store result in a dictionary
    t_test_results = {}
    all_groups_yes_no = {"yes": "all_LFPs", "no": "each_LFP_group"}

    # load the data
    spearman_result = fooof_bip_channel_groups_beta_spearman(cohort=cohort, all_groups_together=all_groups_together)
    spearman_result_df = spearman_result["spearman_result_df"]
    spearman_single_results = spearman_result["single_stn_spearman_DF_copy"]

    # get the fisher transformed correlation coefficients
    if cohort == "group_3":
        filter_comparisons = [
            "fu3m_postop",
            "fu12m_postop",
            "fu18or24m_postop",
            "fu12m_fu3m",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
        ]

    elif cohort == "group_2":
        filter_comparisons = [
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu18or24m",
        ]

    elif cohort == "group_1":
        filter_comparisons = [
            "postop_fu3m",
            "postop_fu12m",
            "fu3m_fu12m",
        ]

    # if cohort == "group_3":
    #     filter_comparisons = [
    #         "postop_fu3m",
    #         "fu3m_fu12m" "fu12m_fu18or24m",
    #     ]

    # elif cohort == "group_2":
    #     filter_comparisons = [
    #         "fu3m_fu12m",
    #         "fu12m_fu18or24m",
    #     ]

    # elif cohort == "group_1":
    #     filter_comparisons = [
    #         "postop_fu3m",
    #         "fu3m_fu12m",
    #     ]

    # filter from dataframe only the comparisons that are needed for the t-test
    spearman_result_df = spearman_result_df[spearman_result_df["comparison"].isin(filter_comparisons)]

    # Create a 6x6 matrix filled with zeros
    comparison_matrix = np.zeros((len(filter_comparisons), len(filter_comparisons)))
    all_p_values = []
    comparison_pairs = []

    # Populate the matrix with comparison values
    for i in range(len(filter_comparisons)):
        for j in range(len(filter_comparisons)):
            if i != j:
                # Compare values and update the matrix
                # get the fisher transformed correlation coefficients for i and j
                fisher_i = spearman_result_df[spearman_result_df["comparison"] == filter_comparisons[i]]
                fisher_i = fisher_i["fisher_transformation_spearman_r"].values[0]

                fisher_j = spearman_result_df[spearman_result_df["comparison"] == filter_comparisons[j]]
                fisher_j = fisher_j["fisher_transformation_spearman_r"].values[0]

                # perform a paired t-test on the transformed correlation coefficients
                # statistic, p_val = stats.ttest_ind(fisher_i, fisher_j) # this test is NOT paired!!!

                differences = fisher_i - fisher_j  # computes differences in Fisher-transformed scores

                # perform a paired t-test on the differences
                statistic, p_val = stats.ttest_rel(differences, np.zeros_like(differences))

                # store the p-value in the matrix
                comparison_matrix[i, j] = p_val
                all_p_values.append(p_val)
                comparison_pairs.append((i, j))

                # and store the statistic and p_val in a dictionary
                t_test_results[f"{filter_comparisons[i]}_vs_{filter_comparisons[j]}"] = [
                    filter_comparisons[i],
                    filter_comparisons[j],
                    statistic,
                    p_val,
                ]

            elif i == j:
                comparison_matrix[i, j] = 1

    if correction:
        corrected_p_values = multipletests(all_p_values, method=correction)[1]

        # Map corrected p-values back to the matrix and dictionary
        for k, (i, j) in enumerate(comparison_pairs):
            comparison_matrix[i, j] = corrected_p_values[k]
            t_test_results[f"{filter_comparisons[i]}_vs_{filter_comparisons[j]}"] = [
                filter_comparisons[i],
                filter_comparisons[j],
                statistic,
                corrected_p_values[k],
            ]

    elif correction is None:
        corrected_p_values = all_p_values

    # write Dataframe from dictionary
    t_test_results_df = pd.DataFrame(t_test_results)
    t_test_results_df.rename(index={0: "comparison_1", 1: "comparison_2", 2: "statistic", 3: "p-value"}, inplace=True)
    t_test_results_df = t_test_results_df.transpose()

    # save as Excel
    t_test_results_df.to_excel(
        os.path.join(
            RESULTS_PATH,
            f"revision_bipolar_LFPs_Spearman_correlations_t-test_fisher_transformed_{cohort}_{all_groups_yes_no[all_groups_together]}.xlsx",
        ),
        sheet_name="t-test_fisher_transformed",
        index=False,
    )

    # plot the matrix in a heatmap
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(comparison_matrix, cmap=plt.cm.YlOrRd)

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(comparison_matrix.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(comparison_matrix.shape[0]) + 0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(filter_comparisons, minor=False, rotation=45)
    ax.set_yticklabels(filter_comparisons, minor=False)

    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label("p-value")

    # Add the cell values to the heatmap
    for i in range(comparison_matrix.shape[0]):
        for j in range(comparison_matrix.shape[1]):
            plt.text(
                j + 0.5, i + 0.5, str("{: .2f}".format(comparison_matrix[i, j])), ha='center', va='center'
            )  # only show 2 numbers after the comma of a float

    # Add a title
    plt.title(f"t-test of fisher transformed Spearman correlation coefficients \nbipolar LFPs")

    fig.tight_layout()

    helpers.save_fig_png_and_svg(
        path=FIGURES_PATH,
        filename=f"revision_t-test_fisher_transformed_Spearman_coeff_{correction}_bipolar_{cohort}_{all_groups_yes_no[all_groups_together]}",
        figure=fig,
    )

    return {
        "t-test_results": t_test_results_df,
        "spearman_group_result": spearman_result_df,
        "spearman_single_results": spearman_single_results,
        "all_p_values": all_p_values,
        "comparison_matrix": comparison_matrix,
        "comparison_pairs": comparison_pairs,
        "corrected_p_values": corrected_p_values,
    }


########################################################
################# MONOPOLAR Beta power #################


def fooof_monopol_psd_spearman_betw_sessions(
    cohort: str,
    mean_or_median: str,
    only_segmental: str,
    values_to_correlate: str,
    similarity_calculation: str,
    fooof_version: str,
):
    """
    Load file:
    containing DF with all monopolar PSD estimates in a frequency band, their ranks along an electrode and their PSD relative to the highest PSD of an electrode.


    Input:
        - cohort: str e.g. "all_included", "group_1", "group_2", "group_3"
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)


        - mean_or_median: str, e.g. "mean", "median"
        - only_segmental:str, "yes" -> will only included segmental contacts
        - values_to_correlate:str  "not_normalized", "rel_to_rank_1", "rel_range_0_to_1" (only "not_normalized" can be used for only segmental, because the relative values were calucalted with ring contacts included)
        - similarity_calculation:str "inverse_distance", "exp_neg_distance", "inverse_sq_distance"
        - fooof_version: "v2"


    1) After loading the data, only select the contacts 0, 1A, 1B, 1C, 2A, 2B, 2C and 3
        - rank again from 1-8 -> column "Rank8contacts"

    2) Use scipy.stats.spearmanr to correlate each STN electrode at two sessions
        - choose between ranks or rel PSD normalized to the highest PSD per electrode
        - STN = one hemisphere of one subject
        - pairs of sessions:
            [('postop', 'postop'),
            ('postop', 'fu3m'),
            ('postop', 'fu12m'),
            ('postop', 'fu18m'),
            ('fu3m', 'postop'),
            ('fu3m', 'fu3m'),
            ('fu3m', 'fu12m'),
            ('fu3m', 'fu18m'),
            ('fu12m', 'postop'),
            ('fu12m', 'fu3m'),
            ('fu12m', 'fu12m'),
            ('fu12m', 'fu18m'),
            ('fu18m', 'postop'),
            ('fu18m', 'fu3m'),
            ('fu18m', 'fu12m'),
            ('fu18m', 'fu18m')]

    3) save values in results_DF with columns:
        - session_1
        - session_2
        - subject_hemisphere
        - spearman_r (r value from -1 to 1, 0=no correlation; 1=positive correlation, ???)
        - pval

    4) Calculate the Mean or Median of all STN correlation r values per session combination

    5) Restructure the column of means or medians to 4x4 matrices with floats

    6) Plot a Heatmap using plotly visualizing the mean or medians of all session combinations


    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

    # load the data
    loaded_fooof_monopolar = loadResults.select_fooof_data(dataset="monopolar_beta", cohort=cohort)
    loaded_fooof_monopolar = loaded_fooof_monopolar["fooof_data"]

    # new column with stn and channel info combined
    # beta_rank_DF_copy = beta_rank_DF.copy()
    # beta_rank_DF_copy["stn_channel"] = (
    #     beta_rank_DF_copy.subject_hemisphere.values + "_" + beta_rank_DF_copy.bipolar_channel.values
    # )

    if cohort == "group_3":
        session_comparison = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "postop_fu18or24m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_postop",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_2":
        session_comparison = [
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu3m",
            "fu12m_fu12m",
            "fu12m_fu18or24m",
            "fu18or24m_fu3m",
            "fu18or24m_fu12m",
            "fu18or24m_fu18or24m",
        ]

    elif cohort == "group_1":
        session_comparison = [
            "postop_postop",
            "postop_fu3m",
            "postop_fu12m",
            "fu3m_postop",
            "fu3m_fu3m",
            "fu3m_fu12m",
            "fu12m_postop",
            "fu12m_fu3m",
            "fu12m_fu12m",
        ]

    # # loaded_fooof_monopolar = loadResults.load_fooof_monoRef_all_contacts_weight_beta(similarity_calculation=similarity_calculation)
    # loaded_fooof_monopolar = loadResults.load_pickle_group_result(
    #     filename=f"fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}_{fooof_version}",
    #     fooof_version=fooof_version,
    # )

    # from the list of all existing sub_hem STNs, get only the STNs with existing sessions 1 + 2
    session_pair_stn_list = {}
    sample_size_dict = {}

    # for each session comparison select STNs that have recordings for both
    for comparison in session_comparison:
        both_sessions = list(comparison.split("_"))

        # define session 1 and 2
        session_1 = both_sessions[0]  # e.g. "postop"
        session_2 = both_sessions[1]  # e.g. "fu3m"

        session_1_df = loaded_fooof_monopolar.loc[loaded_fooof_monopolar.session == session_1]
        session_2_df = loaded_fooof_monopolar.loc[loaded_fooof_monopolar.session == session_2]

        # find STNs with both sessions
        session_1_stns = list(session_1_df.subject_hemisphere.unique())
        session_2_stns = list(session_2_df.subject_hemisphere.unique())

        stn_comparison_list = list(set(session_1_stns) & set(session_2_stns))
        stn_comparison_list.sort()

        comparison_df_1 = session_1_df.loc[session_1_df["subject_hemisphere"].isin(stn_comparison_list)]
        comparison_df_2 = session_2_df.loc[session_2_df["subject_hemisphere"].isin(stn_comparison_list)]

        comparsion_df = pd.concat([comparison_df_1, comparison_df_2], axis=0)

        if only_segmental == "yes":
            comparsion_df = comparsion_df.loc[
                comparsion_df.contact.isin(segmental_contacts)
            ]  # only rows with segmental contacts are included
            print("only segmental contacts included")

        else:
            print("all contacts included")

        # correlate each electrode seperately
        for sub_hem in stn_comparison_list:
            # only run, if sub_hem STN exists in both session Dataframes
            if sub_hem not in comparsion_df.subject_hemisphere.values:
                continue

            # only take one electrode at both sessions and get spearman correlation
            stn_comparison = comparsion_df.loc[comparsion_df["subject_hemisphere"] == sub_hem]

            stn_session1 = stn_comparison.loc[stn_comparison.session == session_1]
            stn_session2 = stn_comparison.loc[stn_comparison.session == session_2]

            # choose which values to correlate
            if values_to_correlate == "not_normalized":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(
                    stn_session1.estimated_monopolar_beta_psd.values, stn_session2.estimated_monopolar_beta_psd.values
                )

            elif values_to_correlate == "rel_to_rank_1":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(
                    stn_session1.beta_psd_rel_to_rank1.values, stn_session2.beta_psd_rel_to_rank1.values
                )

            elif values_to_correlate == "rel_range_0_to_1":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(
                    stn_session1.beta_psd_rel_range_0_to_1.values, stn_session2.beta_psd_rel_range_0_to_1.values
                )

            spearman_beta_r = spearman_psd_stn.statistic

            # make sure to not store r values equal to 1 (this leads to inf value when transofrming to fisher z)
            if spearman_psd_stn.statistic == 1.0:
                # Define epsilon
                epsilon = 1e-10
                # Calculate 1 - epsilon
                spearman_beta_r = 1 - epsilon

            # store values in a dictionary
            session_pair_stn_list[f"{comparison}_{sub_hem}"] = [
                session_1,
                session_2,
                comparison,
                sub_hem,
                spearman_beta_r,
                spearman_psd_stn.pvalue,
            ]

    # save the dictionary as a Dataframe
    results_DF = pd.DataFrame(session_pair_stn_list)
    results_DF.rename(
        index={
            0: "session_1",
            1: "session_2",
            2: "session_comparison",
            3: "subject_hemisphere",
            4: f"spearman_r",
            5: f"pval",
        },
        inplace=True,
    )
    results_DF = results_DF.transpose()

    # save Dataframe to Excel
    results_DF_copy = results_DF.copy()
    results_DF_copy = results_DF_copy.drop(columns=["session_1", "session_2"])

    # delete postop_postop, fu3m_fu3m etc
    filter_comparisons = [
        "postop_postop",
        "fu3m_fu3m",
        "fu12m_fu12m",
        "fu18or24m_fu18or24m",
        "fu3m_postop",
        "fu12m_postop",
        "fu18or24m_postop",
        "fu12m_fu3m",
        "fu18or24m_fu3m",
        "fu18or24m_fu12m",
    ]
    results_DF_copy = results_DF_copy[
        ~results_DF_copy["session_comparison"].isin(filter_comparisons)
    ]  # filters out all rows with comparison values in the given list

    # add new column: significant yes, no
    significant_correlation = results_DF_copy["pval"] < 0.05
    results_DF_copy["significant_correlation"] = ["yes" if cond else "no" for cond in significant_correlation]

    # save as Excel
    results_DF_copy.to_excel(
        os.path.join(
            results_path,
            f"revision_fooof_monopol_{similarity_calculation}_beta_correlations_per_stn_{fooof_version}.xlsx",
        ),
        sheet_name="monopolar_beta_correlations",
        index=False,
    )
    print(
        "file: ",
        f"revision_fooof_monopol_{similarity_calculation}_beta_correlations_per_stn_{fooof_version}.xlsx",
        "\nwritten in: ",
        results_path,
    )

    ################## CALCULATE THE MEAN OR MEDIAN OF ALL SPEARMAN R CORRELATION VALUES OF EACH SESSION COMBINATION ##################
    spearman_m = {}

    # calculate the MEAN or median of each session pair
    for comp in session_comparison:
        # define session 1 and session 2 to correlate
        both_sessions = list(comp.split("_"))

        # define session 1 and 2
        session_1 = both_sessions[0]  # e.g. "postop"
        session_2 = both_sessions[1]  # e.g. "fu3m"

        pairs_df = results_DF.loc[(results_DF.session_1 == session_1)]
        pairs_df = pairs_df.loc[(pairs_df.session_2 == session_2)]

        if mean_or_median == "mean":
            m_spearmanr = pairs_df.spearman_r.mean()
            m_pval = pairs_df.pval.mean()

        elif mean_or_median == "median":
            m_spearmanr = pairs_df.spearman_r.median()
            m_pval = pairs_df.pval.median()

        spearman_std = np.std(pairs_df.spearman_r)

        s_comp_df = results_DF.loc[results_DF.session_comparison == comp]
        s_comp_count = s_comp_df["session_comparison"].count()

        # count the number of significant correlations
        # significant_filter = s_comp_df.loc[s_comp_df.significant_correlation == "yes"]
        # count_significant = len(significant_filter)

        # percentage_significant = (count_significant / s_comp_count) * 100

        spearman_m[f"{comp}_spearman_m"] = [
            session_1,
            session_2,
            comp,
            m_spearmanr,
            m_pval,
            spearman_std,
            s_comp_count,
            # count_significant,
            # percentage_significant,
        ]

    # write a Dataframe with the mean or median spearman values per session combination
    spearman_m_df = pd.DataFrame(spearman_m)
    spearman_m_df.rename(
        index={
            0: "session_1",
            1: "session_2",
            2: "session_comparison",
            3: f"{mean_or_median}_spearmanr",
            4: f"{mean_or_median}_pval",
            5: "spearman_std",
            6: "sample_size",
            # 7: "count_significant",
            # 8: "percentage_significant",
        },
        inplace=True,
    )
    spearman_m_df = spearman_m_df.transpose()

    return {"results_DF": results_DF_copy, "spearman_m_df": spearman_m_df}


def plot_boxplot_monopolar_spearman_r_values(cohort: str, only_segmental: str):
    """
    Plot a boxplot of the spearman r values of all LFPs per session comparison

    Input:
        - cohort: str e.g. "all_included", "group_1", "group_2", "group_3"
        - only_segmental: "yes" or "no"    -> "yes" will calculate the beta correlation only of segmental contacts of each subject
    """

    # palette = sns.color_palette("Paired")

    monopolar_spearman = fooof_monopol_psd_spearman_betw_sessions(
        cohort=cohort,
        mean_or_median="mean",
        only_segmental=only_segmental,
        values_to_correlate="not_normalized",
        similarity_calculation="inverse_sq_distance",
        fooof_version="v2",
    )
    spearman_result_df = monopolar_spearman["results_DF"]
    spearman_group_result = monopolar_spearman["spearman_m_df"]

    # plot a boxplot
    fig, ax = plt.subplots()

    sns.boxplot(
        x="session_comparison",
        y="spearman_r",
        data=spearman_result_df,
        color="white",
        width=0.6,
        showmeans=True,
        meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    # palette="Set2"

    # Overlay the dots (individual data points)
    # sns.stripplot(x="comparison", y="spearman_r", data=spearman_result_df,
    #             jitter=True, color="black", alpha=0.7, size=7)

    sns.stripplot(
        x="session_comparison",
        y="spearman_r",
        data=spearman_result_df,
        jitter=True,
        # hue="subject_hemisphere",  # Different color per sub_id
        color="grey",
        # palette=palette,  # Use a palette that supports multiple colors
        dodge=True,
        size=7,
        alpha=0.4,
    )

    # Customize the legend
    # plt.legend(title="STN", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Customize the plot
    plt.title(f"Spearman Correlation between sessions: {cohort}", fontsize=16)
    plt.xlabel("Comparison", fontsize=14)
    plt.ylabel("Spearman r", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    fig.savefig(
        os.path.join(
            FIGURES_PATH,
            f"revision_{cohort}_monopolar_only_segments_{only_segmental}_beta_correlations_boxplot.png",
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(
            FIGURES_PATH,
            f"revision_{cohort}_monopolar_only_segments_{only_segmental}_beta_correlations_boxplot.svg",
        ),
        bbox_inches="tight",
        format="svg",
    )

    # count how many significant correlations are in the data per comparison
    significant_correlations = {}
    session_comparison_unique = spearman_result_df["session_comparison"].unique()

    for comparison in session_comparison_unique:
        s_comp_df = spearman_result_df.loc[spearman_result_df.session_comparison == comparison]
        significant = s_comp_df.loc[s_comp_df.significant_correlation == "yes"]
        count_significant = len(significant)
        percent_significant = (count_significant / len(s_comp_df)) * 100

        significant_correlations[comparison] = count_significant, percent_significant

    return {
        "spearman_result_df": spearman_group_result,
        "single_stn_spearman_DF_copy": spearman_result_df,
        "significant_correlations": significant_correlations,
    }


def t_test_fisher_transformed_spearman_monopolar(cohort: str, only_segmental: str, correction: str = None):
    """
    Calculate the t-test of the fisher transformed correlation coefficients
    This test only works for cohorts with consistent recording sessions (e.g. group_1, group_2, group_3)

    Input:
        - cohort: str e.g. "group_1", "group_2", "group_3"
        - only_segmental: "yes" or "no"    -> "yes" will calculate the beta correlation only of segmental contacts of each subject
        - correction: "bonferroni", "fdr_bh", "holm" or None


    """

    t_test_results = {}

    spearman_output = fooof_monopol_psd_spearman_betw_sessions(
        cohort=cohort,
        mean_or_median="mean",
        only_segmental=only_segmental,
        values_to_correlate="not_normalized",
        similarity_calculation="inverse_sq_distance",
        fooof_version="v2",
    )

    spearman_m_df = spearman_output["results_DF"]  # single correlations
    spearman_group_df = spearman_output["spearman_m_df"]  # group correlations

    # get the fisher transformed correlation coefficients
    if cohort == "group_3":
        filter_comparisons = [
            "postop_fu3m",
            "postop_fu12m",
            "postop_fu18or24m",
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu18or24m",
        ]

    elif cohort == "group_2":
        filter_comparisons = [
            "fu3m_fu12m",
            "fu3m_fu18or24m",
            "fu12m_fu18or24m",
        ]

    elif cohort == "group_1":
        filter_comparisons = [
            "postop_fu3m",
            "postop_fu12m",
            "fu3m_fu12m",
        ]

    # if cohort == "group_3":
    #     filter_comparisons = [
    #         "postop_fu3m",
    #         "fu3m_fu12m",
    #         "fu12m_fu18or24m",
    #     ]

    # elif cohort == "group_2":
    #     filter_comparisons = [
    #         "fu3m_fu12m",
    #         "fu12m_fu18or24m",
    #     ]

    # elif cohort == "group_1":
    #     filter_comparisons = [
    #         "postop_fu3m",
    #         "fu3m_fu12m",
    #     ]

    # Create a 6x6 matrix with zeros
    comparison_matrix = np.zeros((len(filter_comparisons), len(filter_comparisons)))
    all_p_values = []
    comparison_pairs = []

    # Populate the matrix with comparison values
    for i in range(len(filter_comparisons)):
        for j in range(len(filter_comparisons)):
            if i != j:
                # Compare values and update the matrix

                # get a list of spearman r values for i and j
                spearman_list_i = spearman_m_df[spearman_m_df["session_comparison"] == filter_comparisons[i]]
                spearman_list_i = spearman_list_i["spearman_r"].values.tolist()

                spearman_list_j = spearman_m_df[spearman_m_df["session_comparison"] == filter_comparisons[j]]
                spearman_list_j = spearman_list_j["spearman_r"].values.tolist()

                # get the fisher transformed correlation coefficients for i and j
                fisher_i = np.arctanh(spearman_list_i)
                fisher_j = np.arctanh(spearman_list_j)

                # perform a paired t-test on the transformed correlation coefficients
                differences = fisher_i - fisher_j

                # perform a t-test on the transformed correlation coefficients
                # statistic, p_val = stats.ttest_ind(fisher_i, fisher_j) # independent t-test, assumes different subjects per comparison
                statistic, p_val = stats.ttest_rel(differences, np.zeros_like(differences))

                # store the p-value in the matrix
                comparison_matrix[i, j] = p_val
                all_p_values.append(p_val)
                comparison_pairs.append((i, j))

                # and store the statistic and p_val in a dictionary
                t_test_results[f"{filter_comparisons[i]}_vs_{filter_comparisons[j]}"] = [
                    filter_comparisons[i],
                    filter_comparisons[j],
                    fisher_i,
                    fisher_j,
                    statistic,
                    p_val,
                ]

            elif i == j:
                comparison_matrix[i, j] = 1

    if correction:
        corrected_p_values = multipletests(all_p_values, method=correction)[1]

        # Map corrected p-values back to the matrix and dictionary
        for k, (i, j) in enumerate(comparison_pairs):
            comparison_matrix[i, j] = corrected_p_values[k]
            t_test_results[f"{filter_comparisons[i]}_vs_{filter_comparisons[j]}"] = [
                filter_comparisons[i],
                filter_comparisons[j],
                fisher_i,
                fisher_j,
                statistic,
                corrected_p_values[k],
            ]

    elif correction is None:
        corrected_p_values = all_p_values

    # write Dataframe from dictionary
    t_test_results_df = pd.DataFrame(t_test_results)
    t_test_results_df.rename(
        index={
            0: "comparison_1",
            1: "comparison_2",
            2: "fisher_comparison_1",
            3: "fisher_comparison_2",
            4: "statistic",
            5: "p-value",
        },
        inplace=True,
    )
    t_test_results_df = t_test_results_df.transpose()

    # save as Excel
    t_test_results_df.to_excel(
        os.path.join(
            RESULTS_PATH,
            f"revision_monoolar_only_segmental_{only_segmental}_LFPs_Spearman_correlations_t-test_fisher_transformed_{correction}_{cohort}.xlsx",
        ),
        sheet_name="t-test_fisher_transformed",
        index=False,
    )

    # plot the matrix in a heatmap
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(comparison_matrix, cmap=plt.cm.YlOrRd)

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(comparison_matrix.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(comparison_matrix.shape[0]) + 0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(filter_comparisons, minor=False, rotation=45)
    ax.set_yticklabels(filter_comparisons, minor=False)

    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label("p-value")

    # Add the cell values to the heatmap
    for i in range(comparison_matrix.shape[0]):
        for j in range(comparison_matrix.shape[1]):
            plt.text(
                j + 0.5, i + 0.5, str("{: .2f}".format(comparison_matrix[i, j])), ha='center', va='center'
            )  # only show 2 numbers after the comma of a float

    # Add a title
    plt.title(f"t-test of fisher transformed Spearman correlation coefficients \nmonopolar LFPs")

    fig.tight_layout()

    helpers.save_fig_png_and_svg(
        path=FIGURES_PATH,
        filename=f"revision_t-test_fisher_transformed_Spearman_coeff_monoolar_only_segmental_{only_segmental}_{correction}_{cohort}",
        figure=fig,
    )

    return {
        "spearman_single_results": spearman_m_df,
        "spearman_group_results": spearman_group_df,
        "t_test_results_df": t_test_results_df,
        "all_p_values": all_p_values,
        "comparison_matrix": comparison_matrix,
        "comparison_pairs": comparison_pairs,
        "corrected_p_values": corrected_p_values,
    }


######################################################
################# Beta Rank change ######################


def write_df_xy_changes_of_beta_ranks(cohort: str, ranks_included: list):
    """
    Input:
        - cohort: str - "group_1", "group_2", "group_3", "all_included"
        - ranks_included: [1], [1,2] or [1,2,3,4,5,6] etc depends on how many ranks you want to include into the analysis


    Load the monopolar FOOOF dataframe of estimated beta power at segmental contacts and beta rank 1-6



    """

    # defined variables
    if cohort in ["all_included", "group_3"]:
        comparisons = [
            "0_0",
            "0_3",
            "0_12",
            "0_18",
            "3_0",
            "3_3",
            "3_12",
            "3_18",
            "12_0",
            "12_3",
            "12_12",
            "12_18",
            "18_0",
            "18_3",
            "18_12",
            "18_18",
        ]

    elif cohort == "group_1":
        comparisons = [
            "0_0",
            "0_3",
            "0_12",
            "3_0",
            "3_3",
            "3_12",
            "12_0",
            "12_3",
            "12_12",
        ]

    elif cohort == "group_2":
        comparisons = [
            "3_3",
            "3_12",
            "3_18",
            "12_3",
            "12_12",
            "12_18",
            "18_3",
            "18_12",
            "18_18",
        ]

    coord_difference_data = {}
    sample_size = {}
    sample_size_rank_2 = {}

    loaded_data = loadResults.select_fooof_data(dataset="monopolar_only_segmental", cohort=cohort)
    fooof_monopolar_df = loaded_data["fooof_data"]

    fooof_monopolar_df_copy = fooof_monopolar_df.copy()
    fooof_monopolar_df_copy["rank_beta"] = fooof_monopolar_df["rank"].astype(int)

    # replace session names by integers
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.replace(
        to_replace=["postop", "fu3m", "fu12m", "fu18or24m"], value=[0, 3, 12, 18]
    )

    # add 2 new columns "x_coordinate", "y_coordinate"
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(x_direction=fooof_monopolar_df_copy["contact"]).rename(
        columns={"x_direction": "x_direction"}
    )
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(
        to_replace=["1A", "2A"], value=[1, 1]
    )  # direction A
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(
        to_replace=["1B", "2B"], value=[2, 2]
    )  # direction B
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(
        to_replace=["1C", "2C"], value=[3, 3]
    )  # direction C

    fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(y_level=fooof_monopolar_df_copy["contact"]).rename(
        columns={"y_level": "y_level"}
    )
    fooof_monopolar_df_copy["y_level"] = fooof_monopolar_df_copy["y_level"].replace(
        to_replace=["1A", "1B", "1C"], value=[1, 1, 1]
    )  # level 1
    fooof_monopolar_df_copy["y_level"] = fooof_monopolar_df_copy["y_level"].replace(
        to_replace=["2A", "2B", "2C"], value=[2, 2, 2]
    )  # level 2

    # select only the included ranks
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.loc[fooof_monopolar_df_copy.rank_beta.isin(ranks_included)]

    # check which STNs ad sessions exist in data
    sub_hem_keys = list(fooof_monopolar_df_copy.subject_hemisphere.unique())

    #################   CALCULATE THE DIFFERENCE OF COORDINATES OF DIRECTION AND LEVEL FOR EACH RANK PER SESSION COMPARISON  #################
    for comp in comparisons:
        comp_split = comp.split("_")
        session_1 = int(comp_split[0])  # first session as integer
        session_2 = int(comp_split[1])

        for stn in sub_hem_keys:
            # check for each STN, which ones have both sessions
            stn_dataframe = fooof_monopolar_df_copy.loc[fooof_monopolar_df_copy.subject_hemisphere == stn]

            if session_1 not in stn_dataframe.session.values:
                continue

            elif session_2 not in stn_dataframe.session.values:
                continue

            stn_session_1 = stn_dataframe.loc[stn_dataframe.session == session_1]
            stn_session_2 = stn_dataframe.loc[stn_dataframe.session == session_2]

            # calculate coordinate difference for each included rank
            for rank in ranks_included:
                rank_session_1 = stn_session_1.loc[
                    stn_session_1.rank_beta == rank
                ]  # row of only one rank of one stn of session 1
                rank_session_2 = stn_session_2.loc[stn_session_2.rank_beta == rank]

                # contacts at both sessions with specific rank
                contact_session_1 = rank_session_1.contact.values[0]
                contact_session_2 = rank_session_2.contact.values[0]

                # extract x and y coordinates of a rank contact at two sessions
                x_coord_ses_1 = rank_session_1.x_direction.values[0]
                y_level_ses_1 = rank_session_1.y_level.values[0]
                x_coord_ses_2 = rank_session_2.x_direction.values[0]
                y_level_ses_2 = rank_session_2.y_level.values[0]

                # calculate xy difference between coordinates at both sessions
                x_difference = x_coord_ses_1 - x_coord_ses_2
                y_difference = y_level_ses_1 - y_level_ses_2

                coord_difference_data[f"{comp}_{stn}_{rank}"] = [
                    comp,
                    session_1,
                    session_2,
                    stn,
                    rank,
                    contact_session_1,
                    contact_session_2,
                    x_difference,
                    y_difference,
                ]

    # save as dataframe
    coord_difference_dataframe = pd.DataFrame(coord_difference_data)
    coord_difference_dataframe.rename(
        index={
            0: "session_comparison",
            1: "session_1",
            2: "session_2",
            3: "subject_hemisphere",
            4: "beta_rank",
            5: "contact_session_1",
            6: "contact_session_2",
            7: "x_difference",
            8: "y_difference",
        },
        inplace=True,
    )
    coord_difference_dataframe = coord_difference_dataframe.transpose()

    # considering x_difference: replace all values 2 by value -1, because there can only be a difference of direction of -1, 0 or 1
    coord_difference_dataframe["x_difference"] = coord_difference_dataframe["x_difference"].replace(
        to_replace=[2, -2], value=[-1, +1]
    )

    for comp in comparisons:
        comp_data = coord_difference_dataframe.loc[coord_difference_dataframe.session_comparison == comp]

        comp_data_rank_1 = comp_data.loc[comp_data.beta_rank == 1]

        size = comp_data_rank_1.count()
        size = size["subject_hemisphere"]

        percentage_stable_level = (comp_data_rank_1.y_difference.value_counts()[0]) / size
        percentage_stable_direction = (comp_data_rank_1.x_difference.value_counts()[0]) / size

        sample_size[f"{comp}_beta_rank_1"] = [comp, size, percentage_stable_level, percentage_stable_direction]

        # rank 2
        comp_data_rank_2 = comp_data.loc[comp_data.beta_rank == 2]

        size_2 = comp_data_rank_2.count()
        size_2 = size_2["subject_hemisphere"]

        percentage_rank_2_stable_level = (comp_data_rank_2.y_difference.value_counts()[0]) / size_2
        percentage_rank_2_stable_direction = (comp_data_rank_2.x_difference.value_counts()[0]) / size_2

        sample_size_rank_2[f"{comp}_beta_rank_2"] = [
            comp,
            size_2,
            percentage_rank_2_stable_level,
            percentage_rank_2_stable_direction,
        ]

    # save as dataframe
    sample_size_dataframe = pd.DataFrame(sample_size)
    sample_size_dataframe.rename(
        index={
            0: "session_comparison",
            1: "sample_size",
            2: "percentage_stable_level",
            3: "percentage_stable_direction",
        },
        inplace=True,
    )
    sample_size_dataframe = sample_size_dataframe.transpose()

    sample_size_dataframe_rank_2 = pd.DataFrame(sample_size_rank_2)
    sample_size_dataframe_rank_2.rename(
        index={
            0: "session_comparison",
            1: "sample_size",
            2: "percentage_stable_level",
            3: "percentage_stable_direction",
        },
        inplace=True,
    )
    sample_size_dataframe_rank_2 = sample_size_dataframe_rank_2.transpose()

    return {
        "fooof_monopolar_df_copy": fooof_monopolar_df_copy,
        "coord_difference_dataframe": coord_difference_dataframe,
        "sample_size_dataframe": sample_size_dataframe,
        "sample_size_dataframe_rank_2": sample_size_dataframe_rank_2,
    }


def fooof_beta_rank_coord_difference_scatterplot(cohort: str, ranks_included: list):
    """
    Input:
        -

    """

    # variables
    if cohort in ["all_included", "group_3"]:
        comparisons = [
            "0_3",
            "0_12",
            "0_18",
            "3_12",
            "3_18",
            "12_18",
        ]

    elif cohort == "group_1":
        comparisons = [
            "0_3",
            "0_12",
            "3_12",
        ]

    elif cohort == "group_2":
        comparisons = [
            "3_12",
            "3_18",
            "12_18",
        ]

    jitter = 0.16  # 0.16

    colors = ["sandybrown", "tab:grey", "turquoise", "plum", "cornflowerblue", "yellowgreen"]

    # load the dataframe with coordinate differences of beta rank contacts at different sessions
    df_xy_changes_of_beta_ranks = write_df_xy_changes_of_beta_ranks(cohort=cohort, ranks_included=ranks_included)

    df_xy_changes_of_beta_ranks = df_xy_changes_of_beta_ranks["coord_difference_dataframe"]

    # plot seperately for each session comparison

    for comp in comparisons:
        fig = plt.figure(figsize=[9, 7], layout="tight")  # 9,7

        comp_df = df_xy_changes_of_beta_ranks.loc[df_xy_changes_of_beta_ranks.session_comparison == comp]

        for r, rank in enumerate(ranks_included):
            rank_comp = comp_df.loc[comp_df.beta_rank == rank]  # plot each rank with different color

            x_differences = rank_comp.x_difference.values
            y_differences = rank_comp.y_difference.values

            x_jittered = np.array(x_differences) + np.random.uniform(-jitter, jitter, len(x_differences))
            y_jittered = np.array(y_differences) + np.random.uniform(-jitter, jitter, len(y_differences))

            plt.scatter(
                x_jittered, y_jittered, label=f"beta rank {rank}", c=colors[r], s=280, alpha=0.5, edgecolors="black"
            )  # 50

        plt.xlabel("change in direction", fontdict={"size": 25})
        plt.ylabel("change in level", fontdict={"size": 25})
        plt.xticks(fontsize=20), plt.yticks(fontsize=20)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)

        plt.legend(
            loc='upper right', edgecolor="black", fontsize=20, frameon=True, shadow=True, bbox_to_anchor=(1.6, 1)
        )  # 1.5, 1
        plt.grid(True)

        fig.suptitle(
            f"change of segmental contacts of beta ranks: \ncomparison between sessions {comp}", fontsize=25, y=1.02
        )  # 1.02
        fig.subplots_adjust(wspace=60, hspace=60)

        fig.savefig(
            os.path.join(
                FIGURES_PATH,
                f"revision_fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{cohort}.png",
            ),
            bbox_inches="tight",
        )

        fig.savefig(
            os.path.join(
                FIGURES_PATH,
                f"revision_fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{cohort}.svg",
            ),
            bbox_inches="tight",
            format="svg",
        )
