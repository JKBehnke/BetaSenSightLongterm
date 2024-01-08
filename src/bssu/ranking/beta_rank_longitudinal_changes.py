""" Longitudinal changes of beta ranks """


import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm
import statistics
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


results_path = find_folders.get_local_path(folder="GroupResults")
figures_path = find_folders.get_local_path(folder="GroupFigures")


def write_df_xy_changes_of_beta_ranks(similarity_calculation: str, fooof_version: str, ranks_included: list):
    """
    Input:
        - similarity_calculation: "inverse_distance" or "neg_exp_distance"
        - fooof_version "v2"
        - ranks_included: [1], [1,2] or [1,2,3,4,5,6] etc depends on how many ranks you want to include into the analysis


    Load the monopolar FOOOF dataframe of estimated beta power at segmental contacts and beta rank 1-6



    """

    # defined variables
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

    coord_difference_data = {}
    sample_size = {}
    sample_size_rank_2 = {}

    loaded_fooof_monopolar_data = loadResults.load_fooof_monopolar_weighted_psd(
        fooof_spectrum="periodic_spectrum",
        fooof_version=fooof_version,
        segmental="yes",
        similarity_calculation=similarity_calculation,
    )

    fooof_monopolar_df = pd.concat(
        [
            loaded_fooof_monopolar_data["postop_monopolar_Dataframe"],
            loaded_fooof_monopolar_data["fu3m_monopolar_Dataframe"],
            loaded_fooof_monopolar_data["fu12m_monopolar_Dataframe"],
            loaded_fooof_monopolar_data["fu18or24m_monopolar_Dataframe"],
        ]
    )

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


def fooof_beta_rank_coord_difference_scatterplot(similarity_calculation: str, fooof_version: str, ranks_included: list):
    """
    Input:
        -

    """

    # variables
    comparisons = ["0_3", "0_12", "0_18", "3_12", "3_18", "12_18"]

    jitter = 0.16  # 0.16

    colors = ["sandybrown", "tab:grey", "turquoise", "plum", "cornflowerblue", "yellowgreen"]

    # load the dataframe with coordinate differences of beta rank contacts at different sessions
    df_xy_changes_of_beta_ranks = write_df_xy_changes_of_beta_ranks(
        similarity_calculation=similarity_calculation, fooof_version=fooof_version, ranks_included=ranks_included
    )

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
                figures_path,
                f"fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{similarity_calculation}_{fooof_version}.png",
            ),
            bbox_inches="tight",
        )

        fig.savefig(
            os.path.join(
                figures_path,
                f"fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{similarity_calculation}_{fooof_version}.svg",
            ),
            bbox_inches="tight",
            format="svg",
        )


def permutation_fooof_beta_rank_location_differences(
    ranks_included: list,
):
    """
    Perform a permutation test of beta rank location differences
    to show if beta rank location is significantly stable compared to random beta rank locations, for levels and directions seperately

        - and all session comparisons


    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - ranks_included: list e.g. [1], [1,2] -> these beta ranks will be included into the dataframe with data input for analysis


    1) Load the dataframes from write_df_xy_changes_of_beta_ranks()
        with level and direction differenes of FOOOF beta ranks given in Input
        - columns: session_comparison, session_1, session_2, subject_hemisphere, beta_rank, contact_session_1, contact_session_2, x_difference, y_difference

    2) for each comparison
        comparisons = ["0_0", "0_3", "0_12", "0_18",
                        "3_0", "3_3", "3_12", "3_18",
                        "12_0", "12_3", "12_12", "12_18",
                        "18_0", "18_3", "18_12", "18_18"]

        - per STN:  calculate the MEAN difference of ranks
        - get average of all STN MEAN differences of ranks

    3) loop over each STN and shuffle ranks from session x and session y
        - number of shuffle = 1000
        - calculate the absolute difference between ranks for each BIP recording
        - calculate the MEAN of abs differences for each shuffle and store in a list difference_random_MEANranks

    4) Statistics:
        calculate the distance of the REAL mean from the mean of all randomized means divided by the standard deviation
        - distanceMeanReal_MeanRandom = (mean_difference - np.mean(difference_random_MEANranks)) / np.std(difference_random_MEANranks)

        calculate the p-value
        - (pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung)
        - pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist


    5) Plot the distribution of the permutated MEAN values (should be normally distributed)
        - mark a red line for the REAL MEAN
        - annotation with the p value
        - one figure for each comparison: 3 subplots for 3 channel groups

    6) save a Dataframe with statistics results
        - "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"
        - columns: comparison, channelGroup, MEAN_differenceOfRanks, distanceMEANreal_MEANrandom, p-value


    """

    # load FOOOF beta rank DF
    beta_rank_DF = write_df_xy_changes_of_beta_ranks(
        similarity_calculation="inverse_distance", ranks_included=ranks_included
    )

    beta_rank_DF = beta_rank_DF["coord_difference_dataframe"]

    # new column with stn and channel info combined
    beta_rank_DF_copy = beta_rank_DF.copy()

    # add 2 new columns "x_coordinate", "y_coordinate"
    beta_rank_DF_copy = beta_rank_DF_copy.assign(x_direction_session_1=beta_rank_DF_copy[f"contact_session_1"]).rename(
        columns={"x_direction_session_1": "x_direction_session_1"}
    )
    beta_rank_DF_copy["x_direction_session_1"] = beta_rank_DF_copy["x_direction_session_1"].replace(
        to_replace=["1A", "2A"], value=[1, 1]
    )  # direction A
    beta_rank_DF_copy["x_direction_session_1"] = beta_rank_DF_copy["x_direction_session_1"].replace(
        to_replace=["1B", "2B"], value=[2, 2]
    )  # direction B
    beta_rank_DF_copy["x_direction_session_1"] = beta_rank_DF_copy["x_direction_session_1"].replace(
        to_replace=["1C", "2C"], value=[3, 3]
    )  # direction C

    beta_rank_DF_copy = beta_rank_DF_copy.assign(x_direction_session_2=beta_rank_DF_copy[f"contact_session_2"]).rename(
        columns={"x_direction_session_2": "x_direction_session_2"}
    )
    beta_rank_DF_copy["x_direction_session_2"] = beta_rank_DF_copy["x_direction_session_2"].replace(
        to_replace=["1A", "2A"], value=[1, 1]
    )  # direction A
    beta_rank_DF_copy["x_direction_session_2"] = beta_rank_DF_copy["x_direction_session_2"].replace(
        to_replace=["1B", "2B"], value=[2, 2]
    )  # direction B
    beta_rank_DF_copy["x_direction_session_2"] = beta_rank_DF_copy["x_direction_session_2"].replace(
        to_replace=["1C", "2C"], value=[3, 3]
    )  # direction C

    beta_rank_DF_copy = beta_rank_DF_copy.assign(y_level_session_1=beta_rank_DF_copy[f"contact_session_1"]).rename(
        columns={"y_level_session_1": "y_level_session_1"}
    )
    beta_rank_DF_copy["y_level_session_1"] = beta_rank_DF_copy["y_level_session_1"].replace(
        to_replace=["1A", "1B", "1C"], value=[1, 1, 1]
    )  # level 1
    beta_rank_DF_copy["y_level_session_1"] = beta_rank_DF_copy["y_level_session_1"].replace(
        to_replace=["2A", "2B", "2C"], value=[2, 2, 2]
    )  # level 2

    beta_rank_DF_copy = beta_rank_DF_copy.assign(y_level_session_2=beta_rank_DF_copy[f"contact_session_2"]).rename(
        columns={"y_level_session_2": "y_level_session_2"}
    )
    beta_rank_DF_copy["y_level_session_2"] = beta_rank_DF_copy["y_level_session_2"].replace(
        to_replace=["1A", "1B", "1C"], value=[1, 1, 1]
    )  # level 1
    beta_rank_DF_copy["y_level_session_2"] = beta_rank_DF_copy["y_level_session_2"].replace(
        to_replace=["2A", "2B", "2C"], value=[2, 2, 2]
    )  # level 2

    # defined variables
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

    # x_difference = difference in horizontal direction
    # y_difference = difference in vertical level
    difference_level_or_direction = ["x_difference", "y_difference"]

    ##########################      GET MEAN OF RANK DIFFERENCES PER SESSION COMPARISON AND SHUFFLE RANKS   ##########################
    # 1) get the real mean first of all channels within one session comparison across STNs
    # 2) permute within session comparison, get the permuted mean across STNs

    permutation_fooof_beta_ranks_coord = {}

    # shuffle repetitions: 1000 times
    number_of_shuffle = np.arange(1, 1001, 1)
    fontdict = {"size": 25}

    for rank in ranks_included:
        # dataframe only of one channel group
        rank_df = beta_rank_DF_copy.loc[beta_rank_DF_copy["beta_rank"] == rank]

        for comp in comparisons:
            # Figure Layout per comparison: 2 rows (direction, level), 1 column
            fig, axes = plt.subplots(2, 1, figsize=(10, 15))

            # Dataframe per session comparison
            comp_df = rank_df.loc[(rank_df["session_comparison"] == comp)]

            # list of STNs per session
            comp_stns = list(comp_df.subject_hemisphere.unique())

            for d, diff in enumerate(difference_level_or_direction):
                if diff == "x_difference":
                    coord = "x_direction"

                elif diff == "y_difference":
                    coord = "y_level"

                # Array of real differences: 1 horizontal direction, 2 vertical level
                location_diff_mean = np.mean(
                    comp_df[f"{diff}"].abs()
                )  # absolute values! because otherwise you take the mean of -1, 0 and 1 and it will be close to 0
                location_diff_std = np.std(comp_df[f"{diff}"].abs())

                sample_size = len(comp_df[f"{diff}"].values)

                ############ SHUFFLE ############
                # list of 1000x mean differences between shuffled rank_x and rank_y
                shuffled_mean_differences = []

                # repeat shuffle 1000 times
                for s, shuffle in enumerate(number_of_shuffle):
                    # randomly shuffle the rank contacts at session 1 and session 2 (1 out of 6 potential segmental contacts)
                    # direction: A=1, B=2, C=3
                    # level: 1=1, 2=2
                    random_coordinate_session_1 = list(
                        comp_df[f"{coord}_session_1"].values
                    )  # list of contact coordinates in direction or level
                    random_coordinate_session_2 = list(comp_df[f"{coord}_session_2"].values)

                    # shuffle
                    np.random.shuffle(random_coordinate_session_1)
                    np.random.shuffle(random_coordinate_session_2)

                    # calculate the difference between random random_coordinate_session_1 and random_coordinate_session_2, store in list
                    difference_random_coord = list(
                        abs(np.array(random_coordinate_session_1) - np.array(random_coordinate_session_2))
                    )

                    # only for x_direction: replace all values 2 by -1 and -2 by +1, because there can only be a difference of direction of -1, 0 or 1
                    if diff == "x_difference":
                        difference_random_coord = [
                            -1 if x == 2 else (1 if x == -2 else x) for x in difference_random_coord
                        ]
                        difference_random_coord = [abs(x) for x in difference_random_coord]  # take absolute differences

                    elif diff == "y_difference":
                        difference_random_coord = [abs(x) for x in difference_random_coord]  # take absolute differences

                    # get mean of differences
                    shuffled_mean_differences.append(np.mean(difference_random_coord))

                ############ CALCULATE DISTANCE AND P-VAL OF REAL MEAN FROM MEAN OF ALL RANDOMIZED MEANS  ############
                distance_real_vs_random_mean = (location_diff_mean - np.mean(shuffled_mean_differences)) / np.std(
                    shuffled_mean_differences
                )

                # calculate the p-value
                # pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung
                pval = 1 - norm.cdf(
                    abs(distance_real_vs_random_mean)
                )  # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist

                sample_size_shuffled = len(shuffled_mean_differences)

                # store all values in dictionary
                permutation_fooof_beta_ranks_coord[f"{rank}_{comp}_{diff}"] = [
                    rank,
                    comp,
                    diff,
                    comp_stns,
                    location_diff_mean,
                    location_diff_std,
                    sample_size,
                    shuffled_mean_differences,
                    distance_real_vs_random_mean,
                    "{:.15f}".format(pval),
                    sample_size_shuffled,
                ]

                ############ PLOT ############
                sns.histplot(
                    shuffled_mean_differences,
                    color="tab:blue",
                    ax=axes[d],
                    stat="count",
                    element="bars",
                    label="1000 Permutation repetitions",
                    kde=True,
                    bins=30,
                    fill=True,
                )

                # mark with red line: real mean of the rank differences of comp_group_DF
                axes[d].axvline(location_diff_mean, c="r", linewidth=3)
                axes[d].text(
                    location_diff_mean + 0.02,
                    50,
                    "real mean \nof location difference \n\n p-value: {:.3f}".format(pval),
                    c="k",
                    fontsize=20,
                )

                axes[d].set_title(f"{diff}", fontdict=fontdict)

            for ax in axes:
                ax.set_xlabel(f"Mean location difference between session 1 and session 2", fontsize=25)
                ax.set_ylabel("Count", fontsize=25)

                ax.tick_params(axis="x", labelsize=25)
                ax.tick_params(axis="y", labelsize=25)
                ax.grid(False)

            fig.suptitle(
                f"Permutation analysis of contact location: beta rank {rank}, {comp} session comparison", fontsize=30
            )
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.tight_layout()

            fig.savefig(
                os.path.join(figures_path, f"permutation_location_beta_rank_{rank}_fooof_spectra_{comp}.png"),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(figures_path, f"permutation_location_beta_rank_{rank}_fooof_spectra_{comp}.svg"),
                bbox_inches="tight",
                format="svg",
            )

    # Permutation_BIP transform from dictionary to Dataframe
    permutation_result_df = pd.DataFrame(permutation_fooof_beta_ranks_coord)
    permutation_result_df.rename(
        index={
            0: "beta_rank",
            1: "session_comparison",
            2: "direction_or_level",
            3: "stn_list_included",
            4: "real_difference_mean",
            5: "real_difference_std",
            6: "sample_size_stns",
            7: "shuffled_difference_mean",
            8: "distance_real_vs_random_mean_differences",
            9: "p-value",
            10: "sample_size_random_shuffles",
        },
        inplace=True,
    )
    permutation_result_df = permutation_result_df.transpose()

    ## save the Permutation Dataframe with pickle
    Permutation_filepath = os.path.join(
        results_path, f"permutation_location_difference_between_sessions_beta_ranks.pickle"
    )
    with open(Permutation_filepath, "wb") as file:
        pickle.dump(permutation_result_df, file)

    print(
        "file: ", f"permutation_location_difference_between_sessions_beta_ranks.pickle", "\nwritten in: ", results_path
    )

    return {"permutation_result_df": permutation_result_df, "beta_rank_DF_copy": beta_rank_DF_copy}
