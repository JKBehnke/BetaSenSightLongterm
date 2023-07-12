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
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults


def write_df_xy_changes_of_beta_ranks(
        similarity_calculation:str,
        ranks_included:list

        
):
    """
    Input: 
        - similarity_calculation: "inverse_distance" or "neg_exp_distance"
        - ranks_included: [1], [1,2] or [1,2,3,4,5,6] etc depends on how many ranks you want to include into the analysis

    
    Load the monopolar FOOOF dataframe of estimated beta power at segmental contacts and beta rank 1-6



    """

    # defined variables
    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]
    
    coord_difference_data = {}
    sample_size = {}

    results_path = find_folders.get_local_path(folder="GroupResults")

    loaded_fooof_monopolar_data = loadResults.load_fooof_monopolar_weighted_psd(
        fooof_spectrum="periodic_spectrum",
        segmental="yes",
        similarity_calculation=similarity_calculation
    )

    fooof_monopolar_df = pd.concat([loaded_fooof_monopolar_data["postop_monopolar_Dataframe"],
                                    loaded_fooof_monopolar_data["fu3m_monopolar_Dataframe"],
                                    loaded_fooof_monopolar_data["fu12m_monopolar_Dataframe"],
                                    loaded_fooof_monopolar_data["fu18m_monopolar_Dataframe"]])
    
    fooof_monopolar_df_copy = fooof_monopolar_df.copy()
    fooof_monopolar_df_copy["rank_beta"] = fooof_monopolar_df["rank"].astype(int)

    # replace session names by integers
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.replace(to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18])

    # add 2 new columns "x_coordinate", "y_coordinate"
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(x_direction=fooof_monopolar_df_copy["contact"]).rename(columns={"x_direction":"x_direction"})
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(to_replace=["1A", "2A"], value=[1, 1]) # direction A
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(to_replace=["1B", "2B"], value=[2, 2]) # direction B
    fooof_monopolar_df_copy["x_direction"] = fooof_monopolar_df_copy["x_direction"].replace(to_replace=["1C", "2C"], value=[3, 3]) # direction C

    fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(y_level=fooof_monopolar_df_copy["contact"]).rename(columns={"y_level":"y_level"})
    fooof_monopolar_df_copy["y_level"] = fooof_monopolar_df_copy["y_level"].replace(to_replace=["1A", "1B", "1C"], value=[1, 1, 1]) # level 1
    fooof_monopolar_df_copy["y_level"] = fooof_monopolar_df_copy["y_level"].replace(to_replace=["2A", "2B", "2C"], value=[2, 2, 2]) # level 2


    # select only the included ranks 
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.loc[fooof_monopolar_df_copy.rank_beta.isin(ranks_included)]

    # check which STNs ad sessions exist in data
    sub_hem_keys = list(fooof_monopolar_df_copy.subject_hemisphere.unique())

    #################   CALCULATE THE DIFFERENCE OF COORDINATES OF DIRECTION AND LEVEL FOR EACH RANK PER SESSION COMPARISON  #################
    for comp in comparisons:

        comp_split = comp.split("_")
        session_1 = int(comp_split[0]) # first session as integer
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

                rank_session_1 = stn_session_1.loc[stn_session_1.rank_beta == rank] # row of only one rank of one stn of session 1
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

                coord_difference_data[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, 
                                                                 rank, contact_session_1, contact_session_2,
                                                                 x_difference, y_difference]
                
    
    # save as dataframe
    coord_difference_dataframe = pd.DataFrame(coord_difference_data)
    coord_difference_dataframe.rename(
        index={0: "session_comparison",
               1: "session_1",
               2: "session_2",
               3: "subject_hemisphere",
               4: "beta_rank",
               5: "contact_session_1",
               6: "contact_session_2",
               7: "x_difference",
               8: "y_difference"},
               inplace=True)
    coord_difference_dataframe = coord_difference_dataframe.transpose()

    # considering x_difference: replace all values 2 by value -1, because there can only be a difference of direction of -1, 0 or 1
    coord_difference_dataframe["x_difference"] = coord_difference_dataframe["x_difference"].replace(to_replace=[2, -2], value=[-1, +1])

    for comp in comparisons:

        comp_data = coord_difference_dataframe.loc[coord_difference_dataframe.session_comparison == comp]

        comp_data_rank_1 = comp_data.loc[comp_data.beta_rank == 1]

        size = comp_data_rank_1.count()

        sample_size[f"{comp}_beta_rank_1"] = [comp, size["subject_hemisphere"]]

    # save as dataframe
    sample_size_dataframe = pd.DataFrame(sample_size)
    sample_size_dataframe.rename(
        index={0: "session_comparison",
               1: "sample_size"},
               inplace=True)
    sample_size_dataframe = sample_size_dataframe.transpose()

    return {
        "fooof_monopolar_df_copy": fooof_monopolar_df_copy,
        "coord_difference_dataframe": coord_difference_dataframe,
        "sample_size_dataframe":sample_size_dataframe
    }


def fooof_beta_rank_coord_difference_scatterplot(
        similarity_calculation:str,
        ranks_included:list
):
    """
    Input: 
        - 
    
    """

    # variables
    comparisons = ["0_3", "0_12", "0_18", 
                   "3_12", "3_18", 
                   "12_18"]
    
    jitter = 0.16

    colors = ["turquoise", "tab:grey", "sandybrown", "plum", "cornflowerblue", "yellowgreen"]

    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # load the dataframe with coordinate differences of beta rank contacts at different sessions
    df_xy_changes_of_beta_ranks = write_df_xy_changes_of_beta_ranks(
        similarity_calculation=similarity_calculation,
        ranks_included=ranks_included
    )

    df_xy_changes_of_beta_ranks = df_xy_changes_of_beta_ranks["coord_difference_dataframe"]

    # plot seperately for each session comparison

    for comp in comparisons:

        fig = plt.figure(figsize=[9,7], layout="tight") # 9,7

        comp_df = df_xy_changes_of_beta_ranks.loc[df_xy_changes_of_beta_ranks.session_comparison == comp]

        for r, rank in enumerate(ranks_included):

            rank_comp = comp_df.loc[comp_df.beta_rank == rank] # plot each rank with different color

            x_differences = rank_comp.x_difference.values
            y_differences = rank_comp.y_difference.values

            x_jittered = np.array(x_differences) + np.random.uniform(-jitter, jitter, len(x_differences))
            y_jittered = np.array(y_differences) + np.random.uniform(-jitter, jitter, len(y_differences))

            plt.scatter(x_jittered, y_jittered, label=f"beta rank {rank}", c=colors[r], s=50)
            
        
        plt.xlabel("change in direction", fontdict={"size": 25})
        plt.ylabel("change in level", fontdict={"size": 25})
        plt.xticks(fontsize= 20), plt.yticks(fontsize= 20)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)


        plt.legend(loc= 'upper right', edgecolor="black", fontsize=20, frameon=True, shadow=True, bbox_to_anchor=(1.6, 1)) # 1.5, 1
        plt.grid(True)

        fig.suptitle(f"change of segmental contacts of beta ranks: \ncomparison between sessions {comp}", fontsize=25, y=1.02) # 1.02
        fig.subplots_adjust(wspace=60, hspace=60)

        fig.savefig(os.path.join(figures_path, f"fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{similarity_calculation}.png"),
                    bbox_inches="tight")
        
        fig.savefig(os.path.join(figures_path, f"fooof_beta_ranks_{ranks_included}_change_sessions_{comp}_{similarity_calculation}.svg"),
                    bbox_inches="tight", format="svg")

    


















