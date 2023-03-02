""" Best clinical stimulation contacts longitudinal change in levels and directions """


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns


######### PRIVATE PACKAGES #########
import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults




def correlateActiveClinicalContacts_monopolarPSDRanks(
        incl_sub: list,
        freqBand: str, 
        rank_or_psd: str

):
    """
    Using the monopolar rank results from the monoRef_weightPsdAverageByCoordinateDistance.py

    Input:
        - incl_sub: list, e.g. ["017", "019", "024", "025", "026", "029", "030"]
        - freqBand: str, e.g. "beta", "lowBeta", "highBeta"
        - rank_or_psd: str, e.g. "rank", "rawPsd"



    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    hemispheres = ["Right", "Left"]
    sessions = ["fu3m", "fu12m", "fu18m"]

    ##################### LOAD RANKS OR PSD from monoRef_weightPsdAverageByCoordinateDistance.py #####################

    data_weightedByCoordinates = {}
    keys_weightedByCoordinates ={}
    session_weightedByCoordinate_Dataframe = {}



    for sub in incl_sub:

        for hem in hemispheres:

            data_weightedByCoordinates[f"{sub}_{hem}"] = loadResults.load_monoRef_weightedPsdCoordinateDistance_pickle(
                sub=sub,
                hemisphere=hem,
                freqBand=freqBand,
                normalization="rawPsd",
                filterSignal="band-pass"
                )
            
            # first check, which sessions exist
            keys_weightedByCoordinates[f"{sub}_{hem}"] = data_weightedByCoordinates[f"{sub}_{hem}"].keys()


            for ses in sessions:

                # first check, if session exists in keys
                if f"{ses}_monopolar_Dataframe" in keys_weightedByCoordinates[f"{sub}_{hem}"]:
                    print(f"{sub}_{hem}_{ses}")
                
                else:
                    continue
                
                # get the dataframe per session
                session_weightedByCoordinates = data_weightedByCoordinates[f"{sub}_{hem}"][f"{ses}_monopolar_Dataframe"]

                # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
                session_weightedByCoordinates = session_weightedByCoordinates.loc[["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]]
                session_weightedByCoordinates["Rank8contacts"] = session_weightedByCoordinates["averaged_monopolar_PSD_beta"].rank(ascending=False)
                session_weightedByCoordinates_copy = session_weightedByCoordinates.copy()

                # add column subject_hemisphere_monoChannel
                session_weightedByCoordinates_copy["monopolarChannels"] = np.array(["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"])
                # session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)
                session_weightedByCoordinates_copy.drop(["rank", "coord_z", "coord_xy"], axis=1, inplace=True)

                # save the Dataframe per sub_hem_ses combination
                session_weightedByCoordinate_Dataframe[f"{sub}_{hem}_{ses}"] = session_weightedByCoordinates_copy



    ##### Concatenate all Dataframes together to one
    sub_hem_ses_keys = list(session_weightedByCoordinate_Dataframe.keys())

    MonoBeta8Ranks_DF = pd.DataFrame()

    # loop through all Dataframes and concatenate together
    for sub_hem_ses in sub_hem_ses_keys:
        single_Dataframe = session_weightedByCoordinate_Dataframe[sub_hem_ses]

        # concatenate all DF together
        MonoBeta8Ranks_DF = pd.concat([MonoBeta8Ranks_DF, single_Dataframe], ignore_index=True)
    



    ##################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()

    # get sheet with best clinical contacts
    activeClinicalContacts = bestClinicalStim_file["BestClinicalContacts"]




    ##################### FILTER THE MONOBETA8RANKS_DF: clinically ACTIVE contacts #####################
    activeMonoBeta8Ranks = pd.DataFrame()

    for idx, row in activeClinicalContacts.iterrows():
    
        activeContacts = str(activeClinicalContacts.CathodalContact.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        activeContacts_list = activeContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = activeClinicalContacts.subject_hemisphere.values[idx]
        session = activeClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[(MonoBeta8Ranks_DF["subject_hemisphere"]==sub_hem) & (MonoBeta8Ranks_DF["session"]==session) & (MonoBeta8Ranks_DF["monopolarChannels"].isin(activeContacts_list))]
        

        # concatenate single rows to new Dataframe
        activeMonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)
    
    # add a column "clinicalUse" to the Dataframe and fill with "active"
    activeMonoBeta8Ranks["clinicalUse"] = "active"


    ##################### GET STATISTICALLY IMPORTANT FEATURES FPR CLINICALLY ACTIVE CONTACTS #####################
    statistics_dict = {}

    # clinically active contacts
    fu3m_activeContacts = activeMonoBeta8Ranks.loc[(activeMonoBeta8Ranks.session=="fu3m")]
    fu12m_activeContacts = activeMonoBeta8Ranks.loc[(activeMonoBeta8Ranks.session=="fu12m")]
    fu18m_activeContacts = activeMonoBeta8Ranks.loc[(activeMonoBeta8Ranks.session=="fu18m")]


    if rank_or_psd == "rank":
        # get mean of ranks for each session
        statistics_dict["activeContacts_mean_fu3m"] = fu3m_activeContacts.Rank8contacts.values.mean()
        statistics_dict["activeContacts_mean_fu12m"] = fu12m_activeContacts.Rank8contacts.values.mean()
        statistics_dict["activeContacts_mean_fu18m"] = fu18m_activeContacts.Rank8contacts.values.mean()
    
    elif rank_or_psd == "rawPsd":
        # get mean of ranks for each session
        statistics_dict["activeContacts_mean_fu3m"] = fu3m_activeContacts.averaged_monopolar_PSD_beta.values.mean()
        statistics_dict["activeContacts_mean_fu12m"] = fu12m_activeContacts.averaged_monopolar_PSD_beta.values.mean()
        statistics_dict["activeContacts_mean_fu18m"] = fu18m_activeContacts.averaged_monopolar_PSD_beta.values.mean()

    # get sample size of each session
    statistics_dict["activeContacts_n_fu3m"] = len(fu3m_activeContacts.subject_hemisphere.values)
    statistics_dict["activeContacts_n_fu12m"] = len(fu12m_activeContacts.subject_hemisphere.values)
    statistics_dict["activeContacts_n_fu18m"] = len(fu18m_activeContacts.subject_hemisphere.values)









    ##################### FILTER THE MONOBETA8RANKS_DF: clinically NON-ACTIVE contacts #####################
    nonactiveMonoBeta8Ranks = pd.DataFrame()

    for idx, row in activeClinicalContacts.iterrows():
    
        non_activeContacts = str(activeClinicalContacts.NonActiveContacts.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        non_activeContacts_list = non_activeContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = activeClinicalContacts.subject_hemisphere.values[idx]
        session = activeClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[(MonoBeta8Ranks_DF["subject_hemisphere"]==sub_hem) & (MonoBeta8Ranks_DF["session"]==session) & (MonoBeta8Ranks_DF["monopolarChannels"].isin(non_activeContacts_list))]
        

        # concatenate single rows to new Dataframe
        nonactiveMonoBeta8Ranks = pd.concat([nonactiveMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "non_active"
    nonactiveMonoBeta8Ranks["clinicalUse"] = "non_active"

    ##################### GET STATISTICALLY IMPORTANT FEATURES FPR CLINICALLY ACTIVE CONTACTS #####################
    
    # clinically active contacts
    fu3m_nonactiveContacts = nonactiveMonoBeta8Ranks.loc[(nonactiveMonoBeta8Ranks.session=="fu3m")]
    fu12m_nonactiveContacts = nonactiveMonoBeta8Ranks.loc[(nonactiveMonoBeta8Ranks.session=="fu12m")]
    fu18m_nonactiveContacts = nonactiveMonoBeta8Ranks.loc[(nonactiveMonoBeta8Ranks.session=="fu18m")]


    if rank_or_psd == "rank":
        # get mean of ranks for each session
        statistics_dict["non_activeContacts_mean_fu3m"] = fu3m_nonactiveContacts.Rank8contacts.values.mean()
        statistics_dict["non_activeContacts_mean_fu12m"] = fu12m_nonactiveContacts.Rank8contacts.values.mean()
        statistics_dict["non_activeContacts_mean_fu18m"] = fu18m_nonactiveContacts.Rank8contacts.values.mean()
    
    elif rank_or_psd == "rawPsd":
        # get mean of ranks for each session
        statistics_dict["non_activeContacts_mean_fu3m"] = fu3m_nonactiveContacts.averaged_monopolar_PSD_beta.values.mean()
        statistics_dict["non_activeContacts_mean_fu12m"] = fu12m_nonactiveContacts.averaged_monopolar_PSD_beta.values.mean()
        statistics_dict["non_activeContacts_mean_fu18m"] = fu18m_nonactiveContacts.averaged_monopolar_PSD_beta.values.mean()

    # get sample size of each session
    statistics_dict["non_activeContacts_n_fu3m"] = len(fu3m_nonactiveContacts.subject_hemisphere.values)
    statistics_dict["non_activeContacts_n_fu12m"] = len(fu12m_nonactiveContacts.subject_hemisphere.values)
    statistics_dict["non_activeContacts_n_fu18m"] = len(fu18m_nonactiveContacts.subject_hemisphere.values)




    ##################### CONCATENATE BOTH DATAFRAMES: CLINICALLY ACTIVE and NON-ACTIVE CONTACTS #####################
    active_and_nonactive_MonoBeta8Ranks = pd.concat([nonactiveMonoBeta8Ranks, activeMonoBeta8Ranks], ignore_index=True)




    ##################### PLOT VIOLINPLOT OF RANKS OR RAWPSD OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    if rank_or_psd == "rank":
        y_values = "Rank8contacts"
        y_label = "Beta rank of contact"
        title = "Beta rank of clinically active vs. nonactive stimulation contacts"
        y_lim = 10, -1
       
    
    elif rank_or_psd == "rawPsd":
        y_values = "averaged_monopolar_PSD_beta"
        y_label = "Beta PSD of contact [uV^2/Hz]"
        title = "Beta PSD of clinically active vs. nonactive stimulation contacts"
        y_lim = -0.3, 1.8




    fig =plt.figure()

    ax = sns.violinplot(data=active_and_nonactive_MonoBeta8Ranks, x="session", y=y_values, hue="clinicalUse", palette="Set2")
    ax = sns.despine(left=True, bottom=True) # get rid of figure frame



    plt.title(title)
    plt.ylabel(y_label)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))

    plt.ylim(y_lim)

    plt.show()
    fig.savefig(figures_path + f"\\ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.png", bbox_inches="tight")


    # save dictionaries
    ClinicalActiveVsNonactiveContacts_filepath = os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.pickle")
    with open(ClinicalActiveVsNonactiveContacts_filepath, "wb") as file:
        pickle.dump(active_and_nonactive_MonoBeta8Ranks, file)

    
    print("new file: ", f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.pickle",
          "\nwritten in in: ", results_path,
          f"\nnew figure: ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.png",
          "\nwritten in: ", figures_path)

    

    return {
        "active_and_nonactive_MonoBeta8Ranks": active_and_nonactive_MonoBeta8Ranks,
        "statistics_dict": statistics_dict
    }

    




    





def bestClinicalStimContacts_LevelsComparison():
    
    """

    Load the Excel file with clinical stimulation parameters from the data folder.


    Input:
        - 
    
    """
    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    ##################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()
    compareLevels_sheet = bestClinicalStim_file["compareLevels"]


    ##################### COMPARE CONTACT LEVELS OF FU3M AND FU12M #####################

    # select only fu3m rows and fu12m rows and merge
    fu3m_level = compareLevels_sheet[compareLevels_sheet.session == "fu3m"]
    fu12m_level = compareLevels_sheet[compareLevels_sheet.session == "fu12m"]

    # merge fu3m and fu12m 
    comparefu3m_fu12m = fu3m_level.merge(fu12m_level, left_on="subject_hemisphere", right_on="subject_hemisphere")

    # calculate the difference between levels, add to a new column
    comparefu3m_fu12m["difference_levels"] = (comparefu3m_fu12m["ContactLevel_x"] - comparefu3m_fu12m["ContactLevel_y"]).apply(abs)
    comparefu3m_fu12m.dropna(axis=0)


    ##################### COMPARE CONTACT LEVELS OF FU12M AND LONGTERM #####################

    # select only LONGTERM rows, including fu18m, fu20m, fu22m
    longterm_level = compareLevels_sheet[(compareLevels_sheet.session == "fu18m") | (compareLevels_sheet.session == "fu20m") | (compareLevels_sheet.session == "fu22m")] # use | instead of or here 

    # merge fu12m and longterm 
    comparefu12m_longterm = fu12m_level.merge(longterm_level, left_on="subject_hemisphere", right_on="subject_hemisphere")

    # calculate the difference between levels, add to a new column
    comparefu12m_longterm["difference_levels"] = (comparefu12m_longterm["ContactLevel_x"] - comparefu12m_longterm["ContactLevel_y"]).apply(abs)
    comparefu12m_longterm.dropna(axis=0)


    
    ##################### PLOT BOTH COMPARISONS: FU3M - FU12M AND FU12M - LONGTERM #####################

    colors0 = sns.color_palette("viridis", n_colors=len(comparefu3m_fu12m.index))
    colors1 = sns.color_palette("viridis", n_colors=len(comparefu12m_longterm.index))

    fontdict = {"size": 25}


    fig, axes = plt.subplots(2, 1, figsize=(10,15))


    ##################### PLOT CONTACT LEVELS OF FU3M AND FU12M #####################
    sns.histplot(data=comparefu3m_fu12m, x="difference_levels", stat="count", hue="subject_hemisphere", 
             multiple="stack", bins=np.arange(-0.25,4,0.5),
             palette=colors0, ax=axes[0])

    axes[0].set_title("3MFU vs. 12MFU", fontdict=fontdict)
    
    legend3_12 = axes[0].get_legend()
    handles = legend3_12.legendHandles

    legend3_12_list = list(comparefu3m_fu12m.subject_hemisphere.values)
    axes[0].legend(handles, legend3_12_list, title='subject hemisphere',  title_fontsize=15, fontsize=15)


    ##################### PLOT CONTACT LEVELS OF FU12M AND LONTERM #####################
    sns.histplot(data=comparefu12m_longterm, x="difference_levels", stat="count", hue="subject_hemisphere", 
             multiple="stack", bins=np.arange(-0.25,4,0.5),
             palette=colors1, ax= axes[1])

    axes[1].set_title("12MFU vs. longterm-FU (18, 20, 22MFU)", fontdict=fontdict)

    legend12_longterm = axes[1].get_legend()
    handles = legend12_longterm.legendHandles

    legend12_longterm_list = list(comparefu12m_longterm.subject_hemisphere.values)
    axes[1].legend(handles, legend12_longterm_list, title='subject hemisphere', title_fontsize=15, fontsize=15)



    ##################### ADJUST THE PLOTS #####################

    for ax in axes:

        ax.set_xlabel("difference between contact levels",  fontsize=25)
        ax.set_ylabel("Count", fontsize=25)
    
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)



    fig.suptitle("Difference of active contact levels", fontsize=30)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    
    fig.savefig(figures_path + "\\ActiveClinicalStimContact_Levels_fu3m_fu12m_and_fu12m_longterm.png", bbox_inches="tight")


    ### save the Dataframes with pickle 
    comparefu3m_fu12m_filepath = os.path.join(results_path, f"ActiveClinicalStimContact_Levels_fu3m_fu12m.pickle")
    with open(comparefu3m_fu12m_filepath, "wb") as file:
        pickle.dump(comparefu3m_fu12m, file)

    comparefu12m_longterm_filepath = os.path.join(results_path, f"ActiveClinicalStimContact_Levels_fu12m_longterm.pickle")
    with open(comparefu12m_longterm_filepath, "wb") as file:
        pickle.dump(comparefu12m_longterm, file)



    
    print("new file: ", "ActiveClinicalStimContact_Levels_fu3m_fu12m.pickle",
          "\nwritten in in: ", results_path,
          "\nnew figure: ActiveClinicalStimContact_Levels_fu3m_fu12m_and_fu12m_longterm.png",
          "\nwritten in: ", figures_path)
















    