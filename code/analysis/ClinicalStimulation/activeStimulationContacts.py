""" Best clinical stimulation contacts longitudinal change in levels and directions """


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy
import pingouin as pg
from itertools import combinations
from statannotations.Annotator import Annotator


######### PRIVATE PACKAGES #########
import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults




def correlateActiveClinicalContacts_monopolarPSDRanks(
        incl_sub: list,
        freqBand: str, 
        rank_or_psd: str,
        singleContacts_or_average: str

):
    """
    Using the monopolar rank results from the monoRef_weightPsdAverageByCoordinateDistance.py

    Input:
        - incl_sub: list, e.g. ["017", "019", "024", "025", "026", "029", "030"]
        - freqBand: str, e.g. "beta", "lowBeta", "highBeta"
        - rank_or_psd: str, e.g. "rank", "rawPsd"
        - singleContacts_or_average: str, e.g. "singleContacts", "averageContacts" 
            -> choose if you want to use the average of active contacts vs. average of inactive contacts to have the same sample size in both groups



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
    BestClinicalContacts = bestClinicalStim_file["BestClinicalContacts"]




    ##################### FILTER THE MONOBETA8RANKS_DF: clinically ACTIVE contacts #####################
    activeMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
    
        activeContacts = str(BestClinicalContacts.CathodalContact.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        activeContacts_list = activeContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[(MonoBeta8Ranks_DF["subject_hemisphere"]==sub_hem) & (MonoBeta8Ranks_DF["session"]==session) & (MonoBeta8Ranks_DF["monopolarChannels"].isin(activeContacts_list))]
        

        # concatenate single rows to new Dataframe
        activeMonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)
    
    # add a column "clinicalUse" to the Dataframe and fill with "active"
    activeMonoBeta8Ranks["clinicalUse"] = "active"



    ##################### FILTER THE MONOBETA8RANKS_DF: clinically INACTIVE contacts #####################
    inactiveMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():
    
        inactiveContacts = str(BestClinicalContacts.InactiveContacts.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        inactiveContacts_list = inactiveContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = MonoBeta8Ranks_DF.loc[(MonoBeta8Ranks_DF["subject_hemisphere"]==sub_hem) & (MonoBeta8Ranks_DF["session"]==session) & (MonoBeta8Ranks_DF["monopolarChannels"].isin(inactiveContacts_list))]
        

        # concatenate single rows to new Dataframe
        inactiveMonoBeta8Ranks = pd.concat([inactiveMonoBeta8Ranks, sub_hem_ses_rows], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "non_active"
    inactiveMonoBeta8Ranks["clinicalUse"] = "inactive"

  

    ##################### CONCATENATE BOTH DATAFRAMES: CLINICALLY ACTIVE and INACTIVE CONTACTS #####################
    active_and_inactive_MonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, inactiveMonoBeta8Ranks], ignore_index=True)


    ##################### CHOOSE BETWEEN USING EACH SINGLE CONTACT OR AVERAGE OF ACTIVE OR INACTIVE CONTACTS #####################

    if singleContacts_or_average == "singleContacts":

        data_MonoBeta8Ranks = active_and_inactive_MonoBeta8Ranks

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg('_'.join, axis=1)

    
    # average for each STN: ranks and psd of active or inactive contacts
    elif singleContacts_or_average == "averageContacts":

        STN_average_activeVsInactiveContacts = {} # store averages of ranks and psd values of active vs. inactive contacts per STN

        STN_unique = list(active_and_inactive_MonoBeta8Ranks["subject_hemisphere"].unique()) # list of all existing STNs

        for STN in STN_unique:
            # get dataframe only of one STN
            STN_dataframe = active_and_inactive_MonoBeta8Ranks.loc[(active_and_inactive_MonoBeta8Ranks.subject_hemisphere == STN)]

            # get all existing sessions per STN
            sessions_unique = list(STN_dataframe["session"].unique())

            for ses in sessions_unique:
                STN_session_dataframe = STN_dataframe.loc[(STN_dataframe.session == ses)]

                # get average of active contacts
                STN_session_active = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "active")]
                MEANrank_active = STN_session_active["Rank8contacts"].values.mean()
                MEANpsd_active = STN_session_active["averaged_monopolar_PSD_beta"].values.mean()

                # get average of inactive contacts
                STN_session_inactive = STN_session_dataframe.loc[(STN_session_dataframe.clinicalUse == "inactive")]
                MEANrank_inactive = STN_session_inactive["Rank8contacts"].values.mean()
                MEANpsd_inactive = STN_session_inactive["averaged_monopolar_PSD_beta"].values.mean()

                # store MEAN values in dictionary
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_active"] = [STN, ses, MEANrank_active, MEANpsd_active, "active"]
                STN_average_activeVsInactiveContacts[f"{STN}_{ses}_inactive"] = [STN, ses, MEANrank_inactive, MEANpsd_inactive, "inactive"]



        # transform the dictionary to Dataframe
        STN_average_activeVsInactive_DF = pd.DataFrame(STN_average_activeVsInactiveContacts)
        STN_average_activeVsInactive_DF.rename(index={0: "subject_hemisphere", 1: "session", 2: "MEAN_beta_rank", 3: "MEAN_beta_psd", 4: "clinicalUse"}, inplace=True)
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.transpose()
        # important to transform datatype of columns MEAN_beta_rank and MEAN_beta_psd to float (otherwise Error when plotting with seaborn)
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_rank": float})
        STN_average_activeVsInactive_DF = STN_average_activeVsInactive_DF.astype({"MEAN_beta_psd": float})
        

    	# use the dataframe with MEAN values for plotting
        data_MonoBeta8Ranks = STN_average_activeVsInactive_DF

        # add another column with session and clinical aggregated for statistical tests (Annotatator)
        data_MonoBeta8Ranks["session_clinicalUse"] = data_MonoBeta8Ranks[["session", "clinicalUse"]].agg('_'.join, axis=1)



    ##################### PERFORM MANN-WHITNEY TEST  #####################

    ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    ses_clinicalUse_wilcoxon= [("fu3m_active", "fu3m_inactive"), ("fu12m_active", "fu12m_inactive"), ("fu18m_active", "fu18m_inactive")]
    pairs = list(combinations(ses_clinicalUse, 2))
    all_results_mwu = []
    describe_arrays = {}

    # pair = tuple e.g. fu3m_active, fu3m_inactive
    # for pair in pairs:
    for s_c_wcx in ses_clinicalUse_wilcoxon:

        firstInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[0])]
        secondInPair = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c_wcx[1])]

        if rank_or_psd == "rank":
            if singleContacts_or_average == "singleContacts":
                firstInPair = np.array(firstInPair.Rank8contacts.values)
                secondInPair = np.array(secondInPair.Rank8contacts.values)

            
            elif singleContacts_or_average == "averageContacts":
                firstInPair = np.array(firstInPair.MEAN_beta_rank.values)
                secondInPair = np.array(secondInPair.MEAN_beta_rank.values)

        elif rank_or_psd == "rawPsd":
            if singleContacts_or_average == "singleContacts":
                firstInPair = np.array(firstInPair.averaged_monopolar_PSD_beta.values)
                secondInPair = np.array(secondInPair.averaged_monopolar_PSD_beta.values)

            
            elif singleContacts_or_average == "averageContacts":
                firstInPair = np.array(firstInPair.MEAN_beta_psd.values)
                secondInPair = np.array(secondInPair.MEAN_beta_psd.values)

        
        # Perform Mann-Whitney Test
        results_mwu = pg.wilcoxon(firstInPair, secondInPair) # pair is always a tuple, comparing first and second component of this tuple
        results_mwu[f'comparison_{rank_or_psd}_{singleContacts_or_average}'] = '_'.join(s_c_wcx) # new column "comparison" with the pair being compared e.g. fu3m_active and fu3m_inactive

        all_results_mwu.append(results_mwu)

    significance_results = pd.concat(all_results_mwu)



    ##################### GET STATISTICAL IMPORTANT FEATURES #####################
    # describe all 6 groups
    for s_c in ses_clinicalUse:

        # get array of each group
        group = data_MonoBeta8Ranks.loc[(data_MonoBeta8Ranks.session_clinicalUse == s_c)]

        if rank_or_psd == "rank":
            if singleContacts_or_average == "singleContacts":
                group = np.array(group.Rank8contacts.values)

            
            elif singleContacts_or_average == "averageContacts":
                group = np.array(group.MEAN_beta_rank.values)

        elif rank_or_psd == "rawPsd":
            if singleContacts_or_average == "singleContacts":
                group = np.array(group.averaged_monopolar_PSD_beta.values)

            
            elif singleContacts_or_average == "averageContacts":
                group = np.array(group.MEAN_beta_psd.values)

        description = scipy.stats.describe(group)

        describe_arrays[f"{s_c}_{rank_or_psd}_{singleContacts_or_average}"] = description

    description_results = pd.DataFrame(describe_arrays)
    description_results.rename(index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"}, inplace=True)
    description_results = description_results.transpose()


    ##################### STORE RESULTS IN DICTIONARY AND SAVE #####################

    results_dictionary = {
        "significance_results": significance_results,
        "description_results": description_results
    }

    # save as pickle
    results_filepath = os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_statistics_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle")
    with open(results_filepath, "wb") as file:
        pickle.dump(results_dictionary, file)    
    




    ##################### PLOT VIOLINPLOT OF RANKS OR RAWPSD OF CLINICALLY ACTIVE VS NON-ACTIVE CONTACTS #####################

    if rank_or_psd == "rank":

        if singleContacts_or_average == "singleContacts":
            
            y_values = "Rank8contacts"
            y_label = "Beta rank of contact"
            title = "Beta rank of clinically active vs. inactive stimulation contacts"
            y_lim = 10, -1
        
        elif singleContacts_or_average == "averageContacts": 
            
            y_values = "MEAN_beta_rank"
            y_label = "Mean Beta rank of contact group"
            title = "Mean Beta rank of clinically active vs. inactive stimulation contacts"
            y_lim = 10, -1
       
    
    elif rank_or_psd == "rawPsd":

        if singleContacts_or_average == "singleContacts":

            y_values = "averaged_monopolar_PSD_beta"
            y_label = "Beta PSD of contact [uV^2/Hz]"
            title = "Beta PSD of clinically active vs. inactive stimulation contacts"
            y_lim = -0.3, 1.8

        elif singleContacts_or_average == "averageContacts":

            y_values = "MEAN_beta_psd"
            y_label = "Mean Beta PSD of contact group [uV^2/Hz]"
            title = "Mean Beta PSD of clinically active vs. inactive stimulation contacts"
            y_lim = -0.3, 1.8




    fig=plt.figure()
    ax = fig.add_subplot()

    # sns.violinplot(data=data_MonoBeta8Ranks, x="session_clinicalUse", y=y_values, hue="clinicalUse", palette="Set2", inner="box", ax=ax)
    sns.violinplot(data=data_MonoBeta8Ranks, x="session", y=y_values, hue="clinicalUse", palette="Set3", inner="box", ax=ax)
    
    # statistical test
    # ses_clinicalUse= ["fu3m_active", "fu3m_inactive", "fu12m_active", "fu12m_inactive", "fu18m_active", "fu18m_inactive"]
    # pairs = list(combinations(ses_clinicalUse, 2))

    # annotator = Annotator(ax, pairs, data=active_and_inactive_MonoBeta8Ranks, x='session_clinicalUse', y=y_values)
    # annotator.configure(test='Wilcoxon', text_format='star')
    # annotator.apply_and_annotate()
    
    sns.stripplot(
        data=data_MonoBeta8Ranks,
        x="session",
        y=y_values,
        hue="clinicalUse",
        ax=ax,
        size=6,
        color="black",
        alpha=0.4, # Transparency of dots
        dodge=True, # datapoints of groups active, inactive are plotted next to each other
    )
    
    sns.despine(left=True, bottom=True) # get rid of figure frame

   

    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))

    
    fig.savefig(figures_path + f"\\ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.png", bbox_inches="tight")

    
    # save Dataframe with data 
    ClinicalActiveVsNonactiveContacts_filepath = os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{singleContacts_or_average}.pickle")
    with open(ClinicalActiveVsNonactiveContacts_filepath, "wb") as file:
        pickle.dump(data_MonoBeta8Ranks, file)

    
    # active_and_inactive_MonoBeta8Ranks.to_json(os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.json"))

    
    print("new files: ", f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle",
          f"\nand: ClinicalActiveVsNonactiveContacts_statistics_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.pickle",
          "\nwritten in in: ", results_path,
          f"\nnew figure: ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}_{singleContacts_or_average}.png",
          "\nwritten in: ", figures_path)

    

    return {
        "data_MonoBeta8Ranks": data_MonoBeta8Ranks,
        "description_results": description_results, 
        "significance_results": significance_results

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
















    