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




def correlateActiveClinicalContacts_monopolarPSDRanks():
    """
    Using the monopolar rank results from the monoRef_weightPsdAverageByCoordinateDistance.py




    """

    





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

    
    fig.savefig(figures_path + "\\ActiveClinicalStimContact_Levels_fu3m_fu12m_and_fu12m_longterm.png")


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
















    