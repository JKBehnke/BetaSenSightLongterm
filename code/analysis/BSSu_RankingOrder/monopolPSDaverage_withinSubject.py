""" Comparison PSD average in beta frequency bands over time"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import seaborn as sns

import pingouin as pg
from itertools import combinations
from statannotations.Annotator import Annotator

import json
import os
import mne

# PyPerceive Imports
import PerceiveImport.methods.find_folders as findfolders

# import analysis.loadResults as loadcsv


def compareMonopolarPSDaverage_freqBand(incl_sub:list, freq_band: str):


    """
    Compare the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C) within a subject at different timepoints

    Input:
        - incl_sub: str, e.g. "024"
        - freq_band: str, e.g. "beta","lowBeta", "highBeta"
        - normalization: str, e.g. "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"


            
    Return: 
        {
        
        }
    
    """

    time_points = ['postop', 'fu3m', 'fu12m']
    timepoint_dict = {}
    pairs = list(combinations(time_points,2)) # list of pairs of combinations, e.g. "postop" vs "fu3m"


    # results_sub_path = findfolders.get_local_path(folder="results", sub=incl_sub)
    
    # path to results folder
    path = os.getcwd()
    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path) # path is now leading to Users/username
    
    results_path = os.path.join(path, 'Research', "Longterm_beta_project", "results" )

    # CSV group file of monopolar PSD averages in the correct frequency band
    data = pd.read_csv(os.path.join(results_path, f"monopolPSD_{freq_band}_ALLnormPsdToTotalSum"))
    data.rename(columns={
        "monoRef_fu3m_beta": "fu3m", 
        "monoRef_fu12m_beta": "fu12m", 
        "monoRef_fu18m_beta": "fu18m", 
        "monoRef_postop_beta": "postop"}, 
        inplace=True)
    
    # set figure layout
    # fig, axes = plt.subplots(len(incl_sub)*2, 1) # subplot(rows, columns, panel number), figsize(width,height)
    

    # loop through each unique subject_hemisphere 
    for s_h, sub_hemisphere in enumerate(data.subject_hemisphere.unique()):

        # pick data only from included subjects
        for sub in incl_sub:
            if sub in sub_hemisphere:

                # path to each subject folder within results folder
                results_sub_folder = os.path.join(results_path, f"sub-{sub}")

                # only get rows of a single patient hemisphere
                patient_data = data[data.subject_hemisphere==sub_hemisphere]

                # for each timepoint (column), melt the dataframe of each timepoint column and add the monopolar contact column 
                for t in time_points:

                    timepoint_dict[f"{t}"] = pd.concat(
                        [pd.melt(patient_data.filter(regex= t), var_name='session_timepoint', value_name=f'PSD average in {freq_band} band'), 
                         pd.melt(patient_data.filter(regex='Monopolar_contact'), var_name='column_monopol', value_name='monopolarContact')],
                         axis=1)

                patient_data = pd.concat(timepoint_dict.values(), ignore_index=True)


                #try:
        
                plt.figure()

                ax = sns.boxplot(data=patient_data, x='session_timepoint', y='PSD average in beta band', order=time_points, palette="Blues", width=0.6)
                ax = sns.stripplot(x = "session_timepoint",
                    y = "PSD average in beta band",
                    hue = "monopolarContact",
                    # color = 'black',
                    size=10,
                    data = patient_data,
                    )


                annotator = Annotator(ax, pairs, data=patient_data, x='session_timepoint', y='PSD average in beta band', order=time_points)
                annotator.configure(test='Wilcoxon', text_format='star')
                annotator.apply_and_annotate()

                plt.title(f"{sub_hemisphere}")
                plt.legend(loc= "upper right", bbox_to_anchor=(1.4, 0.4))
                plt.tight_layout()

                plt.savefig(results_sub_folder + f"\\compareMonopolarPSDaverage_{freq_band}_{sub_hemisphere}_Wilcoxon.png")


    ###### LEGEND ######
    # legend = axes[0].legend(loc= 'lower right', edgecolor="black", bbox_to_anchor=(1.4, 0.4)) # only show the first subplot´s legend 
    # # frame the legend with black edges amd white background color 
    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor("white")

    # fig.tight_layout()

    # fig.savefig(figures_path + f"\\RawUnfilteredPSDspectrogram_sub{incl_sub}_{hemisphere}_{pickChannels}.png")
    

                # except:
                #     continue

            

    return {
        "patient_data": patient_data
    }






