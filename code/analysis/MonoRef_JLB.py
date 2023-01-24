""" Monopolar referencing Johannes Busch method """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import mne

# PyPerceive Imports
import PerceiveImport.methods.find_folders as findfolders


def MonoRef_JLB(incl_sub:str, hemisphere:str, normalization:str):

    """
    Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

    Input:
        - incl_sub: str, e.g. "024"
        - hemisphere: str, e.g. "Right"
        - normalization: str, e.g. "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"


    For different versions of PSD: 
        - raw PSD
        - relative PSD to total sum
        - relative PSD to sum 1-100 Hz
        - relative PSD to sum 40-80 Hz

    1) Calculate the percentage of each direction A, B and C:
        - proxy of direction:
            A = 1A2A
            B = 1B2B
            C = 1C2C

        - Percentage of direction = Mean beta power of one direction divided by total mean beta power of all directions 
    
    2) Weight each segmented level 1 and 2 with percentage of direction:
        - proxy of hight:
            1 = 02
            2 = 13

        - Percentage of direction multiplied with mean beta power of each level
        - e.g. 1A = Percentage of direction(A) * mean beta power (02)
    
    """


    time_points = ['postop', 'fu3m', 'fu12m', 'fu18m']
    frequency_range = ["lowBeta", "highBeta", "beta"]
    direction_proxy = ["1A2A", "1B2B", "1C2C"]
    level_proxy = ["13", "02"]
    averagedPSD_dict = {}
    percentagePSD_dict = {}
    monopolar_references = {}

    ############# READ CSV FILE WITH AVERAGED PSD as Dataframe #############
    # get path to .csv file
    results_path = findfolders.get_local_path(folder="results", sub=incl_sub)

    # read .csv file as Dataframe
    psdAverageDataframe = pd.read_csv(os.path.join(results_path, f"psdAverage_{normalization}_{hemisphere}"))

    
    for t, tp in enumerate(time_points):

        for f, fq in enumerate(frequency_range):

            # filter the Dataframe to only get rows within different frequency bands of each session
            session_frequency_Dataframe = psdAverageDataframe[(psdAverageDataframe["frequencyBand"]==fq) & (psdAverageDataframe["session"]==tp)]

            ################### WEIGHT DIRECTION ###################
            # get all relevant averaged psd values for each direction
            for d, dir in enumerate(direction_proxy):

                # get the row that contains the the bipolar channel of interest "1A2A", "1B2B", "1C2C"
                directionDF = session_frequency_Dataframe[session_frequency_Dataframe["bipolarChannel"].str.contains(dir)]

                # store these 3 averaged psd values (in fifth column pf the initial DF) in the dictionary
                averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{dir}"] = [tp, fq, dir, directionDF.iloc[:,4].item()]


            # calculate total mean beta power of all directions: sum of averaged PSD 1A2A + 1B2B + 1C2C
            averagedPsd_A = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1A2A"][3]
            averagedPsd_B = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1B2B"][3]
            averagedPsd_C = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_1C2C"][3]

            sumABC = averagedPsd_A + averagedPsd_B + averagedPsd_C
            
            # calculate the percentage of each direction of the total mean beta power of all directions
            for d, dir in enumerate(direction_proxy):

                percentagePSD = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{dir}"][3]/sumABC

                percentagePSD_dict[f"percentagePSD_{tp}_{fq}_{dir}"] = [tp, fq, dir, percentagePSD]
            
            percentagePSD_A = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1A2A"][3]
            percentagePSD_B = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1B2B"][3]
            percentagePSD_C = percentagePSD_dict[f"percentagePSD_{tp}_{fq}_1C2C"][3]


            ################### WEIGHT LEVEL ###################

            # get both relevant averaged PSD values for the levels 1 and 2
            for l, lev in enumerate(level_proxy):

                # get the row that contains the the bipolar channels of interest "02", "13"
                levelDF = session_frequency_Dataframe[session_frequency_Dataframe["bipolarChannel"].str.contains(lev)]

                # store these 2 averaged psd values (in fifth column) in the dictionary
                averagedPSD_dict[f"averagedPSD_{tp}_{fq}_{lev}"] = [tp, fq, lev, levelDF.iloc[:,4].item()]

    	    # get averaged PSD values for both levels
            averagedPsd_level1 = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_02"][3]
            averagedPsd_level2 = averagedPSD_dict[f"averagedPSD_{tp}_{fq}_13"][3]
            
            # calculate the monopolar reference for each segmented contact
            monopol_1A = percentagePSD_A * averagedPsd_level1
            monopol_1B = percentagePSD_B * averagedPsd_level1
            monopol_1C = percentagePSD_C * averagedPsd_level1
            monopol_2A = percentagePSD_A * averagedPsd_level2
            monopol_2B = percentagePSD_B * averagedPsd_level2
            monopol_2C = percentagePSD_C * averagedPsd_level2

            # store monopolar references in a dictionary
            monopolar_references[f"monoRef_{tp}_{fq}"] = [tp, fq, monopol_1A, monopol_1B, monopol_1C, monopol_2A, monopol_2B, monopol_2C]


    
    #################### WRITE DATAFRAMES TO STORE VALUES ####################

    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(averagedPSD_dict) 
    psdAverageDF.rename(index={0: "session", 1: "frequency_band", 2: "channel", 3: "averagedPSD"}, inplace=True) # rename the rows
    psdAverageDF = psdAverageDF.transpose() # Dataframe with 1 columns and rows for each single power spectrum


    # write DataFrame of percentage psd values in each frequency band depending on the chosen normalization
    psdPercentageDF = pd.DataFrame(percentagePSD_dict)
    psdPercentageDF.rename(index={0: "session", 1: "frequency_band", 2: "direction", 3: "percentagePSD_perDirection"}, inplace=True) # rename the rows
    psdPercentageDF = psdPercentageDF.transpose() # Dataframe with 1 columns and rows for each single power spectrum


    # write DataFrame of percentage psd values in each frequency band depending on the chosen normalization
    monopolRefDF = pd.DataFrame(monopolar_references) 
    monopolRefDF.rename(index={0: "session", 1: "frequency_band", 2: "monopolarRef_1A", 3: "monopolarRef_1B", 4: "monopolarRef_1C", 5: "monopolarRef_2A", 6: "monopolarRef_2B", 7: "monopolarRef_2C"}, inplace=True) # rename the rows
    monopolRefDF = monopolRefDF.transpose() # Dataframe with 8 columns and rows for each single power spectrum


    return {
        "psdAverageDataframe":psdAverageDF,
        "psdPercentagePerDirection":psdPercentageDF,
        "monopolarReference":monopolRefDF
    }




            


















