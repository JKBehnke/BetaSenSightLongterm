""" Ranks within BIP channel Groups Ring, SegmIntra and SegmInter """


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


def Rank_BIPRingSegmGroups(
        result: str,
        filterSignal: str,
        normalization: str,
        freqBand: str
        ):
    
    """

    Loading pickle files with psdAverage or (((Peak values))) from frequency bands alpha, beta, lowBeta and highBeta
    Ranking within each session channelGroup and formatting Dataframes to use in the Permutation function.


    Input:
        - result: str "psdAverage" or "peak" -> this decides what you want to rank 
        - filterSignal: str, "band-pass" or "unfiltered"
        - normalization: str, ""rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str, "alpha", "beta", "lowBeta", "highBeta"

    1) Load the BIPpeak or BIPpsdAverage Pickle file written with functions from BIP_channelGroups.py
        - one pickle file contains either the psdAverage or peak Values from 4 freqency bands: alpha, beta, lowBeta, highBeta

    2) edit all Dataframes from Ring, SegmIntra and SegmInter:
        - from Ring DF: drop irrelevant BIP channels 03, 02, 13 (not adjacent)
        - in all DF: add column "rank" with ranking of psdAverage or PEAK_5HzAverage
        - save new DF with rankings as pickle files in results folder
    
    3) create new version of Dataframes:
        - e.g. Dataframe each for Ring_postop, Ring_fu3m, Ring_fu12m, Ring_18m 
        - columns: session, bipolarChannel, freqBand, absoluteOrRelativePSD, averagedPSD, rank, subject_hemisphere

    

    TODO: does Permutation analysis work if I don't have same number of values within each group?
    Watch out !! doesn't work for peak files, somehow not all channels have a peak value, sometimes only 2 channels in one dataframe.... not sure why

    """
    channelGroups = ["Ring", "SegmInter", "SegmIntra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    Ring_rankDF = {} # same keys as in sub_hem_ses_freqBand_list
    SegmIntra_rankDF = {}
    SegmInter_rankDF = {}


    ##################### LOAD PICKLE FILES WITH PSD AVERAGES OR PEAK VALUES FROM RESULTS FOLDER #####################

    results_path = find_folders.get_local_path(folder="GroupResults")
    
    # load the correct pickle files from all channelGroups
    data = loadResults.load_BIPchannelGroupsPickle(
        result=result,
        channelGroup=channelGroups,
        normalization=normalization,
        filterSignal=filterSignal
        )
    
    # list of all existing combinations (one combination key has one DF with results as value)
    sub_hem_ses_freqBand_all = data["Ring"].keys()

    # filter only keys containing the correct frequency band
    freqBand_keys = [i for i in sub_hem_ses_freqBand_all if f"{freqBand}" in i]



    ##################### EDIT DATAFRAMES; ADD COLUMN WITH RANKS #####################
    # Edit all Dataframes from 3 channelGroups: add column with ranks 
    for group in channelGroups:
        for sub_hem_ses_freq in freqBand_keys:
    
            #first from all Ring DF take only adjacent BIP channels 01,12,23
            if group == "Ring":
                Ring_DF_allChannels = data[group][sub_hem_ses_freq] # containing 6 BIP channels
                
                # drop the BIP channels which contacts are not adjacent 
                Ring_DF_3Channels = Ring_DF_allChannels[Ring_DF_allChannels.bipolarChannel.str.contains("03|13|02") == False]

                # add column rank to DF depending on result input
                if result == "psdAverage":
                    rank_column = Ring_DF_3Channels.averagedPSD.rank(ascending=False) # rank highest psdAverage = 1.0
                    copyDF = Ring_DF_3Channels.copy() # has to be copied, otherwise can´t add a new column
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    Ring_rankDF[sub_hem_ses_freq] = copyDF
                
                elif result == "peak":
                    rank_column = Ring_DF_3Channels.PEAK_5HzAverage.rank(ascending=False) # rank highest PEAK_5HzAverage = 1.0
                    copyDF = Ring_DF_3Channels.copy()
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    Ring_rankDF[sub_hem_ses_freq] = copyDF
            

            elif group == "SegmIntra":
                SegmIntra_DF = data[group][sub_hem_ses_freq] # containing 6 BIP channels

                # add column rank to DF depending on result input
                if result == "psdAverage":
                    rank_column = SegmIntra_DF.averagedPSD.rank(ascending=False) # rank highest psdAverage = 1.0
                    copyDF = SegmIntra_DF.copy()
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    SegmIntra_rankDF[sub_hem_ses_freq] = copyDF
                
                elif result == "peak":
                    rank_column = SegmIntra_DF.PEAK_5HzAverage.rank(ascending=False) # rank highest PEAK_5HzAverage = 1.0
                    copyDF = SegmIntra_DF.copy()
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    SegmIntra_rankDF[sub_hem_ses_freq] = copyDF

            
            elif group == "SegmInter":
                SegmInter_DF = data[group][sub_hem_ses_freq] # containing 6 BIP channels

                # add column rank to DF depending on result input
                if result == "psdAverage":
                    rank_column = SegmInter_DF.averagedPSD.rank(ascending=False) # rank highest psdAverage = 1.0
                    copyDF = SegmInter_DF.copy()
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    SegmInter_rankDF[sub_hem_ses_freq] = copyDF
                
                elif result == "peak":
                    rank_column = SegmInter_DF.PEAK_5HzAverage.rank(ascending=False) # rank highest PEAK_5HzAverage = 1.0
                    copyDF = SegmInter_DF.copy()
                    copyDF["rank"] = rank_column # add new column "rank" to the copied DF
                    SegmInter_rankDF[sub_hem_ses_freq] = copyDF
    


    ##################### PERMUTATION ANALYSIS #####################

    # Restructure Dataframes into different session_channelGroup Dataframes 

    Rank_permutation_dict = {}

    postop_dict = {}
    fu3m_dict = {}
    fu12m_dict = {}
    fu18m_dict = {}

    for group in channelGroups:
        
        # define the dictionary from where to take the dataframes for each channel group 
        if group == "Ring":
            base_dictionary = Ring_rankDF # dictionary that already stores the Dataframes I want to edit
        
        elif group == "SegmIntra":
            base_dictionary = SegmIntra_rankDF
        
        elif group == "SegmInter":
            base_dictionary = SegmInter_rankDF
        
        else:
            print("base_dictionary and dict_to_store not defined. Group must be in: ", channelGroups)


        # seperate Dataframes for each session
        for ses in sessions:

            session_keys = [i for i in freqBand_keys if f"{ses}" in i] # list of keys that contain the correct session

            if ses == "postop":
                session_dict = postop_dict
            
            elif ses == "fu3m":
                session_dict = fu3m_dict
            
            elif ses == "fu12m":
                session_dict = fu12m_dict

            elif ses == "fu18m":
                session_dict = fu18m_dict
            
            #print("session: ", ses)

            for key in session_keys:

                single_dataframe = base_dictionary[key] # this is one dataframe from the channelGroup Dictionary 
                
                # add subject_hemisphere column to the dataframe
                key_split = key.split('_')# list of sub, hem, ses, freq
                sub_hem =  '_'.join([key_split[0], key_split[1]]) # sub_hem information

                session_dict[f"{group}_{key}"] = single_dataframe.assign(subject_hemisphere = sub_hem) # add the column with name subject_hemisphere and the correct value

                # e.g. postop_dict["024_Right_postop_beta"] = DF
                # add this in Ring_permutation_dict["postop"] = 


            # add the complete session dictionary into the channelGroup dictionary
            Rank_permutation_dict[ses] = session_dict # storing all channel Group Dataframes as values from session keys
    


    ###### FROM RANK_PERMUTATION_DICT CREATE FINAL VERSION OF DATAFRAMES CHANNELGROUP_SESSION ######

    Final_Permutation_dictionary = {}

    for group in channelGroups:
        for ses in sessions:

            # get a ses_dictionary e.g. postop dictionary containing all RankDF: Ring+SegmInter+SegmIntra
            ses_dictionary = Rank_permutation_dict[ses]

            # only get keys and values from the ses_dictionary if the correct group e.g. Ring is in keys
            select_dictionary = {key:value for key, value in ses_dictionary.items() if group in key}

            Final_Permutation_dictionary[f"{group}_{ses}"] = pd.concat(select_dictionary.values(), ignore_index=True)


    ### save the Dataframes with pickle 
    # Final_Permutation_dictionary: keys(f"{group}_{ses}") and values(Dataframes)
    Permutation_filepath = os.path.join(results_path, f"BIPranksPermutation_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(Permutation_filepath, "wb") as file:
        pickle.dump(Final_Permutation_dictionary, file)
    

    return {
        "Final_Permutation_dictionary": Final_Permutation_dictionary,
        "Ring_rankDF": Ring_rankDF,
        "SegmIntra_rankDF": SegmIntra_rankDF,
        "SegmInter_rankDF": SegmInter_rankDF

    }
            
            



def Permutation_BIPranksRingSegmGroups(
        result: str,
        filterSignal: str,
        normalization: str,
        freqBand: str
        ):
    
    """

    Loading pickle files with psdAverage or (((Peak values))) from frequency bands alpha, beta, lowBeta and highBeta
    Performing Permutation Tests between different session timepoints for each Channel Group seperately.

    Input:
        - result: str "psdAverage" or "peak" -> this decides what you want to rank 
        - filterSignal: str, "band-pass" or "unfiltered"
        - normalization: str, ""rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str, "alpha", "beta", "lowBeta", "highBeta"

    1) Load the Permutation Pickle file written with function above Rank_BIPRingSegmGroups()
        - one pickle file contains either the psdAverage or peak Values from a specific freqband, normalization and filter.
        - opened pickle file is a dictionary storing several Dataframes e.g. each for Ring_postop, Ring_fu3m, Ring_fu12m, Ring_18m 
        - columns: session, bipolarChannel, freqBand, absoluteOrRelativePSD, averagedPSD, rank, subject_hemisphere

    

    TODO: does Permutation analysis work if I don't have same number of values within each group?
    Watch out !! doesn't work for peak files, somehow not all channels have a peak value, sometimes only 2 channels in one dataframe.... not sure why

    """
    channelGroups = ["Ring", "SegmInter", "SegmIntra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    Ring_rankDF = {} # same keys as in sub_hem_ses_freqBand_list
    SegmIntra_rankDF = {}
    SegmInter_rankDF = {}


    ##################### LOAD PICKLE FILES WITH PSD AVERAGES OR PEAK VALUES FROM RESULTS FOLDER #####################

    results_path = find_folders.get_local_path(folder="GroupResults")
    
    # load the correct pickle file 
    data = loadResults.load_permutation_BIPchannelGroupsPickle(
        result=result,
        freqBand=freqBand,
        normalization=normalization,
        filterSignal=filterSignal
        )

    # data = dictionary with keys(['Ring_postop', 'Ring_fu3m', 'Ring_fu12m', 'Ring_fu18m', 'SegmInter_postop', 'SegmInter_fu3m', 'SegmInter_fu12m', 'SegmInter_fu18m', 'SegmIntra_postop', 'SegmIntra_fu3m', 'SegmIntra_fu12m', 'SegmIntra_fu18m'])
    




                   



            
            








    



