""" Ranks within BIP channel Groups Ring, SegmIntra and SegmInter """


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

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
    


    ##################### RESTRUCTURE DATAFRAMES #####################

    # Restructure Dataframes into different session_channelGroup Dataframes 

    Rank_channelGroup_dict = {}

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
            Rank_channelGroup_dict[ses] = session_dict # storing all channel Group Dataframes as values from session keys
    


    ###### FROM Rank_channelGroup_dict CREATE FINAL VERSION OF DATAFRAMES CHANNELGROUP_SESSION ######
    
    ranks_channelGroup_session_dictionary = {}

    for group in channelGroups:
        for ses in sessions:

            # get a ses_dictionary e.g. postop dictionary containing all RankDF: Ring+SegmInter+SegmIntra
            ses_dictionary = Rank_channelGroup_dict[ses]

            # only get keys and values from the ses_dictionary if the correct group e.g. Ring is in keys
            select_dictionary = {key:value for key, value in ses_dictionary.items() if group in key}

            ranks_channelGroup_session_dictionary[f"{group}_{ses}"] = pd.concat(select_dictionary.values(), ignore_index=True)


    ### save the Dataframes with pickle 
    # Final_Permutation_dictionary: keys(f"{group}_{ses}") and values(Dataframes)
    channelgroup_ses_filepath = os.path.join(results_path, f"BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(channelgroup_ses_filepath, "wb") as file:
        pickle.dump(ranks_channelGroup_session_dictionary, file)
    
    print("new file: ", f"BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle",
          "\nin: ", results_path)
    

    return {
        "ranks_channelGroup_session_dictionary": ranks_channelGroup_session_dictionary,
        "Ring_rankDF": Ring_rankDF,
        "SegmIntra_rankDF": SegmIntra_rankDF,
        "SegmInter_rankDF": SegmInter_rankDF

    }
            
            



def Permutation_BIPranksRingSegmGroups(
        result: str,
        difference: str,
        filterSignal: str,
        normalization: str,
        freqBand: str,
        ):
    
    """

    Loading pickle files with psdAverage or (((Peak values))) from frequency bands alpha, beta, lowBeta and highBeta
    Performing Permutation Tests between different session timepoints for each Channel Group seperately.

    Input:
        - result: str "psdAverage" or "peak" -> this decides what you want to rank 
        - difference: str "rank", "psdAverage", "peak" -> decide what you want to compare, ranks will only compare the ranks of psdAverage or peaks
        - filterSignal: str, "band-pass" or "unfiltered"
        - normalization: str, ""rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str, "alpha", "beta", "lowBeta", "highBeta"

    1) Load the Permutation Pickle file written with function above Rank_BIPRingSegmGroups()
        - one pickle file contains either the psdAverage or peak Values from a specific freqband, normalization and filter.
        - opened pickle file is a dictionary storing several Dataframes e.g. each for Ring_postop, Ring_fu3m, Ring_fu12m, Ring_18m 
        - columns: session, bipolarChannel, freqBand, absoluteOrRelativePSD, averagedPSD, rank, subject_hemisphere
    
    2) Restructure Dataframes:
        - in total 3 channelgroups x 4 sessions = 12 Dataframes 
        ('Ring_postop', 'Ring_fu3m', 'Ring_fu12m', 'Ring_fu18m', 
        'SegmInter_postop', 'SegmInter_fu3m', 'SegmInter_fu12m', 'SegmInter_fu18m', 
        'SegmIntra_postop', 'SegmIntra_fu3m', 'SegmIntra_fu12m', 'SegmIntra_fu18m')

        - each Dataframe will be changed, so only 3 columns are left: "session", "rank", "averagedPSD", "sub_hem_BIPchannel"
    
    3) Add column to Dataframe with calculation of relative PSD to rank1 Psd:
        - e.g. Ring Group: 01 = rank1
        - calculate for each psd of one subject hemisphere: PSD / PSD of rank1 -> rank1 is always 1 (100%)
        - for all other channels the relative amount compared to rank 1 PSD will be given -> this gives more detailed information on the distribution of beta 

    4) Merge Dataframes that should be compared, only keep rows with matching sub_hem_BIPchannel values
        - postop - fu3m
        - postop - fu12m
        - postop - fu18m
        - fu3m - fu12m
        - fu3m - fu18m
        - fu12m - fu18m

    

    TODO: does Permutation analysis work if I don't have same number of values within each group?
    Watch out !! doesn't work for peak files, somehow not all channels have a peak value, sometimes only 2 channels in one dataframe.... not sure why

    """

    # Error check:
    assert result in [ "psdAverage", "peak"], f'Result ({result}) INCORRECT' 
    assert difference in ["rank", "psdAverage", "peak"], f'difference ({difference}) INCORRECT' 
    assert filterSignal in ["band-pass", "unfiltered"], f'filterSignal ({filterSignal}) INCORRECT' 
    assert normalization in ["rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"], f'normalization ({normalization}) INCORRECT' 
    assert freqBand in ["alpha", "beta", "lowBeta", "highBeta"], f'freqBand ({freqBand}) INCORRECT' 
    


    channelGroups = ["Ring", "SegmInter", "SegmIntra"]
    Ring_channels = ["12", "01", "23"]
    SegmIntra_channels = ["1A1B", "1B1C", "1A1C", "2A2B", "2B2C", "2A2C",]
    SegmInter_channels = ["1A2A", "1B2B", "1C2C"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    

    ##################### LOAD PICKLE FILES WITH PSD AVERAGES OR PEAK VALUES FROM RESULTS FOLDER #####################

    results_path = find_folders.get_local_path(folder="GroupResults")
    
    # load the correct pickle file 
    data = loadResults.load_BIPchannelGroup_sessionPickle(
        result=result,
        freqBand=freqBand,
        normalization=normalization,
        filterSignal=filterSignal
        )

    # data = dictionary with keys(['Ring_postop', 'Ring_fu3m', 'Ring_fu12m', 'Ring_fu18m', 'SegmInter_postop', 'SegmInter_fu3m', 'SegmInter_fu12m', 'SegmInter_fu18m', 'SegmIntra_postop', 'SegmIntra_fu3m', 'SegmIntra_fu12m', 'SegmIntra_fu18m'])
    data_keys = data.keys()


    ##################### RESTRUCTURE DATAFRAMES AND STORE IN DICTIONARY DF_storage #####################
    DF_storage = {} # this will store single Dataframes stored as values in a dictionary
    DF_all_in_one = pd.DataFrame() # this will store all Dataframes concatenated together


    # Permutation test for each channelGroup seperately
    for group in channelGroups:

        comb_keys = [i for i in data_keys if group in i] # e.g. ['Ring_postop', 'Ring_fu3m', 'Ring_fu12m', 'Ring_fu18m']


        for group_ses in comb_keys: # e.g. ['Ring_postop', 'Ring_fu3m', 'Ring_fu12m', 'Ring_fu18m']
            DF_storage[f"{group_ses}"] = data[group_ses]


            #problem with merging!! not all channel names are the same: therefore exchange ch_name each by structure "BIP_03"
            # rename channels to BIP_xx

            if group == "Ring":
                channelnames = Ring_channels

            elif group == "SegmIntra":
                channelnames = SegmIntra_channels
            
            elif group == "SegmInter":
                channelnames = SegmInter_channels

        
            # rename each channel: make sure the original DF is changed globally!!??
            for chan in channelnames:

                # replace the str in column bipolarChannel, if it contains chan e.g. "12"
                DF_storage[f"{group_ses}"].loc[DF_storage[f"{group_ses}"].bipolarChannel.str.contains(chan), "bipolarChannel"] = f"BIP_{chan}"


            # add new column "sub_hem_BIPchannel" by aggregating columns
            DF_storage[f"{group_ses}"]["sub_hem_BIPchannel"] = DF_storage[f"{group_ses}"][['subject_hemisphere', 'bipolarChannel']].agg('_'.join, axis=1)

            # add new column: recording_type
            DF_storage[f"{group_ses}"].insert(0, "recording_type", group)

            # drop unnecessary columns
            DF_storage[f"{group_ses}"].drop(columns=["frequencyBand", "absoluteOrRelativePSD", "subject_hemisphere", "bipolarChannel"], inplace=True)


            ################### NORMALIZE PSD TO RANK 1 ###################
            sub_hem_BIPchannel = [] # stores all values from this column
            sub_hem_all = [] # stores all sub_hem combinations
            sub_hem_unique = [] # stores only unique sub_hem combinations
 
            
            # get all values in column "sub_hem_BIPchannel"
            for idx, row in DF_storage[f"{group_ses}"].iterrows():
                sub_hem_BIPchannel.append(row["sub_hem_BIPchannel"])
            
            # split each string into 4 parts, only extract 0, 1 = sub, hem
            for c, comb in enumerate(sub_hem_BIPchannel):
                four_parts_split = comb.split("_")

                sub_hem = "_".join([four_parts_split[0], four_parts_split[1]]) # e.g. 017_Right
                sub_hem_all.append(sub_hem)

            # now only take unique sub_hem combinations
            for s_h in sub_hem_all:
                if s_h not in sub_hem_unique:
                    sub_hem_unique.append(s_h)
            
            # for each unique sub_hem combination - calculate the PSD relative to the ranked 1 beta
            for s_h_unique in sub_hem_unique:

                df_to_edit = DF_storage[f"{group_ses}"][DF_storage[f"{group_ses}"].sub_hem_BIPchannel.str.contains(s_h_unique)]

                # if rank = 1.0 define averagedPSD value of the same row as beta_rank_1
                beta_rank_1 = df_to_edit[df_to_edit["rank"] == 1.0] # taking the row containing 1.0 in rank
                beta_rank_1 = beta_rank_1["averagedPSD"].values[0] # just taking psdAverage of rank 1.0
                
                # create a new column and fill the rows by calculating averagedPSD / beta_rank_1
                df_to_edit_copy = df_to_edit.copy() # somehow copying is needed, otherwise warning
                df_to_edit_copy["relative_to_rank1"] = df_to_edit_copy.apply(lambda row: row.averagedPSD / beta_rank_1, axis=1) # in each row add to new column psd/beta_rank1

                DF_all_in_one = pd.concat([DF_all_in_one, df_to_edit_copy], ignore_index=True) # all dataframes will be added up together to one big dataframe


    # save DF_all_in_one as pickle
    DF_all_in_one_filepath = os.path.join(results_path, f"BIP_relativePsdToRank1_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(DF_all_in_one_filepath, "wb") as file:
        pickle.dump(DF_all_in_one, file)

    ##################### MERGE DATAFRAMES TO PAIRED DATAFRAMES AND GET DIFFERENCES OF RANKS/psdAverage/peaks #####################

    # 3 comparisons for each channel group: store all channel groups in comparison dictionary
    comparisons = ["comparePostop_Fu3m", "comparePostop_Fu12m", "comparePostop_Fu18m", "compareFu3m_Fu12m", "compareFu3m_Fu18m", "compareFu12m_Fu18m"]

    # dictionaries to store DF of each channel Group
    comparePostop_Fu3m = {} # keys: Ring, SegmIntra, SegmInter
    comparePostop_Fu12m = {}
    comparePostop_Fu18m = {}
    compareFu3m_Fu12m = {}
    compareFu3m_Fu18m = {}
    compareFu12m_Fu18m = {}
    mean_differenceOfComparison = {}


    for group in channelGroups:

        # merge DF postop and fu3m, only keep matching rows in column "sub_hem_BIPchannel"
        comparePostop_Fu3m[group] = DF_storage[f"{group}_postop"].merge(DF_storage[f"{group}_fu3m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')
        comparePostop_Fu12m[group] = DF_storage[f"{group}_postop"].merge(DF_storage[f"{group}_fu12m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')
        comparePostop_Fu18m[group] = DF_storage[f"{group}_postop"].merge(DF_storage[f"{group}_fu18m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')
        compareFu3m_Fu12m[group] = DF_storage[f"{group}_fu3m"].merge(DF_storage[f"{group}_fu12m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')
        compareFu3m_Fu18m[group] = DF_storage[f"{group}_fu3m"].merge(DF_storage[f"{group}_fu18m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')
        compareFu12m_Fu18m[group] = DF_storage[f"{group}_fu12m"].merge(DF_storage[f"{group}_fu18m"], left_on='sub_hem_BIPchannel', right_on='sub_hem_BIPchannel')


        # add new column to each DF: Difference_rank_x_y, 
        # or depending on result: Difference_averagedPSD_x_y or Difference_peak_x_y
        
        for c in comparisons:

            if c == "comparePostop_Fu3m":
                comparison_DF = comparePostop_Fu3m[group]

            elif c == "comparePostop_Fu12m":
                comparison_DF = comparePostop_Fu12m[group] 
            
            elif c == "comparePostop_Fu18m":
                comparison_DF = comparePostop_Fu18m[group] 
            
            elif c == "compareFu3m_Fu12m":
                comparison_DF = compareFu3m_Fu12m[group]
            
            elif c == "compareFu3m_Fu18m":
                comparison_DF = compareFu3m_Fu18m[group]
            
            elif c == "compareFu12m_Fu18m":
                comparison_DF = compareFu12m_Fu18m[group]
            
            # new column calculating difference between ranks (abs only gives absolute values, so no negative values)
            comparison_DF['Difference_rank_x_y']  = (comparison_DF["rank_x"] - comparison_DF["rank_y"]).apply(abs)

            if result == "psdAverage":
                comparison_DF['Difference_psdAverage_x_y'] = (comparison_DF["averagedPSD_x"] - comparison_DF["averagedPSD_y"]).apply(abs)
            
            elif result == "peak":
                comparison_DF['Difference_peak5Hz_x_y'] = (comparison_DF["PEAK_5HzAverage_x"] - comparison_DF["PEAK_5HzAverage_y"]).apply(abs)
            

            # calculate the mean of each Difference column and store into dictionary
            mean_differenceOfComparison[f"meanDiff_rank_{group}_{c}"] = comparison_DF['Difference_rank_x_y'].mean()

            if result == "psdAverage":
                mean_differenceOfComparison[f"meanDiff_psdAverage_{group}_{c}"] = comparison_DF['Difference_psdAverage_x_y'].mean() 
            
            elif result == "peak":
                mean_differenceOfComparison[f"meanDiff_peak_{group}_{c}"] = comparison_DF['Difference_peak5Hz_x_y'].mean()
            

    # perform different Permutation tests depending on what difference you choose
    mean_difference_keys = mean_differenceOfComparison.keys()
    
    if difference == "rank":
        mean_diff_list = [i for i in mean_difference_keys if "rank" in i] # list of 3 channel groups x 3 comparisons (postop-3, 3-12, 12-18)

    elif difference == "psdAverage":
        mean_diff_list = [i for i in mean_difference_keys if "psdAverage" in i]
    
    elif difference == "peak":
        mean_diff_list = [i for i in mean_difference_keys if "peak" in i]
    

    
    




    ## save the Permutation structured Dataframes with pickle 
    
    comparePostop_Fu3m_filepath = os.path.join(results_path, f"BIPpermutationDF_Postop_Fu3m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(comparePostop_Fu3m_filepath, "wb") as file:
        pickle.dump(comparePostop_Fu3m, file)
    
    comparePostop_Fu12m_filepath = os.path.join(results_path, f"BIPpermutationDF_Postop_Fu12m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(comparePostop_Fu12m_filepath, "wb") as file:
        pickle.dump(comparePostop_Fu12m, file)
    
    comparePostop_Fu18m_filepath = os.path.join(results_path, f"BIPpermutationDF_Postop_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(comparePostop_Fu18m_filepath, "wb") as file:
        pickle.dump(comparePostop_Fu18m, file)

    compareFu3m_Fu12m_filepath = os.path.join(results_path, f"BIPpermutationDF_Fu3m_Fu12m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(compareFu3m_Fu12m_filepath, "wb") as file:
        pickle.dump(compareFu3m_Fu12m, file)
    
    compareFu3m_Fu18m_filepath = os.path.join(results_path, f"BIPpermutationDF_Fu3m_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(compareFu3m_Fu18m_filepath, "wb") as file:
        pickle.dump(compareFu3m_Fu18m, file)

    compareFu12m_Fu18m_filepath = os.path.join(results_path, f"BIPpermutationDF_Fu12m_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(compareFu12m_Fu18m_filepath, "wb") as file:
        pickle.dump(compareFu12m_Fu18m, file)
    
    print("files: ", 
          f"BIP_relativePsdToRank1_{result}_{freqBand}_{normalization}_{filterSignal}.pickle",
          f"\nBIPpermutationDF_Postop_Fu3m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle", 
          f"\nBIPpermutationDF_Postop_Fu12m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle", 
          f"\nBIPpermutationDF_Postop_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle", 
          f"\nBIPpermutationDF_Fu3m_Fu12m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle", 
          f"\nBIPpermutationDF_Fu3m_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle", 
          f"\nBIPpermutationDF_Fu12m_Fu18m_{result}_{freqBand}_{normalization}_{filterSignal}.pickle",
          "\nwritten in: ", results_path
          )
    


    return {
        "DF_all_in_one": DF_all_in_one,
        "comparePostop_Fu3m": comparePostop_Fu3m,
        "comparePostop_Fu12m": comparePostop_Fu12m,
        "comparePostop_Fu18m": comparePostop_Fu18m,
        "compareFu3m_Fu12m": compareFu3m_Fu12m,
        "compareFu3m_Fu18m": compareFu3m_Fu18m,
        "compareFu12m_Fu18m": compareFu12m_Fu18m, 
        "mean_differenceOfComparison": mean_differenceOfComparison
    }



def BIPranks_plotDifferenceBetweenSessions(
        data2plot: str,
        filterSignal: str,
        normalization: str,
        freqBand: str,
    ):

    """
    Plot comparisons of ranks or absolute values of either psdAverage or peak 5Hz average amplitude
        - Postop - Fu3m
        - Fu3m - Fu12m
        - Fu12m - Longterm (18, 20, 22MFU)
    
    Input: 
        - data: str "psdAverage_ranks", "psdAverage_absolute",  "peak_ranks",  "peak_absolute" -> this decides what you want to plot 
        - filterSignal: str, "band-pass" or "unfiltered"
        - normalization: str, ""rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str, "alpha", "beta", "lowBeta", "highBeta"
    
    


    
    """

    # Error check:
    assert data2plot in ["psdAverage_ranks", "psdAverage_absolute",  "peak_ranks",  "peak_absolute"], f'data ({data2plot}) INCORRECT' 
    assert filterSignal in ["band-pass", "unfiltered"], f'filterSignal ({filterSignal}) INCORRECT' 
    assert normalization in ["rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"], f'normalization ({normalization}) INCORRECT' 
    assert freqBand in ["alpha", "beta", "lowBeta", "highBeta"], f'freqBand ({freqBand}) INCORRECT' 


    ##################### LOAD PICKLE FILES WITH PSD AVERAGES OR PEAK VALUES FROM RESULTS FOLDER #####################

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    if "psdAverage" in data2plot:
        result = "psdAverage"
    
    elif "peak" in data2plot:
        result = "peak"

    
    # load the correct pickle file 
    data = loadResults.load_BIPpermutationComparisonsPickle(
        result=result,
        filterSignal=filterSignal,
        normalization=normalization,
        freqBand=freqBand
        )

    # data = dictionary with keys(['Postop_Fu3m', 'Fu3m_Fu12m', 'Fu12m_Fu18m'])
    comparisons = data.keys() # ['Postop_Fu3m', 'Fu3m_Fu12m', 'Fu12m_Fu18m']
    channelGroups = ["Ring", "SegmIntra", "SegmInter"]


    fontdict = {"size": 25}

    # plot one figure each for comparison
    for comp in comparisons:

        fig, axes = plt.subplots(3,1,figsize=(10,15))

        # 3 rows: Ring, SegmIntra, SegmInter
        for g, group in enumerate(channelGroups):

            # get data
            comparison_group_dataframe = data[comp][group]

            colors = sns.color_palette("viridis", n_colors=len(comparison_group_dataframe.index))

            if group == "Ring":
                BIP_number = 3
            
            elif group == "SegmIntra":
                BIP_number = 6
            
            elif group == "SegmInter":
                BIP_number = 3


            # plot per group
            sns.histplot(data=comparison_group_dataframe, x="Difference_rank_x_y", stat="count", bins=np.arange(-0.25, BIP_number+0.5, 0.5),
                         hue="sub_hem_BIPchannel", multiple="stack", 
                         palette=colors, ax=axes[g])
            

            axes[g].set_title(f"{group}", fontdict=fontdict)
            
            legend = axes[g].get_legend()
            axes[g].legend().remove()
            # handles = legend.legendHandles
            # legend_list = list(comparison_group_dataframe.sub_hem_BIPchannel.values)
            # axes[g].legend(handles, legend_list, title='subject, hemisphere, BIP channel',  title_fontsize=15, fontsize=15)

        ##################### ADJUST THE PLOTS #####################

        for ax in axes:

            ax.set_xlabel(f"difference between {data2plot}",  fontsize=25)
            ax.set_ylabel("Count", fontsize=25)
        
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)


        fig.suptitle(f"{comp} Difference of {data2plot}", fontsize=30)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.show()

        fig.savefig(figures_path + f"\\BIP{comp}_{data2plot}_comparisons.png")












    







