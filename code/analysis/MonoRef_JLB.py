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

import analysis.loadCSVresults as loadcsv


def MonoRef_JLB(incl_sub:str, hemisphere:str, normalization:str):

    """
    Calculate the monopolar average of beta power (13-35 Hz) for segmented contacts (1A,1B,1C and 2A,2B,2C)

    Input:
        - incl_sub: str, e.g. "024"
        - hemisphere: str, e.g. "Right"
        - normalization: str, e.g. "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"


    Four different versions of PSD: 
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
    
        

     Selecting the monopolar referenced contact with #1 Rank (postop #1 or fu3m #1)

        1) restructure the monopolar Rank and References Dataframes
            - select the monopolar contact with Rank #1 in every session_freqBand column of the monoRankDF
            - store every Rank #1 monopolar contact in a dictionary monopolarFirstRank, transform to DF
            - for every #1 ranked channel in every session and for every frequency band, add all monopolarly referenced PSD values
            - FirstRankChannel_PSD_DF with columns: "session_frequencyBand", "numberOneRank_monopolarChannel", "monoRef_postop_lowBeta", "monoRef_postop_highBeta" etc.
        
        2) For postop and fu3m baselineRank Channel choose highest ranked channel of each frequency band with corresponding averaged psd values of the same frequency band
            - BetaPsdAverage
            - LowBetaPsdAverage
            - HighBetaPsdAverage

            (e.g. #1 channel postop in lowBeta band = 1A, get all monopolarly averaged PSD values from channel 1A in lowBeta band at all timepoints)

            
    Return: 
        {
        "psdAverageDataframe":psdAverageDF,
        "psdPercentagePerDirection":psdPercentageDF,
        "monopolarReference":monopolRefDF,
        "monopolarRanks":monopolRankDF,
        "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
        "PostopBaseline_beta": postopBaseline_beta,
        "PostopBaseline_lowBeta": postopBaseline_lowBeta,
        "PostopBaseline_highBeta": postopBaseline_highBeta,
        "Fu3mBaseline_beta": fu3mBaseline_beta,
        "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
        "Fu3mBaseline_highBeta": fu3mBaseline_highBeta
        }

    
    Pitfalls: 
        - Error if a patient only had 3 timepoints, especially when postop or fu3m recording is missing
        - so check beforehand how many timepoints the patient went through and adjust the code manually


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
    psdAverageDataframe = pd.read_csv(os.path.join(results_path, f"SPECTROGRAMpsdAverageFrequencyBands_{normalization}_{hemisphere}"))

    
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
            monopolar_references[f"monoRef_{tp}_{fq}"] = [monopol_1A, monopol_1B, monopol_1C, monopol_2A, monopol_2B, monopol_2C]


    
    #################### WRITE DATAFRAMES TO STORE VALUES ####################

    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(averagedPSD_dict) 
    psdAverageDF.rename(index={0: "session", 1: "frequency_band", 2: "channel", 3: "averagedPSD"}, inplace=True) # rename the rows
    psdAverageDF = psdAverageDF.transpose() # Dataframe with 1 columns and rows for each single power spectrum


    # write DataFrame of percentage psd values in each frequency band depending on the chosen normalization
    psdPercentageDF = pd.DataFrame(percentagePSD_dict)
    psdPercentageDF.rename(index={0: "session", 1: "frequency_band", 2: "direction", 3: "percentagePSD_perDirection"}, inplace=True) # rename the rows
    psdPercentageDF = psdPercentageDF.transpose() # Dataframe with 1 columns and rows for each single power spectrum


    # write DataFrame of monopolar reference values in each frequency band and session timepoint
    monopolRefDF = pd.DataFrame(monopolar_references) 
    monopolRefDF.rename(index={0: "monopolarRef_1A", 1: "monopolarRef_1B", 2: "monopolarRef_1C", 3: "monopolarRef_2A", 4: "monopolarRef_2B", 5: "monopolarRef_2C"}, inplace=True) # rename the rows


    # write DataFrame of ranks of monopolar references in each frequency band and session timepoint
    monopolRankDF = monopolRefDF.rank(ascending=False) # new Dataframe ranking monopolar values from monopolRefDF from high to low


    # save Dataframes as csv in the results folder
    psdAverageDF.to_csv(os.path.join(results_path,f"averagedPSD_{normalization}_{hemisphere}"), sep=",")
    psdPercentageDF.to_csv(os.path.join(results_path,f"percentagePsdDirection_{normalization}_{hemisphere}"), sep=",")
    monopolRefDF.to_csv(os.path.join(results_path,f"monopolarReference_{normalization}_{hemisphere}"), sep=",")
    monopolRankDF.to_csv(os.path.join(results_path,f"monopolarRanks_{normalization}_{hemisphere}"), sep=",")
    



    ################ Restructure the Dataframes ################ 
    # Replace every rank #1 in monoRankDF with the monopolar channel (in index)
    monoRank_replace1DF = monopolRankDF.apply(lambda x: x.where(x != 1.0, monopolRankDF.index), axis=0)

    # drop first column "Unnamed: 0" with all monopolar Ref channel names 
    monoRank_replace1DF.drop(columns=monoRank_replace1DF.columns[0], axis=1,  inplace=True)

    # only select the strings values with the monopolar channel #1 rank for each column (session_frequencyBand)
    monopolarFirstRank = {}

    # loop over each column
    for col in monoRank_replace1DF.columns:
        # extract the column as a series
        column = monoRank_replace1DF[col]
        
        # exclude float values and replace floats by nan
        # lambda function returns the value if it is a string, otherwise it returns np.nan
        column = column.apply(lambda x: x if isinstance(x, str) else np.nan)
        
        # drop all NaN values
        column.dropna(how='all', inplace=True)
        
        # find the first value that is a string (e.g. "monopolarRef_1A")
        #The next function returns the first value of each column that is a string. If the sequence is empty, the default value None is returned.
        value = next((value for value in column.values if isinstance(value, str)), None)
        
        # add the result to the dictionary
        monopolarFirstRank[col] = value

    # convert the dictionary to a dataframe
    monopolarFirstRankDF = pd.DataFrame(list(monopolarFirstRank.items()), columns=['session_frequencyBand', 'numberOneRank_monopolarChannel'])
    

    # from monoRefDF extract only the row equal to the value of the column 'numberOneRank_monopolarChannel' 
    
    # loop through each #1 rank value and store the matching dataframe row from monoRefDF in a dictionary
    FirstRankRef_dict = {}
    
    for index, value in monopolarFirstRankDF['numberOneRank_monopolarChannel'].iteritems():
        FirstRankRef_dict[f"{index}_{value}"] = monopolRefDF[monopolRefDF.index.str.contains(value)]


    # first make a new Dataframe of selected monoRef rows, by concatenating all values of FirstRankRef_dict
    FirstRankRef_DF = pd.concat(FirstRankRef_dict.values(), keys=FirstRankRef_dict.keys(), ignore_index=True) # keys need to be specified

    # drop the first column with monopolar channel names, because this column already exists in the monopolarFirstRank DF that will be concatenated
    FirstRankRef_DF.drop(columns=FirstRankRef_DF.columns[0], axis=1, inplace=True) # inplace=True will modify the original DF and will not create a new DF

    # now concatenate the FirstRankDF (with #1 ranked monopolar contacts) with the FirstRankRefDF (with all referenced psd values of this #1 ranked contact)
    FirstRankChannel_PSD_DF = pd.concat([monopolarFirstRankDF, FirstRankRef_DF], axis=1)


    ################ SELECT HIGHEST RANK FOR POSTOP AND FU3M BASELINE ################

    # Dataframes for postop Baseline first ranks in each frequency band
    postop_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("postop")
    fu3m_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("fu3m")
    beta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_beta")
    lowBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_lowBeta")
    highBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_highBeta")


    #######    POSTOP BASELINE  #######
    # first select the row from the highest ranked channel in each frequency band
    postopBaseline_beta_channelRow = FirstRankChannel_PSD_DF[postop_mask & beta_mask]
    postopBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & lowBeta_mask]
    postopBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & highBeta_mask]

    # # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    postopBaseline_beta = postopBaseline_beta_channelRow.loc[:,postopBaseline_beta_channelRow.columns.str.contains("_beta")]
    postopBaseline_lowBeta = postopBaseline_lowBeta_channelRow.loc[:,postopBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    postopBaseline_highBeta = postopBaseline_highBeta_channelRow.loc[:,postopBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]
    

    #######    FU3M BASELINE  #######
     # first select the row from the highest ranked channel in each frequency band
    fu3mBaseline_beta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & beta_mask]
    fu3mBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & lowBeta_mask]
    fu3mBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & highBeta_mask]

    # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    fu3mBaseline_beta = fu3mBaseline_beta_channelRow.loc[:,fu3mBaseline_beta_channelRow.columns.str.contains("_beta")]
    fu3mBaseline_lowBeta = fu3mBaseline_lowBeta_channelRow.loc[:,fu3mBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    fu3mBaseline_highBeta = fu3mBaseline_highBeta_channelRow.loc[:,fu3mBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]
    
    # save Dataframes as csv in the results folder
    FirstRankChannel_PSD_DF.to_csv(os.path.join(results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
    postopBaseline_beta.to_csv(os.path.join(results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
    postopBaseline_lowBeta.to_csv(os.path.join(results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
    postopBaseline_highBeta.to_csv(os.path.join(results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
    fu3mBaseline_beta.to_csv(os.path.join(results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
    fu3mBaseline_lowBeta.to_csv(os.path.join(results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
    fu3mBaseline_highBeta.to_csv(os.path.join(results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
    

    return {
        "psdAverageDataframe":psdAverageDF,
        "psdPercentagePerDirection":psdPercentageDF,
        "monopolarReference":monopolRefDF,
        "monopolarRanks":monopolRankDF,
        "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
        "PostopBaseline_beta": postopBaseline_beta,
        "PostopBaseline_lowBeta": postopBaseline_lowBeta,
        "PostopBaseline_highBeta": postopBaseline_highBeta,
        "Fu3mBaseline_beta": fu3mBaseline_beta,
        "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
        "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
    }
    



def MonoRefPsd_highestRank(sub: str, normalization: str, hemisphere: str):
    """
    Selecting the monopolar referenced contact with #1 Rank (postop #1 or fu3m #1)

        - sub: str, 
        - normalization: str, 
        - hemisphere: str, 

        1) Load and restructure the monopolar Rank and References Dataframes
            - select the monopolar contact with Rank #1 in every session_freqBand column of the monoRankDF
            - store every Rank #1 monopolar contact in a dictionary monopolarFirstRank, transform to DF
            - for every #1 ranked channel in every session and for every frequency band, add all monopolarly referenced PSD values
            - FirstRankChannel_PSD_DF with columns: "session_frequencyBand", "numberOneRank_monopolarChannel", "monoRef_postop_lowBeta", "monoRef_postop_highBeta" etc.
        
        2) For postop and fu3m baselineRank Channel choose highest ranked channel of each frequency band with corresponding averaged psd values of the same frequency band
            - BetaPsdAverage
            - LowBetaPsdAverage
            - HighBetaPsdAverage

            (e.g. #1 channel postop in lowBeta band = 1A, get all monopolarly averaged PSD values from channel 1A in lowBeta band at all timepoints)


    """

    ############# READ CSV FILE of monopolar referenced AVERAGED PSD as Dataframe #############
    # get path to results folder of subject
    results_path = findfolders.get_local_path(folder="results", sub=sub)

    # read .csv file as Dataframe
    monoRef_result = loadcsv.load_MonoRef_JLBresultCSV(
    sub=sub,
    normalization=normalization,
    hemisphere=hemisphere
    )
    
    # get Dataframe of monopolar references/ monopolarly referenced PSD values
    monoRefDF = monoRef_result["monopolRefDF"]

    # get Dataframe of monopolar ranks (column "Unnamed: 0" with monopolar contact names n=6)
    monoRankDF = monoRef_result["monopolRankDF"]


    ################ Restructure the Dataframes ################ 
    # Replace every rank #1 in monoRankDF with the monopolar channel (in first column: 'Unnamed: 0')
    monoRank_replace1DF = monoRankDF.apply(lambda x: x.where(x != 1.0, monoRankDF['Unnamed: 0']), axis=0)

    # drop first column "Unnamed: 0" with all monopolar Ref channel names 
    monoRank_replace1DF.drop(columns=monoRank_replace1DF.columns[0], axis=1,  inplace=True)

    # only select the strings values with the monopolar channel #1 rank for each column (session_frequencyBand)
    monopolarFirstRank = {}

    # loop over each column
    for col in monoRank_replace1DF.columns:
        # extract the column as a series
        column = monoRank_replace1DF[col]
        
        # exclude float values and replace floats by nan
        # lambda function returns the value if it is a string, otherwise it returns np.nan
        column = column.apply(lambda x: x if isinstance(x, str) else np.nan)
        
        # drop all NaN values
        column.dropna(how='all', inplace=True)
        
        # find the first value that is a string (e.g. "monopolarRef_1A")
        #The next function returns the first value of each column that is a string. If the sequence is empty, the default value None is returned.
        value = next((value for value in column.values if isinstance(value, str)), None)
        
        # add the result to the dictionary
        monopolarFirstRank[col] = value

    # convert the dictionary to a dataframe
    monopolarFirstRankDF = pd.DataFrame(list(monopolarFirstRank.items()), columns=['session_frequencyBand', 'numberOneRank_monopolarChannel'])
    

    # from monoRefDF extract only the row equal to the value of the column 'numberOneRank_monopolarChannel' 
    
    # loop through each #1 rank value and store the matching dataframe row from monoRefDF in a dictionary
    FirstRankRef_dict = {}
    
    for index, value in monopolarFirstRankDF['numberOneRank_monopolarChannel'].iteritems():
        FirstRankRef_dict[f"{index}_{value}"] = monoRefDF[monoRefDF['Unnamed: 0'].str.contains(value)]


    # first make a new Dataframe of selected monoRef rows, by concatenating all values of FirstRankRef_dict
    FirstRankRef_DF = pd.concat(FirstRankRef_dict.values(), keys=FirstRankRef_dict.keys(), ignore_index=True) # keys need to be specified

    # drop the first column with monopolar channel names, because this column already exists in the monopolarFirstRank DF that will be concatenated
    FirstRankRef_DF.drop(columns=FirstRankRef_DF.columns[0], axis=1, inplace=True) # inplace=True will modify the original DF and will not create a new DF

    # now concatenate the FirstRankDF (with #1 ranked monopolar contacts) with the FirstRankRefDF (with all referenced psd values of this #1 ranked contact)
    FirstRankChannel_PSD_DF = pd.concat([monopolarFirstRankDF, FirstRankRef_DF], axis=1)


    ################ SELECT HIGHEST RANK FOR POSTOP AND FU3M BASELINE ################

    # Dataframes for postop Baseline first ranks in each frequency band
    postop_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("postop")
    fu3m_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("fu3m")
    beta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_beta")
    lowBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_lowBeta")
    highBeta_mask = FirstRankChannel_PSD_DF["session_frequencyBand"].str.contains("_highBeta")


    #######    POSTOP BASELINE  #######
    # first select the row from the highest ranked channel in each frequency band
    postopBaseline_beta_channelRow = FirstRankChannel_PSD_DF[postop_mask & beta_mask]
    postopBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & lowBeta_mask]
    postopBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[postop_mask & highBeta_mask]

    # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    postopBaseline_beta = postopBaseline_beta_channelRow.loc[:,postopBaseline_beta_channelRow.columns.str.contains("_beta")]
    postopBaseline_lowBeta = postopBaseline_lowBeta_channelRow.loc[:,postopBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    postopBaseline_highBeta = postopBaseline_highBeta_channelRow.loc[:,postopBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]
    

    #######    FU3M BASELINE  #######
     # first select the row from the highest ranked channel in each frequency band
    fu3mBaseline_beta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & beta_mask]
    fu3mBaseline_lowBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & lowBeta_mask]
    fu3mBaseline_highBeta_channelRow = FirstRankChannel_PSD_DF[fu3m_mask & highBeta_mask]

    # then select the columns with the averaged PSD values of the corresponding frequency band of every session
    fu3mBaseline_beta = fu3mBaseline_beta_channelRow.loc[:,postopBaseline_beta_channelRow.columns.str.contains("_beta")]
    fu3mBaseline_lowBeta = fu3mBaseline_lowBeta_channelRow.loc[:,postopBaseline_lowBeta_channelRow.columns.str.contains("_lowBeta")]
    fu3mBaseline_highBeta = fu3mBaseline_highBeta_channelRow.loc[:,postopBaseline_highBeta_channelRow.columns.str.contains("_highBeta")]
    
    

    return {
        "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
        "PostopBaseline_beta": postopBaseline_beta,
        "PostopBaseline_lowBeta": postopBaseline_lowBeta,
        "PostopBaseline_highBeta": postopBaseline_highBeta,
        "Fu3mBaseline_beta": fu3mBaseline_beta,
        "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
        "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
    }

    


















