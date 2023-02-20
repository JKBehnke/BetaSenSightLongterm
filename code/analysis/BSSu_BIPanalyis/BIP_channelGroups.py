""" BIP comparisons within Ring, SegmIntra and SegmInter groups """


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns


######### PRIVATE PACKAGES #########
import analysis.classes.mainAnalysis_class as mainAnalysis_class
import analysis.utils.find_folders as find_folders


def PsdAverage_RingSegmGroups(
        incl_sub: list, 
        signalFilter: str,
        normalization: str,
        freqBands: list
        ):
    
    """
    Plot the mean PSD average of all channels of an electrode within a frequency band (alpha, beta, highBeta, lowBeta)
    in 3 seperate groups: Ring (6 channels), SegmIntra (6 channels), SegmInter (3 channels)

    Input:
        - sub: list e.g. ["017", "019", "024", "025", "026", "029", "030"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBands: list e.g. ["beta", "highBeta", "lowBeta"]
    
    1) Load the JSON files in the correct format of all subject hemispheres from MainClass
        - example of filename in sub_results folder: "SPECTROGRAMpsdAverageFrequencyBands_Left_band-pass.json"

    2) select the correct normalization and divide the dataframes into 3 groups: 
        - Ring_DF 
        - SegmIntra_DF
        - SegmInter_DF

        each is a dict 
            - with keys (f"{sub}_{hem}_{ses}_{freq}") 
            - and values (DataFrame with columns: session, bipolarChannel, frequencyBand, absoluteOrRelativePSD, averagedPSD)
            - each Dataframe with length of rows depending on Channels in the group

    3) Calculate the mean of column "averagedPSD" of each Dataframe

    4) 


    """

    figures_path = find_folders.get_local_path(folder="GroupFigures")
    results_path = find_folders.get_local_path(folder="GroupResults")

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    channelGroup = ["Ring", "SegmIntra", "SegmInter"]
    Ring = ['03', '13', '02', '12', '01', '23']
    SegmIntra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C'] 
    SegmInter = ['1A2A', '1B2B', '1C2C']
    
    data = {}
    freqBand_norm_session_DF = {} # keys with f"{sub}_{hem}_{freq}": value = sub_hem DF selected for correct session, normalization and frequency band
    Ring_DF = {}
    SegmIntra_DF = {}
    SegmInter_DF = {}


    ##################### LOAD DATA for all subject hemispheres #####################
    for sub in incl_sub:

        subject_results_path = find_folders.get_local_path(folder="results", sub=sub)

        for hem in hemispheres:

            # load the data from each subject hemisphere
            data[f"{sub}_{hem}"] = mainAnalysis_class.MainClass(
                sub = sub,
                hemisphere = hem,
                filter = signalFilter,
                result = "PSDaverageFrequencyBands",
                incl_session = sessions,
                pickChannels = ['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization = [normalization],
                freqBands = freqBands,
                feature= ["averagedPSD"]
            )


                
            # for each timepoint and frequency band seperately, get the Dataframes of the correct normalization and frequency band
            for ses in sessions:
                for freq in freqBands:

                    # get the Dataframe
                    if ses == "postop":
                        # Error check: check if session exists as an attribute of one subject_hemisphere object
                        try: 
                            data[f"{sub}_{hem}"].postop
                        
                        except AttributeError:
                            continue
                            
                        session_DF = data[f"{sub}_{hem}"].postop.Result_DF # select for session
                        norm_session_DF = session_DF[session_DF.absoluteOrRelativePSD == normalization] # select for correct normalization e.g. absolute PSD instead of relative
                        freqBand_norm_session_DF = norm_session_DF[norm_session_DF.frequencyBand == freq] # select  for correct frequency band
                    
                    # get the Dataframe
                    if ses == "fu3m":
                        # Error check: check if session exists as an attribute of one subject_hemisphere object
                        try: 
                            data[f"{sub}_{hem}"].fu3m
                        
                        except AttributeError:
                            continue
                            
                        session_DF = data[f"{sub}_{hem}"].fu3m.Result_DF # select for session
                        norm_session_DF = session_DF[session_DF.absoluteOrRelativePSD == normalization] # select for absolute PSD instead of relative
                        freqBand_norm_session_DF = norm_session_DF[norm_session_DF.frequencyBand == freq] # select  for beta frequency band
                    
                    # get the Dataframe
                    if ses == "fu12m":
                        # Error check: check if session exists as an attribute of one subject_hemisphere object
                        try: 
                            data[f"{sub}_{hem}"].fu12m
                        
                        except AttributeError:
                            continue
                            
                        session_DF = data[f"{sub}_{hem}"].fu12m.Result_DF # select for session
                        norm_session_DF = session_DF[session_DF.absoluteOrRelativePSD == normalization] # select for absolute PSD instead of relative
                        freqBand_norm_session_DF = norm_session_DF[norm_session_DF.frequencyBand == freq] # select  for beta frequency band
                    
                    # get the Dataframe
                    if ses == "fu18m":
                        # Error check: check if session exists as an attribute of one subject_hemisphere object
                        try: 
                            data[f"{sub}_{hem}"].fu18m
                        
                        except AttributeError:
                            continue
                            
                        session_DF = data[f"{sub}_{hem}"].fu18m.Result_DF # select for session
                        norm_session_DF = session_DF[session_DF.absoluteOrRelativePSD == normalization] # select for absolute PSD instead of relative
                        freqBand_norm_session_DF = norm_session_DF[norm_session_DF.frequencyBand == freq] # select  for beta frequency band


                    ############# divide each Dataframe in 3 groups: Ring, SegmIntra, SegmInter #############
                
                    # Dataframe with all Ring Channels
                    Ring_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.DataFrame()
                    for chan in Ring:
                        channel_DF = freqBand_norm_session_DF[freqBand_norm_session_DF.bipolarChannel.str.contains(chan)] # if e.g. 03 isin bipolarChannel of DF, 
                        Ring_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.concat([Ring_DF[f"{sub}_{hem}_{ses}_{freq}"], channel_DF] ) # add the row to the empty Ring_DF 

                    # Dataframe with all SegmIntra Channels
                    SegmIntra_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.DataFrame()
                    for chan in SegmIntra:
                        channel_DF = freqBand_norm_session_DF[freqBand_norm_session_DF.bipolarChannel.str.contains(chan)]
                        SegmIntra_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.concat([SegmIntra_DF[f"{sub}_{hem}_{ses}_{freq}"], channel_DF])

                    # Dataframe with all SegmInter Channels
                    SegmInter_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.DataFrame()
                    for chan in SegmInter:
                        channel_DF = freqBand_norm_session_DF[freqBand_norm_session_DF.bipolarChannel.str.contains(chan)]
                        SegmInter_DF[f"{sub}_{hem}_{ses}_{freq}"] = pd.concat([SegmInter_DF[f"{sub}_{hem}_{ses}_{freq}"], channel_DF])
    


    ### save the Dataframes with pickle 
    # Ring_DF is a dictionary containing DF for each key f"{sub}_{hem}_{ses}_{freq}"
    Ring_filepath = os.path.join(results_path, f"BIPpsdAverage_Ring_{normalization}_{signalFilter}.pickle")
    with open(Ring_filepath, "wb") as file:
        pickle.dump(Ring_DF, file)
    
    SegmIntra_filepath = os.path.join(results_path, f"BIPpsdAverage_SegmIntra_{normalization}_{signalFilter}.pickle")
    with open(SegmIntra_filepath, "wb") as file:
        pickle.dump(SegmIntra_DF, file)

    SegmInter_filepath = os.path.join(results_path, f"BIPpsdAverage_SegmInter_{normalization}_{signalFilter}.pickle")
    with open(SegmInter_filepath, "wb") as file:
        pickle.dump(SegmInter_DF, file)


    ############# get the AVERAGE of channelGroup of a sub_hem_session_freq combination #############
    # e.g. average of all Ring channels (n=6) of sub-24, Right, postop, beta

    meanPSD_RingChannels_dict = {}
    meanPSD_SegmIntraChannels_dict = {}
    meanPSD_SegmInterChannels_dict = {}

    # list of all existing combinations, e.g. ["024_Right_postop", "024_Right_fu3m"]
    sub_hem_ses_freq_combinations = list(Ring_DF.keys())

    # get the mean from column averagedPSD of each combination for Ring, SegmIntra, SegmInter
    # also get Standard deviation for each sub_hem_ses ???
    for combination in sub_hem_ses_freq_combinations:

        # split the sub_hem_ses_freq combinations into sub_hem (will be label of plot), ses (will be x axis) and freq (select)
        combination_split = combination.split("_")
        subject_hemisphere = "".join([combination_split[0], "_", combination_split[1]]) # e.g. "029_Right" will be used for labels
        tp = combination_split[2] # e.g. "postop" will be used as x axis
        freq_band = combination_split[3] # e.g. "beta"

        # for each group, calculate the mean over the averagedPSD column, add to dictionary together with subject_hemisphere and session 
        meanPSD_RingChannels_dict[combination] = [Ring_DF[combination].averagedPSD.mean(), subject_hemisphere, tp, freq_band]
        meanPSD_SegmIntraChannels_dict[combination] = [SegmIntra_DF[combination].averagedPSD.mean(), subject_hemisphere, tp, freq_band]
        meanPSD_SegmInterChannels_dict[combination] = [SegmInter_DF[combination].averagedPSD.mean(), subject_hemisphere, tp, freq_band]


    # new Dataframes for averaged PSD over all Channels per group Ring, SegmIntra, SegmInter
    Ring_DF_meanChannels = pd.DataFrame.from_dict(meanPSD_RingChannels_dict, orient="index", columns=["PSDaverage_ChannelsPerHemisphere", "subject_hemisphere", "session", "freqBand"]) # keys of the dictionary will be indeces
    #Ring_DF_meanChannels.session.replace("postop", "fu0m", inplace=True)

    SegmIntra_DF_meanChannels = pd.DataFrame.from_dict(meanPSD_SegmIntraChannels_dict, orient="index", columns=["PSDaverage_ChannelsPerHemisphere", "subject_hemisphere", "session", "freqBand"]) # keys of the dictionary will be indeces
    #SegmIntra_DF_meanChannels.session.replace("postop", "fu0m", inplace=True)

    SegmInter_DF_meanChannels = pd.DataFrame.from_dict(meanPSD_SegmInterChannels_dict, orient="index", columns=["PSDaverage_ChannelsPerHemisphere", "subject_hemisphere", "session", "freqBand"]) # keys of the dictionary will be indeces
    #SegmInter_DF_meanChannels.session.replace("postop", "fu0m", inplace=True)



    ############# Divide the meanChannel dataframes in different freq band subgroups #############
    meanChannelRing_freqBand = {}
    meanChannelSegmIntra_freqBand = {}
    meanChannelSegmInter_freqBand = {}


    for f in freqBands:
        meanChannelRing_freqBand[f"{f}"] = Ring_DF_meanChannels[Ring_DF_meanChannels.freqBand == f]
        meanChannelSegmIntra_freqBand[f"{f}"] = SegmIntra_DF_meanChannels[SegmIntra_DF_meanChannels.freqBand == f]
        meanChannelSegmInter_freqBand[f"{f}"] = SegmInter_DF_meanChannels[SegmInter_DF_meanChannels.freqBand == f]

    ############# Plot the average of each Channel group at every session timepoint #############

    # Create a list of 15 colors and add it to the cycle of matplotlib 
    cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
    plt.rc('axes', prop_cycle=cycler_colors)
    
    
    for f in freqBands:
        fig = plt.figure(figsize=(5, 10), layout="tight")
        channelGroup_DF = {}

        for g, group in enumerate(channelGroup):
            if group == "Ring":
                channelGroup_DF[group] = meanChannelRing_freqBand[f]
                
            elif group == "SegmIntra":
                channelGroup_DF[group] = meanChannelSegmIntra_freqBand[f]
                
            elif group == "SegmInter":
                channelGroup_DF[group] = meanChannelSegmInter_freqBand[f]
        
            plt.subplot(3,1,g+1)

            #sns.lineplot(data=channelGroup_DF[group], x='session', y='PSDaverage_beta_ChannelsPerHemisphere', size_order=sessions)
    
            sns.boxplot(data=channelGroup_DF[group], 
                        x='session', 
                        y='PSDaverage_ChannelsPerHemisphere', 
                        order=sessions, 
                        palette="Blues", 
                        width=0.8,
                        )
            # whis = whiskers are makers to define outliers
            sns.stripplot(x = "session",
                y = 'PSDaverage_ChannelsPerHemisphere',
                hue = "subject_hemisphere",
                order=sessions,
                size=5,
                jitter=True,
                data = channelGroup_DF[group],
                )
            
            plt.title(f"{group} Channels: PSD in {f} band")
            plt.legend(loc= "upper right", bbox_to_anchor=(1.4, 1))
    
            # y label depends on normalization:
            if normalization == "rawPsd":
                ylabel = "mean absolute PSD [uV^2/Hz]"
            
            elif normalization == "normPsdToTotalSum":
                ylabel = "mean rel. PSD to total sum[%]"
            
            elif normalization == "normPsdToSum1_100Hz":
                ylabel = "mean rel. PSD to 1-100 Hz[%]"

            elif normalization == "normPsdToSum40_90Hz":
                ylabel = "mean rel. PSD to 40_90Hz[%]"
            
            else:
                print(normalization, "has to be in ['rawPsd', 'normPsdToTotalSum', 'normPsdToSum1_100Hz', 'normPsdToSum40_90Hz']")


            plt.ylabel(ylabel)


        fig.tight_layout()

        fig.savefig(figures_path + f"\\BIP_PsdAverage_RingSegmGroups_{f}_{normalization}_{signalFilter}.png")
        
    
    return {
        "meanChannelRing_freqBand": meanChannelRing_freqBand,


    }
                    





