""" PSD analysis per bipolar channel of BSSu """

import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

import seaborn as sns


######### PRIVATE PACKAGES #########
from ..classes import mainAnalysis_class
from ..utils import find_folders


def BIP_channelNormalizedToSession(
        incl_sub: list,
        normalization: str,
        freqBand: str,
        plot: str

):

    """
    Analysis of PSD in frequency Band of each bipolar channel across STNs, normalized to Postop or 3MFU


    Using classes to extract the PSD values 

    Input: 
        - incl_sub: list, e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - normalization: str, e.g. "rawPsd"
        - freqBand: str, e.g. "beta"
        - plot: str, e.g. "plotPerSTN", "plotPerBipolarChannel"




    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    hemispheres = ["Right", "Left"]
    sessions_postop = ["fu3m", "fu12m", "fu18m"]
    sessions_fu3m = ["fu12m", "fu18m"]
    BIP_channels = ['BIP_03', 'BIP_13', 'BIP_02', 'BIP_12', 'BIP_01', 'BIP_23', 
                    'BIP_1A1B', 'BIP_1B1C', 'BIP_1A1C', 'BIP_2A2B', 'BIP_2B2C', 'BIP_2A2C',
                    'BIP_1A2A', 'BIP_1B2B', 'BIP_1C2C']
    


    ####################### NORMALIZE PSD TO POSTOP #######################
    normalizedToPostop_psd_Dict = {}

    for sub in incl_sub:
        for hem in hemispheres:

            sub_hem_data = mainAnalysis_class.MainClass(
                sub=sub,
                hemisphere=hem,
                filter="band-pass",
                result="PSDaverageFrequencyBands",
                incl_session=["postop", "fu3m", "fu12m", "fu18m"],
                pickChannels=['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization=[normalization],
                freqBands=["beta", "lowBeta", "highBeta"],
                feature=["averagedPSD"]
            )

            # for every channel get PSD for each session and normalize to postop
            for c, chan in enumerate(BIP_channels):

                # check if postop exists 
                try:
                    getattr(sub_hem_data, "postop")
                
                except AttributeError:
                    continue

                # first get postop PSD
                postop_PSD = getattr(sub_hem_data.postop, chan)
                postop_PSD = getattr(postop_PSD, normalization)
                postop_PSD = getattr(postop_PSD, freqBand)
                postop_PSD = postop_PSD.averagedPSD.data

                normalized_postop = postop_PSD/postop_PSD

                # store in dictionary
                normalizedToPostop_psd_Dict[f"{sub}_{hem}_{chan}_postop"] = [sub, hem, "postop", chan, normalized_postop]

                # normalize all other session PSD values to postop PSD
                for ses in sessions_postop:

                    # check if session exists
                    try: 
                        getattr(sub_hem_data, ses)
                    
                    except AttributeError:
                        continue

                    session_PSD = getattr(sub_hem_data, ses)
                    session_PSD = getattr(session_PSD, chan)
                    session_PSD = getattr(session_PSD, normalization)
                    session_PSD = getattr(session_PSD, freqBand)
                    session_PSD = session_PSD.averagedPSD.data

                    normalized_session_PSD = session_PSD/postop_PSD
                    
                    # store in dictionary
                    normalizedToPostop_psd_Dict[f"{sub}_{hem}_{chan}_{ses}"] = [sub, hem, ses, chan, normalized_session_PSD]

    # make a dataframe out of the dictionary, store sub, hem, ses, chan and normalized PSD into columns
    normalizedToPostop_psd_Dataframe = pd.DataFrame(normalizedToPostop_psd_Dict)
    normalizedToPostop_psd_Dataframe.rename(index={0: "subject", 1: "hemisphere", 2: "session", 3: "bipolarChannel", 4: f"{freqBand}_Psd_normalized_to_postop"}, inplace=True)
    normalizedToPostop_psd_Dataframe = normalizedToPostop_psd_Dataframe.transpose()
    
    # add column subject_hem to get info per STN
    normalizedToPostop_psd_Dataframe["subject_hemisphere"] = normalizedToPostop_psd_Dataframe[["subject", "hemisphere"]].agg('_'.join, axis=1)

    # add also a column for each single bipolar channel from each STN
    normalizedToPostop_psd_Dataframe["subject_hemisphere_BIPchannel"] = normalizedToPostop_psd_Dataframe[["subject", "hemisphere", "bipolarChannel"]].agg('_'.join, axis=1)

    normalizedToPostop_psd_Dataframe.drop(["subject", "hemisphere", "bipolarChannel"], axis=1, inplace=True)


    ####################### Plot the normalized PSD over sessions, two versions depending on Input -> plot #######################
    if plot == "plotPerSTN":

        fig_1 = plt.figure()

        ax = sns.barplot(data=normalizedToPostop_psd_Dataframe, x="session", y=f"{freqBand}_Psd_normalized_to_postop", hue="subject_hemisphere", palette="viridis")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
        plt.title("Relative PSD to postop per bipolar channel across STNs")
        plt.ylabel("relative PSD to postop per bipolar channel")
        plt.ylim(0, 30)

    elif plot == "plotPerBipolarChannel":

        fig_1 = plt.figure()

        ax = sns.barplot(data=normalizedToPostop_psd_Dataframe, x="session", y="beta_Psd_normalized_to_postop", hue="subject_hemisphere_BIPchannel", palette="rocket")
        # plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
        ax.get_legend().remove() # legend is too large, so take it out
        plt.title("Relative PSD to postop per bipolar channel across STNs")
        plt.ylabel("relative PSD to postop per bipolar channel")
        plt.ylim(0, 60)


    
    fig_1.savefig(figures_path + f"\\relativeTo_Postop_{normalization}_{freqBand}_barplot_{plot}.png", bbox_inches="tight")




    ####################### NORMALIZE PSD TO 3MFU #######################
    normalizedToFu3m_psd_Dict = {}

    for sub in incl_sub:
        for hem in hemispheres:

            sub_hem_data = mainAnalysis_class.MainClass(
                sub=sub,
                hemisphere=hem,
                filter="band-pass",
                result="PSDaverageFrequencyBands",
                incl_session=["fu3m", "fu12m", "fu18m"],
                pickChannels=['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization=[normalization],
                freqBands=["beta", "lowBeta", "highBeta"],
                feature=["averagedPSD"]
            )

            # for every channel get PSD for each session and normalize to postop
            for c, chan in enumerate(BIP_channels):

                # check if postop exists 
                try:
                    getattr(sub_hem_data, "fu3m")
                
                except AttributeError:
                    continue

                # first get postop PSD
                fu3m_PSD = getattr(sub_hem_data.fu3m, chan)
                fu3m_PSD = getattr(fu3m_PSD, normalization)
                fu3m_PSD = getattr(fu3m_PSD, freqBand)
                fu3m_PSD = fu3m_PSD.averagedPSD.data

                normalized_fu3m = fu3m_PSD/fu3m_PSD

                # store in dictionary
                normalizedToFu3m_psd_Dict[f"{sub}_{hem}_{chan}_fu3m"] = [sub, hem, "fu3m", chan, normalized_fu3m]

                # normalize all other session PSD values to fu3m PSD
                for ses in sessions_fu3m:

                    # check if session exists
                    try: 
                        getattr(sub_hem_data, ses)
                    
                    except AttributeError:
                        continue

                    session_PSD = getattr(sub_hem_data, ses)
                    session_PSD = getattr(session_PSD, chan)
                    session_PSD = getattr(session_PSD, normalization)
                    session_PSD = getattr(session_PSD, freqBand)
                    session_PSD = session_PSD.averagedPSD.data

                    normalized_session_PSD = session_PSD/fu3m_PSD
                    
                    # store in dictionary
                    normalizedToFu3m_psd_Dict[f"{sub}_{hem}_{chan}_{ses}"] = [sub, hem, ses, chan, normalized_session_PSD]


    # make a Dataframe out of the dictionary
    normalizedToFu3m_psd_Dataframe = pd.DataFrame(normalizedToFu3m_psd_Dict)
    normalizedToFu3m_psd_Dataframe.rename(index={0: "subject", 1: "hemisphere", 2: "session", 3: "bipolarChannel", 4: f"{freqBand}_Psd_normalized_to_fu3m"}, inplace=True)
    normalizedToFu3m_psd_Dataframe = normalizedToFu3m_psd_Dataframe.transpose()

    # add columns sub_hem and sub_hem_chan
    normalizedToFu3m_psd_Dataframe["subject_hemisphere"] = normalizedToFu3m_psd_Dataframe[["subject", "hemisphere"]].agg('_'.join, axis=1)
    normalizedToFu3m_psd_Dataframe["subject_hemisphere_BIPchannel"] = normalizedToFu3m_psd_Dataframe[["subject", "hemisphere", "bipolarChannel"]].agg('_'.join, axis=1)
    normalizedToFu3m_psd_Dataframe.drop(["subject", "hemisphere", "bipolarChannel"], axis=1, inplace=True)

    ####################### Plot the normalized PSD over sessions, two versions depending on Input -> plot #######################
    if plot == "plotPerSTN":

        fig_2 = plt.figure()

        ax = sns.barplot(data=normalizedToFu3m_psd_Dataframe, x="session", y=f"{freqBand}_Psd_normalized_to_fu3m", hue="subject_hemisphere", palette="viridis")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
        plt.title("Relative PSD to 3MFU per bipolar channel across STNs")
        plt.ylabel("relative PSD to 3MFU per bipolar channel")
        plt.ylim(0, 30)
    
    elif plot == "plotPerBipolarChannel":

        fig_2 = plt.figure()

        ax = sns.barplot(data=normalizedToFu3m_psd_Dataframe, x="session", y="beta_Psd_normalized_to_fu3m", hue="subject_hemisphere_BIPchannel", palette="rocket")
        # plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
        ax.get_legend().remove()

        plt.title("Relative PSD to 3MFU per bipolar channel across STNs")
        plt.ylabel("relative PSD to 3MFU per bipolar channel")
        plt.ylim(0, 60)

    fig_2.savefig(figures_path + f"\\relativeTo_Fu3m_{normalization}_{freqBand}_barplot_{plot}.png", bbox_inches="tight")



    # save Dataframes
    normalizedToPostop_psd_filepath = os.path.join(results_path, f"normalizedToPostop_{normalization}_{freqBand}.pickle")
    with open(normalizedToPostop_psd_filepath, "wb") as file:
        pickle.dump(normalizedToPostop_psd_Dataframe, file)

    normalizedToFu3m_psd_filepath = os.path.join(results_path, f"normalizedToFu3m_{normalization}_{freqBand}.pickle")
    with open(normalizedToFu3m_psd_filepath, "wb") as file:
        pickle.dump(normalizedToFu3m_psd_Dataframe, file)

    
    # active_and_inactive_MonoBeta8Ranks.to_json(os.path.join(results_path, f"ClinicalActiveVsNonactiveContacts_{freqBand}_{rank_or_psd}.json"))

    
    print("new files: ", f"normalizedToPostop_{normalization}_{freqBand}.pickle",
          f"\nand: normalizedToFu3m_{normalization}_{freqBand}.pickle",
          "\nwritten in: ", results_path,
          f"\nnew figure: relativeTo_Fu3m_{normalization}_{freqBand}_barplot_{plot}.png",
          "\nwritten in: ", figures_path)

    

    return {
        "normalizedToPostop_psd_Dataframe": normalizedToPostop_psd_Dataframe,
        "normalizedToFu3m_psd_Dataframe": normalizedToFu3m_psd_Dataframe, 
    }









