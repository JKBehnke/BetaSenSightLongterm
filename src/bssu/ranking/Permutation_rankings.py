""" Permutation tests of ranks """


import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statistics
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import plotly.express as px

import itertools
import seaborn as sns


######### PRIVATE PACKAGES #########
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults


def PermutationTest_BIPchannelGroups(
        data2permute: str,
        filterSignal: str,
        normalization: str,
        freqBand: str,
        ):
    
    """
    Perform a permutation test 
    data2permute will choose which file will be loaded: "psdAverage" will load "BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle"
        - for all channel groups: Ring, SegmIntra, SegmInter
        - rawPsd and band-pass filtered data defined in this code

    Watchout: only works for "ranks" as data2permute so far!!!!
        
    Input: 
        - data2permute: str e.g. "psdAverage",  "peak"
        - filterSignal: str e.g. "band-pass"
        - normalization: str e.g. "rawPsd"
        - freqBand: str e.g. "beta"
    
    1) Load the comparison dataframes: e.g. BIPpermutationDF_Fu12m_Fu18m_psdAverage_beta_rawPsd_band-pass.pickle

    2) for each comparison 
                    ["Postop_Postop", "Postop_Fu3m", "Postop_Fu12m", "Postop_Fu18m", 
                   "Fu3m_Postop", "Fu3m_Fu3m", "Fu3m_Fu12m", "Fu3m_Fu18m", 
                   "Fu12m_Postop", "Fu12m_Fu3m", "Fu12m_Fu12m", "Fu12m_Fu18m",
                   "Fu18m_Postop", "Fu18m_Fu3m", "Fu18m_Fu12m", "Fu18m_Fu18m"]

        and for each group ("Ring", "SegmInter", "SegmIntra")
        calculate the MEAN difference of ranks over all STNs
    
    3) loop over each STN and shuffle ranks from session x and session y
        - number of shuffle = 1000
        - calculate the absolute difference between ranks for each BIP recording
        - calculate the MEAN of abs differences for each shuffle and store in a list difference_random_MEANranks
    
    4) Statistics:
        calculate the distance of the REAL mean from the mean of all randomized means divided by the standard deviation
        - distanceMeanReal_MeanRandom = (mean_difference - np.mean(difference_random_MEANranks)) / np.std(difference_random_MEANranks)

        calculate the p-value 
        - (pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung)
        - pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist
    
        
    5) Plot the distribution of the permutated MEAN values (should be normally distributed)
        - mark a red line for the REAL MEAN
        - annotation with the p value
        - one figure for each comparison: 3 subplots for 3 channel groups
    
    6) save a Dataframe with statistics results
        - "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"
        - columns: comparison, channelGroup, MEAN_differenceOfRanks, distanceMEANreal_MEANrandom, p-value
    

    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # comparisons for each channel group
    # comparisons = ["Postop_Fu3m", "Fu3m_Fu12m", "Fu12m_Fu18m", "Postop_Fu12m", "Postop_Fu18m", "Fu3m_Fu18m"]
    
    comparisons = ["Postop_Postop", "Postop_Fu3m", "Postop_Fu12m", "Postop_Fu18m", 
                   "Fu3m_Postop", "Fu3m_Fu3m", "Fu3m_Fu12m", "Fu3m_Fu18m", 
                   "Fu12m_Postop", "Fu12m_Fu3m", "Fu12m_Fu12m", "Fu12m_Fu18m",
                   "Fu18m_Postop", "Fu18m_Fu3m", "Fu18m_Fu12m", "Fu18m_Fu18m"]

    channelGroups = ["Ring", "SegmInter", "SegmIntra"]
    Ring_channels = ["12", "01", "23"]
    SegmIntra_channels = ["1A1B", "1B1C", "1A1C", "2A2B", "2B2C", "2A2C",]
    SegmInter_channels = ["1A2A", "1B2B", "1C2C"]


    ################# LOAD THE COMPARISON DATAFRAMES #################
    # load files: "BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle"
    BIPcomparison_data = loadResults.load_BIPpermutationComparisonsPickle(
        result=data2permute,
        freqBand=freqBand,
        normalization=normalization,
        filterSignal=filterSignal
    )

    fontdict = {"size": 25}

    # store 
    Permutation_BIP = {}

    for comp in comparisons:
        
        # Figure Layout per comparison: 3 rows (Ring, SegmIntra, SegmInter), 1 column
        fig, axes = plt.subplots(3,1,figsize=(10,15)) 

        for g, group in enumerate(channelGroups):

            comp_group_DF = BIPcomparison_data[comp][group]

            # calculate the mean of a difference of ranks
            mean_difference = comp_group_DF["Difference_rank_x_y"].mean()
            
            ################# CREATE RANDOMLY SHUFFLED ARRAYS OF RANK-X AND RANK-Y #################

            # list of mean differences between shuffled rank_x and rank_y
            difference_random_MEANranks = []

            # shuffle within STNs!! first get unique list of all STNs within the dataframe
            STN_channel_list = list(comp_group_DF["sub_hem_BIPchannel"].unique())
            STN_list = [] # list of unique STNs
            STN_data = {} # Dataframes of each unique STN

            for STN_channel in STN_channel_list:
                split = STN_channel.split("_")
                STN = "_".join([split[0], split[1]]) # e.g. "024_Right"
                STN_list.append(STN)

            STN_list = list(set(STN_list)) # set() gets the unique values from a list
            STN_list.sort() # sorts the strings 

            # shuffle repetitions: 1000 times
            numberOfShuffle = np.arange(1, 1001, 1)

            # repeat shuffle 1000 times
            for s, shuffle in enumerate(numberOfShuffle):
                difference_random_STN_ranks = [] # list with all differences, will contain lists, so later on need to merge lists to one list

                for STN in STN_list:
                    all_STNs = comp_group_DF.copy()
                    STN_data = all_STNs.loc[(all_STNs["sub_hem_BIPchannel"].str.contains(STN))] # Dataframe only of one STN

                    rank_x = list(STN_data.rank_x.values)
                    rank_y = list(STN_data.rank_y.values)

                    # shuffle within one STN
                    np.random.shuffle(rank_x)
                    np.random.shuffle(rank_y)

                    # calculate the difference between random rank_x and rank_y, store in list
                    difference_random_STN_ranks.append(list(abs(np.array(rank_x) - np.array(rank_y))))

                # one merged list with all differences of one shuffle
                difference_random_STN_ranks = list(itertools.chain.from_iterable(difference_random_STN_ranks)) 

                # store the MEAN of differences of one shuffle in list
                difference_random_MEANranks.append(np.mean(difference_random_STN_ranks))


            # calculate the distance of the real mean from the mean of all randomized means divided by the standard deviation
            distanceMeanReal_MeanRandom = (mean_difference - np.mean(difference_random_MEANranks)) / np.std(difference_random_MEANranks)

            # calculate the p-value 
            # pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung
            pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist
            

            # store all values in dictionary
            Permutation_BIP[f"{comp}_{group}"] = [comp, group, mean_difference, distanceMeanReal_MeanRandom, "{:.15f}".format(pval)]
        


            # plot the distribution of randomized difference MEAN values
            # axes[g].hist(difference_random_ranks,bins=100)

            # make the normal distribution fit of the data
            # mu, std = norm.fit(difference_random_ranks)
            # xmin, xmax = plt.xlim()
            # x = np.linspace(xmin,xmax,100)
            # p = norm.pdf(x, mu, std)
            # axes[g].plot(x, p, 'b', linewidth= 2)

            sns.histplot(difference_random_MEANranks, color="tab:blue", ax=axes[g], stat="count", element="bars", label="1000 Permutation repetitions", kde=True, bins=30, fill=True)

            # mark with red line: real mean of the rank differences of comp_group_DF
            axes[g].axvline(mean_difference, c="r")
            axes[g].text(mean_difference +0.02, 50, 
             "Mean difference between \nranks of both sessions \n\n p-value: {:.2f}".format(pval),
             c="r", fontsize=15)

            axes[g].set_title(f"{group} channels", fontdict=fontdict)

        for ax in axes:

            ax.set_xlabel(f"MEAN Difference between {freqBand} ranks", fontsize=25)
            ax.set_ylabel("Count", fontsize=25)
            #ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.0), fontsize=15)
            
            # if group == "Ring":
            #     ax.set_xlim(0,1.3)
            
            # elif group == "SegmInter":
            #     ax.set_xlim(0,1.3)
            
            # elif group =="SegmIntra":
            #     ax.set_xlim(0, 2.6)


            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)
        
        fig.suptitle(f"Permutation analysis: {comp} comparisons", fontsize=30)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.show()

        fig.savefig(figures_path + f"\\PermutationAnalysis_BIP_{comp}_{data2permute}_{freqBand}_{normalization}_{filterSignal}.png")


    # Permutation_BIP transform from dictionary to Dataframe
    Permutation_BIP_DF = pd.DataFrame(Permutation_BIP)
    Permutation_BIP_DF.rename(index={0: "comparison", 1: "channelGroup", 2: "MEAN_differenceOfRanks", 3: "distanceMEANreal_MEANrandom", 4: "p-value"}, inplace=True)
    Permutation_BIP_DF = Permutation_BIP_DF.transpose()

    ## save the Permutation Dataframes with pickle 
    Permutation_BIP_filepath = os.path.join(results_path, f"Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(Permutation_BIP_filepath, "wb") as file:
        pickle.dump(Permutation_BIP_DF, file)

    print("file: ", 
          f"Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle",
          "\nwritten in: ", results_path
          )
    

    return Permutation_BIP_DF
    


def heatmap_distances_to_permutated_mean(
        data2permute:str,
        filterSignal:str,
        normalization:str,
        freqBand:str
):
    """
    
    Input:
        - data2permute= e.g. "psdAverage",
        - filterSignal= e.g. "band-pass",
        - normalization= e.g. "rawPsd",
        - freqBand= e.g. "beta"
    
    Load the pickle file written with function PermutationTest_BIPchannelGroups() in Permutation_rankings.py 
        filename: "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle" 


    """

    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # load the Permutation results
    permutation_results = loadResults.load_BIPpermutation_ranks_result(
        data2permute=data2permute,
        filterSignal=filterSignal,
        normalization=normalization,
        freqBand=freqBand
        )
    
    # restructure the dataframe for a heatmap
    # filter for each channel group

    channel_group = ["Ring", "SegmInter", "SegmIntra"]

    for group in channel_group:

        group_df = permutation_results.loc[permutation_results.channelGroup==group]

        distance_to_plot = group_df["p-value"].values.astype(float)
        distance_to_plot = distance_to_plot.reshape(4,4)

        # fig = px.imshow(distance_to_plot,
        #                 labels=dict(x="session 1", y="session 2", color=f"p-value of difference real vs. random"),
        #                 x=['postop', 'fu3m', 'fu12m', 'fu18m'],
        #                 y=['postop', 'fu3m', 'fu12m', 'fu18m'],
        #                 title=f"{group} channel group, p-values {freqBand} power ranks",
        #                 text_auto=False)

        # # fig = px.imshow(distance_to_plot,
        # #                 labels=dict(x="session 1", y="session 2", color=f"distance between real and permutated mean differences"),
        # #                 x=['postop', 'fu3m', 'fu12m', 'fu18m'],
        # #                 y=['postop', 'fu3m', 'fu12m', 'fu18m'],
        # #                 title=f"{group} channel group, difference of {freqBand} power ranks",
        # #                 text_auto=True)
        # fig.update_xaxes(side="top")
        # fig.update_layout(title={
        #     'y':0.98,
        #     'x':0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'})
        
        # fig.show()

         # plot a heatmap
        fig, ax = plt.subplots()

        heatmap = ax.pcolor(distance_to_plot, cmap=plt.cm.YlOrRd)
        # other color options: GnBu, YlOrRd, YlGn, Greys, Blues, PuBuGn, YlGnBu

        # Set the x and y ticks to show the indices of the matrix
        ax.set_xticks(np.arange(distance_to_plot.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(distance_to_plot.shape[0])+0.5, minor=False)

        # Set the tick labels to show the values of the matrix
        ax.set_xticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)
        ax.set_yticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)

        
        # Add a colorbar to the right of the heatmap
        cbar = plt.colorbar(heatmap)
        cbar.set_label(f"p-value of difference real vs. random")

        # Add the cell values to the heatmap
        for i in range(distance_to_plot.shape[0]):
            for j in range(distance_to_plot.shape[1]):
                plt.text(j + 0.5, i + 0.5, str("{: .2f}".format(distance_to_plot[i, j])), ha='center', va='center') # only show 2 numbers after the comma of a float

        # Add a title
        plt.title(f"{group} channel group, p-values {freqBand} power ranks")

        fig.tight_layout()
        fig.savefig(figures_path + f"\\heatmap_distances_to_permutated_mean_{group}_{freqBand}_{data2permute}_{normalization}_{filterSignal}", bbox_inches="tight")

            


def permutation_fooof_beta_ranks(
        fooof_spectrum:str,
        ):
    
    """
    Perform a permutation test 
    
        - for all channel groups: Ring, SegmIntra, SegmInter
        - and all session comparisons

        
    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

    
    1) Load the fooof beta rank dataframes: e.g. beta_ranks_all_channels_fooof_periodic_spectrum.pickle

    2) for each comparison 
                    ["postop_postop", "postop_fu3m", "postop_fu12m", "postop_fu18m", 
                   "fu3m_postop", "fu3m_fu3m", "fu3m_fu12m", "fu3m_fu18m", 
                   "fu12m_postop", "fu12m_fu3m", "fu12m_fu12m", "fu12m_fu18m",
                   "fu18m_postop", "fu18m_fu3m", "fu18m_fu12m", "fu18m_fu18m"]

        and for each group ["ring", "segm_inter", "segm_intra"]
        
        - per STN:  calculate the MEAN difference of ranks 
        - get average of all STN MEAN differences of ranks
    
    3) loop over each STN and shuffle ranks from session x and session y
        - number of shuffle = 1000
        - calculate the absolute difference between ranks for each BIP recording
        - calculate the MEAN of abs differences for each shuffle and store in a list difference_random_MEANranks
    
    4) Statistics:
        calculate the distance of the REAL mean from the mean of all randomized means divided by the standard deviation
        - distanceMeanReal_MeanRandom = (mean_difference - np.mean(difference_random_MEANranks)) / np.std(difference_random_MEANranks)

        calculate the p-value 
        - (pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung)
        - pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist
    
        
    5) Plot the distribution of the permutated MEAN values (should be normally distributed)
        - mark a red line for the REAL MEAN
        - annotation with the p value
        - one figure for each comparison: 3 subplots for 3 channel groups
    
    6) save a Dataframe with statistics results
        - "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"
        - columns: comparison, channelGroup, MEAN_differenceOfRanks, distanceMEANreal_MEANrandom, p-value
    

    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # load FOOOF beta rank DF
    beta_rank_DF = loadResults.load_fooof_beta_ranks(
        fooof_spectrum=fooof_spectrum,
        all_or_one_chan="beta_ranks_all"
        )
    
    # new column with stn and channel info combined
    beta_rank_DF_copy = beta_rank_DF.copy()
    beta_rank_DF_copy["stn_channel"] = beta_rank_DF_copy.subject_hemisphere.values + "_" + beta_rank_DF_copy.bipolar_channel.values

    compare_sessions = ["postop_postop", "postop_fu3m", "postop_fu12m", "postop_fu18m", 
                   "fu3m_postop", "fu3m_fu3m", "fu3m_fu12m", "fu3m_fu18m", 
                   "fu12m_postop", "fu12m_fu3m", "fu12m_fu12m", "fu12m_fu18m",
                   "fu18m_postop", "fu18m_fu3m", "fu18m_fu12m", "fu18m_fu18m"]

    channel_groups = ["ring", "segm_inter", "segm_intra"]


    ##########################    WRITE COMPARISON DATAFRAMES PER SESSION COMPARISON  ##########################
    # for each session comparison, get STNs that have recordings at both sessions
    # subtract rank_session1 from rank_session2 and take the absolute value

    comparisons_storage = {}
    sample_size_dict = {}

    for group in channel_groups:

        if group == "ring":
            channels = ['01', '12', '23']
        
        elif group == "segm_inter":
            channels = ["1A2A", "1B2B", "1C2C"]
        
        elif group == "segm_intra":
            channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

        # dataframe only of one channel group
        group_df = beta_rank_DF_copy.loc[beta_rank_DF_copy["bipolar_channel"].isin(channels)]

        for comparison in compare_sessions:

            two_sessions = comparison.split("_")
            session_1 = two_sessions[0]
            session_2 = two_sessions[1]

            # Dataframe per session
            session_1_df = group_df.loc[(group_df["session"]==session_1)]
            session_2_df = group_df.loc[(group_df["session"]==session_2)]

            # list of STNs per session
            session_1_stns = list(session_1_df.subject_hemisphere.unique())
            session_2_stns = list(session_2_df.subject_hemisphere.unique())

            # list of STNs included in both sessions
            STN_list = list(set(session_1_stns) & set(session_2_stns))
            STN_list.sort()

            # get the rows with STNs with both sessions
            comparison_df_1 = session_1_df.loc[session_1_df["subject_hemisphere"].isin(STN_list)]
            comparison_df_2 = session_2_df.loc[session_2_df["subject_hemisphere"].isin(STN_list)]

            # subtract ranks from each other row by row from session 1 to session 2
            abs_difference_ranks = np.absolute(comparison_df_1.beta_rank.values - comparison_df_2.beta_rank.values) # array with differences of ranks of one session comparison
            sample_size = len(abs_difference_ranks)

            comparison_df_merged = comparison_df_1.merge(comparison_df_2, left_on="stn_channel", right_on="stn_channel")
            comparison_df_merged_copy = comparison_df_merged.copy()
            comparison_df_merged_copy["abs_difference_ranks"] = abs_difference_ranks

            sample_size_dict[f"{group}_{comparison}"] = [group, comparison, sample_size]
            comparisons_storage[f"{group}_{comparison}"] = comparison_df_merged_copy



    ##########################      GET MEAN OF RANK DIFFERENCES PER SESSION COMPARISON AND SHUFFLE RANKS   ##########################
    # 1) get the mean first of all channels within one STN and then across STNs
    # 2) permute within STNs, get mean within STNs and then across permuted STNs

    permutation_fooof_beta_ranks = {}

    # shuffle repetitions: 1000 times
    number_of_shuffle = np.arange(1, 1001, 1)
    fontdict = {"size": 25}

    for comp in compare_sessions:
        
        # Figure Layout per comparison: 3 rows (Ring, SegmIntra, SegmInter), 1 column
        fig, axes = plt.subplots(3,1,figsize=(10,15)) 

        for g, group in enumerate(channel_groups):

            # Dataframe of one comparison and one channel group
            comp_group_DF = comparisons_storage[f"{group}_{comp}"]

            # list of available STNs
            stn_list = list(comp_group_DF["subject_hemisphere_x"].unique())

            # list with all mean differences per comparison and channel group
            stn_real_mean_differences = []

            for stn in stn_list:

                # store the mean of differences per stn
                stn_real_mean = np.mean(comp_group_DF.abs_difference_ranks.values)
                stn_real_mean_differences.append(stn_real_mean)
            
            # real mean of all stns
            mean_comp_group = np.mean(stn_real_mean_differences)
            std_comp_group = np.std(stn_real_mean_differences)
            sample_size = len(stn_real_mean_differences) # number of STNs in one session comparison 
            

            ############ SHUFFLE ############
            # list of 1000x mean differences between shuffled rank_x and rank_y
            all_shuffled_mean_differences = []

            # repeat shuffle 1000 times
            for s, shuffle in enumerate(number_of_shuffle):
                # will contain mean differences per stn of one shuffles, so later on need to merge lists to one list
                difference_random_stn_mean_ranks = [] 

                for STN in stn_list:

                    stn_data = comp_group_DF.loc[comp_group_DF.subject_hemisphere_x == STN]

                    rank_x = list(stn_data.beta_rank_x.values)
                    rank_y = list(stn_data.beta_rank_y.values)

                    # shuffle within one STN
                    np.random.shuffle(rank_x)
                    np.random.shuffle(rank_y)

                    # calculate the difference between random rank_x and rank_y, store in list
                    difference_random_STN_ranks = list(abs(np.array(rank_x) - np.array(rank_y)))

                    # get mean of differences of one stn
                    difference_random_stn_mean_ranks.append(np.mean(difference_random_STN_ranks))

                # store the MEANs of differences of one shuffle in list
                all_shuffled_mean_differences.append(np.mean(difference_random_stn_mean_ranks))

        
            ############ CALCULATE DISTANCE AND P-VAL OF REAL MEAN FROM MEAN OF ALL RANDOMIZED MEANS  ############
            distance_real_vs_random_mean = (mean_comp_group - np.mean(all_shuffled_mean_differences)) / np.std(all_shuffled_mean_differences)

            # calculate the p-value 
            # pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung
            pval = 1-norm.cdf(abs(distance_real_vs_random_mean)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist
            
            sample_size_shuffled = len(all_shuffled_mean_differences)

            # store all values in dictionary
            permutation_fooof_beta_ranks[f"{comp}_{group}"] = [comp, group, sample_size, std_comp_group, mean_comp_group, distance_real_vs_random_mean, "{:.15f}".format(pval), sample_size_shuffled]
            

            ############ PLOT ############
            sns.histplot(all_shuffled_mean_differences, color="tab:blue", ax=axes[g], stat="count", element="bars", label="1000 Permutation repetitions", kde=True, bins=30, fill=True)

            # mark with red line: real mean of the rank differences of comp_group_DF
            axes[g].axvline(mean_comp_group, c="r")
            axes[g].text(mean_comp_group +0.02, 50, 
                "Mean difference between \nranks of both sessions \n\n p-value: {:.2f}".format(pval),
                c="r", fontsize=15)

            axes[g].set_title(f"{group} channels", fontdict=fontdict)

        for ax in axes:

            ax.set_xlabel(f"MEAN Difference between beta ranks", fontsize=25)
            ax.set_ylabel("Count", fontsize=25)

            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)

        fig.suptitle(f"Permutation analysis: {comp} comparisons", fontsize=30)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()

        fig.savefig(figures_path + f"\\permutation_beta_ranks_fooof_spectra_{comp}.png", bbox_inches="tight")

        
    # Permutation_BIP transform from dictionary to Dataframe
    permutation_result_df = pd.DataFrame(permutation_fooof_beta_ranks)
    permutation_result_df.rename(index={
        0: "comparison", 
        1: "channel_group", 
        2: "sample_size_stn",
        3: "standard_deviation_real_different_ranks",
        4: "mean_real_different_ranks",
        5: "distance_real_vs_random_mean_different_ranks",
        6: "p-value",
        7: "sample_size_random_shuffles"
        }, 
        inplace=True)
    permutation_result_df = permutation_result_df.transpose()

    ## save the Permutation Dataframes with pickle 
    Permutation_BIP_filepath = os.path.join(results_path, f"permutation_beta_ranks_fooof_spectra.pickle")
    with open(Permutation_BIP_filepath, "wb") as file:
        pickle.dump(permutation_result_df, file)

    print("file: ", 
            f"permutation_beta_ranks_fooof_spectra.pickle",
            "\nwritten in: ", results_path
            )

    
    return {
        "comparisons_storage":comparisons_storage,
        "permutation_result_df":permutation_result_df
    }


def Permutation_monopolarRanks_compareMethods(
        incl_sub: list,
        freqBand: str,

):
    """

    Permutation analysis of differences between two methods resulting in beta ranks for directional contacts (1A,1B,1C,2A,2B,2C)

    Input: 
        - incl_sub: list of all subjects
        - freqBand: str e.g. "beta"
    
    1) Load and restructure Dataframes from 
        - Robert's method: monoRef_weightPsdAverageByCoordinateDistance.py and 
        - Johannes method: MonoRef_JLB.py
    
    2) Calculate the Difference between ranks of both methods for each directional contact and the MEAN of differences

    3) Create a normally distributed permutation of ranks by shuffling the ranks extracted from one method
        - 1000 repetitions
        - shuffle existing ranks twice and get the difference between two shuffled ranks
        - get the mean of differences of all shuffled pairs
    
    4) Plot the normally distributed shuffled Means of differences 
        - Mark with red line the real Mean of differences between the two methods
        - calculate the distance from the real Mean to the random mean and divide by the standard deviation
        - with this distance calculate the p-value: 
            zweiseitig: pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom))
            oder einseitig: pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom))


    """

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    data_weightedByCoordinates = {}
    keys_weightedByCoordinates = {}
    JLB_mono = {}
    comparisonDataframe = {}
    

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")


    ###################### LOAD ROBERTS AND JOHANNES METHOD RESULTS ######################


    for sub in incl_sub:

        for hem in hemispheres:

            data_weightedByCoordinates[f"{sub}_{hem}"] = loadResults.load_monoRef_weightedPsdCoordinateDistance_pickle(
                sub=sub,
                hemisphere=hem,
                freqBand=freqBand,
                normalization="rawPsd",
                filterSignal="band-pass"
                )
            
            # to check, which sessions exist
            keys_weightedByCoordinates[f"{sub}_{hem}"] = data_weightedByCoordinates[f"{sub}_{hem}"].keys()

            
            data_JLB_mono_directional = loadResults.load_monoRef_JLB_pickle(
                sub=sub,
                hemisphere=hem,
                normalization="rawPsd",
                filterSignal="band-pass"
                )
            
            # get monopolar ranks and filter only for correct freqBand in column
            JLB_ranks = data_JLB_mono_directional["monopolar_psdRank"].filter(like=f"_{freqBand}", axis=1) 

            # copy in order to modify
            JLB_ranks_copy = JLB_ranks.copy()

            # add column subject_hemisphere_monoChannel
            JLB_ranks_copy["subject_hemisphere"] = f"{sub}_{hem}"
            JLB_ranks_copy["monopolarChannels"] = np.array(["1A", "1B", "1C", "2A", "2B", "2C"])
            JLB_ranks_copy["subject_hemisphere_monoChannel"] = JLB_ranks_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)
            JLB_ranks_copy.drop(["subject_hemisphere", "monopolarChannels"], axis=1, inplace=True)

            # store in dictionary
            JLB_mono[f"{sub}_{hem}"] = JLB_ranks_copy



            for ses in sessions:

                # first check, if session exists in keys
                if f"{ses}_monopolar_Dataframe" in keys_weightedByCoordinates[f"{sub}_{hem}"]:
                    print(f"{sub}_{hem}_{ses}")
                
                else:
                    continue
                
                # get the dataframe per session
                session_weightedByCoordinates = data_weightedByCoordinates[f"{sub}_{hem}"][f"{ses}_monopolar_Dataframe"]

                # choose only directional contacts and rank again only the directional contacts
                session_weightedByCoordinates = session_weightedByCoordinates.loc[["1A", "1B", "1C", "2A", "2B", "2C"]]
                session_weightedByCoordinates["directionalRank"] = session_weightedByCoordinates["averaged_monopolar_PSD_beta"].rank(ascending=False)
                session_weightedByCoordinates_copy = session_weightedByCoordinates.copy()

                # add column subject_hemisphere_monoChannel
                session_weightedByCoordinates_copy["monopolarChannels"] = np.array(["1A", "1B", "1C", "2A", "2B", "2C"])
                session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)
                session_weightedByCoordinates_copy.drop(["subject_hemisphere", "monopolarChannels", "rank", "coord_z", "coord_xy"], axis=1, inplace=True)


                # merge together Roberts session Dataframe and Johannes dataframe
                compare_DF = JLB_mono[f"{sub}_{hem}"].merge(session_weightedByCoordinates_copy, left_on="subject_hemisphere_monoChannel", right_on="subject_hemisphere_monoChannel")

                # add new column and calculate Difference between ranks
                compare_DF["Difference_ranks"] = (compare_DF[f"monoRef_{ses}_beta"] - compare_DF["directionalRank"]).apply(abs)

                # store finished Dataframe in dictionary
                comparisonDataframe[f"{sub}_{hem}_{ses}"] = compare_DF


    
    # from every Dataframe only keep 3 relevant columns: "subject_hemisphere_monoChannel", "session", "Difference_ranks"
    comparison_sub_hem_ses = list(comparisonDataframe.keys())

    Dataframe_all_sub_hem_ses = pd.DataFrame()

    for c, comp in enumerate(comparison_sub_hem_ses):
        
        combination_DF = comparisonDataframe[comp]
        filtered_combination_DF = combination_DF[["subject_hemisphere_monoChannel", "session", "directionalRank", "Difference_ranks"]]

        # concatenate all Dataframes to one
        Dataframe_all_sub_hem_ses = pd.concat([Dataframe_all_sub_hem_ses, filtered_combination_DF], ignore_index=True)


    ###################### CALCULATE THE MEAN OF ALL DIFFERENCES ######################

    Mean_differences_monoRanks_JLB_vs_weightedCoordinates = Dataframe_all_sub_hem_ses["Difference_ranks"].mean()



    ###################### PERMUTATION TEST ######################


    # column to shuffle
    directional_rank_x = list(Dataframe_all_sub_hem_ses.directionalRank.values)
    directional_rank_y = list(Dataframe_all_sub_hem_ses.directionalRank.values)

    # shuffle repetitions: 
    numberOfShuffle = np.arange(1, 1001, 1)

    # list of mean differences between shuffled rank_x and rank_y
    difference_random_ranks = []

    # repeat shuffle 1000 times
    for s, shuffle in enumerate(numberOfShuffle):

        np.random.shuffle(directional_rank_x)
        np.random.shuffle(directional_rank_y)

        # calculate the mean of the difference between random rank_x and rank_y, store in list: 1000 MEAN values after all
        difference_random_ranks.append(np.mean(abs(np.array(directional_rank_x) - np.array(directional_rank_y))))


    # calculate the distance of the real mean from the mean of all randomized means divided by the standard deviation
    distanceMeanReal_MeanRandom = (Mean_differences_monoRanks_JLB_vs_weightedCoordinates - np.mean(difference_random_ranks)) / np.std(difference_random_ranks)

    # calculate the p-value 
    # pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom)) # zweiseitige Berechnung
    pval = 1-norm.cdf(abs(distanceMeanReal_MeanRandom)) # einseitige Berechnung: wieviele Standardabweichungen der Real Mean vom randomized Mean entfernt ist
    

    # store all Permutation values
    Permutation_monopolarMethods = [Mean_differences_monoRanks_JLB_vs_weightedCoordinates, distanceMeanReal_MeanRandom, "{:.15f}".format(pval)]
    # Permutation_BIP transform from dictionary to Dataframe
    Permutation_monopolarMethods = pd.DataFrame(Permutation_monopolarMethods)
    Permutation_monopolarMethods.rename(index={0: "MEAN_differenceRanks_JLB_vs_weightedByCoordinates", 1: "distanceMeanReal_MeanRandom", 2: "p-value"}, inplace=True)
    Permutation_monopolarMethods = Permutation_monopolarMethods.transpose()

    fig = plt.figure()
    fontdict = {"size": 25} 

    # plot the distribution of randomized difference MEAN values
    sns.histplot(difference_random_ranks, color="dodgerblue", stat="count", label="1000 Permutation repetitions", kde=True, bins=40)

    # mark with red line: real mean of the rank differences of comp_group_DF
    plt.axvline(Mean_differences_monoRanks_JLB_vs_weightedCoordinates, c="r")
    plt.text(Mean_differences_monoRanks_JLB_vs_weightedCoordinates +0.02, 2, 
             "Mean difference between \nranks of both methods \n\n p-value: {:.2f}".format(pval),
             c="r")



    plt.title(f"Difference between two methods: \nbeta psd ranks of directional contacts ", fontdict=fontdict)

    plt.legend(loc="upper center")

    plt.xlabel("MEAN difference of ranks", fontsize=25)
    plt.ylabel("Count", fontsize=25)
    #ax.set_ylim(0,25)

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)

    fig.tight_layout()

    fig.savefig(figures_path + f"\\Permutation_monopolarMethods_{freqBand}_rawPsd_band-pass.png", bbox_inches="tight")


    ## save all Permutation Dataframes with pickle 
    Permutation_results = {
        "comparisonDataframe_per_sub_hem_ses": comparisonDataframe,
        "Dataframe_all_sub_hem_ses": Dataframe_all_sub_hem_ses,
        "Permutation_monopolarMethods": Permutation_monopolarMethods
    }

    Permutation_monopolardirectionalRanks_filepath = os.path.join(results_path, f"Permutation_monopolarMethods_{freqBand}_rawPsd_band-pass.pickle")
    with open(Permutation_monopolardirectionalRanks_filepath, "wb") as file:
        pickle.dump(Permutation_results, file)

    print("file: ", 
          f"Permutation_monopolarMethods_{freqBand}_rawPsd_band-pass.pickle",
          "\nwritten in: ", results_path
          )



            
    return {
        "comparisonDataframe_per_sub_hem_ses": comparisonDataframe,
        "Dataframe_all_sub_hem_ses": Dataframe_all_sub_hem_ses,
        "Permutation_monopolarMethods": Permutation_monopolarMethods
    }
