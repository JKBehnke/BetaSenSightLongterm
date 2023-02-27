""" Permutation tests of ranks """


import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns


######### PRIVATE PACKAGES #########
import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults


def PermutationTest_BIPchannelGroups(
        data2permute: str,
        filterSignal: str,
        normalization: str,
        freqBand: str,
        ):
    
    """
    Perform a permutation test 

    Input: 
        - data2permute: str e.g. "psdAverage",  "peak"
        - filterSignal: str e.g. "band-pass"
        - normalization: str e.g. "rawPsd"
        - freqBand: str e.g. "beta"
    
    Watchout: only works for "ranks" as data2permute so far!!!!

    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # 3 comparisons for each channel group
    comparisons = ["Postop_Fu3m", "Fu3m_Fu12m", "Fu12m_Fu18m"]
    channelGroups = ["Ring", "SegmInter", "SegmIntra"]
    Ring_channels = ["12", "01", "23"]
    SegmIntra_channels = ["1A1B", "1B1C", "1A1C", "2A2B", "2B2C", "2A2C",]
    SegmInter_channels = ["1A2A", "1B2B", "1C2C"]


    ################# LOAD THE COMPARISON DATAFRAMES #################
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
            

            # columns from the dataframe to shuffle 
            rank_x = list(comp_group_DF.rank_x.values)
            rank_y = list(comp_group_DF.rank_x.values)


            ################# CREATE RANDOMLY SHUFFLED ARRAYS OF RANK-X AND RANK-Y #################

            # shuffle repetitions: 1000 times
            numberOfShuffle = np.arange(1, 100001, 1)

            # list of differences between rank_x and rank_y
            difference_random_ranks = []

            # repeat shuffle 1000 times
            for s, shuffle in enumerate(numberOfShuffle):

                np.random.shuffle(rank_x)
                np.random.shuffle(rank_y)

                # calculate the mean of the difference between random rank_x and rank_y, store in list
                difference_random_ranks.append(np.mean(abs(np.array(rank_x) - np.array(rank_y))))

            
            # plot the distribution of randomized difference MEAN values
            axes[g].hist(difference_random_ranks,bins=100)
            # mark with red line: real mean of the rank differences of comp_group_DF
            axes[g].axvline(mean_difference, c="r")

            axes[g].set_title(f"{group} channels", fontdict=fontdict)

            # calculate the distance of the real mean from the mean of all randomized means divided by the standard deviation
            distanceMeanReal_MeanRandom = (mean_difference - np.mean(difference_random_ranks) / np.std(difference_random_ranks))

            # calculate the p-value 
            pval = 2-2*norm.cdf(abs(distanceMeanReal_MeanRandom))
            # TODO: save more decimals in pval, only 0.0 is shown...


            # store all values in dictionary
            Permutation_BIP[f"{comp}_{group}"] = [comp, group, mean_difference, distanceMeanReal_MeanRandom, pval]
        
        for ax in axes:

            ax.set_xlabel("distribution of randomized MEAN values of rank differences", fontsize=25)
            ax.set_ylabel("Count", fontsize=25)

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
    





            


            













    