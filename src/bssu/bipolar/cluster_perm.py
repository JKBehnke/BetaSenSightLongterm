 # load the json file
import pandas as pd
import os
import numpy as np
import pickle

import scipy.stats as st
import mne
from mne.stats import permutation_cluster_test

from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults



def cluster_permutation_power_spectra_betw_sesssions(
        incl_channels:str,
        signalFilter:str
):
    
    """
    Load the file f"power_spectra_{signalFilter}_{incl_channels}_session_comparisons.pickle"
    from the group result folder

    Input:
        - incl_channels: str, e.g. "SegmInter", "SegmIntra", "Ring"
        - signalFilter: str, e.g. "band-pass" or "unfiltered" 
    
    1) Get the Dataframes of each session for each session comparison 
        - within each session comparison -> only STNs are included, that have recordings at both sessions (same sample size per comparison)
    
    2) for each comparison make a list with two arrays, one per session
        - use np.vstack to create one big array with power spectra vectors per session
        - only include power up until maximal frequency (90 Hz) by indexing
        - shape of one array/one session = (n_observations, )
    
    3) perform cluster permutation per session comparison:
        - comparisons: ["postop_fu3m", "postop_fu12m", "postop_fu18m", 
                        "fu3m_fu12m", "fu3m_fu18m", "fu12m_fu18m"]
        - number of Permutations = 1000
        
    Output of mne.stats.permutation_cluster_test:
        - F_obs, shape (p[, q][, r]): 
            Statistic (F by default) observed for all variables.

        - clusterslist: 
            List type defined by out_type above.

        - cluster_pvarray
            P-value for each cluster.

        - H0array, shape (n_permutations,)
            Max cluster level stats observed under permutation.


    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    # load all session comparisons
    loaded_session_comparisons = loadResults.load_power_spectra_session_comparison(
        incl_channels=incl_channels,
        signalFilter=signalFilter)
    
    # get the dataframe for each session comparison
    compare_sessions = ["postop_fu3m", "postop_fu12m", "postop_fu18m", 
                        "fu3m_fu12m", "fu3m_fu18m", "fu12m_fu18m"]
    
    # maximal frequency to perform cluster permutation
    max_freq = 90

    permutation_results = {}
    
    for comparison in compare_sessions:

        two_sessions = comparison.split("_")
        session_1 = two_sessions[0] # e.g."postop"
        session_2 = two_sessions[1] # e.g. "fu3m"

        comparison_df = loaded_session_comparisons[f"{comparison}_df"] 

        # only get part of df with session_1 and df with session_2
        session_1_df = comparison_df.loc[comparison_df.session==session_1]
        session_2_df = comparison_df.loc[comparison_df.session==session_2]

        # from each session df only take the values from column with power spectra only until maximal frequency
        x_session_1 = np.vstack(session_1_df['power_spectrum'].values)[:,:max_freq]
        # e.g. for shape x_18mfu comparison to 12mfu = (10, 90), 10 STNs, 90 values per STN
        x_session_2 = np.vstack(session_2_df['power_spectrum'].values)[:,:max_freq]

        list_for_cluster_permutation = [x_session_1, x_session_2]

        # perform cluster permutation
        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(list_for_cluster_permutation, n_permutations=1000)

        # get the sample size
        sample_size = len(session_1_df.power_spectrum.values)

        # save results
        permutation_results[f"{comparison}"] = [comparison, F_obs, clusters, cluster_pv, H0, sample_size]

    results_df = pd.DataFrame(permutation_results)
    results_df.rename(index={
        0: "session_comparison",
        1: "F_obs",
        2: "clusters",
        3: "cluster_pv",
        4: "H0",
        5: "sample_size"
    }, inplace=True)
    results_df = results_df.transpose()

    # save the results DF into results path
    results_df_filepath = os.path.join(results_path, f"cluster_permutation_session_comparisons_{incl_channels}_{signalFilter}.pickle")
    with open(results_df_filepath, "wb") as file:
        pickle.dump(results_df, file)
    
    print("file: ", 
          f"cluster_permutation_session_comparisons_{incl_channels}_{signalFilter}.pickle",
          "\nwritten in: ", results_path
          )

    return results_df
        
        
        
        
    
    
    
    
    
    
    
