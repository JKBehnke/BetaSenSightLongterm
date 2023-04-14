 # load the json file
import pandas as pd
import json
import os
import numpy as np

import scipy.stats as st
from mne.stats import permutation_cluster_test

from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults



def cluster_permutation_power_spectra_betw_sesssions(
        session_1: str,
        session_2: str
):
    
    """
    
    """

    results_path = find_folders.get_local_path(folder="GroupResults")
    filenames = [f'cluster_permutation_{session_1}_to_{session_2}.json', f'cluster_permutation_{session_2}_to_{session_1}.json', ]

    max_freq = 90

    results = []
    for filename in filenames:
        with open(os.path.join(results_path_sub, filename)) as file:
            json_data = json.load(file)
        
        result_df = pd.DataFrame(json_data)    
        results.append(np.vstack(result_df['power_spectrum_average_segm_inter'].values)[:,:max_freq])
    
        
    F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(results)
        
        
        
        
    
    
    
    
    
    
    
