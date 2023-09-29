""" Comparisons between monopolar estimation methods """


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import mne
import pickle

# internal Imports
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults



results_path = find_folders.get_local_path(folder="GroupResults")
figures_path = find_folders.get_local_path(folder="GroupFigures")

incl_sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]
segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    

################## method weighted by euclidean coordinates ##################

monopolar_fooof_euclidean_segmental = loadResults.load_fooof_monopolar_weighted_psd(
    fooof_spectrum = "periodic_spectrum",
    segmental = "yes",
    similarity_calculation = "inverse_distance"
)

monopolar_fooof_euclidean_segmental = pd.concat([
    monopolar_fooof_euclidean_segmental["postop_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu3m_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu12m_monopolar_Dataframe"],
    monopolar_fooof_euclidean_segmental["fu18or24m_monopolar_Dataframe"],])

# columns: coord_z, coord_xy, session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank


################## method by JLB ##################

monopolar_fooof_JLB = loadResults.load_fooof_monopolar_JLB()
# columns: session, subject_hemisphere, estimated_monopolar_beta_psd, contact, rank



def spearman_monopol_fooof_beta_methods(
        method_1:str,
        method_2:str
):
    
    """
    Spearman correlation between monopolar beta power estimations between 2 methods

    Input: define methods to compare
        - method_1: "JLB", "euclidean", "Strelow"
        - method_2: "JLB", "euclidean", "Strelow"
    """
    
    # get data from method 1
    if method_1 == "JLB":
        method_1_data = monopolar_fooof_JLB
            
    elif method_1 == "euclidean":
        method_1_data = monopolar_fooof_euclidean_segmental
    
    elif method_1 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # get data from method 2
    if method_2 == "JLB":
        method_2_data = monopolar_fooof_JLB
            
    elif method_2 == "euclidean":
        method_2_data = monopolar_fooof_euclidean_segmental
    
    elif method_2 == "Strelow":
        print("Strelow method only gives optimal contacts with beta ranks 1-3")

    # Perform spearman correlation for every session separately and within each STN
    for ses in incl_sessions:

        method_1_session = method_1_data.loc[method_1_data.session == ses]
        method_2_session = method_2_data.loc[method_2_data.session == ses]





        





