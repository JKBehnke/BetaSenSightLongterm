""" Is there a decrease of beta power over time at active stimulation contacts? """

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy
import pingouin as pg
from itertools import combinations
from statannotations.Annotator import Annotator


######### PRIVATE PACKAGES #########
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..stimulation import activeStimulationContacts as active_stim_contacts

RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
FIGURES_PATH = find_folders.get_local_path(folder="GroupFigures")

SESSIONS = ["fu3m", "fu12m", "fu18or24m"]

def hemispheres_active_beta_higher_than_inactive():
    """
    This function counts how many hemispheres show higher beta average at their active stimulation contacts 
    compared to their average beta power at inactive stimulation contacts
    """

    summary_all = pd.DataFrame()
    no_beta = []

    active_and_inactive_contacts_beta = active_stim_contacts.fooof_mono_beta_and_clinical_activity_write_dataframes(
        fooof_version="v2"
    ) # this loads the FOOOF monopolar beta power (inverse sq distance) at active and inactive contacts

    electrode_average_data = active_and_inactive_contacts_beta["electrode_average"]
    sub_hem_unique = electrode_average_data.subject_hemisphere.unique()


    for stn in sub_hem_unique:

        stn_data = electrode_average_data.loc[electrode_average_data.subject_hemisphere == stn]

        for ses in SESSIONS:

            # check if session exists for this stn: 
            if ses not in stn_data.session.values:
                continue

            ses_data = stn_data.loc[stn_data.session == ses]

            active_data = ses_data.loc[ses_data.clinical_activity == "active"]
            inactive_data = ses_data.loc[ses_data.clinical_activity == "inactive"]

            active_mean_beta = active_data.electrode_mean_beta_psd.values[0]
            active_mean_beta_rel_to_max = active_data.electrode_mean_beta_psd_rel_to_rank1.values[0]

            inactive_mean_beta = inactive_data.electrode_mean_beta_psd.values[0]
            inactive_mean_beta_rel_to_max = inactive_data.electrode_mean_beta_psd_rel_to_rank1.values[0]

            # check if no beta values exist:
            if np.isnan(active_mean_beta) or np.isnan(inactive_mean_beta):
                no_beta.append(f"{stn}_{ses}")
                continue


            # check if active mean beta is higher than inactive mean beta for both absolute and relative values
            active_mean_above_inactive_mean = "no"
            rel_active_mean_above_inactive_mean = "no"
            
            if active_mean_beta > inactive_mean_beta:
                active_mean_above_inactive_mean = "yes"
            
            if active_mean_beta_rel_to_max > inactive_mean_beta_rel_to_max:
                rel_active_mean_above_inactive_mean = "yes"

            summary_dict = {
                "subject_hemisphere": [stn],
                "session": [ses],
                "active_mean_beta": [active_mean_beta],
                "active_mean_beta_rel": [active_mean_beta_rel_to_max],
                "inactive_mean_beta": [inactive_mean_beta],
                "inactive_mean_beta_rel": [inactive_mean_beta_rel_to_max],
                "active_mean_beta_above_inactive": [active_mean_above_inactive_mean],
                "active_mean_beta_above_inactive_rel": [rel_active_mean_above_inactive_mean],
            }
            summary_single_stn = pd.DataFrame(summary_dict)
            summary_all = pd.concat([summary_all, summary_single_stn], ignore_index=True)

    # count how many hemispheres show higher beta average mean at active vs. inactive contacts
    # first drop all rows with NaN because they don´t have beta values, collect the stn and session 
    

    
    return summary_all, no_beta


        





