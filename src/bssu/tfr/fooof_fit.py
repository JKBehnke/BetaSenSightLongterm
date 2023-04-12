""" FOOOF Model """


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scipy
from scipy.signal import spectrogram, hann, butter, filtfilt, freqz

import sklearn
from sklearn.preprocessing import normalize

import json
import os
import mne

# PyPerceive Imports
import py_perceive
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults


import fooof
# Import required code for visualizing example models
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model



def FOOOF(incl_sub: list, normalization: str):
    """

    Input: 
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
      
    1) Load the Power Spectrum from main Class:
        - unfiltered
        - rawPSD (not normalized)
        - all channels: Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']


    2) 

    """

    # define variables 
    hemispheres = ["Right", "Left"]
    time_points = ['postop', 'fu3m', 'fu12m', 'fu18m']
    frequency_range = ["alpha", "lowBeta", "highBeta", "beta", "gamma"]

    

    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    for sub, subject in enumerate(incl_sub):

        # get path to results folder of each subject
            local_figures_path = findfolders.get_local_path(folder="figures", sub=subject)


        for hem, hemisphere in enumerate(hemispheres):

            # get power spectrum and frequencies from each STN
            data_power_spectrum = mainAnalysis_class.MainClass(
                sub=subject,
                hemisphere=hemisphere,
                filter="unfiltered",
                result="PowerSpectrum",
                incl_session=["postop", "fu3m", "fu12m", "fu18m"],
                pickChannels=['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization=["rawPsd"],
                feature=["frequency", "time_sectors", "rawPsd", "SEM_rawPsd"]
            )
            
            
            #subject_dictionary[f"sub{subject}_{hemisphere}"] = [subject, local_results_path, df]
        


    ################### DIVIDE EACH DATAFRAME INTO COLLECTIONS OF EACH SESSION TIMEPOINT ###################



    return {
        #"subject_dictionary": subject_dictionary, 
        }
    

    



