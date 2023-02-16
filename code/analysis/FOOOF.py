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
import PerceiveImport.classes.main_class as mainclass
import PerceiveImport.methods.find_folders as findfolders
import analysis.loadResults as loadcsv


import fooof
# Import required code for visualizing example models
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model



def FOOOF(incl_sub: list, psdMethod: str, incl_hemisphere: list):
    """

    Input: 
        - incl_sub: list e.g. ["024"]
        - psdMethod: str "Welch" or "Spectrogram"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - incl_contact: a list of contacts to include ["RingR", "SegmIntraR", "SegmInterR", "RingL", "SegmIntraL", "SegmInterL"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - incl_hemisphere: list e.g. ["Right", "Left"]
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    1) Load csv file from results folder for each subject

    2) 

    """


    # dictionary to fill in for each subject
    subject_dictionary = {}

    # define variables 
    time_points = ['postop', 'fu3m', 'fu12m', 'fu18m']
    frequency_range = ["lowBeta", "highBeta", "beta"]

    

    ################### Load csv file from results folder for each subject ###################
    for sub, subject in enumerate(incl_sub):
        for hem, hemisphere in enumerate(incl_hemisphere):

            # get path to results folder of each subject
            local_results_path = findfolders.get_local_path(folder="results", sub=subject)

            # get result CSV from each subject
            df = loadcsv.load_PSDresultCSV(
                sub=subject,
                psdMethod=psdMethod,
                normalization=normalization,
                hemisphere=hemisphere
                )
            
            subject_dictionary[f"sub{subject}_{hemisphere}"] = [subject, local_results_path, df]
        


    ################### DIVIDE EACH DATAFRAME INTO COLLECTIONS OF EACH SESSION TIMEPOINT ###################



    return {
        "subject_dictionary": subject_dictionary, 
        }
    

    



