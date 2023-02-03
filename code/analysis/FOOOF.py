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
import analysis.loadCSVresults as loadcsv


import fooof
# Import required code for visualizing example models
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model



def FOOOF(incl_sub: str, psdMethod: str, normalization: str, hemisphere: str, incl_session: list, incl_condition: list, incl_contact: list, pickChannels: list):
    """

    Input: 
        - incl_sub: str e.g. "024"
        - psdMethod: str "Welch" or "Spectrogram"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - incl_contact: a list of contacts to include ["RingR", "SegmIntraR", "SegmInterR", "RingL", "SegmIntraL", "SegmInterL"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"

    """


    # dictionary to fill in for each subject
    subject_dictionary = {}

    
    for sub, subject in enumerate(incl_sub):

        # get path to results folder of each subject
        local_results_path = findfolders.get_local_path(folder="results", sub=subject)

        # get result CSV from each subject
        df = loadcsv.load_resultCSV(
            sub=subject,
            psdMethod=psdMethod,
            normalization=normalization,
            hemisphere=hemisphere
            )
        
        subject_dictionary[f"{subject}_path_csvDF"] = [subject, local_results_path, df]
    

    

