""" BSSu SSD filter"""


import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import numpy as np

import scipy
from scipy import signal
from scipy.signal import spectrogram, hann, butter, filtfilt, freqz

import sklearn
from sklearn.preprocessing import normalize

import json
import pickle
import os
import mne

# PyPerceive Imports
import PerceiveImport.classes.main_class as mainclass

# local analysis functions
import analysis.utils.find_folders as find_folders
import analysis.BSSu_processing.feats_ssd as feats_ssd


def SSD_filter_groupChannels(
        incl_sub=list, 
        f_band=str,
):

    """
    Loads unfiltered raw time domain data with PyPerceive

    Input:
        - incl_sub = ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - f_range = tuple e.g. beta: 13, 35
    
    1) Load the BSSu rest Time domain data of each existing subject, hemisphere, session, and recording group (e.g. right: "RingR", "SegmIntraR", "SegmInterR")
        - in the Ring Group only choose bipolar recording montages with same distances between contacts (01, 12, 23) --> caveat: probably still not comparable because segmented contacts have higher impedance than circular contacts

    2) apply the SSD filter by using feats_ssd.get_SSD_component() function from Jeroen (based on Gunnar Waterstraat's meet Toolbox)
        for each recording group seperately!

        Input
            - data_2d: 2d array of channels x time domain data
            - fband_interest: tuple f_range e.g. 13, 35 (beta band)
            - s_rate: 250 Hz usually for BSSu
            - use_freqBand_filtered=True (uses only the filtered frequency band, not the flanks)
            - return_comp_n=0 -> only the first component is returned, which is the most important component
    
    3) Plot: 
        - first plot (axes[0]): for each recording group the unfiltered raw Power Spectra of all included bipolar channels
        - second plot (axes[1]): the Powewr Spectrum of the most important first component (should be a clear Peak in the given f_band of interest)
    
    4) Dataframe returned and saved: columns 
        - subject
        - hemisphere
        - session
        - recording_group (e.g. RingR)
        - bipolarChannel
        - ssd_filtered_timedomain (array)
        - ssd_pattern (weights of a channel contributing to the first component)
        - ssd_eigvals
    
    TODO: 
        - add the weights of each channel in the plots
        - add new functionality to filter SSD for all 15 bipolar channels?? useful??

    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    # load the PSD data with classes
    hemisphere =["Right", "Left"]
    incl_session=["postop", "fu3m", "fu12m", "fu18m"]
    incl_cond=["m0s0"]

    incl_contact={}
    rec_group_input_array={}
    SSD_results_storage = {}

    # set the frequency range depending on the f_band of interest
    if f_band == "beta":
        f_range = [13,35]

    elif f_band == "lowBeta":
        f_range = [13,20]
    
    elif f_band == "highBeta":
        f_range = [21,35]
    


    for sub in incl_sub:

        sub_results_path = find_folders.get_local_path(folder="figures", sub=sub) 

        for hem in hemisphere:
            if hem == "Right":
                incl_contact["Right"] = ["RingR", "SegmIntraR", "SegmInterR"]

            elif hem == "Left":
                incl_contact["Left"] = ["RingL", "SegmIntraL", "SegmInterL"]

            mainclass_sub = mainclass.PerceiveData(
                sub = sub, 
                incl_modalities= ["survey"],
                incl_session = incl_session,
                incl_condition = incl_cond,
                incl_task = ["rest"],
                incl_contact=incl_contact[f"{hem}"]
                )

            for ses in incl_session:
                for cond in incl_cond:
                    for recording_group in incl_contact[f"{hem}"]:

                        # check if session exists 
                        try: 
                            getattr(mainclass_sub.survey, ses)
                        except AttributeError:
                            continue

                        timedomain_data = getattr(mainclass_sub.survey, ses)
                        

                        # check if condition exists
                        try:
                            getattr(timedomain_data, cond)
                        except AttributeError:
                            continue

                        timedomain_data = getattr(timedomain_data, cond)
                        timedomain_data = getattr(timedomain_data.rest, recording_group) # for each rec group in incl_contact
                        timedomain_data = timedomain_data.run1.data

                        # channel names (list) and sampling frequency
                        ch_names = timedomain_data.ch_names
                        sfreq = timedomain_data.info["sfreq"]

                        # Time domain arrays of all channels in a recording group                
                        rec_group_data = timedomain_data.get_data() # 2D array of all rec channels in a rec group

                        # for Ring contacts only choose channels with adjacent contacts (01,12,23)
                        if (recording_group == "RingR" or recording_group == "RingL"):

                            rec_group_data = rec_group_data[[3,4,5], :]
                            ch_names = ch_names[3:6]


                        rec_group_input_array[f"{sub}_{hem}_{ses}_{cond}_{recording_group}"] = rec_group_data

                        # apply SSD filter on time domain of each recording group
                        (ssd_filt_data, ssd_pattern, ssd_eigvals 
                                    ) = feats_ssd.get_SSD_component( 
                                        data_2d=rec_group_data, 
                                        fband_interest=f_range, 
                                        s_rate=sfreq, 
                                        use_freqBand_filtered=True, 
                                        return_comp_n=0, ) 
                        
                        # store SSD outcome in Dataframe
                        for c, chan in enumerate(ch_names):
                            SSD_results_storage[f"{sub}_{hem}_{ses}_{chan}"] = [sub, hem, ses, recording_group, chan, ssd_filt_data, ssd_pattern[0][c], ssd_eigvals[c]]
                            # ssd_filt_data is ssd filtered time domain of a channel
                            # ssd_pattern[0] is weight of a channel, contributing to FIRST component 

                        # Figure with 2 subplots (2 rows, 1 column)
                        fig, axes = plt.subplots(2, 1, figsize=(10,15))
                        
                        # plot the PSD channels in a recording group
                        for c, chan in enumerate(ch_names):

                            f_raw, psd_raw = signal.welch(rec_group_data[c], axis=-1, nperseg=sfreq, fs=sfreq) 

                            axes[0].plot(f_raw, psd_raw, label=f"{chan}_ssd_pattern_{ssd_pattern[0][c]}")        
                            axes[0].set_title(f'Power Spectra {recording_group}: sub{sub} {hem} hemisphere, {ses}')
                            axes[0].set_xlim(0, 100) 
                            axes[0].set_xlabel("Frequency")
                            axes[0].set_ylabel("PSD [uV^2/Hz]")
                            axes[0].axvline(x=13, color='black', linestyle='--')
                            axes[0].axvline(x=35, color='black', linestyle='--')
                            axes[0].legend()


                        # use Welch to transform the time domain data to frequency domain data            
                        f, psd = signal.welch(ssd_filt_data, axis=-1, nperseg=sfreq, fs=sfreq) # ssd_filt_data is only one array with the PSD of the FIRST component
                        

                        # plot the first component Power spectrum of each recording group
                        axes[1].plot(f, psd) 
                                    
                        axes[1].set_xlim(0, 100)   
                        axes[1].set_xlabel("Frequency")
                        axes[1].set_title(f'Power Spectrum of first component: sub{sub} {hem} hemisphere, {ses}, {recording_group}') 
                        axes[1].axvline(x=13, color='black', linestyle='--')
                        axes[1].axvline(x=35, color='black', linestyle='--')

                        fig.savefig(sub_results_path + f"\\sub-{sub}_{hem}_SSD_firstComponent_PowerSpectrum_{ses}_{recording_group}")

                        plt.close()

                    

    SSD_results_Dataframe = pd.DataFrame(SSD_results_storage)
    SSD_results_Dataframe.rename(index={0:"subject", 1:"hemisphere", 2:"session", 3:"recording_group", 4:"bipolarChannel", 5:"ssd_filtered_timedomain", 6:"ssd_pattern", 7:"ssd_eigvals"}, inplace=True)
    SSD_results_Dataframe = SSD_results_Dataframe.transpose()

    ### save the Dataframes with pickle 
    SSD_results_Dataframe_filepath = os.path.join(results_path, f"SSD_results_Dataframe_{f_band}.pickle")
    with open(SSD_results_Dataframe_filepath, "wb") as file:
        pickle.dump(SSD_results_Dataframe, file)
    
    print("file: ", 
          f"SSD_results_Dataframe_{f_band}.pickle",
          "\nwritten in: ", results_path)

    return SSD_results_Dataframe


        
    

