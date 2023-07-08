""" Movement artifact cleaning before computing power spectra """


import os

import matplotlib.pyplot as plt
import mplcursors
import mpld3
import mne
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
from scipy.signal import hann


# PyPerceive Imports
# import py_perceive
from PerceiveImport.classes import main_class
from .. utils import find_folders as findfolders


def plot_raw_time_series(incl_sub: list, incl_session: list, incl_condition: list, hemisphere: str, filter: str):
    """

    Input: 
        - incl_sub: list e.g. ["024"]
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        
        - hemisphere: str e.g. "Right"
        - filter: str "unfiltered", "band-pass"

    
    1) load data from main_class.PerceiveData using the input values.

    2) pick channels
    
    
   

    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "normPsdToTotalSumDataFrame":normToTotalSumPsdDataFrame,
        "normPsdToSum1_100Hz": normToSum1_100Hz,
        "normPsdToSum40_90Hz":normToSum40_90Hz,
        "psdAverage_dict": psdAverage_dict,
        "highestPeakRawPSD": highestPeakRawPsdDF,
    }
    
    """

    # sns.set()
    plt.style.use('seaborn-whitegrid')  

    # depending on hemisphere: define incl_contact
    incl_contact = {}
    if hemisphere == "Right":
        incl_contact["Right"] = ["RingR", "SegmIntraR", "SegmInterR"]
    
    elif hemisphere == "Left":
        incl_contact["Left"] = ["RingL", "SegmIntraL", "SegmInterL"]
    
    channel_groups = ["ring", "segm_inter", "segm_intra"]


    for sub in incl_sub:
        mainclass_sub = main_class.PerceiveData(
            sub = sub, 
            incl_modalities= ["survey"],
            incl_session = incl_session,
            incl_condition = incl_condition,
            incl_task = ["rest"],
            incl_contact=incl_contact[f"{hemisphere}"]
            )

        
        figures_path = findfolders.get_local_path(folder="figures", sub=incl_sub)
        results_path = findfolders.get_local_path(folder="results", sub=incl_sub)

        # add error correction for sub and task??
        
        f_rawPsd_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject
    

        # Create a list of 15 colors and add it to the cycle of matplotlib 
        cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
        plt.rc('axes', prop_cycle=cycler_colors)

        # one figure for each session per STN
        for t, tp in enumerate(incl_session):

            for group in channel_groups:

                if group == "ring":
                    channels = ['03', '13', '02', '12', '01', '23']
                
                elif group == "segm_intra":
                    channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                
                elif group == "segm_inter":
                    channels = ['1A2A', '1B2B', '1C2C']

                # set layout for figures: using the object-oriented interface
                fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)
            
                for c, cond in enumerate(incl_condition):

                    for cont, contact in enumerate(incl_contact[f"{hemisphere}"]): 

                        # avoid Attribute Error, continue if session attribute doesn´t exist
                        if getattr(mainclass_sub.survey, tp) is None:
                            continue

                    
                        # apply loop over channels
                        temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                        
                        # avoid Attribute Error, continue if attribute doesn´t exist
                        if getattr(temp_data, cond) is None:
                            continue
                    
                        # try:
                        #     temp_data = getattr(temp_data, cond)
                        #     temp_data = temp_data.rest.data[tasks[tk]]
                        
                        # except AttributeError:
                        #     continue

                        temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                        temp_data = getattr(temp_data.rest, contact)
                        temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
            

                        #################### CREATE A BUTTERWORTH FILTER ####################
                        # sampling frequency: 250 Hz
                        fs = temp_data.info['sfreq']

                        # only if filter == "band-pass"
                        if filter == "band-pass":

                            # set filter parameters for band-pass filter
                            filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
                            frequency_cutoff_low = 5 # 5Hz high-pass filter
                            frequency_cutoff_high = 95 # 95 Hz low-pass filter

                            # create the filter
                            b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
            
                        else:
                            print("no filter applied")
                        
                        
                        # get new channel names
                        ch_names = temp_data.info.ch_names


                        #################### PICK CHANNELS ####################
                        include_channelList = [] # this will be a list with all channel names selected

                        for n, names in enumerate(ch_names):
                            
                            # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
                            for picked in channels:
                                if picked in names:
                                    include_channelList.append(names)

                            
                        # Error Checking: 
                        if len(include_channelList) == 0:
                            continue

                        # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
                        ch_names_indices = mne.pick_channels(ch_names, include=include_channelList)

                        
                        for i, ch in enumerate(ch_names):
                            
                            # only get picked channels
                            if i not in ch_names_indices:
                                continue

                            #################### FILTER ####################
                            signal = {}
                            if filter == "band-pass":
                                # filter the signal by using the above defined butterworth filter
                                signal["band-pass"] = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 
                            
                            elif filter == "unfiltered": 
                                signal["unfiltered"] = temp_data.get_data()[i, :]
                            

                            #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################
                            x = signal[f"{filter}"]
                            y = np.arange(1, len(signal[f"{filter}"])+1)

                            # the title of each plot is set to the timepoint e.g. "postop"
                            axes[i].set_title(f"{tp}, {group}, {channels[i]}", fontsize=15) 

                            # get y-axis label and limits
                            # axes[t].get_ylabel()
                            # axes[t].get_ylim()

                            # .plot() method for creating the plot, axes[0] refers to the first plot, the plot is set on the appropriate object axes[t]
                            lines = axes[i].plot(signal[f"{filter}"], label=f"{channels[i]}_{cond}", color="k")  # or np.log10(px) 
                            # colors of each line in different color, defined at the beginning
                            # axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}", color=colors[i])

                            # Add cursor hover functionality
                            #mplcursors.cursor(axes[i]).connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]:.2f}, {sel.target[1]:.2f})'))
                            #mplcursors.cursor(hover=True)
                            # Enable interactive tooltips
                            mpld3.plugins.connect(fig, mpld3.plugins.PointHTMLTooltip(lines[0], labels=['({:.2f}, {:.2f})'.format(xi, yi) for xi, yi in zip(x,y)]))

                            # Display the plot
                            mpld3.display(fig)


    return {
        "time_series": signal[f"{filter}"],
    }