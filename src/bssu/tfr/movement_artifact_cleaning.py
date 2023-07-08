""" Movement artifact cleaning before computing power spectra """


import os

import matplotlib.pyplot as plt
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


def get_input_y_n(message: str) -> str:
    """Get `y` or `n` user input."""
    while True:
        user_input = input(f"{message} (y/n)? ")
        if user_input.lower() in ["y", "n"]:
            break
        print(
            f"Input must be `y` or `n`. Got: {user_input}."
            " Please provide a valid input."
        )
    return user_input



def plot_raw_time_series(incl_sub: list, incl_session: list, incl_condition: list, filter: str):
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

    #channel_groups = ["ring", "segm_inter", "segm_intra"]
    hemispheres = ["Right", "Left"]


    for sub in incl_sub:

        for hem in hemispheres:

            if hem == "Right":
                channel_groups = ["RingR", "SegmIntraR", "SegmInterR"]

            elif hem == "Left":
                channel_groups = ["RingL", "SegmIntraL", "SegmInterL"]

            mainclass_sub = main_class.PerceiveData(
                sub = sub, 
                incl_modalities= ["survey"],
                incl_session = incl_session,
                incl_condition = incl_condition,
                incl_task = ["rest"],
                incl_contact=channel_groups
                )

            
            figures_path = findfolders.get_local_path(folder="figures", sub=incl_sub)
            results_path = findfolders.get_local_path(folder="results", sub=incl_sub)

            # add error correction for sub and task??
            
            move_artifact_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject
        

            # Create a list of 15 colors and add it to the cycle of matplotlib 
            #cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
            #plt.rc('axes', prop_cycle=cycler_colors)

            # one figure for each session per STN
            for t, tp in enumerate(incl_session):

                for g, group in enumerate(channel_groups):

                    if g == 0:
                        channels = ['03', '13', '02', '12', '01', '23']
                    
                    elif g == 1:
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                    
                    elif g == 2:
                        channels = ['1A2A', '1B2B', '1C2C']

                    
                    for c, cond in enumerate(incl_condition):

                        # set layout for figures: using the object-oriented interface
                        fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)

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
                        temp_data = getattr(temp_data.rest, group)
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
                            
                            if filter == "band-pass":
                                # filter the signal by using the above defined butterworth filter
                                signal = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 
                            
                            elif filter == "unfiltered": 
                                signal = temp_data.get_data()[i, :]
                            

                            #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################
                            # x = signal[f"{filter}"]
                            # y = np.arange(1, len(signal[f"{filter}"])+1)

                            axes[i].set_title(f"{tp}, {group}, {channels[i]}", fontsize=15) 
                            axes[i].plot(signal, label=f"{channels[i]}_{cond}", color="k")  


                    # interaction: when a movement artifact is found first click = x1, second click = x2
                    pos = [] # collecting the clicked x and y values for one channel group of stn at one session
                    def onclick(event):
                        pos.append([event.xdata,event.ydata])
                                
                    fig.canvas.mpl_connect('button_press_event', onclick)
                    fig.tight_layout()
                    fig.show()

                    # input_y_or_n = get_input_y_n("Finished viewing plots?") # interrups run and asks for input

                    # if input_y_or_n == "y":

                    #     # store results
                    #     number_of_artifacts = len(pos) / 2

                    #     artifact_x = [x_list[0] for x_list in pos]
                    #     artifact_y = [y_list[1] for y_list in pos]

                    #     move_artifact_dict[f"{sub}_{hem}_{tp}_{group}_{cond}"] = [sub, hem, tp, group, cond,
                    #                                                             number_of_artifacts]
                        
                    #     # fig.savefig()

                    #     for a, art in enumerate(artifact_x):

                    #         # all even numbers reflect one new artifact
                    #         if a % 2 == 0:

                    #             # get index of where artifact starts and ends
                    #             artifact_x_start = round(artifact_x[a]) # index of x value in artifact x list
                    #             artifact_x_end = round(artifact_x[a+1])

                    #             # clean each channel by removing the array with the detected artifact
                    #             for i, ch in enumerate(ch_names):
                            
                    #                 # only get picked channels
                    #                 if i not in ch_names_indices:
                    #                     continue

                    #                 #################### FILTER ####################
                                    
                    #                 if filter == "band-pass":
                    #                     # filter the signal by using the above defined butterworth filter
                    #                     signal = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 
                                    
                    #                 elif filter == "unfiltered": 
                    #                     signal = temp_data.get_data()[i, :]
                                    
                    #                 # cut out the artifact, keep everything before and after
                    #                 cleaned_signal = np.concatenate([signal[:artifact_x_start-1], signal[artifact_x_end:]])
                                    

                    #                 #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################
                    #                 # x = signal[f"{filter}"]
                    #                 # y = np.arange(1, len(signal[f"{filter}"])+1)

                    #                 axes[i].set_title(f"{tp}, {group}, {channels[i]}", fontsize=15) 
                    #                 axes[i].plot(cleaned_signal, label=f"{channels[i]}_{cond}", color="k")  

                    #         else:
                    #             continue





                        # cut artifacts out
                        # cleaned_signal = 





                    #plt.close(fig)





    return {
        "time_series": signal,
        "pos": pos,
        "move_artifact_dict":move_artifact_dict
    }