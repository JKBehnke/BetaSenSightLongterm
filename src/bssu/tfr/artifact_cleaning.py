""" artifact cleaning before computing power spectra """


import os

import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, create_ecg_epochs
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
from scipy.signal import hann
import pickle


# PyPerceive Imports
# import py_perceive
from PerceiveImport.classes import main_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults


HEMISPHERES = ["Right", "Left"]

def load_mne_object_pyPerceive(sub: str, session: str, hemisphere: str, channel_group: str):
    """
    
    """

    if hemisphere == "Right":
        channel_groups = ["RingR", "SegmIntraR", "SegmInterR"]

    elif hemisphere == "Left":
        channel_groups = ["RingL", "SegmIntraL", "SegmInterL"]


    mainclass_sub = main_class.PerceiveData(
        sub = sub, 
        incl_modalities= ["survey"],
        incl_session = [session],
        incl_condition = ["m0s0"],
        incl_task = ["rest"],
        incl_contact=channel_groups
        )

    temp_data = getattr(mainclass_sub.survey, session) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
    temp_data = getattr(temp_data, "m0s0") # gets attribute e.g. "m0s0"
    temp_data = getattr(temp_data.rest, channel_group)
    temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"

    return temp_data


def plot_raw_time_series(incl_sub: list, incl_session: list, incl_condition: list):
    """
    Function to plot raw time series of all channels for each subject, session, condition and channel group.
    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    # sns.set()
    plt.style.use('seaborn-whitegrid')  

    hemispheres = ["Right", "Left"]

    artifact_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject

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

            
            figures_path = findfolders.get_local_path(folder="figures", sub=sub)

            # add error correction for sub and task??
            
            # one figure for each STN, session, channel group
            for t, tp in enumerate(incl_session):

                for g, group in enumerate(channel_groups):

                    if g == 0:
                        channels = ['03', '13', '02', '12', '01', '23']
                        group_name = "ring"
                        fig_size = (120, 40)
                    
                    elif g == 1:
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        group_name = "segm_intra"
                        fig_size = (120, 40)
                    
                    elif g == 2:
                        channels = ['1A2A', '1B2B', '1C2C']
                        group_name = "segm_inter"
                        fig_size = (120, 20)

                    for c, cond in enumerate(incl_condition):

                        # set layout for figures: using the object-oriented interface
                        # fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)


                        # avoid Attribute Error, continue if session attribute doesn´t exist
                        try:
                            getattr(mainclass_sub.survey, tp)
                        
                        except AttributeError:
                            continue

                        # if getattr(mainclass_sub.survey, tp) is None:
                        #     continue
        
                        # apply loop over channels
                        temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                        
                        # avoid Attribute Error, continue if attribute doesn´t exist
                        # if getattr(temp_data, cond) is None:
                        #     continue
                    
                        try:
                            getattr(temp_data, cond)
                            #temp_data = temp_data.rest.data[tasks[tk]]
                        
                        except AttributeError:
                            continue

                        temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                        temp_data = getattr(temp_data.rest, group)
                        temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
                        
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

                        fig, axes = plt.subplots(len(channels), 1, figsize=fig_size) # subplot(rows, columns, panel number), figsize(width,height)

                        for i, ch in enumerate(ch_names):
                            
                            # only get picked channels
                            if i not in ch_names_indices:
                                continue

                            signal = temp_data.get_data()[i, :]

                            # save signals in dictionary
                            artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}_{channels[i]}"] = [sub, hem, tp, group_name, cond, channels[i], signal]
                            

                            #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################
                            # x = signal[f"{filter}"]
                            # y = np.arange(1, len(signal[f"{filter}"])+1)

                            axes[i].set_title(f"{tp}, {group}, {channels[i]}", fontsize=40) 
                            axes[i].plot(signal, label=f"{channels[i]}_{cond}", color="k")  

                        for ax in axes:
                            ax.set_xlabel("timestamp", fontsize=40)
                            ax.set_ylabel("amplitude", fontsize=40)
                            ax.tick_params(axis='both', which='major', labelsize=40)
                        
                        for ax in axes.flat[:-1]:
                            ax.set(xlabel='')

                        # # interaction: when a movement artifact is found first click = x1, second click = x2
                        # pos = [] # collecting the clicked x and y values for one channel group of stn at one session
                        # def onclick(event):
                        #     pos.append([event.xdata,event.ydata])
                                    
                        # fig.canvas.mpl_connect('button_press_event', onclick)

                        fig.suptitle(f"raw time series sub-{sub}, {hem} hemisphere", ha="center", fontsize=40)
                        fig.tight_layout()
                        plt.subplots_adjust(wspace=0, hspace=0)

                        plt.show(block=False)
                        #plt.gcf().canvas.draw()

                        #input_y_or_n = get_input_y_n("Artifacts found?") # interrups run and asks for input

                        # if input_y_or_n == "y":

                        #     # save figure
                        #     fig.savefig(os.path.join(figures_path, f"raw_time_series_{filter}_sub-{sub}_{hem}_{tp}_{cond}_{group_name}_with_artifact.png"), bbox_inches="tight")

                        #     # store results
                        #     number_of_artifacts = len(pos) / 2

                        #     artifact_x = [x_list[0] for x_list in pos] # list of all clicked x values
                        #     artifact_y = [y_list[1] for y_list in pos] # list of all clicked y values

                        #     move_artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}"] = [sub, hem, tp, group_name, cond,
                        #                                                             number_of_artifacts, artifact_x, artifact_y]
                        

                        # save figure
                        fig.savefig(os.path.join(figures_path, f"raw_time_series_sub-{sub}_{hem}_{tp}_{cond}_{group_name}.png"), bbox_inches="tight")


                        #number_of_artifacts = len(pos)

                        plt.close()


    move_artifact_result_df = pd.DataFrame(artifact_dict)
    move_artifact_result_df.rename(index={0: "subject", 
                                          1: "hemisphere",
                                          2: "session",
                                          3: "channel_group",
                                          4: "condition",
                                          5: "channel",
                                          6: "signal",
                                          }, inplace=True)
    move_artifact_result_df = move_artifact_result_df.transpose()

    # join two columns sub and hem to one -> subject_hemisphere
    move_artifact_result_df["subject_hemisphere"] = move_artifact_result_df["subject"] + "_" + move_artifact_result_df["hemisphere"]

    # save dataframe as pickle
    # results_filepath = os.path.join(results_path, f"raw_time_series.pickle")
    # with open(results_filepath, "wb") as file:
    #     pickle.dump(move_artifact_result_df, file)   

    return move_artifact_result_df, temp_data



