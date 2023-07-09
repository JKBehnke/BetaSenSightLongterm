""" Movement artifact cleaning before computing power spectra """


import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
from scipy.signal import hann
import pickle


# PyPerceive Imports
# import py_perceive
from PerceiveImport.classes import main_class
from .. utils import find_folders as findfolders
from ..utils import loadResults as loadResults


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
        
    }
    
    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    # sns.set()
    plt.style.use('seaborn-whitegrid')  

    hemispheres = ["Right", "Left"]

    move_artifact_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject

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
                    
                    elif g == 1:
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        group_name = "segm_intra"
                    
                    elif g == 2:
                        channels = ['1A2A', '1B2B', '1C2C']
                        group_name = "segm_inter"

                    
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

                        fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)

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

                        for ax in axes:
                            ax.set_xlabel("time", fontsize=12)
                            ax.set_ylabel("amplitude", fontsize=12)
                        
                        for ax in axes.flat[:-1]:
                            ax.set(xlabel='')

                        # interaction: when a movement artifact is found first click = x1, second click = x2
                        pos = [] # collecting the clicked x and y values for one channel group of stn at one session
                        def onclick(event):
                            pos.append([event.xdata,event.ydata])
                                    
                        fig.canvas.mpl_connect('button_press_event', onclick)

                        fig.suptitle(f"raw time series ({filter}) sub-{sub}, {hem} hemisphere", ha="center", fontsize=20)
                        fig.tight_layout()
                        plt.subplots_adjust(wspace=0, hspace=0)

                        plt.show(block=False)
                        plt.gcf().canvas.draw()
                        

                        input_y_or_n = get_input_y_n("Artifacts found?") # interrups run and asks for input

                        if input_y_or_n == "y":

                            # save figure
                            fig.savefig(os.path.join(figures_path, f"raw_time_series_{filter}_sub-{sub}_{hem}_{tp}_{cond}_{group_name}_with_artifact.png"), bbox_inches="tight")

                            # store results
                            number_of_artifacts = len(pos) / 2

                            artifact_x = [x_list[0] for x_list in pos] # list of all clicked x values
                            artifact_y = [y_list[1] for y_list in pos] # list of all clicked y values

                            move_artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}"] = [sub, hem, tp, group_name, cond,
                                                                                    number_of_artifacts, artifact_x, artifact_y]
                            
                        
                        elif input_y_or_n == "n":

                            # save figure
                            fig.savefig(os.path.join(figures_path, f"raw_time_series_{filter}_sub-{sub}_{hem}_{tp}_{cond}_{group_name}_no_artifact.png"), bbox_inches="tight")

                            print("no artifacts")

                            number_of_artifacts = len(pos)

                            move_artifact_dict[f"{sub}_{hem}_{tp}_{group_name}_{cond}"] = [sub, hem, tp, group_name, cond,
                                                                                    number_of_artifacts, 0, 0]

                        plt.close()


    move_artifact_result_df = pd.DataFrame(move_artifact_dict)
    move_artifact_result_df.rename(index={0: "subject", 
                                          1: "hemisphere",
                                          2: "session",
                                          3: "channel_group",
                                          4: "condition",
                                          5: "number_move_artifacts",
                                          6: "artifact_x",
                                          7: "artifact_y"}, inplace=True)
    move_artifact_result_df = move_artifact_result_df.transpose()

    # join two columns sub and hem to one -> subject_hemisphere
    move_artifact_result_df["subject_hemisphere"] = move_artifact_result_df["subject"] + "_" + move_artifact_result_df["hemisphere"]

    # save dataframe as pickle
    results_filepath = os.path.join(results_path, f"movement_artifacts_from_raw_time_series_{filter}.pickle")
    with open(results_filepath, "wb") as file:
        pickle.dump(move_artifact_result_df, file)   


    # return {
    #     "time_series": signal,
    #     "pos": pos,
    #     "move_artifact_dict": move_artifact_dict
    # }







def clean_time_series_move_artifact(
):
    """ 
    
    
    Two sources:
        - artifact x values: load the movement_artifacts_from_raw_time_series_{filter}.pickle file
        - raw signals loaded with PyPerceive
    
    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    loaded_move_artifact_table = loadResults.load_preprocessing_files(table="movement_artifact_coord") 

    # list of existing STNs 
    sub_hem_list = list(loaded_move_artifact_table.subject_hemisphere.unique())
    sessions = list(loaded_move_artifact_table.session.unique())
    number_artifacts_unique = list(loaded_move_artifact_table.number_move_artifacts.unique())
    max_number_artifacts = max(number_artifacts_unique)

    clean_power_spectra_dict = {}

    for stn in sub_hem_list:

        stn_data = loaded_move_artifact_table.loc[loaded_move_artifact_table.subject_hemisphere == stn]
        sub = stn_data.subject.values[0]
        hem = stn_data.hemisphere.values[0]

        figures_path = findfolders.get_local_path(folder="figures", sub=sub)

        if hem == "Right":
            incl_contacts = ["RingR", "SegmIntraR", "SegmInterR"]
    
        if hem == "Left":
            incl_contacts = ["RingL", "SegmIntraL", "SegmInterL"]
        
        # load the signal from this given combination of sub, hem
        mainclass_sub = main_class.PerceiveData(
                sub = sub, 
                incl_modalities= ["survey"],
                incl_session = sessions,
                incl_condition = ["m0s0"],
                incl_task = ["rest"],
                incl_contact=incl_contacts
                )

        for ses in sessions:

            # check if session exists
            if ses not in stn_data.session.values:
                continue

            else:
                ses_data = stn_data.loc[stn_data.session == ses]


            for g, group in enumerate(incl_contacts): 
                # group = e.g. "RingR"
                if g == 0:
                    group_name = "ring"
                
                elif g == 1:
                    group_name = "segm_intra"
                
                elif g == 2:
                    group_name = "segm_inter"
                
                ############## extract x variables from the artifact table ##############
                group_data = ses_data.loc[ses_data.channel_group == group_name] # should only be one row left

                # check how many artifacts 
                number_artifacts = group_data.number_move_artifacts.values[0]
                artifact_x_values = group_data.artifact_x.values[0] # list with even number of x values (number_artifacts x 2 because beginning and end)
                
                
                ############## RAW SIGNALS for each channel in the group ##############
                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(mainclass_sub.survey, ses) is None:
                    continue

                raw_data = getattr(mainclass_sub.survey, ses) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                
                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(raw_data, "m0s0") is None:
                    continue

                raw_data = getattr(raw_data, "m0s0") # gets attribute e.g. "m0s0"
                raw_data = getattr(raw_data.rest, group)
                raw_data = raw_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"

                fs = raw_data.info['sfreq']
    
                ############### CREATE FILTER if filter band-pass 5-95 Hz ###############
                # set filter parameters for band-pass filter
                filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
                frequency_cutoff_low = 5 # 5Hz high-pass filter
                frequency_cutoff_high = 95 # 95 Hz low-pass filter

                # create the filter
                b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
                
                ############### GET SIGNAL FOR EACH CHANNEL###############     
                ch_names = raw_data.info.ch_names

                # set layout for figures: using the object-oriented interface
                fig_cleaned, axes_cleaned = plt.subplots(len(ch_names), 1, figsize=(25, 15)) # subplot(rows, columns, panel number), figsize(width,height)

                for i, chan in enumerate(ch_names):
                    # only extract integers in channel 
                    split_chan = chan.split("_") # ['LFP', 'R', '03', 'STN', 'MT']
                    channel = split_chan[2]

                    # filter the signal by using the above defined butterworth filter
                    filtered_signal = scipy.signal.filtfilt(b, a, raw_data.get_data()[i, :]) 
                    
                    unfiltered_signal = raw_data.get_data()[i, :]

                    ############### CLEAR EACH SIGNAL DEPENDING ON HOW MANY ARTIFACTS WERE DETECTED ###############
                    if number_artifacts == 0:
                        cleaned_filtered_signal = filtered_signal
                        cleaned_unfiltered_signal = unfiltered_signal
                    
                    elif number_artifacts == 1:
                        cleaned_filtered_signal = np.concatenate([filtered_signal[:round(artifact_x_values[0])], filtered_signal[round(artifact_x_values[1]+1):]])
                        cleaned_unfiltered_signal = np.concatenate([unfiltered_signal[:round(artifact_x_values[0])], unfiltered_signal[round(artifact_x_values[1]+1):]])
                    
                    elif number_artifacts == 2:
                        cleaned_filtered_signal = np.concatenate([filtered_signal[:round(artifact_x_values[0])], 
                                                        filtered_signal[round(artifact_x_values[1]+1):round(artifact_x_values[2])],
                                                        filtered_signal[round(artifact_x_values[3]+1):]])
                        
                        cleaned_unfiltered_signal = np.concatenate([unfiltered_signal[:round(artifact_x_values[0])], 
                                                        unfiltered_signal[round(artifact_x_values[1]+1):round(artifact_x_values[2])],
                                                        unfiltered_signal[round(artifact_x_values[3]+1):]])
                    
                    elif number_artifacts == 3:
                        cleaned_filtered_signal = np.concatenate([filtered_signal[:round(artifact_x_values[0])], # cut out artifact 1 
                                                        filtered_signal[round(artifact_x_values[1]+1):round(artifact_x_values[2])], # cut out artifact 2
                                                        filtered_signal[round(artifact_x_values[3]+1):round(artifact_x_values[4])], # cut out artifact 3
                                                        filtered_signal[round(artifact_x_values[5]+1):]])
                        
                        cleaned_unfiltered_signal = np.concatenate([unfiltered_signal[:round(artifact_x_values[0])], # cut out artifact 1 
                                                        unfiltered_signal[round(artifact_x_values[1]+1):round(artifact_x_values[2])], # cut out artifact 2
                                                        unfiltered_signal[round(artifact_x_values[3]+1):round(artifact_x_values[4])], # cut out artifact 3
                                                        unfiltered_signal[round(artifact_x_values[5]+1):]])
                        
                    # plot the cleaned filtered signal 
                    axes_cleaned[i].set_title(f"{ses}, {group_name}, {ch_names[i]}, band-pass 5-95 Hz", fontsize=15) 
                    axes_cleaned[i].plot(cleaned_filtered_signal, color="k")  

                    for ax in axes_cleaned:
                        ax.set_xlabel("time", fontsize=12)
                        ax.set_ylabel("amplitude", fontsize=12)
                    
                    for ax in axes_cleaned.flat[:-1]:
                        ax.set(xlabel='')
                    
                    ########################### FAST FOURIER TRANSFORM ###########################
                    window = hann(250, sym=False)
                    noverlap = 0.5
                    
                    ############## filtered and unfiltered version ##############
                    f,time_sectors,Sxx_filtered = scipy.signal.spectrogram(x=cleaned_filtered_signal, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
                    f,time_sectors,Sxx_unfiltered = scipy.signal.spectrogram(x=cleaned_unfiltered_signal, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
                    # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
                    # time_sectors = sectors 0.5 - 20.5 s in 1.0 steps (in total 21 time sectors)
                    # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
                    # Sxx = 126 frequency rows, 21 time sector columns

                    # average all 21 Power spectra of all time sectors 
                    average_Sxx_filtered = np.mean(Sxx_filtered, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                    average_Sxx_unfiltered = np.mean(Sxx_unfiltered, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency

                    #################### CALCULATE THE STANDARD ERROR OF MEAN ####################
                    # SEM = standard deviation / square root of sample size
                    Sxx_std_filtered = np.std(Sxx_filtered, axis=1) # standard deviation of each frequency row
                    semRawPsd_filtered = Sxx_std_filtered / np.sqrt(Sxx_filtered.shape[1]) # sample size = 21 time vectors -> sem with 126 values

                    Sxx_std_unfiltered = np.std(Sxx_unfiltered, axis=1) # standard deviation of each frequency row
                    semRawPsd_unfiltered = Sxx_std_unfiltered / np.sqrt(Sxx_unfiltered.shape[1]) # sample size = 21 time vectors -> sem with 126 values

                    duration_cleaned_signal = len(cleaned_filtered_signal)

                    
                    
                    
                    
                    
                    # store frequency, time vectors and psd values in a dictionary, together with session timepoint and channel
                    clean_power_spectra_dict[f'{stn}_{ses}_{group}_{chan}'] = [stn, ses, group_name, channel, f, time_sectors, 
                                                                               average_Sxx_filtered, semRawPsd_filtered, average_Sxx_unfiltered, semRawPsd_unfiltered,
                                                                               duration_cleaned_signal, number_artifacts] 
                

                    
                # save figure of cleaned time series
                fig_cleaned.suptitle(f"cleaned time series (band-pass 5-95 Hz) sub-{sub}, {hem} hemisphere", ha="center", fontsize=20)
                fig_cleaned.tight_layout()
                plt.subplots_adjust(wspace=0, hspace=0)

                # save figure
                if number_artifacts == 0:
                    fig_cleaned.savefig(os.path.join(figures_path, f"no_movement_artifact_in_time_series_band-pass_sub-{sub}_{hem}_{ses}_m0s0_{group_name}.png"), bbox_inches="tight")
                
                else:
                    fig_cleaned.savefig(os.path.join(figures_path, f"movement_artifact_removed_time_series_band-pass_sub-{sub}_{hem}_{ses}_m0s0_{group_name}.png"), bbox_inches="tight")


    # save power spectra result 
    clean_power_spectra = pd.DataFrame(clean_power_spectra_dict)
    clean_power_spectra.rename(index={0: "subject_hemisphere",
                                      1: "session",
                                      2: "channel_group",
                                      3: "channel",
                                      4: "frequencies",
                                      5: "time_sectors",
                                      6: "clean_psd_band_pass",
                                      7: "sem_psd_band_pass",
                                      8: "clean_psd_unfiltered",
                                      9: "sem_psd_unfiltered",
                                      10: "duration_clean_time_series",
                                      11: "number_movement_artifacts_removed"}, inplace=True)
    clean_power_spectra = clean_power_spectra.transpose()

    # save as json
    clean_power_spectra.to_json(os.path.join(results_path, f"clean_power_spectra.json"))



def plot_clean_power_spectra(signal_filter:str):

    """
    Input:
        - signal_filter: "band-pass" or "unfiltered"
    load the cleaned power spectra table and plot all existing clean power spectra
    
    """

    # load power spectra table
    clean_power_spectra_table = loadResults.load_preprocessing_files(
        table="cleaned_power_spectra")
    
    # list of existing STNs 
    sub_hem_list = list(clean_power_spectra_table.subject_hemisphere.unique())

    for stn in sub_hem_list:

        split_stn = stn.split("_")
        sub = split_stn[0]

        figures_path = findfolders.get_local_path(folder="figures", sub=sub)

        stn_data = clean_power_spectra_table.loc[clean_power_spectra_table.subject_hemisphere == stn]

        sessions = list(stn_data.session.unique())

        # set layout for figures: using the object-oriented interface
        fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)

        # Create a list of 15 colors and add it to the cycle of matplotlib 
        cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
        plt.rc('axes', prop_cycle=cycler_colors)

     
        for s, ses in enumerate(sessions): 

            ses_data = stn_data.loc[stn_data.session == ses]

            channel_names = list(ses_data.channel.values)

            for chan in channel_names:

                chan_data = ses_data.loc[ses_data.channel == chan]

                f = chan_data.frequencies.values[0]

                if signal_filter == "band-pass":
                    cleaned_psd = np.array(chan_data.clean_psd_band_pass.values[0])
                    cleaned_sem_psd = np.array(chan_data.sem_psd_band_pass.values[0])
                
                elif signal_filter == "unfiltered":
                    cleaned_psd = np.array(chan_data.clean_psd_unfiltered.values[0])
                    cleaned_sem_psd = np.array(chan_data.sem_psd_unfiltered.values[0])

                axes[s].set_title(ses, fontsize=15) 
                axes[s].plot(f, cleaned_psd, label=f"{chan}")
                axes[s].fill_between(f, cleaned_psd-cleaned_sem_psd, cleaned_psd+cleaned_sem_psd, color='lightgray', alpha=0.5)


        #################### PLOT SETTINGS ####################
        fig.suptitle(f"clean power spectra sub-{stn}, Filter: {signal_filter}", ha="center", fontsize= 20)
        plt.subplots_adjust(wspace=0, hspace=0)
    
        font = {"size": 20}

        for ax in axes: 
            # ax.legend(loc= 'upper right') # Legend will be in upper right corner
            ax.grid() # show grid

            # different xlim depending on filtered or unfiltered signal
            if signal_filter == "band-pass":
                ax.set(xlim=[3, 50]) # no ylim for rawPSD and normalization to sum 40-90 Hz

            elif signal_filter == "unfiltered":
                ax.set(xlim=[-2, 50])

            # ax.set(xlim=[-5, 60] ,ylim=[0,7]) for normalizations to total sum or to sum 1-100Hz set ylim to zoom in
            ax.set_xlabel("Frequency", fontsize=12)
            ax.set_ylabel("uV^2/Hz+-SEM", fontsize=12)
            ax.set(ylim=[0, 3])
            ax.axvline(x=8, color='black', linestyle='--')
            ax.axvline(x=13, color='black', linestyle='--')
            ax.axvline(x=20, color='black', linestyle='--')
            ax.axvline(x=35, color='black', linestyle='--')
        
        # remove x ticks and labels from all but the bottom subplot
        for ax in axes.flat[:-1]:
            ax.set(xlabel='')
    

        ###### LEGEND ######
        legend = axes[0].legend(loc= 'lower right', edgecolor="black", bbox_to_anchor=(1.5, -0.1)) # only show the first subplot´s legend 
        # frame the legend with black edges amd white background color 
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor("white")

        fig.tight_layout()

        plt.show()
        fig.savefig(os.path.join(figures_path, f"clean_power_spectra_{stn}_{signal_filter}.png"), bbox_inches="tight")
          



       


    

    




