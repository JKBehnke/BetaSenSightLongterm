""" Read and preprocess externalized LFPs"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy import signal
from scipy.signal import spectrogram, hann, butter, filtfilt, freqz

import json
import os
import mne
import pickle

import fooof
from fooof.plts.spectra import plot_spectrum

# internal Imports
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults
from .. utils import load_data_files as load_data



group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")
group_figures_path = find_folders.get_monopolar_project_path(folder="GroupFigures")

# patient_metadata = load_data.load_patient_metadata_externalized()
patient_metadata = load_data.load_excel_data(filename="patient_metadata")
hemispheres = ["Right", "Left"]

# get index of each channel and get the corresponding LFP data
# plot filtered channels 1-8 [0]-[7] Right and 9-16 [8]-[15] 
# butterworth filter: band pass -> filter order = 5, high pass 5 Hz, low-pass 95 Hz
def band_pass_filter_externalized(
        fs: int,
        signal: np.array
):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a band pass filter to the signal
        - 5 Hz high pass
        - 95 Hz low pass
        - filter order: 3

    """
    # parameters
    filter_order = 3 # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 5 # 5Hz high-pass filter 
    frequency_cutoff_high = 95 # 95 Hz low-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal) 

    return band_pass_filtered


# notch filter: 50 Hz
def notch_filter_externalized(
        fs: int,
        signal: np.array
):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a notch filter to the signal
    
    
    """

    # parameters
    notch_freq = 50 # 50 Hz line noise in Europe
    Q = 30 # Q factor for notch filter

    # apply notch filter
    b, a = scipy.signal.iirnotch(w0=notch_freq, Q=Q, fs=fs)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)

    return filtered_signal


# detect artefacts 
# remove artefacts (cut out)

# save the cleaned data unfiltered

####### plot all channels and check for any artefacts ####### TF and raw data plots

####### Fourier Transform, plot averaged Power Spectra

####### FOOOF unfiltered power spectra

# perform FOOOF to extract only periodic component

def preprocess_externalized_lfp(
        sub:list
):
    """
    Input:
        - sub: str e.g. [
            "25", "30", "32", "47", "52", "59", 
            "61", "64", "67", "69", "71", 
            "72", "75", "77", "79", "80"]

    Load the BIDS .vhdr files with mne_bids.read_raw_bids(bids_path=bids_path)

    - crop the data to only 2 minutes: 1min - 3min
    - downsample the data to 
        1) sfreq=4000 Hz (if it was 4096Hz originally)
        2) sfreq=250 Hz 
    
    - Filter both versions:
        1) notch filter 50 Hz
        2) band-pass filter 5-95 Hz (filterorder 2)
        
    - save the data of all contacts into Dataframe:
        1) externalized_preprocessed_data -> all versions of the data filtered, unfiltered, 250 Hz, 4000 Hz
        2) externalized_recording_info_original -> information about the original recording
        3) mne_objects_cropped_2_min -> MNE objects of the 4000 Hz and 250 Hz data, unfiltered


    """

    group_data = pd.DataFrame()
    group_originial_rec_info = pd.DataFrame()
    mne_objects = {}
    
    for patient in sub:

        mne_data = load_data.load_BIDS_externalized_vhdr_files(
            sub=patient
        )

        recording_info = {}
        processed_recording = {}

        # get info
        ch_names = mne_data.info["ch_names"]
        ch_names_LFP = [chan for chan in ch_names if "LFP" in chan]
        bads = mne_data.info["bads"] # channel L_01 is mostly used as reference, "bad" channel is the reference
        chs = mne_data.info["chs"] # list, [0] is dict
        sfreq = mne_data.info["sfreq"]
        subject_info = mne_data.info["subject_info"]
        n_times = mne_data.n_times # number of timestamps
        rec_duration = (n_times / sfreq) / 60 # duration in minutes

        subject = f"0{patient}"
        bids_ID = mne_data.info["subject_info"]["his_id"]

        # pick LFP channels of both hemispheres
        mne_data.pick_channels(ch_names_LFP) 

        recording_info["original_information"] = [subject, bids_ID, ch_names, bads, sfreq, subject_info, n_times, rec_duration]
        originial_rec_info = pd.DataFrame(recording_info)
        originial_rec_info.rename(index={
            0: "subject",
            1: "BIDS_id",
            2: "ch_names",
            3: "bads",
            4: "sfreq",
            5: "subject_info",
            6: "number_time_stamps",
            7: "recording_duration"
        }, inplace=True)
        originial_rec_info = originial_rec_info.transpose()

        # plot the filtered channels, to visually detect artefacts
        # mne_data.plot(highpass=5.0, lowpass=95.0, filtorder=5.0)
        
        # select a period of 2 minutes with no aratefacts, default start at 1 min until 3 min
        mne_data.crop(60,180)

        # downsample all to 4000 Hz
        if int(sfreq) != 4000:
            mne_data = mne_data.copy().resample(sfreq=4000)
            sfreq = mne_data.info["sfreq"]
        
        # downsample from TMSi sampling frequency to 250 sfreq (like Percept)
        resampled_250 = mne_data.copy().resample(sfreq=250)
        sfreq_250 = resampled_250.info["sfreq"]
        # cropped data should have 30000 samples (2 min of sfreq 250)

        # save the mne object
        mne_objects[f"{patient}_4000Hz_2min"] = mne_data
        mne_objects[f"{patient}_resampled_250Hz"] = resampled_250

        # from bids_id only keep the part after sub-
        bids_ID = bids_ID.split('-')
        bids_ID = bids_ID[1]

        ########## save processed LFP data in dataframe ##########
        for idx, chan in enumerate(ch_names_LFP):

            lfp_data = mne_data.get_data(picks = chan)[0]
            time_stamps = mne_data[idx][1]

            lfp_data_250 = resampled_250.get_data(picks = chan)[0]
            time_stamps_250 = resampled_250[idx][1]

            # ch_name corresponding to Percept -> TODO: is the order always correct???? 02 = 1A? could it also be 1B?
            if "_01_" in chan:
                monopol_chan_name = "0"
            
            elif "_02_" in chan:
                monopol_chan_name = "1A"
            
            elif "_03_" in chan:
                monopol_chan_name = "1B"

            elif "_04_" in chan:
                monopol_chan_name = "1C"

            elif "_05_" in chan:
                monopol_chan_name = "2A"

            elif "_06_" in chan:
                monopol_chan_name = "2B"

            elif "_07_" in chan:
                monopol_chan_name = "2C"

            elif "_08_" in chan:
                monopol_chan_name = "3"

            # hemisphere
            if "_L_" in chan:
                hemisphere = "Left"
            
            elif "_R_" in chan:
                hemisphere = "Right"
            
            # subject_hemisphere
            subject_hemisphere = f"{subject}_{hemisphere}"

            # notch filter 50 Hz
            notch_filtered_lfp_4000 = notch_filter_externalized(fs=sfreq, signal=lfp_data)
            notch_filtered_lfp_250 = notch_filter_externalized(fs=sfreq_250, signal=lfp_data_250)

            # band pass filter 5-95 Hz, Butter worth filter order 3
            filtered_lfp_4000 = band_pass_filter_externalized(fs=sfreq, signal=notch_filtered_lfp_4000)
            filtered_lfp_250 = band_pass_filter_externalized(fs=sfreq_250, signal=notch_filtered_lfp_250)

            # number of samples
            n_samples_250 = len(filtered_lfp_250)



            processed_recording[f"{chan}"] = [bids_ID, subject, hemisphere, subject_hemisphere, chan, monopol_chan_name, 
                                              lfp_data, time_stamps, sfreq, sfreq_250, lfp_data_250, time_stamps_250,
                                              filtered_lfp_4000, filtered_lfp_250, n_samples_250]
        
        preprocessed_dataframe = pd.DataFrame(processed_recording)
        preprocessed_dataframe.rename(index={
            0: "BIDS_id",
            1: "subject",
            2: "hemisphere",
            3: "subject_hemisphere",
            4: "original_ch_name",
            5: "contact",
            6: "lfp_2_min",
            7: "time_stamps",
            8: "sfreq",
            9: "sfreq_250Hz",
            10: "lfp_resampled_250Hz",
            11: "time_stamps_250Hz",
            12: "filtered_lfp_4000Hz",
            13: "filtered_lfp_250Hz", 
            14: "n_samples_250Hz"
 
        }, inplace=True)
        preprocessed_dataframe = preprocessed_dataframe.transpose()
    
        group_data = pd.concat([group_data, preprocessed_dataframe])
        group_originial_rec_info = pd.concat([group_originial_rec_info, originial_rec_info])
    

    # save dataframes
    group_data_path = os.path.join(group_results_path, f"externalized_preprocessed_data.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(group_data, file)

    print(f"externalized_preprocessed_data.pickle",
            f"\nwritten in: {group_results_path}" )
    
    group_rec_info_path = os.path.join(group_results_path, f"externalized_recording_info_original.pickle")
    with open(group_rec_info_path, "wb") as file:
        pickle.dump(group_originial_rec_info, file)

    print(f"externalized_recording_info_original.pickle",
            f"\nwritten in: {group_results_path}" )
    
    group_mne_objects_path = os.path.join(group_results_path, f"mne_objects_cropped_2_min.pickle")
    with open(group_mne_objects_path, "wb") as file:
        pickle.dump(mne_objects, file)

    print(f"mne_objects_cropped_2_min.pickle",
            f"\nwritten in: {group_results_path}" )

    return {
        "group_originial_rec_info": group_originial_rec_info,
        "group_data": group_data,
        "mne_objects": mne_objects
    }


def fourier_transform_time_frequency_plots(
        
):
    """
    Load the preprocessed data: externalized_preprocessed_data.pickle

    - For each subject, plot a Time Frequency figure of all channels for Left and Right hemisphere
        1) extract only the filtered data, sfreq=250 Hz
    
    - save figure into subject figures folder

    """

    # load the dataframe with all filtered LFP data
    preprocessed_data = load_data.load_externalized_pickle(
        filename="externalized_preprocessed_data"
    )

    # get all subject_hemispheres
    BIDS_id_unique = list(preprocessed_data.BIDS_id.unique())
    sub_hem_unique = list(preprocessed_data.subject_hemisphere.unique())

    # plot all time frequency plots of the 250 Hz sampled filtered LFPs
    for BIDS_id in BIDS_id_unique:
        figures_path = find_folders.get_monopolar_project_path(folder="figures", sub=BIDS_id)
        subject_data = preprocessed_data.loc[preprocessed_data.BIDS_id == BIDS_id]
        sub = subject_data.subject.values[0]

        for hem in hemispheres:
            sub_hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(sub_hem_data.contact.values)

            # Figure of one subject_hemisphere with all 8 channels 
            # 4 columns, 2 rows

            fig = plt.figure(figsize= (30, 30), layout="tight")

            for c, contact in enumerate(contacts):

                contact_data = sub_hem_data.loc[sub_hem_data.contact == contact]

                # filtered LFP from one contact, resampled to 250 Hz
                filtered_lfp_250 = contact_data.filtered_lfp_250Hz.values[0]
                sfreq = 250

                # Calculate the short time Fourier transform (STFT) using hamming window
                window_length = int(sfreq) # 1 second window length
                overlap = window_length // 4 # 25% overlap

                frequencies, times, Zxx = signal.stft(filtered_lfp_250, fs=sfreq, nperseg=window_length, noverlap=overlap, window='hamming')
                # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
                # times: len=161, 0, 0.75, 1.5 .... 120.75
                # Zxx: 126 arrays, each len=161
                # Zxx with imaginary values -> take the absolute!
                # to get power -> **2

                plt.subplot(4, 2, c+1) # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                plt.title(f"Channel {contact}", fontdict={"size": 40})
                plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='auto', cmap='viridis')

                plt.xlabel("Time [s]", fontdict={"size": 30})
                plt.ylabel("Frequency [Hz]", fontdict={"size": 30})
                plt.yticks(np.arange(0, 512, 30), fontsize= 20)
                plt.ylim(1, 100)
                plt.xticks(fontsize= 20)
            
            fig.suptitle(f"Time Frequency sub-{sub}, {hem} hemisphere, fs = 250 Hz", fontsize=55, y=1.02)
            plt.show()

            fig.savefig(os.path.join(figures_path, f"Time_Frequency_sub{sub}_{hem}_filtered_250Hz_resampled.png"),
                        bbox_inches="tight")



def clean_artefacts(
        
):
    """
    Clean artefacts:

    - Load the artefact Excel sheet with the time in seconds of when visually inspected artefacts start and end
    - load the preprocessed data

    - clean the artefacts from the two versions of 250 Hz resampled data: 
        - lfp_resampled_250 Hz -> unfiltered LFP, resampled to 250 Hz
        - filtered_lfp_250Hz -> notch, band-pass filtered, resampled to 250 Hz
    
    - Plot again the clean Time Frequency plots (sfreq=250 Hz, filtered, artefact-free) to check, if artefacts are gone

    - copy the old preprocessed dataframe, replace the original data by the clean artefact-free data:
        externalized_preprocessed_data_artefact_free.pickle

    """
    sfreq = 250

    # load data
    artefacts_excel = load_data.load_excel_data(filename="movement_artefacts")
    preprocessed_data = load_data.load_externalized_pickle(filename="externalized_preprocessed_data")

    # artefact_free_dataframe= pd.DataFrame()
    artefact_free_dataframe = preprocessed_data.copy()
    artefact_free_dataframe = artefact_free_dataframe.reset_index(drop=True)

    # check which subjects have artefacts
    artefacts_excel = artefacts_excel.loc[artefacts_excel.contacts == "all"]
    BIDS_id_artefacts = list(artefacts_excel.BIDS_key.unique())

    for bids_id in BIDS_id_artefacts:

        figures_path = find_folders.get_monopolar_project_path(folder="figures", sub=bids_id)
        
        # data only of one subject
        subject_artefact_data = artefacts_excel.loc[artefacts_excel.BIDS_key == bids_id]
        subject_data = preprocessed_data.loc[preprocessed_data.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        # check which hemispheres have artefacts
        hemispheres_with_artefacts = []

        if "Right" in subject_artefact_data.hemisphere.values:
            hemispheres_with_artefacts.append("Right")

        
        if "Left" in subject_artefact_data.hemisphere.values:
            hemispheres_with_artefacts.append("Left")
            
        
        for hem in hemispheres_with_artefacts:

            # get artefact data from one subject hemisphere
            hem_artefact_data = subject_artefact_data.loc[subject_artefact_data.hemisphere == hem]

            artefact_samples_list = []
            artefact1_start = hem_artefact_data.artefact1_start.values[0]
            artefact1_stop = hem_artefact_data.artefact1_stop.values[0]
            
            # calculate the samples: X sample = 250 Hz * second
            sample_start1 = int(sfreq * artefact1_start)
            sample_stop1 = int(sfreq * artefact1_stop)

            artefact_samples_list.append(sample_start1)

            # check if there are more artefacts
            if hem_artefact_data["artefact2_start"].notna().any():
                artefact2_start = hem_artefact_data.artefact2_start.values[0]
                artefact2_stop = hem_artefact_data.artefact2_stop.values[0]

                sample_start2 = int(sfreq * artefact2_start)
                sample_stop2 = int(sfreq * artefact2_stop)

                artefact_samples_list.append(sample_start2)
            
            if hem_artefact_data["artefact3_start"].notna().any():
                artefact3_start = hem_artefact_data.artefact3_start.values[0]
                artefact3_stop = hem_artefact_data.artefact3_stop.values[0]

                sample_start3 = int(sfreq * artefact3_start)
                sample_stop3 = int(sfreq * artefact3_stop)

                artefact_samples_list.append(sample_start3)

            # get lfp data from one subject hemisphere
            hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(hem_data.contact.values)

            # Figure of one subject_hemisphere with all 8 channels 
            # 4 columns, 2 rows

            fig = plt.figure(figsize= (30, 30), layout="tight")

            for c, contact in enumerate(contacts):

                contact_data = hem_data.loc[hem_data.contact == contact]

                # get LFP data from one contact
                filtered_lfp_250 = contact_data.filtered_lfp_250Hz.values[0] # to plot
                
                lfp_resampled_250Hz = contact_data.lfp_resampled_250Hz.values[0]
                # lfp_2_min = contact_data.lfp_2_min.values[0]
                # filtered_lfp_4000Hz = contact_data.filtered_lfp_4000Hz.values[0]

                loop_over_data = [1,2]

                for data in loop_over_data:

                    if data == 1:
                        lfp_data = filtered_lfp_250
                        column_name = "filtered_lfp_250Hz"
                    
                    elif data == 2:
                        lfp_data = lfp_resampled_250Hz
                        column_name = "lfp_resampled_250Hz"
                    
                    # elif data == 3:
                    #     lfp_data = lfp_2_min
                    #     column_name = "lfp_2_min"
                    
                    # elif data == 4:
                    #     lfp_data = filtered_lfp_4000Hz
                    #     column_name = "filtered_lfp_4000Hz"
                    

                    # clean artefacts from LFP data
                    # check how many artefacts 1-3?
                    if len(artefact_samples_list) == 1:

                        data_clean_1 = lfp_data[0 : sample_start1+1]
                        data_clean_2 = lfp_data[sample_stop1 : 30000]
                        clean_data = np.concatenate([data_clean_1, data_clean_2]) 


                    elif len(artefact_samples_list) == 2:
                        
                        data_clean_1 = lfp_data[0 : sample_start1+1]
                        data_clean_2 = lfp_data[sample_stop1 : sample_start2+1]
                        data_clean_3 = lfp_data[sample_stop2 : 30000]
                        clean_data = np.concatenate([data_clean_1, data_clean_2, data_clean_3]) 

                    
                    elif len(artefact_samples_list) == 3:
                        
                        data_clean_1 = lfp_data[0 : sample_start1+1]
                        data_clean_2 = lfp_data[sample_stop1 : sample_start2+1]
                        data_clean_3 = lfp_data[sample_stop2 : sample_start3+1]
                        data_clean_4 = lfp_data[sample_stop3 : 30000]
                        clean_data = np.concatenate([data_clean_1, data_clean_2, data_clean_3, data_clean_4]) 


                    # replace artefact_free data in the copied original dataframe 
                    # get the index of the contact you're in
                    row_index = artefact_free_dataframe[(artefact_free_dataframe['BIDS_id'] == bids_id) & (artefact_free_dataframe['hemisphere'] == hem) & (artefact_free_dataframe['contact'] == contact)]
                    row_index = row_index.index[0]

                    # filtered LFP, resampled to 250 Hz
                    artefact_free_dataframe.loc[row_index, column_name] = clean_data
                    artefact_free_dataframe.loc[row_index, "n_samples_250Hz"] = len(clean_data)

                    if data == 1: 
                
                        ############################# Calculate the short time Fourier transform (STFT) using hamming window #############################
                        window_length = int(sfreq) # 1 second window length
                        overlap = window_length // 4 # 25% overlap

                        frequencies, times, Zxx = signal.stft(clean_data, fs=sfreq, nperseg=window_length, noverlap=overlap, window='hamming')
                        # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
                        # times: len=161, 0, 0.75, 1.5 .... 120.75
                        # Zxx: 126 arrays, each len=161
                        # Zxx with imaginary values -> take the absolute!
                        # to get power -> **2

                        plt.subplot(4, 2, c+1) # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                        plt.title(f"Channel {contact}", fontdict={"size": 40})
                        plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='auto', cmap='viridis')

                        plt.xlabel("Time [s]", fontdict={"size": 30})
                        plt.ylabel("Frequency [Hz]", fontdict={"size": 30})
                        plt.yticks(np.arange(0, 512, 30), fontsize= 20)
                        plt.ylim(1, 100)
                        plt.xticks(fontsize= 20)
            
            fig.suptitle(f"Time Frequency sub-{sub}, {hem} hemisphere, fs = 250 Hz, artefact-free", fontsize=55, y=1.02)
            plt.show()

            fig.savefig(os.path.join(figures_path, f"Time_Frequency_sub{sub}_{hem}_filtered_250Hz_resampled_artefact_free.png"),
                        bbox_inches="tight")
    
    # save dataframes
    group_data_path = os.path.join(group_results_path, f"externalized_preprocessed_data_artefact_free.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(artefact_free_dataframe, file)

    print(f"externalized_preprocessed_data_artefact_free.pickle",
            f"\nwritten in: {group_results_path}" )
    
    return artefact_free_dataframe
                    


def fourier_transform_to_psd(
        
):
    
    """
    Load the artefact free data (only the sfreq=250 Hz data is artefact-free!!): 
        - 2 min rest
        - artefacts removed
        - resampled to 250 Hz
        - 2 versions: filtered (notch, band-pass) and unfiltered
    
    calculate the power spectrum for both filtered and unfiltered LFP:
        - window length = 250 # 1 second window length
        - overlap = window_length // 4 # 25% overlap
        - window = hann(window_length, sym=False)
        - frequencies, times, Zxx = scipy.signal.spectrogram(band_pass_filtered, fs=fs, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0)
   
    Plot the Power Spectra only of the filtered sfreq=250Hz data

    Save the Power Spectra of all filtered and unfiltered data in one dataframe:
        - externalized_power_spectra_250Hz_artefact_free.pickle
    
    """

    sfreq = 250
    hemispheres = ["Right", "Left"]

    power_spectra_dict = {}

    artefact_free_lfp = load_data.load_externalized_pickle(filename="externalized_preprocessed_data_artefact_free")

    BIDS_id_unique = list(artefact_free_lfp.BIDS_id.unique())

    for bids_id in BIDS_id_unique:

        figures_path = find_folders.get_monopolar_project_path(folder="figures", sub=bids_id)
        
        # data only of one subject
        subject_data = artefact_free_lfp.loc[artefact_free_lfp.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        for hem in hemispheres:

            hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(hem_data.contact.values)
            subject_hemisphere = f"{sub}_{hem}"

            # Figure of one subject_hemisphere with all 8 channels 
            # 4 columns, 2 rows
            fig = plt.figure(figsize= (30, 30), layout="tight")

            for c, contact in enumerate(contacts):

                contact_data = hem_data.loc[hem_data.contact == contact]

                original_ch_name = contact_data.original_ch_name.values[0]

                # get LFP data from one contact
                filtered_lfp_250 = contact_data.filtered_lfp_250Hz.values[0] # to plot
                lfp_resampled_250Hz = contact_data.lfp_resampled_250Hz.values[0]
                
                loop_over_data = ["filtered", "unfiltered"]

                for filt in loop_over_data:

                    if filt == "filtered":
                        lfp_data = filtered_lfp_250
                    
                    elif filt == "unfiltered":
                        lfp_data = lfp_resampled_250Hz
                    
                    ######### short time fourier transform to calculate PSD #########
                    window_length = int(sfreq) # 1 second window length
                    overlap = window_length // 4 # 25% overlap

                    # Calculate the short-time Fourier transform (STFT) using Hann window
                    window = hann(window_length, sym=False)

                    frequencies, times, Zxx = scipy.signal.spectrogram(lfp_data, fs=sfreq, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0)
                    # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
                    # times: len=161, 0, 0.75, 1.5 .... 120.75
                    # Zxx: 126 arrays, each len=161

                    # average PSD across duration of the recording
                    average_Zxx = np.mean(Zxx, axis=1)
                    std_Zxx = np.std(Zxx, axis=1)
                    sem_Zxx = std_Zxx / np.sqrt(Zxx.shape[1])

                    # save power spectra values
                    power_spectra_dict[f"{bids_id}_{hem}_{contact}_{filt}"] = [bids_id, sub, hem, subject_hemisphere, contact, original_ch_name,
                                                                               filt, lfp_data, frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx]

                    if filt == "filtered": 
                        plt.subplot(4, 2, c+1) # row 1: 1, 2, 3, 4; row 2: 5, 6, 7, 8
                        plt.title(f"Channel {contact}", fontdict={"size": 40})
                        plt.plot(frequencies, average_Zxx)
                        plt.fill_between(frequencies, average_Zxx-sem_Zxx, average_Zxx+sem_Zxx, color='lightgray', alpha=0.5)

                        plt.xlabel("Frequency [Hz]", fontdict={"size": 30})
                        plt.ylabel("PSD", fontdict={"size": 30})
                        #plt.ylim(1, 100)
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)

            fig.suptitle(f"Power Spectrum sub-{sub}, {hem} hemisphere, fs = 250 Hz, filtered and artefact-free", fontsize=55, y=1.02)
            plt.show()

            fig.savefig(os.path.join(figures_path, f"Power_spectrum_sub{sub}_{hem}_filtered_250Hz_resampled_artefact_free.png"),
                        bbox_inches="tight") 
    

    power_spectra_df = pd.DataFrame(power_spectra_dict)
    power_spectra_df.rename(index={
        0: "BIDS_id",
        1: "subject",
        2: "hemisphere",
        3: "subject_hemisphere",
        4: "contact",
        5: "original_ch_name",
        6: "filtered",
        7: "lfp_data",
        8: "frequencies",
        9: "times",
        10: "power",
        11: "power_average_over_time",
        12: "power_std",
        13: "power_sem", 

    }, inplace=True)
    power_spectra_df = power_spectra_df.transpose()

    # save dataframes
    group_data_path = os.path.join(group_results_path, f"externalized_power_spectra_250Hz_artefact_free.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(power_spectra_df, file)

    print(f"externalized_power_spectra_250Hz_artefact_free.pickle",
            f"\nwritten in: {group_results_path}" )
    
    return power_spectra_df


def externalized_fooof_fit(
        
):
    """
    Load the Power Spectra data
        - externalized_power_spectra_250Hz_artefact_free.pickle
        - resampled sfreq=250 Hz
        - artefact-free
        - unfiltered Power Spectra

    1) First set and fit a FOOOF model without a knee -> within a frequency range from 1-95 Hz (broad frequency range for fitting the aperiodic component)
        - peak_width_limits=[2, 15.0],  # must be a list, low limit should be more than twice as frequency resolution, usually not more than 15Hz bw
        - max_n_peaks=6,                # 4, 5 sometimes misses important peaks, 6 better even though there might be more false positives in high frequencies
        - min_peak_height=0.2,          # 0.2 detects false positives in gamma but better than 0.35 missing relevant peaks in low frequencies
        - peak_threshold=2.0,           # default 2.0, lower if necessary to detect peaks more sensitively
        - aperiodic_mode="fixed",       # fitting without knee component, because there are no knees found so far in the STN
        - verbose=True,

        frequency range for parameterization: 1-95 Hz

        plot a figure with the raw Power spectrum and the fitted model 

    2) save figure into figure folder of each subject:
        figure filename: fooof_externalized_sub{subject}_{hemisphere}_{chan}_250Hz_clean.png
    
    3) Extract following parameters and save as columns into DF 
        - 0: "subject_hemisphere", 
        - 1: "subject",
        - 2: "hemisphere",
        - 3: "BIDS_id", 
        - 4: "original_ch_name",
        - 5: "contact", 
        - 6: "fooof_error", 
        - 7: "fooof_r_sq", 
        - 8: "fooof_exponent", 
        - 9: "fooof_offset", 
        - 10: "fooof_power_spectrum", # with 95 values, 1 Hz frequency resolution, so 1-95 Hz
        - 11: "periodic_plus_aperiodic_power_log",
        - 12: "fooof_periodic_flat",
        - 13: "fooof_number_peaks",
        - 14: "alpha_peak_CF_power_bandWidth",
        - 15: "low_beta_peak_CF_power_bandWidth",
        - 16: "high_beta_peak_CF_power_bandWidth",
        - 17: "beta_peak_CF_power_bandWidth",
        - 18: "gamma_peak_CF_power_bandWidth", 
        
         
    5) save Dataframe into results folder of each subject
        - filename: "fooof_externalized_sub{subject}.json"
    
    """

    freq_range = [1, 95] # frequency range to fit FOOOF model

    fooof_results_df = pd.DataFrame()
    
    common_reference_contacts = {}


    power_spectra_data = load_data.load_externalized_pickle(filename="externalized_power_spectra_250Hz_artefact_free")

    # first select only the rows with UNFILTERED DATA 
    power_spectra_data = power_spectra_data.loc[power_spectra_data.filtered == "unfiltered"]

    BIDS_id_unique = list(power_spectra_data.BIDS_id.unique())

    for bids_id in BIDS_id_unique:

        figures_path = find_folders.get_monopolar_project_path(folder="figures", sub=bids_id)
        results_path = find_folders.get_monopolar_project_path(folder="results", sub=bids_id)
        
        # data only of one subject
        subject_data = power_spectra_data.loc[power_spectra_data.BIDS_id == bids_id]
        sub = subject_data.subject.values[0]

        for hem in hemispheres:

            hem_data = subject_data.loc[subject_data.hemisphere == hem]
            contacts = list(hem_data.contact.values)
            subject_hemisphere = f"{sub}_{hem}"

            for c, contact in enumerate(contacts):

                contact_data = hem_data.loc[hem_data.contact == contact]
                original_ch_name = contact_data.original_ch_name.values[0]

                # get the data to fit
                power_spectrum = contact_data.power_average_over_time.values[0]
                freqs = contact_data.frequencies.values[0]

                # check if the power spectrum contains only 0 -> in case of a contact used as common reference
                if np.all(power_spectrum == 0):
                    common_reference_contacts[f"{sub}_{hem}_{contact}"] = [bids_id, sub, hem, contact, original_ch_name]
                    print(f"Sub-{sub}, {hem}: contact {contact} is used as common reference.")
                    continue

                ############ SET PLOT LAYOUT ############
                fig, ax = plt.subplots(4,1, figsize=(7,20))

                # Plot the unfiltered Power spectrum in first ax
                plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False,
                                ax=ax[0])
                ax[0].grid(False)

                ############ SET FOOOF MODEL ############
                
                model = fooof.FOOOF(
                        peak_width_limits=[2, 15.0],
                        max_n_peaks=6,
                        min_peak_height=0.2,
                        peak_threshold=2.0,
                        aperiodic_mode="fixed", # fitting without knee component
                        verbose=True,
                    )

                # always fit a large Frequency band, later you can select Peaks within specific freq bands
                model.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                # Plot an example power spectrum, with a model fit in second ax
                # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                model.plot(ax=ax[1], plt_log=True) # to evaluate the aperiodic component
                model.plot(ax=ax[2], plt_log=False) # To see the periodic component better without log in frequency axis
                ax[1].grid(False)
                ax[2].grid(False)

                # check if fooof attributes are None:
                if model._peak_fit is None:
                    print(f"subject {sub}, {hem}, {contact}: model peak fit is None.")
                    continue

                if model._ap_fit is None:
                    print(f"subject {sub}, {hem}, {contact}: model aperiodic fit is None.")
                    continue


                # plot only the fooof spectrum of the periodic component
                fooof_power_spectrum = 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
                plot_spectrum(np.arange(1, (len(fooof_power_spectrum)+1)), fooof_power_spectrum, log_freqs=False, log_powers=False, ax=ax[3])
                # frequencies: 1-95 Hz with 1 Hz resolution

                
                # titles
                fig.suptitle(f"sub {sub}, {hem} hemisphere, contact: {contact}",
                                        fontsize=25)

                ax[0].set_title("unfiltered, artefact-free power spectrum", fontsize=20, y=0.97, pad=-20)
                ax[3].set_title("power spectrum of periodic component", fontsize=20)

                # mark beta band
                x1 = 13
                x2 = 35
                ax[3].axvspan(x1, x2, color="whitesmoke")
                ax[3].grid(False)

                fig.tight_layout()
                fig.savefig(os.path.join(figures_path, f"fooof_externalized_sub{sub}_{hem}_{contact}_250Hz_clean.svg"), bbox_inches="tight", format="svg")
                fig.savefig(os.path.join(figures_path, f"fooof_externalized_sub{sub}_{hem}_{contact}_250Hz_clean.png"), bbox_inches="tight")

                # extract parameters from the chosen model
                # model.print_results()

                ############ SAVE APERIODIC PARAMETERS ############
                # goodness of fit
                err = model.get_params('error')
                r_sq = model.r_squared_

                # aperiodic components
                exp = model.get_params('aperiodic_params', 'exponent')
                offset = model.get_params('aperiodic_params', 'offset')

                # periodic component
                log_power_fooof_periodic_plus_aperiodic = model._peak_fit + model._ap_fit # periodic+aperiodic component in log Power axis
                fooof_periodic_component = model._peak_fit # just periodic component, flattened spectrum

                ############ SAVE ALL PEAKS IN ALPHA; HIGH AND LOW BETA ############

                number_peaks = model.n_peaks_

                # get the highest Peak of each frequency band as an array: CF center frequency, Power, BandWidth
                alpha_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(8.0, 12.0),
                    select_highest=True,
                    attribute="peak_params"
                    )

                low_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 20.0),
                    select_highest=True,
                    attribute="peak_params",
                    )

                high_beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(21.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                    )

                beta_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 35.0),
                    select_highest=True,
                    attribute="peak_params",
                    )

                gamma_peak = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(60.0, 90.0),
                    select_highest=True,
                    attribute="peak_params",
                    )

                # save all results in dictionary
                fooof_results = {
                    "BIDS_id": [bids_id],
                    "subject": [sub],
                    "hemisphere": [hem],
                    "subject_hemisphere": [subject_hemisphere],
                    "contact": [contact],
                    "original_ch_name": [original_ch_name],
                    "fooof_error": [err],
                    "fooof_r_sq": [r_sq],
                    "fooof_exponent": [exp],
                    "fooof_offset": [offset],
                    "fooof_power_spectrum": [fooof_power_spectrum],
                    "periodic_plus_aperiodic_power_log": [log_power_fooof_periodic_plus_aperiodic],
                    "fooof_periodic_flat": [fooof_periodic_component],
                    "fooof_number_peaks": [number_peaks],
                    "alpha_peak_CF_power_bandWidth": [alpha_peak],
                    "low_beta_peak_CF_power_bandWidth": [low_beta_peak],
                    "high_beta_peak_CF_power_bandWidth": [high_beta_peak],
                    "beta_peak_CF_power_bandWidth": [beta_peak],
                    "gamma_peak_CF_power_bandWidth": [gamma_peak]
                }

                fooof_results_single = pd.DataFrame(fooof_results)
                fooof_results_df = pd.concat([fooof_results_df, fooof_results_single], ignore_index=True)

    # save DF as pickle
    fooof_results_df_path = os.path.join(group_results_path, f"fooof_externalized_group.pickle")
    with open(fooof_results_df_path, "wb") as file:
        pickle.dump(fooof_results_df, file)

    print(f"fooof_externalized_group.pickle",
            f"\nwritten in: {group_results_path}" )

    
    # bids_id, sub, hem, contact, original_ch_name
    common_reference_contacts_df = pd.DataFrame(common_reference_contacts)
    common_reference_contacts_df.rename(index={
        0: "BIDS_id",
        1: "subject",
        2: "hemisphere",
        3: "contact",
        4: "original_ch_name"}, inplace=True)
    
    common_reference_contacts_df = common_reference_contacts_df.transpose()

    # save DF
    common_reference_contacts_path = os.path.join(group_results_path, f"externalized_contacts_common_reference.pickle")
    with open(common_reference_contacts_path, "wb") as file:
        pickle.dump(common_reference_contacts_df, file)

    print(f"externalized_contacts_common_reference.pickle",
            f"\nwritten in: {group_results_path}" )
    
    return fooof_results_df



# def write_group_fooof(
        
# ):
#     """
#     Loop through each subject folder in the results path, check if there is a FOOOF JSON file, 
#         - Load the fooof_externalized_sub{sub}.json file from each subject

#     1) 



#     """

#     group_fooof_dataframe = pd.DataFrame()

#     # loop through each folder in results path
#     sub_folders = os.listdir(group_results_path)
#     sub_folders = [folder for folder in sub_folders if folder.startswith("sub-")]

#     for sub_dir in sub_folders:

#         bids_id = sub_dir.split('-')
#         bids_id = bids_id[1]

#         sub_path = os.path.join(group_results_path, sub_dir)
#         files = os.listdir(sub_path)

#         found = False

#         for file in files:

#             if file.startswith("fooof_externalized") and file.endswith(".json"):
#                 found = True
#                 filename = file
#                 break

#         # check if the file was not found
#         if not found:
#             print(f"No FOOOF externalized JSON file in the results folder: {sub_dir}")
#             continue

#         # continues only, if file exists
#         # load the file
#         with open(os.path.join(sub_path, filename)) as f:
#             data = json.load(f)
#             data = pd.DataFrame(data)
        
#         # concatenate all dataframes together
#         group_fooof_dataframe = pd.concat([group_fooof_dataframe, data], ignore_index=True)
    
#     # save the new dataframe
#     group_fooof_dataframe.to_json(os.path.join(group_results_path, f"fooof_externalized_group_data.json"))

#     return group_fooof_dataframe









        
















            
            








            



    















    














   



    






