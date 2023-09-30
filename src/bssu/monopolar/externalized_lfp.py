""" Read and preprocess externalized LFPs"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import mne
import pickle

# internal Imports
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults
from .. utils import load_data_files as load_data



group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")
group_figures_path = find_folders.get_monopolar_project_path(folder="GroupFigures")

patient_metadata = load_data.load_patient_metadata_externalized()

# get index of each channel and get the corresponding LFP data
# plot filtered channels 1-8 [0]-[7] Right and 9-16 [8]-[15] 
# butterworth filter: band pass -> filter order = 5, high pass 5 Hz, low-pass 95 Hz
# nodge filter: 48-52 Hz


# detect artefacts 
# remove artefacts (cut out)

# save the cleaned data unfiltered

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



    """

    group_data = pd.DataFrame()
    group_originial_rec_info = pd.DataFrame()
    
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

        # downsample from TMSi sampling frequency to 250 sfreq (like Percept)
        resampled_data = mne_data.copy().resample(sfreq=250)
        # cropped data should have 30000 samples (2 min of sfreq 250)

        ########## save processed LFP data in dataframe ##########
        for idx, lfp in enumerate(ch_names_LFP):

            lfp_data = resampled_data[idx][0][0] # LFP data from each channel
            time_stamps = resampled_data[idx][1] # time stamps
            sfreq = resampled_data.info["sfreq"]

            # ch_name corresponding to Percept -> TODO: is the order always correct???? 02 = 1A? could it also be 1B?
            if "_01_" in lfp:
                monopol_chan_name = "0"
            
            elif "_02_" in lfp:
                monopol_chan_name = "1A"
            
            elif "_03_" in lfp:
                monopol_chan_name = "1B"

            elif "_04_" in lfp:
                monopol_chan_name = "1C"

            elif "_05_" in lfp:
                monopol_chan_name = "2A"

            elif "_06_" in lfp:
                monopol_chan_name = "2B"

            elif "_07_" in lfp:
                monopol_chan_name = "2C"

            elif "_08_" in lfp:
                monopol_chan_name = "3"

            # hemisphere
            if "_L_" in lfp:
                hemisphere = "Left"
            
            elif "_R_" in lfp:
                hemisphere = "Right"

            processed_recording[f"{lfp}"] = [bids_ID, subject, hemisphere, lfp, monopol_chan_name, lfp_data, time_stamps, sfreq]
        
        preprocessed_dataframe = pd.DataFrame(processed_recording)
        preprocessed_dataframe.rename(index={
            0: "BIDS_id",
            1: "subject",
            2: "hemisphere",
            3: "original_ch_name",
            4: "contact",
            5: "cropped_lfp",
            6: "time_stamps",
            7: "sfreq"
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

    return {
        "group_originial_rec_info": group_originial_rec_info,
        "group_data": group_data
    }














    ####### plot all channels and check for any artefacts #######



    






