""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram

from ..tfr import feats_ssd as feats_ssd
from ..utils import find_folders as find_folders
from ..utils import load_data_files as load_data
from ..utils import loadResults as loadResults

GROUP_RESULTS_PATH = find_folders.get_monopolar_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_monopolar_project_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

# list of subjects with no BIDS transformation yet -> load these via poly5reader instead of BIDS
SUBJECTS_NO_BIDS = ["24", "28", "29", "48", "49", "56"]


# get index of each channel and get the corresponding LFP data
# plot filtered channels 1-8 [0]-[7] Right and 9-16 [8]-[15]
# butterworth filter: band pass -> filter order = 5, high pass 5 Hz, low-pass 95 Hz
def band_pass_filter_externalized(fs: int, signal: np.array):
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
    filter_order = 3  # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 5  # 5Hz high-pass filter
    frequency_cutoff_high = 95  # 95 Hz low-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(
        filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs
    )
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal)

    return band_pass_filtered


def high_pass_filter_externalized(fs: int, signal: np.array):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a band pass filter to the signal
        - 1 Hz high pass
        - filter order: 3
    """
    # parameters
    filter_order = 5  # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 1  # 1Hz high-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low), btype='highpass', output='ba', fs=fs)
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal)

    return band_pass_filtered


# notch filter: 50 Hz
def notch_filter_externalized(fs: int, signal: np.array):
    """
    Input:
        - fs: sampling frequency of the signal
        - signal: array of the signal

    Applying a notch filter to the signal


    """

    # parameters
    notch_freq = 50  # 50 Hz line noise in Europe
    Q = 30  # Q factor for notch filter

    # apply notch filter
    b, a = scipy.signal.iirnotch(w0=notch_freq, Q=Q, fs=fs)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)

    return filtered_signal


def load_patient_data(patient: str):
    """
    Input:
        - patient: str, e.g. "25"

    First check, if the patient is in the list with no BIDS data yet.
    If no BIDS data exists:
        - load data with Poly5Reader
        - rename the channels, so that they match to the BIDS channel names

    If BIDS data exists:
        - load data with mne bids


    return:
        - mne_data as an MNE raw object
        - subject info (empty if not loaded with bids)
        - bids_ID

    """

    # rename channel names, if files were loaded via Poly5reader
    channel_mapping_1 = {
        'LFPR1STNM': 'LFP_R_01_STN_MT',
        'LFPR2STNM': 'LFP_R_02_STN_MT',
        'LFPR3STNM': 'LFP_R_03_STN_MT',
        'LFPR4STNM': 'LFP_R_04_STN_MT',
        'LFPR5STNM': 'LFP_R_05_STN_MT',
        'LFPR6STNM': 'LFP_R_06_STN_MT',
        'LFPR7STNM': 'LFP_R_07_STN_MT',
        'LFPR8STNM': 'LFP_R_08_STN_MT',
        'LFPL1STNM': 'LFP_L_01_STN_MT',
        'LFPL2STNM': 'LFP_L_02_STN_MT',
        'LFPL3STNM': 'LFP_L_03_STN_MT',
        'LFPL4STNM': 'LFP_L_04_STN_MT',
        'LFPL5STNM': 'LFP_L_05_STN_MT',
        'LFPL6STNM': 'LFP_L_06_STN_MT',
        'LFPL7STNM': 'LFP_L_07_STN_MT',
        'LFPL8STNM': 'LFP_L_08_STN_MT',
    }

    channel_mapping_2 = {
        'LFP_0_R_S': 'LFP_R_01_STN_MT',
        'LFP_1_R_S': 'LFP_R_02_STN_MT',
        'LFP_2_R_S': 'LFP_R_03_STN_MT',
        'LFP_3_R_S': 'LFP_R_04_STN_MT',
        'LFP_4_R_S': 'LFP_R_05_STN_MT',
        'LFP_5_R_S': 'LFP_R_06_STN_MT',
        'LFP_6_R_S': 'LFP_R_07_STN_MT',
        'LFP_7_R_S': 'LFP_R_08_STN_MT',
        'LFP_0_L_S': 'LFP_L_01_STN_MT',
        'LFP_1_L_S': 'LFP_L_02_STN_MT',
        'LFP_2_L_S': 'LFP_L_03_STN_MT',
        'LFP_3_L_S': 'LFP_L_04_STN_MT',
        'LFP_4_L_S': 'LFP_L_05_STN_MT',
        'LFP_5_L_S': 'LFP_L_06_STN_MT',
        'LFP_6_L_S': 'LFP_L_07_STN_MT',
        'LFP_7_L_S': 'LFP_L_08_STN_MT',
    }

    # check if patient is in the list with no BIDS yet
    if patient in SUBJECTS_NO_BIDS:
        mne_data = load_data.load_externalized_Poly5_files(sub=patient)

        # rename channels, first check which channel_mapping is correct
        found = False
        for name in mne_data.info["ch_names"]:
            if name in channel_mapping_1:
                found = True
                channel_mapping = channel_mapping_1
                break

            elif name in channel_mapping_2:
                found = True
                channel_mapping = channel_mapping_2
                break

        if found == False:
            print(f"Channel names of sub-{patient} are not in channel_mapping_1 or channel_mapping_2.")

        mne_data.rename_channels(channel_mapping)

        subject_info = "no_bids"

        # bids_ID
        bids_ID = f"sub-noBIDS{patient}"
        print(f"subject {patient} with bids ID {bids_ID} was loaded.")

    else:
        mne_data = load_data.load_BIDS_externalized_vhdr_files(sub=patient)

        subject_info = mne_data.info["subject_info"]
        bids_ID = mne_data.info["subject_info"]["his_id"]
        print(f"subject {patient} with bids ID {bids_ID} was loaded.")

    return {"mne_data": mne_data, "subject_info": subject_info, "bids_ID": bids_ID}


def save_result_dataframe_as_pickle(data: pd.DataFrame, filename: str):
    """
    Input:
        - data: must be a pd.DataFrame()
        - filename: str, e.g."externalized_preprocessed_data"

    picklefile will be written in the group_results_path:

    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"{filename}.pickle", f"\nwritten in: {GROUP_RESULTS_PATH}")


def save_fig_png_and_svg(path: str, filename: str, figure=None):
    """
    Input:
        - path: str
        - filename: str
        - figure: must be a plt figure

    """

    figure.savefig(
        os.path.join(path, f"{filename}.svg"),
        bbox_inches="tight",
        format="svg",
    )

    figure.savefig(
        os.path.join(path, f"{filename}.png"),
        bbox_inches="tight",
    )

    print(f"Figures {filename}.svg and {filename}.png", f"\nwere written in: {path}.")


def assign_cluster(value):
    """
    This function takes an input float value and assigns a mathing cluster value between 1 and 3

        - value <= 0.4:         cluster 3
        - 0.4 < value <= 0.7:   cluster 2
        - 0.7 < value:          cluster 1

    """

    if value <= 0.4:
        return 3

    elif 0.4 < value <= 0.7:
        return 2

    elif 0.7 < value:
        return 1
