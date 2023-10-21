""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram

from ..utils import find_folders as find_folders

GROUP_RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_local_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

SUBJECTS = [
    "017",
    "019",
    "021",
    "024",
    "025",
    "028",
    "029",
    "030",
    "031",
    "032",
    "033",
    "036",
    "040",
    "041",
    "045",
    "047",
    "048",
    "049",
    "050",
    "052",
    "055",
    "059",
    "060",
    "061",
    "062",
    "063",
    "065",
    "066",
]
# excluded subjects (ECG artifacts): "026", "038",


# get index of each channel and get the corresponding LFP data
# plot filtered channels 1-8 [0]-[7] Right and 9-16 [8]-[15]
# butterworth filter: band pass -> filter order = 5, high pass 5 Hz, low-pass 95 Hz
def band_pass_filter_percept(fs: int, signal: np.array):
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
    filter_order = 5  # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 5  # 5Hz high-pass filter
    frequency_cutoff_high = 95  # 95 Hz low-pass filter

    # create and apply the filter
    b, a = scipy.signal.butter(
        filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs
    )
    band_pass_filtered = scipy.signal.filtfilt(b, a, signal)

    return band_pass_filtered


def high_pass_filter_percept(fs: int, signal: np.array):
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
