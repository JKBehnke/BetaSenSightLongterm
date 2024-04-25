""" Helper functions to Read and preprocess externalized LFPs"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import shapiro
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
    return scipy.signal.filtfilt(b, a, signal)


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
    return scipy.signal.filtfilt(b, a, signal)


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

    else:
        return 1

def get_statistics(data_info:str, data=None):
    """
    Caculates statistical information of the data
    Input:
        - data_info: str, information about the data
        - data: pd.Series, data to calculate statistics
    """

    # caculate Outliers
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    threshold = 1.5 

    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr

    if type(data) == pd.Series:
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers_indices = outliers.index
        outliers_values = outliers.values

    elif type(data) == list:
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        outliers_indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        outliers_values = [x for x in data if x < lower_bound or x > upper_bound]

    else: 
        print("Data type not supported, must be pd.Series or list")

    # calculate shapiro wilk test
    stat, p = shapiro(np.array(data))

    if p > 0.05:
        print(f'Data is normally distributed (p={p})')
    else:
        print(f'Data is not normally distributed (p={p})')

    stats_dict = {
            "data_info": [data_info],
            "sample_size": [len(data)],
            "mean": [np.mean(data)],
            "std": [np.std(data)],
            "median": [np.median(data)],
            "outliers_indices": [outliers_indices],
            "outliers_values": [outliers_values],
            "n_outliers": [len(outliers)],
            "min": [np.min(data)],
            "max": [np.max(data)],
            "25%": [q25],
            "50%": [np.percentile(data, 50)],
            "75%": [q75],
            "shapiro_wilk_stat": [stat],
            "shapiro_wilk_p": [p],
            "normal_distribution": [p > 0.05],
        }
    
    stats_df = pd.DataFrame(stats_dict)

    return stats_df