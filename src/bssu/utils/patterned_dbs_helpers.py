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

# import py_perceive
from PerceiveImport.classes import main_class

GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

HEMISPHERES = ["Right", "Left"]

SUBJECTS = ["075"]


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


def load_source_json_patterned_dbs(sub: str, incl_session: list, run: str):
    """ """

    # load with pyPerceive
    py_perceive_data = main_class.PerceiveData(
        sub="075",
        incl_modalities=["streaming"],
        incl_session=["fu3m"],
        incl_condition=["m0s1"],
        incl_task=["rest"],
        import_json=True,
        # warn_for_metaNaNs = True,
        # allow_NaNs_in_metadata = True,
    )

    # get JSON
    json_data = {
        "1": py_perceive_data.streaming.fu3m.m0s1.rest.run1.json,
        "2": py_perceive_data.streaming.fu3m.m0s1.rest.run2.json,
    }

    return json_data[run]


def load_pickle_files(filename: str):
    """
    Input:
        - filename: str, e.g.
            "streaming_info_patterned_pilot_sub-075"
            "raw_objects_patterned_pilot_sub-075"
            "beta_baseline_patterned_pilot_sub-075"

    Returns:
        - data
    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "rb") as file:
        data = pickle.load(file)

    return data


def load_excel_files(filename: str):
    """ """

    dbs_turned_off_sheet = ["streaming_dbs_turned_OFF"]

    # find the path to the results folder
    path = find_folders.get_patterned_dbs_project_path(folder="data")

    # create filename
    f_name = f"{filename}.xlsx"

    if filename in dbs_turned_off_sheet:
        sheet_name = "dbs_OFF"

    filepath = os.path.join(path, f_name)

    # load the file
    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=sheet_name)
    print("Excel file loaded: ", f_name, "\nloaded from: ", path)

    return data
