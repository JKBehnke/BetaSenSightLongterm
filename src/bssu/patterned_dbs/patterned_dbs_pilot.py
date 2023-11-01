""" Patterned DBS Pilot"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram


# internal Imports
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import patterned_dbs_helpers as helpers

HEMISPHERES = ["Right", "Left"]
SAMPLING_FREQ = 250

FREQUENCY_BANDS = {
    "beta": [13, 36],
    "low_beta": [13, 21],
    "high_beta": [21, 36],
}

GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

sub_075_pilot_streaming_dict = {
    "0": ["pre", "burstDBS", "1min", "Left"],
    "1": ["pre", "burstDBS", "1min", "Right"],
    "2": ["post", "burstDBS", "1min", "Left"],
    "3": ["post", "burstDBS", "1min", "Right"],
    "4": ["pre", "cDBS", "1min", "Left"],
    "5": ["pre", "cDBS", "1min", "Right"],
    "6": ["post", "cDBS", "1min", "Left"],
    "7": ["post", "cDBS", "1min", "Right"],
    "8": ["pre", "burstDBS", "5min", "Left"],
    "9": ["pre", "burstDBS", "5min", "Right"],
    "10": ["post", "burstDBS", "5min", "Left"],
    "11": ["post", "burstDBS", "5min", "Right"],
    "12": ["pre", "cDBS", "5min", "Left"],
    "13": ["pre", "cDBS", "5min", "Right"],
    "16": ["post", "cDBS", "5min", "Left"],
    "17": ["post", "cDBS", "5min", "Right"],
    "18": ["pre", "burstDBS", "30min", "Left"],
    "19": ["pre", "burstDBS", "30min", "Right"],
    "22": ["post", "burstDBS", "30min", "Left"],  # 8.45 min
    "23": ["post", "burstDBS", "30min", "Right"],  # 8.45 min
}


def write_json_streaming_info(sub: str, incl_session: list, run: str):
    """ """
    streaming_info = pd.DataFrame()
    raw_objects = {}

    load_json = helpers.load_source_json_patterned_dbs(sub=sub, incl_session=incl_session, run=run)

    # number of BrainSense Streamings
    # n_streamings = len(load_json["BrainSenseTimeDomain"])

    # get info of each recording
    for streaming in list(sub_075_pilot_streaming_dict.keys()):
        time_domain_data = load_json["BrainSenseTimeDomain"][int(streaming)]["TimeDomainData"]
        channel = load_json["BrainSenseTimeDomain"][int(streaming)]["Channel"]

        pre_or_post = sub_075_pilot_streaming_dict[streaming][0]
        burstDBS_or_cDBS = sub_075_pilot_streaming_dict[streaming][1]
        DBS_duration = sub_075_pilot_streaming_dict[streaming][2]
        hemisphere = sub_075_pilot_streaming_dict[streaming][3]

        # transform to mne
        units = ["µVolt"]
        scale = np.array([1e-6 if u == "µVolt" else 1 for u in units])

        info = mne.create_info(ch_names=[channel], sfreq=250, ch_types="dbs")
        raw = mne.io.RawArray(time_domain_data * np.expand_dims(scale, axis=1), info)

        # save raw
        raw_objects[streaming] = raw

        # get more info
        time_domain_dataframe = raw.to_data_frame()
        rec_duration = raw.tmax

        # save into dataframe
        streaming_data = {
            "streaming_index": [streaming],
            "original_time_domain_data": [time_domain_data],
            "channel": [channel],
            "time_domain_dataframe": [time_domain_dataframe],
            "rec_duration": [rec_duration],
            "pre_or_post": [pre_or_post],
            "burstDBS_or_cDBS": [burstDBS_or_cDBS],
            "DBS_duration": [DBS_duration],
            "hemisphere": [hemisphere],
        }

        streaming_data_single = pd.DataFrame(streaming_data)
        streaming_info = pd.concat([streaming_info, streaming_data_single], ignore_index=True)

    # save data as pickle
    helpers.save_result_dataframe_as_pickle(data=streaming_info, filename="streaming_info_patterned_pilot_sub-075")
    helpers.save_result_dataframe_as_pickle(data=raw_objects, filename="raw_objects_patterned_pilot_sub-075")

    return {
        "streaming_info": streaming_info,
        "raw": raw_objects,
    }


def figure_layout_time_frequency():
    """ """
    ########################### Figure Layout ###########################
    # set layout for figures: using the object-oriented interface

    cols = ["pre-DBS", "post-DBS"]
    rows = ["c-DBS", "burst-DBS"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.setp(axes.flat, xlabel='Time [sec]', ylabel='Frequency [Hz]')

    pad = 5  # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords='axes fraction',
            textcoords='offset points',
            size='large',
            ha='center',
            va='baseline',
        )

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            size='large',
            ha='right',
            va='center',
        )

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    # fig.suptitle(f"sub-075 3MFU pilot")

    return fig, axes


def plot_time_frequency(dbs_duration: str):
    """
    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"


    """
    ########################### Load data ###########################

    cDBS_or_burst_DBS = ["cDBS", "burstDBS"]

    streaming_data = helpers.load_pickle_files(filename="streaming_info_patterned_pilot_sub-075")

    # select for dbs_duration
    streaming_data = streaming_data[streaming_data["DBS_duration"] == dbs_duration]

    for hem in HEMISPHERES:
        # figure layout
        fig, axes = figure_layout_time_frequency()

        # select for hemisphere
        hem_data = streaming_data[streaming_data["hemisphere"] == hem]

        if dbs_duration == "30min":
            # select for cDBS and burstDBS and pre and post DBS

            DBS_data = hem_data[hem_data["burstDBS_or_cDBS"] == "burstDBS"]
            pre_DBS = DBS_data[DBS_data["pre_or_post"] == "pre"]
            post_DBS = DBS_data[DBS_data["pre_or_post"] == "post"]

            # get data
            pre_DBS_data = pre_DBS["original_time_domain_data"].values[0]
            post_DBS_data = post_DBS["original_time_domain_data"].values[0]

            # band-pass filter
            pre_DBS_filtered = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=pre_DBS_data)
            post_DBS_filtered = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=post_DBS_data)

            # plot TF
            noverlap = 0

            axes[1, 0].specgram(
                x=pre_DBS_filtered, Fs=SAMPLING_FREQ, noverlap=noverlap, cmap='viridis', vmin=-25, vmax=10
            )
            axes[1, 1].specgram(
                x=post_DBS_filtered, Fs=SAMPLING_FREQ, noverlap=noverlap, cmap='viridis', vmin=-25, vmax=10
            )

            axes[1, 0].grid(False)
            axes[1, 1].grid(False)

            fig.suptitle(f"sub-075 3MFU pilot {hem} hemisphere {dbs_duration} DBS duration")
            helpers.save_fig_png_and_svg(
                path=GROUP_FIGURES_PATH,
                filename=f"time_frequency_plot_sub-075_{hem}_3MFU_pilot_{dbs_duration}",
                figure=fig,
            )

        else:
            for dbs, dbs_type in enumerate(cDBS_or_burst_DBS):
                # select for cDBS and burstDBS and pre and post DBS

                DBS_data = hem_data[hem_data["burstDBS_or_cDBS"] == dbs_type]
                pre_DBS = DBS_data[DBS_data["pre_or_post"] == "pre"]
                post_DBS = DBS_data[DBS_data["pre_or_post"] == "post"]

                # get data
                pre_DBS_data = pre_DBS["original_time_domain_data"].values[0]
                post_DBS_data = post_DBS["original_time_domain_data"].values[0]

                # band-pass filter
                pre_DBS_filtered = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=pre_DBS_data)
                post_DBS_filtered = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=post_DBS_data)

                # plot TF
                noverlap = 0

                axes[dbs, 0].specgram(
                    x=pre_DBS_filtered, Fs=SAMPLING_FREQ, noverlap=noverlap, cmap='viridis', vmin=-25, vmax=10
                )
                axes[dbs, 1].specgram(
                    x=post_DBS_filtered, Fs=SAMPLING_FREQ, noverlap=noverlap, cmap='viridis', vmin=-25, vmax=10
                )

                axes[dbs, 0].grid(False)
                axes[dbs, 1].grid(False)

            fig.suptitle(f"sub-075 3MFU pilot {hem} hemisphere {dbs_duration} DBS duration")
            helpers.save_fig_png_and_svg(
                path=GROUP_FIGURES_PATH,
                filename=f"time_frequency_plot_sub-075_{hem}_3MFU_pilot_{dbs_duration}",
                figure=fig,
            )


def fourier_transform(time_domain_data: np.array):
    """ """

    window_length = int(SAMPLING_FREQ)  # 1 second window length
    overlap = window_length // 2  # 50% overlap e.g. 2min pre-DBS baseline -> 239 x 0.5 seconds = 120 seconds

    # Calculate the short-time Fourier transform (STFT) using Hann window
    window = hann(window_length, sym=False)

    frequencies, times, Zxx = scipy.signal.spectrogram(
        time_domain_data, fs=SAMPLING_FREQ, window=window, noverlap=overlap, scaling="density", mode="psd", axis=0
    )

    # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
    # times: len=161, 0, 0.75, 1.5 .... 120.75
    # Zxx: 126 arrays, each len=239

    # average PSD across duration of the recording
    average_Zxx = np.mean(Zxx, axis=1)
    std_Zxx = np.std(Zxx, axis=1)
    sem_Zxx = std_Zxx / np.sqrt(Zxx.shape[1])

    return frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx


def normalize_to_psd(to_sum: str, power_spectrum: np.array):
    """ """

    if to_sum == "40_to_90":
        # sum frequencies 40-90 Hz
        sum_40_90 = np.sum(power_spectrum[40:90])
        # normalize
        normalized_power_spectrum = power_spectrum / sum_40_90

    elif to_sum == "5_to_95":
        # sum frequencies 5-95 Hz
        sum_5_95 = np.sum(power_spectrum[5:95])
        # normalize
        normalized_power_spectrum = power_spectrum / sum_5_95

    return normalized_power_spectrum


def calculate_beta_baseline(DBS_duration: str, burstDBS_or_cDBS: str, filtered: str):
    """
    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"
        - cDBS_or_burst_DBS: str, e.g. "cDBS", "burstDBS"
        - filtered: str, e.g. "band-pass_5_95" or "unfiltered"

    1) select for dbs_duration, cDBS or burstDBS, pre DBS
    2) get time domain data, only take the last 120 seconds of the original time domain data (30000 samples = 2 minutes)
    3) band-pass filter 5-95 Hz
    4) calculate PSD


    """

    pre_DBS_baseline = pd.DataFrame()

    streaming_data = helpers.load_pickle_files(filename="streaming_info_patterned_pilot_sub-075")

    # select for dbs_duration
    streaming_data = streaming_data[streaming_data["DBS_duration"] == DBS_duration]

    # select for cDBS and burstDBS and pre DBS
    streaming_data = streaming_data[streaming_data["burstDBS_or_cDBS"] == burstDBS_or_cDBS]
    streaming_data = streaming_data[streaming_data["pre_or_post"] == "pre"]

    for hem in HEMISPHERES:
        hem_data = streaming_data[streaming_data["hemisphere"] == hem]

        # get data
        time_domain_data = hem_data.original_time_domain_data.values[0]

        # only take the last 120 seconds of the original time domain data (30000 samples = 2 minutes)
        time_domain_data = np.array(time_domain_data[-30000:])

        if filtered == "band-pass_5_95":
            # band-pass filter 5-95 Hz
            time_domain_data = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=time_domain_data)

        # calculate PSD
        frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx = fourier_transform(time_domain_data)

        # normalize PSD
        normalized_to_5_95 = normalize_to_psd(to_sum="5_to_95", power_spectrum=average_Zxx)
        normalized_to_40_90 = normalize_to_psd(to_sum="40_to_90", power_spectrum=average_Zxx)

        # calculate average of beta range 13-35 Hz
        for freq in FREQUENCY_BANDS.keys():
            f_average_rel_to_5_95 = np.mean(normalized_to_5_95[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]])
            f_average_rel_to_40_90 = np.mean(normalized_to_40_90[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]])
            f_average_raw = np.mean(average_Zxx[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]])

            # save data
            pre_DBS_baseline_dict = {
                "hemisphere": [hem],
                "DBS_duration": [DBS_duration],
                "burstDBS_or_cDBS": [burstDBS_or_cDBS],
                "filtered": [filtered],
                "freq_band": [freq],
                "frequencies": [frequencies],
                "times": [times],
                "Zxx": [Zxx],
                "average_Zxx": [average_Zxx],
                "std_Zxx": [std_Zxx],
                "sem_Zxx": [sem_Zxx],
                "normalized_to_5_95": [normalized_to_5_95],
                "normalized_to_40_90": [normalized_to_40_90],
                "f_average_rel_to_5_95": [f_average_rel_to_5_95],
                "f_average_rel_to_40_90": [f_average_rel_to_40_90],
                "f_average_raw": [f_average_raw],
            }

            pre_DBS_baseline_single = pd.DataFrame(pre_DBS_baseline_dict)
            pre_DBS_baseline = pd.concat([pre_DBS_baseline, pre_DBS_baseline_single], ignore_index=True)

    return pre_DBS_baseline


def concatenate_beta_baseline_data():
    """ """

    DBS_duration = ["1min", "5min"]
    burstDBS_or_cDBS = ["cDBS", "burstDBS"]
    filtered = ["band-pass_5_95", "unfiltered"]

    beta_baseline = pd.DataFrame()

    for duration in DBS_duration:
        for dbs_type in burstDBS_or_cDBS:
            for filt in filtered:
                pre_DBS_baseline = calculate_beta_baseline(
                    DBS_duration=duration, burstDBS_or_cDBS=dbs_type, filtered=filt
                )

                beta_baseline = pd.concat([beta_baseline, pre_DBS_baseline], ignore_index=True)

    for filt in filtered:
        burstDBS_30min = calculate_beta_baseline(DBS_duration="30min", burstDBS_or_cDBS="burstDBS", filtered=filt)
        beta_baseline = pd.concat([beta_baseline, burstDBS_30min], ignore_index=True)

    # save data as pickle
    helpers.save_result_dataframe_as_pickle(data=beta_baseline, filename="beta_baseline_patterned_pilot_sub-075")

    return beta_baseline


def load_value_beta_baseline(
    DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str, normalized: str, freq_band: str, filtered: str
):
    """
    Input:
        - DBS_duration: str, e.g. "1min", "5min", "30min"
        - burstDBS_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"
        - normalized: str, e.g. "normalized_to_5_95", "normalized_to_40_90", "not_normalized"
        - freq_band: str, e.g. "beta", "low_beta", "high_beta"
        - filtered: str, e.g. "band-pass_5_95", "unfiltered"

    """

    normalize_dict = {
        "normalized_to_5_95": "f_average_rel_to_5_95",
        "normalized_to_40_90": "f_average_rel_to_40_90",
        "not_normalized": "f_average_raw",
    }

    ############## load the beta baseline of the given recording ##############
    beta_baseline = helpers.load_pickle_files(filename="beta_baseline_patterned_pilot_sub-075")
    beta_baseline = beta_baseline[beta_baseline["DBS_duration"] == DBS_duration]
    beta_baseline = beta_baseline[beta_baseline["burstDBS_or_cDBS"] == burstDBS_or_cDBS]
    beta_baseline = beta_baseline[beta_baseline["hemisphere"] == hemisphere]
    beta_baseline = beta_baseline[beta_baseline["filtered"] == filtered]
    beta_baseline = beta_baseline[beta_baseline["freq_band"] == freq_band]

    return beta_baseline[normalize_dict[normalized]].values[0]


def plot_post_dbs_time_series(DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str):
    """
    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"
        - burst_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"

    plot the raw time series, manually write the time in seconds when stimulation is turned off = start for collecting beta post-DBS


    """

    # load the raw object
    all_raw_objects = helpers.load_pickle_files(filename="raw_objects_patterned_pilot_sub-075")

    # find correct key by input values
    for key, value in sub_075_pilot_streaming_dict.items():
        if value[0] == "post" and value[1] == burstDBS_or_cDBS and value[2] == DBS_duration and value[3] == hemisphere:
            raw = all_raw_objects[key]

    raw.plot()

    return raw


def get_post_dbs_time_series(DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str):
    """
    Input:
        - DBS_duration: str, e.g. "1min", "5min", "30min"
        - burstDBS_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"
    """

    ############## load the excel file to get the index when stimulation is turned OFF ##############
    dbs_turned_off = helpers.load_excel_files(filename="streaming_dbs_turned_OFF")

    # select
    dbs_turned_off = dbs_turned_off[dbs_turned_off["DBS_duration"] == DBS_duration]
    dbs_turned_off = dbs_turned_off[dbs_turned_off["burstDBS_or_cDBS"] == burstDBS_or_cDBS]
    dbs_turned_off = dbs_turned_off[dbs_turned_off["hemisphere"] == hemisphere]

    # get index when stimulation is turned off
    dbs_OFF_sec = dbs_turned_off["DBS_OFF_sec"].values[0]

    ############## load the time domain data ##############
    streaming_data = helpers.load_pickle_files(filename="streaming_info_patterned_pilot_sub-075")

    # select
    streaming_data = streaming_data[streaming_data["pre_or_post"] == "post"]
    streaming_data = streaming_data[streaming_data["DBS_duration"] == DBS_duration]
    streaming_data = streaming_data[streaming_data["burstDBS_or_cDBS"] == burstDBS_or_cDBS]
    streaming_data = streaming_data[streaming_data["hemisphere"] == hemisphere]

    # get the time domain data
    streaming_data = streaming_data["original_time_domain_data"].values[0]

    # now get the index of the time domain matching the dbs_OFF_sec
    dbs_OFF_index = int(dbs_OFF_sec * SAMPLING_FREQ)  # this is start of the beta post-DBS readout

    # cut the time domain data at the dbs_OFF_index and take the following 3 min (45000 samples)
    streaming_data = streaming_data[dbs_OFF_index : dbs_OFF_index + 45000]

    return streaming_data


def calculate_rel_beta_post_dbs(
    DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str, normalized: str, freq_band: str, filtered: str
):
    """

    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"
        - burst_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"
        - normalized: str, e.g. "normalized_to_5_95", "normalized_to_40_90", "not_normalized"
        - freq_band: str, e.g. "beta", "low_beta", "high_beta"
        - filtered: str, e.g. "band-pass_5_95", "unfiltered"

    1) load the post DBS time series
    2) load the beta baseline of the given recording: band-pass-filtered 5-95 Hz
    """

    post_DBS_dataframe = pd.DataFrame()

    ############## load the post DBS time series ##############

    normalized_dict = {
        "normalized_to_5_95": "5_to_95",
        "normalized_to_40_90": "40_to_90",
    }
    # length = 3 minutes (45000 samples)
    streaming_data = get_post_dbs_time_series(
        DBS_duration=DBS_duration, burstDBS_or_cDBS=burstDBS_or_cDBS, hemisphere=hemisphere
    )

    ############## load the beta baseline of the given recording ##############
    beta_baseline = load_value_beta_baseline(
        DBS_duration=DBS_duration,
        burstDBS_or_cDBS=burstDBS_or_cDBS,
        hemisphere=hemisphere,
        normalized=normalized,
        freq_band=freq_band,
        filtered=filtered,
    )

    if filtered == "band-pass_5_95":
        # band-pass filter 5-95 Hz
        time_domain_data = helpers.band_pass_filter_percept(fs=SAMPLING_FREQ, signal=np.array(streaming_data))

    elif filtered == "unfiltered":
        time_domain_data = np.array(streaming_data)

    # calculate PSD from the postDBS time series (3 minutes = 45000 samples)
    frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx = fourier_transform(time_domain_data)
    # Zxx = 126 arrays, each len=359 -> 359*0.5s = 179.5s = 2.99 min

    # from Zxx get the power spectra for each 0.5 seconds
    power_spectra_half_sec = np.transpose(Zxx)  # 359 arrays, each len=126

    # save data
    post_DBS_dict = {
        "hemisphere": [hemisphere],
        "DBS_duration": [DBS_duration],
        "burstDBS_or_cDBS": [burstDBS_or_cDBS],
        "filtered": [filtered],
        "freq_band": [freq_band],
        "frequencies": [frequencies],
        "times": [times],
        "Zxx": [Zxx],
        "average_Zxx": [average_Zxx],
        "std_Zxx": [std_Zxx],
        "sem_Zxx": [sem_Zxx],
        "power_spectra_half_sec": [power_spectra_half_sec],
    }

    post_DBS_single_dataframe = pd.DataFrame(post_DBS_dict)
    post_DBS_dataframe = pd.concat([post_DBS_dataframe, post_DBS_single_dataframe], ignore_index=True)

    # normalize PSD for each 0.5 seconds

    half_sec_spectra_df = pd.DataFrame()

    for half_sec_psd in np.arange(0, len(power_spectra_half_sec), 1):
        single_spectrum = power_spectra_half_sec[half_sec_psd]
        seconds_post_DBS_OFF = times[half_sec_psd]

        # normalize PSD
        if normalized != "not_normalized":
            normalized_psd = normalize_to_psd(to_sum=normalized_dict[normalized], power_spectrum=single_spectrum)

        elif normalized == "not_normalized":
            normalized_psd = single_spectrum

        # calculate average of frequency range of interest
        freq_average = np.mean(normalized_psd[FREQUENCY_BANDS[freq_band][0] : FREQUENCY_BANDS[freq_band][1]])

        # calculate relative frequency power to beta baseline
        rel_freq_power = freq_average / beta_baseline

        half_sec_spectra_dict = {
            "half_sec_psd": [half_sec_psd],
            "seconds_post_DBS_OFF": [seconds_post_DBS_OFF],
            "single_spectrum": [normalized_psd],
            "freq_average": [freq_average],
            "rel_freq_average_to_preDBS": [rel_freq_power],
            "beta_baseline": [beta_baseline],
        }

        half_sec_spectra_single = pd.DataFrame(half_sec_spectra_dict)
        half_sec_spectra_df = pd.concat([half_sec_spectra_df, half_sec_spectra_single], ignore_index=True)

    return post_DBS_dataframe, half_sec_spectra_df
