""" Analysis of FOOOF beta peak parameters """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
import scipy
from scipy import stats
from scipy.integrate import simps
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import fooof
from fooof.plts.spectra import plot_spectrum

# Local Imports
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults


def highest_beta_channels_fooof(fooof_spectrum: str, fooof_version: str, highest_beta_session: str):
    """
    Load the file "fooof_model_group_data.json"
    from the group result folder

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version: "v2"
        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

    1) calculate beta average for each channel and rank within 1 stn, 1 session and 1 channel group

    2) rank beta averages and only select the channels with rank 1.0

    Output highest_beta_df
        - containing all stns, all sessions, all channels with rank 1.0 within their channel group

    """

    # load the group dataframe
    fooof_group_result = loadResults.load_group_fooof_result(fooof_version=fooof_version)

    # create new column: first duplicate column fooof power spectrum, then apply calculation to each row -> average of indices [13:36] so averaging the beta range
    fooof_group_result_copy = fooof_group_result.copy()

    if fooof_spectrum == "periodic_spectrum":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_power_spectrum"]

    elif fooof_spectrum == "periodic_plus_aperiodic":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["periodic_plus_aperiodic_power_log"]

    elif fooof_spectrum == "periodic_flat":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_periodic_flat"]

    fooof_group_result_copy["beta_average"] = fooof_group_result_copy["beta_average"].apply(
        lambda row: np.mean(row[13:36])
    )

    ################################ WRITE DATAFRAME ONLY WITH HIGHEST BETA CHANNELS PER STN | SESSION | CHANNEL_GROUP ################################
    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]

    stn_unique = fooof_group_result_copy.subject_hemisphere.unique().tolist()

    beta_rank_df = pd.DataFrame()

    for stn in stn_unique:
        stn_df = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for ses in sessions:
            # check if session exists
            if ses not in stn_df.session.values:
                continue

            else:
                stn_ses_df = stn_df.loc[stn_df.session == ses]  # df of only 1 stn and 1 session

            for group in channel_group:
                if group == "ring":
                    channels = ['01', '12', '23']

                elif group == "segm_inter":
                    channels = ["1A2A", "1B2B", "1C2C"]

                elif group == "segm_intra":
                    channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                group_comp_df = stn_ses_df.loc[
                    stn_ses_df["bipolar_channel"].isin(channels)
                ].reset_index()  # df of only 1 stn, 1 session and 1 channel group

                # rank beta average of channels within one channel group
                group_comp_df_copy = group_comp_df.copy()
                group_comp_df_copy["beta_rank"] = group_comp_df_copy["beta_average"].rank(ascending=False)

                # save to ranked_beta_df
                beta_rank_df = pd.concat([beta_rank_df, group_comp_df_copy])

    # depending on input: keep only rank 1.0 or keep postop rank 1 or 3MFU rank 1 channel
    if highest_beta_session == "highest_each_session":
        # only keep the row with beta rank 1.0
        highest_beta_df = beta_rank_df.loc[beta_rank_df.beta_rank == 1.0]

    elif highest_beta_session == "highest_postop":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:
            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]

            for ses in sessions:
                # check if postop exists
                if "postop" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue

                else:
                    postop_rank1_channels = stn_data.loc[stn_data.session == "postop"]
                    postop_rank1_channels = postop_rank1_channels.loc[postop_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]

                for group in channel_group:
                    if group == "ring":
                        channels = ['01', '12', '23']

                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]

                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in postop in this channel group
                    postop_1_row = postop_rank1_channels.loc[postop_rank1_channels["bipolar_channel"].isin(channels)]
                    postop_1_channelname = postop_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == postop_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, postop_1_row, selected_rows])

        # drop index columns
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(
            keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"]
        )

    elif highest_beta_session == "highest_fu3m":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:
            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]

            for ses in sessions:
                # # if session is postop, continue, because we´re only interested in follow ups here
                # if ses == "postop":
                #     continue

                # check if fu3m exists
                if "fu3m" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue

                else:
                    fu3m_rank1_channels = stn_data.loc[stn_data.session == "fu3m"]
                    fu3m_rank1_channels = fu3m_rank1_channels.loc[fu3m_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]

                for group in channel_group:
                    if group == "ring":
                        channels = ['01', '12', '23']

                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]

                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in fu3m in this channel group
                    fu3m_1_row = fu3m_rank1_channels.loc[fu3m_rank1_channels["bipolar_channel"].isin(channels)]
                    fu3m_1_channelname = fu3m_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == fu3m_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, fu3m_1_row, selected_rows])

        # drop index columns
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(
            keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"]
        )

    le = LabelEncoder()

    # define split array function
    split_array = lambda x: pd.Series(x)

    channel_group = ["ring", "segm_inter", "segm_intra"]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    group_dict = {}  # data with predictions

    ############################## create a single dataframe for each channel group with only one highest beta channels per STN ##############################
    for group in channel_group:
        if group == "ring":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(ring)]

        elif group == "segm_inter":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(segm_inter)]

        elif group == "segm_intra":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(segm_intra)]

        # session values have to be integers, add column group with integers for each STN electrode
        group_df_copy = group_df.copy()
        group_df_copy["group"] = le.fit_transform(
            group_df_copy["subject_hemisphere"]
        )  # adds a column "group" with integer values for each subject_hemisphere
        group_df_copy["session"] = group_df_copy.session.replace(
            to_replace=["postop", "fu3m", "fu12m", "fu18m", "fu24m"], value=[0, 3, 12, 18, 24]
        )

        # split beta, low beta and high beta peak columns into three columns each
        group_df_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = group_df_copy[
            "beta_peak_CF_power_bandWidth"
        ].apply(split_array)
        group_df_copy[["low_beta_center_frequency", "low_beta_peak_power", "low_beta_band_width"]] = group_df_copy[
            "low_beta_peak_CF_power_bandWidth"
        ].apply(split_array)
        group_df_copy[["high_beta_center_frequency", "high_beta_peak_power", "high_beta_band_width"]] = group_df_copy[
            "high_beta_peak_CF_power_bandWidth"
        ].apply(split_array)

        group_df_copy = group_df_copy.drop(columns=["alpha_peak_CF_power_bandWidth", "gamma_peak_CF_power_bandWidth"])

        # group_df_copy = group_df_copy.dropna()
        # TODO: DON`T DROP NAN!!!! instead check, if low or high beta peak exist, otherwise you loose the whole row and maybe there is no low peak but a high beta peak that you want to include into the analysis!!!!!

        group_dict[group] = group_df_copy

    return group_dict


def calculate_auc_beta_power(fooof_spectrum: str, fooof_version: str, highest_beta_session: str, around_cf: str):
    """
    calculating the area under the curve of ± 3 Hz around the center frequency of the highest beta peak in the selected FU

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version: "v2"

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - around_cf: "around_cf_at_each_session", "around_cf_at_fixed_session"



    """

    # Load the dataframe with only highest beta channels
    highest_beta_channels = highest_beta_channels_fooof(
        fooof_spectrum=fooof_spectrum, fooof_version=fooof_version, highest_beta_session=highest_beta_session
    )
    # output is a dictionary with keys "ring", "segm_inter", "segm_intra"

    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = [0, 3, 12, 18, 24]

    group_dict = {}
    no_beta_peak_dict = {}
    no_low_beta_peak_dict = {}
    no_high_beta_peak_dict = {}

    if highest_beta_session == "highest_postop":
        session_selection = 0

    elif highest_beta_session == "highest_fu3m":
        session_selection = 3

    ############################## select the center frequency of the highest peak of each session and get the area under the curve of power in a freq range +- 3 Hz around that center frequency ##############################
    # from each channel with highest beta per session get CF and area under the curve around that CF
    if highest_beta_session == "highest_each_session":
        for group in channel_group:
            group_df = highest_beta_channels[group]

            stn_unique = list(group_df.subject_hemisphere.unique())

            group_df_with_power_in_frange = pd.DataFrame()
            no_beta_peak = []
            no_low_beta_peak = []
            no_high_beta_peak = []

            # select the beta center frequency of each session seperately for each stn
            for stn in stn_unique:
                stn_data = group_df.loc[group_df.subject_hemisphere == stn]

                # check which sessions exist for this stn
                stn_ses_unique = list(stn_data.session.unique())

                for ses in stn_ses_unique:
                    ses_data = stn_data.loc[stn_data.session == ses]

                    # power spectrum
                    power = ses_data.fooof_power_spectrum.values[0]
                    ses_data_copy = ses_data.copy()

                    ####### HIGHEST BETA PEAK #######
                    # check if highest beta exist
                    if pd.isna((ses_data.iloc[0]["beta_center_frequency"])):
                        ses_beta_nan = f"{stn}_{ses}"
                        no_beta_peak.append(ses_beta_nan)

                    else:
                        # get center frequency, frequency range and power spectrum of the highest beta peak of that session
                        ses_beta_cf = round(ses_data.beta_center_frequency.values[0])
                        beta_cf_range = np.arange(ses_beta_cf - 3, ses_beta_cf + 4, 1)

                        # calculate area under the curve of power
                        beta_power_in_freq_range = power[
                            beta_cf_range[0] : (beta_cf_range[6] + 1)
                        ]  # select the power values by indexing from frequency range first until last value
                        beta_power_area_under_curve = simps(beta_power_in_freq_range, beta_cf_range)

                        ses_data_copy["round_beta_cf"] = ses_beta_cf
                        ses_data_copy["beta_power_auc_around_cf"] = beta_power_area_under_curve

                    ####### LOW BETA PEAK #######
                    # get center frequency, frequency range of the low beta peak of that session
                    # check if low beta exist
                    if pd.isna((ses_data.iloc[0]["low_beta_center_frequency"])):
                        ses_low_beta_nan = f"{stn}_{ses}"
                        no_low_beta_peak.append(ses_low_beta_nan)

                    else:
                        ses_low_beta_cf = round(ses_data.low_beta_center_frequency.values[0])
                        low_beta_cf_range = np.arange(ses_low_beta_cf - 3, ses_low_beta_cf + 4, 1)

                        # calculate area under the curve of power
                        low_beta_power_in_freq_range = power[
                            low_beta_cf_range[0] : (low_beta_cf_range[6] + 1)
                        ]  # select the power values by indexing from frequency range first until last value
                        low_beta_power_area_under_curve = simps(low_beta_power_in_freq_range, low_beta_cf_range)

                        ses_data_copy["round_low_beta_cf"] = ses_low_beta_cf
                        ses_data_copy["low_beta_power_auc_around_cf"] = low_beta_power_area_under_curve

                    ####### HIGH BETA PEAK #######
                    # get center frequency, frequency range of the high beta peak of that session
                    # check if a high beta peak exists!!
                    if pd.isna((ses_data.iloc[0]["high_beta_center_frequency"])):
                        ses_high_beta_nan = f"{stn}_{ses}"
                        no_high_beta_peak.append(ses_high_beta_nan)

                    else:
                        ses_high_beta_cf = round(ses_data.high_beta_center_frequency.values[0])
                        high_beta_cf_range = np.arange(ses_high_beta_cf - 3, ses_high_beta_cf + 4, 1)

                        # calculate area under the curve of power
                        high_beta_power_in_freq_range = power[
                            high_beta_cf_range[0] : (high_beta_cf_range[6] + 1)
                        ]  # select the power values by indexing from frequency range first until last value
                        high_beta_power_area_under_curve = simps(high_beta_power_in_freq_range, high_beta_cf_range)

                        ses_data_copy["round_high_beta_cf"] = ses_high_beta_cf
                        ses_data_copy["high_beta_power_auc_around_cf"] = high_beta_power_area_under_curve

                    group_df_with_power_in_frange = pd.concat([group_df_with_power_in_frange, ses_data_copy])

                group_dict[group] = group_df_with_power_in_frange

                # dictionary with keys for each lfp group with lists of sub-ses with no beta peak, low beta or high beta peak
                no_beta_peak_dict[group] = no_beta_peak
                no_low_beta_peak_dict[group] = no_low_beta_peak
                no_high_beta_peak_dict[group] = no_high_beta_peak

    ############################## select the center frequency of Postop OR 3MFU and get the area under the curve of power in a freq range +- 3 Hz around that center frequency ##############################
    else:
        for group in channel_group:
            group_df = highest_beta_channels[group]

            stn_unique = list(group_df.subject_hemisphere.unique())

            group_df_with_power_in_frange = pd.DataFrame()

            no_beta_peak = []
            no_low_beta_peak = []
            no_high_beta_peak = []

            # select the beta center frequency at Postop or 3MFU for every stn
            for stn in stn_unique:
                stn_data = group_df.loc[group_df.subject_hemisphere == stn]

                # check if session_selection exists for this stn
                if session_selection not in stn_data.session.values:
                    continue

                if around_cf == "around_cf_at_fixed_session":
                    # select the center frequency of the desired session selection (postop or fu3m)

                    fu_data = stn_data.loc[stn_data.session == session_selection]

                    ######### HIGHEST BETA PEAK #########
                    # check if highest beta exist
                    if pd.isna((fu_data.iloc[0]["beta_center_frequency"])):
                        ses_beta_nan = f"{stn}_{session_selection}"
                        no_beta_peak.append(ses_beta_nan)

                    else:
                        fum_beta_peak_center_frequency = round(fu_data.beta_center_frequency.values[0])
                        # now get +- 3 Hz frequency range around peak center frequency
                        fum_beta_cf_range = np.arange(
                            fum_beta_peak_center_frequency - 3, fum_beta_peak_center_frequency + 4, 1
                        )

                    ######### LOW BETA PEAK #########
                    # check if highest beta exist
                    if pd.isna((fu_data.iloc[0]["low_beta_center_frequency"])):
                        ses_low_beta_nan = f"{stn}_{session_selection}"
                        no_low_beta_peak.append(ses_low_beta_nan)

                    else:
                        fum_low_beta_peak_center_frequency = round(fu_data.low_beta_center_frequency.values[0])
                        # now get +- 3 Hz frequency range around peak center frequency
                        fum_low_beta_cf_range = np.arange(
                            fum_low_beta_peak_center_frequency - 3, fum_low_beta_peak_center_frequency + 4, 1
                        )

                    ######### HIGH BETA PEAK #########
                    # check if highest beta exist
                    if pd.isna((fu_data.iloc[0]["high_beta_center_frequency"])):
                        ses_high_beta_nan = f"{stn}_{session_selection}"
                        no_high_beta_peak.append(ses_high_beta_nan)

                    else:
                        fum_high_beta_peak_center_frequency = round(fu_data.high_beta_center_frequency.values[0])
                        # now get +- 3 Hz frequency range around peak center frequency
                        fum_high_beta_cf_range = np.arange(
                            fum_high_beta_peak_center_frequency - 3, fum_high_beta_peak_center_frequency + 4, 1
                        )

                    print(
                        f"The center frequency of session {session_selection} was taken for every session to calculate AUC"
                    )

                else:
                    print(
                        "The center frequency of the beta peak was taken from each session independently to calculate AUC."
                    )

                for (
                    ses
                ) in (
                    sessions
                ):  # for each session collect the area under the curve for the selected frequency range of one stn
                    if ses not in stn_data.session.values:
                        continue

                    else:  # now loop over each session to extract the area under the curve around the selected center frequency
                        ses_data = stn_data.loc[stn_data.session == ses]
                        power = ses_data.fooof_power_spectrum.values[0]
                        ses_data_copy = ses_data.copy()

                        if around_cf == "around_cf_at_fixed_session":
                            ########## HIGHEST BETA ##########
                            # get power area under the curve
                            # first check if variable fum_beta_cf_range exist (only if there wasn´t a NaN)
                            if pd.isna((fu_data.iloc[0]["beta_center_frequency"])):
                                ses_beta_nan = f"{stn}_{ses}"
                                no_beta_peak.append(ses_beta_nan)
                                print("no beta peak")

                            else:
                                beta_power_in_freq_range = power[
                                    fum_beta_cf_range[0] : (fum_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                beta_power_area_under_curve = simps(beta_power_in_freq_range, fum_beta_cf_range)
                                ses_data_copy[f"round_beta_cf"] = fum_beta_peak_center_frequency
                                ses_data_copy[f"beta_power_auc"] = beta_power_area_under_curve

                            ########## LOW BETA ##########
                            if pd.isna((fu_data.iloc[0]["low_beta_center_frequency"])):
                                ses_low_beta_nan = f"{stn}_{ses}"
                                no_low_beta_peak.append(ses_low_beta_nan)
                                print("no low beta peak")

                            else:
                                low_beta_power_in_freq_range = power[
                                    fum_low_beta_cf_range[0] : (fum_low_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                low_beta_power_area_under_curve = simps(
                                    low_beta_power_in_freq_range, fum_low_beta_cf_range
                                )
                                ses_data_copy[f"round_low_beta_cf"] = fum_low_beta_peak_center_frequency
                                ses_data_copy[f"low_beta_power_auc"] = low_beta_power_area_under_curve

                            ########## HIGH BETA ##########
                            if pd.isna((fu_data.iloc[0]["high_beta_center_frequency"])):
                                ses_high_beta_nan = f"{stn}_{ses}"
                                no_high_beta_peak.append(ses_high_beta_nan)
                                print("no high beta peak")

                            else:
                                high_beta_power_in_freq_range = power[
                                    fum_high_beta_cf_range[0] : (fum_high_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                high_beta_power_area_under_curve = simps(
                                    high_beta_power_in_freq_range, fum_high_beta_cf_range
                                )
                                ses_data_copy[f"round_high_beta_cf"] = fum_high_beta_peak_center_frequency
                                ses_data_copy[f"high_beta_power_auc"] = high_beta_power_area_under_curve

                        elif around_cf == "around_cf_at_each_session":
                            # select the cf of each session

                            # check if highest beta exist
                            if pd.isna((ses_data.iloc[0]["beta_center_frequency"])):
                                ses_beta_nan = f"{stn}_{ses}"
                                no_beta_peak.append(ses_beta_nan)

                            else:
                                fum_beta_peak_center_frequency = round(ses_data.beta_center_frequency.values[0])
                                # now get +- 3 Hz frequency range around peak center frequency
                                fum_beta_cf_range = np.arange(
                                    fum_beta_peak_center_frequency - 3, fum_beta_peak_center_frequency + 4, 1
                                )
                                # get power area under the curve
                                beta_power_in_freq_range = power[
                                    fum_beta_cf_range[0] : (fum_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                beta_power_area_under_curve = simps(beta_power_in_freq_range, fum_beta_cf_range)

                                ses_data_copy[f"round_beta_cf"] = fum_beta_peak_center_frequency
                                ses_data_copy[f"beta_power_auc"] = beta_power_area_under_curve

                            # check if low beta exist
                            if pd.isna((ses_data.iloc[0]["low_beta_center_frequency"])):
                                ses_low_beta_nan = f"{stn}_{ses}"
                                no_low_beta_peak.append(ses_low_beta_nan)

                            else:
                                fum_low_beta_peak_center_frequency = round(ses_data.low_beta_center_frequency.values[0])
                                # now get +- 3 Hz frequency range around peak center frequency
                                fum_low_beta_cf_range = np.arange(
                                    fum_low_beta_peak_center_frequency - 3, fum_low_beta_peak_center_frequency + 4, 1
                                )
                                # get power area under the curve
                                low_beta_power_in_freq_range = power[
                                    fum_low_beta_cf_range[0] : (fum_low_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                low_beta_power_area_under_curve = simps(
                                    low_beta_power_in_freq_range, fum_low_beta_cf_range
                                )

                                ses_data_copy[f"round_low_beta_cf"] = fum_low_beta_peak_center_frequency
                                ses_data_copy[f"low_beta_power_auc"] = low_beta_power_area_under_curve

                            # check if high beta exist
                            if pd.isna((ses_data.iloc[0]["high_beta_center_frequency"])):
                                ses_high_beta_nan = f"{stn}_{ses}"
                                no_high_beta_peak.append(ses_high_beta_nan)

                            else:
                                fum_high_beta_peak_center_frequency = round(
                                    ses_data.high_beta_center_frequency.values[0]
                                )
                                # now get +- 3 Hz frequency range around peak center frequency
                                fum_high_beta_cf_range = np.arange(
                                    fum_high_beta_peak_center_frequency - 3, fum_high_beta_peak_center_frequency + 4, 1
                                )
                                # get power area under the curve
                                high_beta_power_in_freq_range = power[
                                    fum_high_beta_cf_range[0] : (fum_high_beta_cf_range[6] + 1)
                                ]  # select the power values by indexing from frequency range first until last value
                                high_beta_power_area_under_curve = simps(
                                    high_beta_power_in_freq_range, fum_high_beta_cf_range
                                )

                                ses_data_copy[f"round_high_beta_cf"] = fum_high_beta_peak_center_frequency
                                ses_data_copy[f"high_beta_power_auc"] = high_beta_power_area_under_curve

                        group_df_with_power_in_frange = pd.concat([group_df_with_power_in_frange, ses_data_copy])

                group_dict[group] = group_df_with_power_in_frange

                # dictionary with keys for each lfp group with lists of sub-ses with no beta peak, low beta or high beta peak
                no_beta_peak_dict[group] = no_beta_peak
                no_low_beta_peak_dict[group] = no_low_beta_peak
                no_high_beta_peak_dict[group] = no_high_beta_peak

    return group_dict
    # return {
    #     "group_dict": group_dict,
    #     "no_beta_peak_dict": no_beta_peak_dict,
    #     "no_low_beta_peak_dict": no_low_beta_peak_dict,
    #     "no_high_beta_peak_dict": no_high_beta_peak_dict}


def calculate_auc_beta_power_fu18or24(
    fooof_spectrum: str, fooof_version: str, highest_beta_session: str, around_cf: str
):
    """
    Taking the output dataframes from the function calculate_auc_beta_power_fu18or24()
    and if there are two longterm sessions fu18m AND fu24m from one STN -> fu24m will be deleted and only fu18m stays

    This way only one longterm session will be evaluated. To make analysis easier, the remaining fu18m and fu24m will be renamed to 18.

    - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

    - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

    - around_cf: "around_cf_at_each_session", "around_cf_at_fixed_session"

    """

    channel_group = ["ring", "segm_inter", "segm_intra"]
    dataframe_longterm_all = {}

    data_all = calculate_auc_beta_power(
        fooof_spectrum=fooof_spectrum,
        fooof_version=fooof_version,
        highest_beta_session=highest_beta_session,
        around_cf=around_cf,
    )

    for group in channel_group:
        data_group = data_all[group]

        # per stn and session: if fu18m and fu24m exist -> delete fu24m
        stn_unique = list(data_group.subject_hemisphere.unique())
        longterm_sessions = [18, 24]
        dataframe_longterm_group = pd.DataFrame()

        for stn in stn_unique:
            stn_data = data_group.loc[data_group.subject_hemisphere == stn]

            # check if both fu18m and fu24m exist, if yes: delete fu24m
            if all(l_ses in stn_data["session"].values for l_ses in longterm_sessions):
                # exclude all rows including "fu24m"
                stn_data = stn_data[stn_data["session"] != 24]

            # append stn dataframe to the longterm dataframe
            dataframe_longterm_group = pd.concat([dataframe_longterm_group, stn_data])

        # replace all "fu18m" and "fu24m" by 18
        dataframe_longterm_group["session"] = dataframe_longterm_group["session"].replace(longterm_sessions, 18)

        dataframe_longterm_all[f"{group}"] = dataframe_longterm_group

    return dataframe_longterm_all


def fooof_mixedlm_highest_beta_channels(
    fooof_spectrum: str,
    fooof_version: str,
    highest_beta_session: str,
    data_to_fit: str,
    around_cf: str,
    incl_sessions: list,
    shape_of_model: str,
):
    """

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc"
                                "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"

        - around_cf: "around_cf_at_each_session", "around_cf_at_fixed_session"

        - incl_sessions: [0,3] or [3,12,18] o [0,3,12,18]

        - shape_of_model: e.g. "straight", "curved", "asymptotic"


    Load the dataframe with highest beta channels in a given baseline session

    ASYMPTOTIC LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_asymptotic", data=data_analysis, groups=data_analysis["group"], re_formula="1")
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session_asymptotic
        - groups = electrode ID
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group

    The model will fit a curved line, the sum of a linear function (ax + b) and an asymptotic function a * (x / (1 + x)) + b
        y = a1 * x + a2 * (x / (1 + x))  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the asymptotic model (output as Coef. of session_asymptotic)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)


    CURVED LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_sq", data=data_analysis, groups=data_analysis["group"], re_formula="1")
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session squared
        - groups = electrode ID
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group

    The model will fit a curved line, the sum of a linear function (ax + b) and an exponential function (ax^2 + b)
        y = a1 * x + a2 * x**2  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the squared model (output as Coef. of session_sq)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)


    Two figures


    TODO: for low beta, high beta and beta seperately: take the dataframe, drop the other columns not used for the analysis (low or high or all beta), then drop NaN!!!!!


    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")
    fontdict = {"size": 25}

    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = [0, 3, 12, 18]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    sample_size_dict = {}  #
    model_output = {}  # md.fit()

    cropped_data = (
        {}
    )  # cropped, because depending on data_to_fit low, high beta or beta was dropped and then NaNs were dropped

    ############################## select the center frequency of 3MFU and get the area under the curve of power in a freq range +- 3 Hz around that center frequency ##############################

    beta_peak_auc_data = calculate_auc_beta_power_fu18or24(
        fooof_spectrum=fooof_spectrum,
        fooof_version=fooof_version,
        highest_beta_session=highest_beta_session,
        around_cf=around_cf,
    )

    ############################## perform linear mixed effects model ##############################
    for g, group in enumerate(channel_group):
        data_analysis = beta_peak_auc_data[group]
        data_analysis_copy = data_analysis.copy()

        # depending on data_to_fit, drop unnecessary columns and subsequently drop NaN
        if data_to_fit == "beta_center_frequency" or data_to_fit == "beta_power_auc":
            data_analysis_copy = data_analysis_copy.drop(
                columns=[
                    "low_beta_peak_CF_power_bandWidth",
                    "high_beta_peak_CF_power_bandWidth",
                    "low_beta_center_frequency",
                    "high_beta_center_frequency",
                    "low_beta_peak_power",
                    "high_beta_peak_power",
                    "low_beta_band_width",
                    "high_beta_band_width",
                    "round_low_beta_cf",
                    "round_high_beta_cf",
                    "low_beta_power_auc",
                    "high_beta_power_auc",
                ]
            )
            data_analysis_copy = data_analysis_copy.dropna()

        elif data_to_fit == "low_beta_center_frequency" or data_to_fit == "low_beta_power_auc":
            data_analysis_copy = data_analysis_copy.drop(
                columns=[
                    "beta_peak_CF_power_bandWidth",
                    "high_beta_peak_CF_power_bandWidth",
                    "beta_center_frequency",
                    "high_beta_center_frequency",
                    "beta_peak_power",
                    "high_beta_peak_power",
                    "beta_band_width",
                    "high_beta_band_width",
                    "round_beta_cf",
                    "round_high_beta_cf",
                    "beta_power_auc",
                    "high_beta_power_auc",
                ]
            )
            data_analysis_copy = data_analysis_copy.dropna()

        elif data_to_fit == "high_beta_center_frequency" or data_to_fit == "high_beta_power_auc":
            data_analysis_copy = data_analysis_copy.drop(
                columns=[
                    "low_beta_peak_CF_power_bandWidth",
                    "beta_peak_CF_power_bandWidth",
                    "low_beta_center_frequency",
                    "beta_center_frequency",
                    "low_beta_peak_power",
                    "beta_peak_power",
                    "low_beta_band_width",
                    "beta_band_width",
                    "round_low_beta_cf",
                    "round_beta_cf",
                    "low_beta_power_auc",
                    "beta_power_auc",
                ]
            )
            data_analysis_copy = data_analysis_copy.dropna()

        # predictor_x = data_analysis.session.values

        data_analysis_copy["session_sq"] = data_analysis_copy.session**2
        data_analysis_copy["session_asymptotic"] = data_analysis_copy.session / (1 + data_analysis_copy.session)

        if shape_of_model == "asymptotic":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session + session_asymptotic",
                data=data_analysis_copy,
                groups=data_analysis_copy["group"],
                re_formula="1",
            )

        elif shape_of_model == "curved":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session + session_sq",
                data=data_analysis_copy,
                groups=data_analysis_copy["group"],
                re_formula="1",
            )

        elif shape_of_model == "straight":
            md = smf.mixedlm(
                f"{data_to_fit} ~ session", data=data_analysis_copy, groups=data_analysis_copy["group"], re_formula="1"
            )

        # re_formula defining the random effect
        # re_formula = 1 specifying random intercept model, assuming same effect of predictor for all groups
        # re_formula = f"1 + session" specifying random intercept and slope model
        mdf = md.fit()

        # save linear model result
        print(mdf.summary())
        model_output[group] = mdf

        # add predictions column to dataframe
        yp = mdf.fittedvalues
        # beta_peak_auc_data[group]["predictions"] = yp # TODO
        data_analysis_copy["predictions"] = yp
        cropped_data[group] = data_analysis_copy

        for ses in incl_sessions:
            ses_data = data_analysis_copy.loc[data_analysis_copy.session == ses]
            count = ses_data.subject_hemisphere.count()

            # save sample size
            sample_size_dict[f"{group}_{ses}mfu"] = [group, ses, count]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(
        index={
            0: "channel_group",
            1: "session",
            2: "count",
        },
        inplace=True,
    )
    sample_size_df = sample_size_df.transpose()

    ############################## plot the observed values and the model ##############################
    fig_1, axes_1 = plt.subplots(3, 1, figsize=(10, 15))
    fig_2, axes_2 = plt.subplots(3, 1, figsize=(10, 15))

    for g, group in enumerate(channel_group):
        # data_analysis = beta_peak_auc_data[group] # this is the dataframe with data
        data_analysis = cropped_data[group]
        mdf_group = model_output[group]  # this is md.fit()

        # get the results
        result_part_2 = mdf_group.summary().tables[1]  # part containing model intercept, slope, std.error
        model_intercept = float(result_part_2["Coef."].values[0])
        model_slope = float(result_part_2["Coef."].values[1])

        if shape_of_model == "asymptotic" or "curved":
            model_slope_2 = float(result_part_2["Coef."].values[2])
            # group_variance = float(result_part_2["Coef."].values[3])

        std_error_intercept = float(result_part_2["Std.Err."].values[0])
        # p_val_intercept = float(result_part_2["P>|z|"].values[0])
        # p_val_session = float(result_part_2["P>|z|"].values[1])
        # p_val_session2 = float(result_part_2["P>|z|"].values[2])
        conf_int = mdf_group.conf_int(
            alpha=0.05
        )  # table with confidence intervals for intercept, session, session2 and group var

        # one subplot per channel group
        axes_1[g].set_title(f"{group} channel group", fontdict=fontdict)
        axes_2[g].set_title(f"{group} channel group", fontdict=fontdict)

        ################## plot the result for each electrode ##################
        for id, group_id in enumerate(data_analysis.group.unique()):
            sub_data = data_analysis[data_analysis.group == group_id]

            # axes[g].scatter(sub_data[f"{data_to_fit}"], sub_data["session"] ,color=plt.cm.twilight_shifted(group_id*10)) # color=plt.cm.tab20(group_id)
            # axes[g].plot(sub_data[f"{data_to_fit}"], sub_data["predictions"], color=plt.cm.twilight_shifted(group_id*10))

            axes_1[g].scatter(
                sub_data["session"], sub_data[f"{data_to_fit}"], color=plt.cm.twilight_shifted((id + 1) * 10), alpha=0.3
            )  # color=plt.cm.tab20(group_id)
            # plot the predictions
            # axes[g].plot(sub_data["session"], sub_data["predictions"], color=plt.cm.twilight_shifted((id+1)*10), linewidth=1, alpha=0.5)
            axes_1[g].plot(
                sub_data["session"],
                sub_data[f"{data_to_fit}"],
                color=plt.cm.twilight_shifted((id + 1) * 10),
                linewidth=1,
                alpha=0.3,
            )

        # plot the model regression line
        if 0 in incl_sessions:
            if 18 in incl_sessions:
                x = np.arange(0, 19)

            else:
                x = np.arange(0, 4)

        elif 0 not in incl_sessions:
            x = np.arange(3, 19)

        ################## plot the modeled curved line ##################
        if shape_of_model == "curved":
            y = x * model_slope + x**2 * model_slope_2 + model_intercept  # curved model

        elif shape_of_model == "asymptotic":
            y = x * model_slope + (x / (1 + x)) * model_slope_2 + model_intercept  # asymptotic model

        elif shape_of_model == "straight":
            y = x * model_slope + model_intercept  # straight line

        axes_1[g].plot(x, y, color="k", linewidth=5)
        # linear model: coef*x (=linear) + coef*x^2 (=exponential) + intercept
        # coef defines the slope

        # pred = mdf_group.predict(exog=dict(x=x))
        # calculate the confidence interval
        # cov_params = mdf_group.cov_params()
        # mse = np.mean(mdf_group.resid.values**2)
        # t_value = stats.t.ppf(0.975, df=mdf_group.df_resid)
        # standard_errors = np.sqrt(np.diag(cov_params))
        # lower_bound = y - t_value * standard_errors * np.sqrt(mse)
        # upper_bound = y + t_value * standard_errors * np.sqrt(mse)

        # lower_bound = y + 1.96 *

        # axes[g].plot(prediction_data["session"], prediction_data["mean_yp"], color="k", linewidth=5)
        # axes[g].fill_between(prediction_data["session"], prediction_data["mean_yp"]-prediction_data["sem_yp"], prediction_data["mean_yp"]+prediction_data["sem_yp"], color='lightgray', alpha=0.5)
        # axes_1[g].fill_between(x, lower_bound, upper_bound, color="k", linewidth=5, alpha=0.3)

        ################### plot the residuals against predictions ##################
        # number of residuals equals the number of observations (number of channels I input)
        resid = mdf_group.resid
        predicted_values = mdf_group.fittedvalues
        axes_2[g].scatter(predicted_values, resid, color="k", alpha=0.2)
        axes_2[g].axhline(y=0, color="red", linestyle="--")

    for ax in axes_1:
        ax.set_ylabel(f"{data_to_fit}", fontsize=25)
        ax.set_xlabel("months post-surgery", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_1.suptitle(f"Linear mixed effects model: {highest_beta_session} beta channels", fontsize=30)
    fig_1.subplots_adjust(wspace=0, hspace=0)

    fig_1.tight_layout()

    if data_to_fit == "beta_power_auc":
        fig1_filename_png = f"lme_{shape_of_model}_{data_to_fit}_{around_cf}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.png"
        fig1_filename_svg = f"lme_{shape_of_model}_{data_to_fit}_{around_cf}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.svg"

        fig2_filename_png = f"lme_{shape_of_model}_residuals_{data_to_fit}_{around_cf}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.png"
        fig2_filename_svg = f"lme_{shape_of_model}_residuals_{data_to_fit}_{around_cf}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.svg"

        mdf_result_filename = f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{around_cf}_{highest_beta_session}_sessions{incl_sessions}_{fooof_version}.pickle"

    else:
        fig1_filename_png = f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.png"
        fig1_filename_svg = f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.svg"

        fig2_filename_png = f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.png"
        fig2_filename_svg = f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}_{fooof_version}.svg"

        mdf_result_filename = f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}_{fooof_version}.pickle"

    fig_1.savefig(os.path.join(figures_path, fig1_filename_png), bbox_inches="tight")
    fig_1.savefig(os.path.join(figures_path, fig1_filename_svg), bbox_inches="tight", format="svg")

    print("figure: ", f"{fig1_filename_png}", "\nwritten in: ", figures_path)

    for ax in axes_2:
        ax.set_xlabel("Predicted Values", fontsize=25)
        ax.set_ylabel("Residuals", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_2.suptitle(f"Linear mixed effects model residuals: {highest_beta_session} beta channels", fontsize=30)
    fig_2.subplots_adjust(wspace=0, hspace=0)
    fig_2.tight_layout()

    fig_2.savefig(os.path.join(figures_path, fig2_filename_png), bbox_inches="tight")
    fig_2.savefig(os.path.join(figures_path, fig2_filename_svg), bbox_inches="tight", format="svg")

    # save results
    mdf_result_filepath = os.path.join(results_path, mdf_result_filename)
    with open(mdf_result_filepath, "wb") as file:
        pickle.dump(model_output, file)

    print("file: ", f"{mdf_result_filename}", "\nwritten in: ", results_path)

    return {
        "beta_peak_auc_data": beta_peak_auc_data,
        "cropped_data": cropped_data,
        "sample_size_df": sample_size_df,
        "conf_int": conf_int,
        "model_output": model_output,
        "md": md,
    }


def calculate_mean_squared_error(
    fooof_spectrum: str,
    fooof_version: str,
    highest_beta_session: str,
    data_to_fit: str,
    around_cf: str,
):
    """
    Input
        - fooof_spectrum:
                "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
                "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
                "periodic_flat"             -> model._peak_fit
        - fooof_version: "v2"
        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc"
                                "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"

        - around_cf: "around_cf_at_each_session", "around_cf_at_fixed_session"


    calculates the mean squared error of all models: "straight", "curved", "asymptotic"

        - mean of (predicted - real)**2

    """

    shape_of_model_all = ["straight", "curved", "asymptotic"]
    mean_squared_error_result = {}

    for model in shape_of_model_all:
        mixedlm_highest_beta_channels = fooof_mixedlm_highest_beta_channels(
            fooof_spectrum=fooof_spectrum,
            fooof_version=fooof_version,
            highest_beta_session=highest_beta_session,
            data_to_fit=data_to_fit,
            around_cf=around_cf,
            incl_sessions=[0, 3, 12, 18],
            shape_of_model=model,
        )

        # merge dataframes of all LFP groups together
        ring_DF = mixedlm_highest_beta_channels["cropped_data"]["ring"]
        segm_inter_DF = mixedlm_highest_beta_channels["cropped_data"]["segm_inter"]
        segm_intra_DF = mixedlm_highest_beta_channels["cropped_data"]["segm_intra"]

        data_all_LFP_groups = pd.concat([ring_DF, segm_inter_DF, segm_intra_DF])

        # calculate the mean square error of the model: mean of (predicted - real)**2
        mean_squared_error = np.mean((data_all_LFP_groups["predictions"] - data_all_LFP_groups[f"{data_to_fit}"]) ** 2)

        mean_squared_error_result[f"{model}"] = [model, mean_squared_error]

    mean_squared_error_df = pd.DataFrame(mean_squared_error_result)
    mean_squared_error_df.rename(index={0: "model", 1: "mean_squared_error"}, inplace=True)

    mean_squared_error_df = mean_squared_error_df.transpose()

    return mean_squared_error_df


def change_beta_peak_power_or_cf_violinplot(
    fooof_spectrum: str,
    fooof_version: str,
    highest_beta_session: str,
    data_to_analyze: str,
    around_cf: str,
    absolute_change: str,
    session_comparisons: list,
):
    """
    Load the fooof data of the selected highest beta channels

    Input:
        - fooof_spectrum:
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - fooof_version: "v2"
        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_analyze: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc",
                                    "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"


        - around_cf: "around_cf_at_each_session", "around_cf_at_fixed_session"

        - absolute_change: "yes" or "no"
                            -> if "yes": the change is quantified independent of what direction this change was
                            -> if "no": the change is quantified as session 2 - session 1
        - session_comparisons: list ["0_3", "0_12", "0_18", "3_12", "3_18", "12_18"] or ["0_3", "3_12", "12_18"]

    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")
    fontdict = {"size": 25}

    # Load the dataframe with only highest beta channels and calculated area under the curve of highest beta peaks
    beta_data = calculate_auc_beta_power_fu18or24(
        fooof_spectrum=fooof_spectrum,
        fooof_version=fooof_version,
        highest_beta_session=highest_beta_session,
        around_cf=around_cf,
    )
    # output is a dictionary with keys "ring", "segm_inter", "segm_intra"

    channel_group = ["ring", "segm_inter", "segm_intra"]

    statistics_dict = {}  #
    description_dict = {}
    difference_session_comparison_dict = {}

    ############################## calculate difference between sessions ##############################
    for g, group in enumerate(channel_group):
        group_data = beta_data[group]  # all data only from one channel group

        group_data_copy = group_data.copy()

        # depending on data_to_fit, drop unnecessary columns and subsequently drop NaN
        if data_to_analyze == "beta_center_frequency" or data_to_analyze == "beta_power_auc":
            group_data_copy = group_data_copy.drop(
                columns=[
                    "low_beta_peak_CF_power_bandWidth",
                    "high_beta_peak_CF_power_bandWidth",
                    "low_beta_center_frequency",
                    "high_beta_center_frequency",
                    "low_beta_peak_power",
                    "high_beta_peak_power",
                    "low_beta_band_width",
                    "high_beta_band_width",
                    "round_low_beta_cf",
                    "round_high_beta_cf",
                    "low_beta_power_auc",
                    "high_beta_power_auc",
                ]
            )
            group_data_copy = group_data_copy.dropna()

        elif data_to_analyze == "low_beta_center_frequency" or data_to_analyze == "low_beta_power_auc":
            group_data_copy = group_data_copy.drop(
                columns=[
                    "beta_peak_CF_power_bandWidth",
                    "high_beta_peak_CF_power_bandWidth",
                    "beta_center_frequency",
                    "high_beta_center_frequency",
                    "beta_peak_power",
                    "high_beta_peak_power",
                    "beta_band_width",
                    "high_beta_band_width",
                    "round_beta_cf",
                    "round_high_beta_cf",
                    "beta_power_auc",
                    "high_beta_power_auc",
                ]
            )
            group_data_copy = group_data_copy.dropna()

        elif data_to_analyze == "high_beta_center_frequency" or data_to_analyze == "high_beta_power_auc":
            group_data_copy = group_data_copy.drop(
                columns=[
                    "low_beta_peak_CF_power_bandWidth",
                    "beta_peak_CF_power_bandWidth",
                    "low_beta_center_frequency",
                    "beta_center_frequency",
                    "low_beta_peak_power",
                    "beta_peak_power",
                    "low_beta_band_width",
                    "beta_band_width",
                    "round_low_beta_cf",
                    "round_beta_cf",
                    "low_beta_power_auc",
                    "beta_power_auc",
                ]
            )
            group_data_copy = group_data_copy.dropna()

        stn_unique = list(group_data_copy.subject_hemisphere.unique())

        for stn in stn_unique:
            stn_data = group_data_copy.loc[group_data_copy.subject_hemisphere == stn]  # data only from one stn

            ses_unique = list(stn_data.session.unique())  # sessions existing from this stn

            for comparison in session_comparisons:
                split_comparison = comparison.split("_")  # ["0", "3"]
                session_1 = int(split_comparison[0])  # e.g. 0
                session_2 = int(split_comparison[1])  # e.g. 3

                # check if both sessions exist for this stn
                if session_1 not in ses_unique:
                    continue

                elif session_2 not in ses_unique:
                    continue

                # select only the data from session 1 and session 2
                session_1_data = stn_data.loc[stn_data.session == session_1]
                session_2_data = stn_data.loc[stn_data.session == session_2]
                channel = stn_data.bipolar_channel.values[0]

                # depending on what you want to analyze, pick the data of interest
                if data_to_analyze == "beta_average":
                    session_1_data_of_interest = session_1_data.beta_average.values[0]
                    session_2_data_of_interest = session_2_data.beta_average.values[0]

                elif data_to_analyze == "beta_peak_power":
                    session_1_data_of_interest = session_1_data.beta_peak_power.values[0]
                    session_2_data_of_interest = session_2_data.beta_peak_power.values[0]

                elif data_to_analyze == "beta_center_frequency":
                    session_1_data_of_interest = session_1_data.beta_center_frequency.values[0]
                    session_2_data_of_interest = session_2_data.beta_center_frequency.values[0]

                elif data_to_analyze == "low_beta_center_frequency":
                    session_1_data_of_interest = session_1_data.low_beta_center_frequency.values[0]
                    session_2_data_of_interest = session_2_data.low_beta_center_frequency.values[0]

                elif data_to_analyze == "high_beta_center_frequency":
                    session_1_data_of_interest = session_1_data.high_beta_center_frequency.values[0]
                    session_2_data_of_interest = session_2_data.high_beta_center_frequency.values[0]

                elif data_to_analyze == "beta_power_auc":
                    session_1_data_of_interest = session_1_data.beta_power_auc.values[0]
                    session_2_data_of_interest = session_2_data.beta_power_auc.values[0]

                elif data_to_analyze == "low_beta_power_auc":
                    session_1_data_of_interest = session_1_data.low_beta_power_auc.values[0]
                    session_2_data_of_interest = session_2_data.low_beta_power_auc.values[0]

                elif data_to_analyze == "high_beta_power_auc":
                    session_1_data_of_interest = session_1_data.high_beta_power_auc.values[0]
                    session_2_data_of_interest = session_2_data.high_beta_power_auc.values[0]

                # calculate difference between two sessions: session 2 - session 1
                if absolute_change == "yes":
                    difference_ses1_ses2 = abs(session_2_data_of_interest - session_1_data_of_interest)

                else:
                    difference_ses1_ses2 = session_2_data_of_interest - session_1_data_of_interest

                # store data in dataframe
                difference_session_comparison_dict[f"{group}_{stn}_{comparison}"] = [
                    group,
                    stn,
                    channel,
                    comparison,
                    session_1_data_of_interest,
                    session_2_data_of_interest,
                    difference_ses1_ses2,
                ]

    # save as dataframe
    difference_dataframe = pd.DataFrame(difference_session_comparison_dict)
    difference_dataframe.rename(
        index={
            0: "channel_group",
            1: "subject_hemisphere",
            2: "bipolar_channel",
            3: "session_comparison",
            4: f"session_1_{data_to_analyze}",
            5: f"session_2_{data_to_analyze}",
            6: "difference_ses1-ses2",
        },
        inplace=True,
    )
    difference_dataframe = difference_dataframe.transpose()

    if len(session_comparisons) == 6:
        difference_dataframe["session_comp_group"] = difference_dataframe.session_comparison.replace(
            to_replace=session_comparisons, value=[1, 2, 3, 4, 5, 6]
        )
        group_comparisons = [1, 2, 3, 4, 5, 6]

    elif len(session_comparisons) == 3:
        difference_dataframe["session_comp_group"] = difference_dataframe.session_comparison.replace(
            to_replace=session_comparisons, value=[1, 2, 3]
        )
        group_comparisons = [1, 2, 3]

    else:
        print("length of session_comparisons should be 3 or 6.")

    difference_dataframe["difference_ses1-ses2"] = difference_dataframe["difference_ses1-ses2"].astype(float)

    ######################### VIOLINPLOT OF CHANGE, seperately for each channel group #########################

    for group in channel_group:
        group_data_to_plot = difference_dataframe.loc[difference_dataframe.channel_group == group]

        fig = plt.figure()
        ax = fig.add_subplot()

        sns.violinplot(
            data=group_data_to_plot,
            x="session_comp_group",
            y="difference_ses1-ses2",
            palette="coolwarm",
            inner="box",
            ax=ax,
        )

        # statistical test:
        pairs = list(combinations(group_comparisons, 2))

        annotator = Annotator(ax, pairs, data=group_data_to_plot, x='session_comp_group', y="difference_ses1-ses2")
        annotator.configure(test='Mann-Whitney', text_format='star')  # or t-test_ind ??
        annotator.apply_and_annotate()

        sns.stripplot(
            data=group_data_to_plot,
            x="session_comp_group",
            y="difference_ses1-ses2",
            ax=ax,
            size=6,
            color="black",
            alpha=0.2,  # Transparency of dots
        )

        sns.despine(left=True, bottom=True)  # get rid of figure frame

        if highest_beta_session == "all_channels":
            title_name = f"{group} group: change of {data_to_analyze} between sessions \n(of all channels)"

        elif highest_beta_session == "highest_postop":
            title_name = f"{group} group: change of {data_to_analyze} between sessions \n(of highest beta channel, baseline postop)"

        elif highest_beta_session == "highest_fu3m":
            title_name = (
                f"{group} group: change of {data_to_analyze} between sessions \n(highest beta channel, baseline 3MFU)"
            )

        elif highest_beta_session == "highest_each_session":
            title_name = f"{group} group: change of {data_to_analyze} between sessions \n(only highest beta channels)"

        plt.title(title_name)
        plt.ylabel(f"difference of {data_to_analyze} \nin beta band (13-35 Hz)")
        plt.xlabel("session comparison")
        plt.xticks(range(len(session_comparisons)), session_comparisons)

        fig.tight_layout()
        if absolute_change == "yes":
            absolute = "absolute_"

        else:
            absolute = ""

        if data_to_analyze == "beta_power_auc":
            fig_filename_png = f"change_of_{data_to_analyze}_{around_cf}_fooof_beta_{highest_beta_session}_{group}_{absolute}{session_comparisons}_{fooof_version}.png"
            fig_filename_svg = f"change_of_{data_to_analyze}_{around_cf}_fooof_beta_{highest_beta_session}_{group}_{absolute}{session_comparisons}_{fooof_version}.svg"

        else:
            fig_filename_png = f"change_of_{data_to_analyze}_fooof_beta_{highest_beta_session}_{group}_{absolute}{session_comparisons}_{fooof_version}.png"
            fig_filename_svg = f"change_of_{data_to_analyze}_fooof_beta_{highest_beta_session}_{group}_{absolute}{session_comparisons}_{fooof_version}.svg"

        fig.savefig(os.path.join(figures_path, fig_filename_png), bbox_inches="tight")
        fig.savefig(os.path.join(figures_path, fig_filename_svg), bbox_inches="tight", format="svg")

        print("figure: ", f"{fig_filename_png}", "\nwritten in: ", figures_path)

        ######################### STATISTICS AND DESCRIPTION OF DATA #########################

        for pair in pairs:
            # pair e.g. (1, 2) for comparing group 1 (postop-3mfu) and group 2 (postop-12mfu)

            group_1 = pair[0]  # e.g. postop-3mfu
            group_2 = pair[1]  # e.g. postop-12mfu

            group_1_data = group_data_to_plot.loc[group_data_to_plot.session_comp_group == group_1]
            group_1_data = group_1_data["difference_ses1-ses2"].values
            group_2_data = group_data_to_plot.loc[group_data_to_plot.session_comp_group == group_2]
            group_2_data = group_2_data["difference_ses1-ses2"].values

            # mann-whitney U test
            statistic, p_value = stats.mannwhitneyu(
                group_1_data, group_2_data
            )  # by default two-sided test, so testing for significant difference between two distributions regardless of the direction of the difference

            statistics_dict[f"{group}_{pair}"] = [group, pair, statistic, p_value]

        # describe each group
        for descr_group in group_comparisons:
            data_to_describe = group_data_to_plot.loc[group_data_to_plot.session_comp_group == descr_group]
            data_to_describe = data_to_describe["difference_ses1-ses2"].values

            description_1 = scipy.stats.describe(data_to_describe)
            description_dict[f"{group}_{descr_group}"] = description_1

    # save as dataframe
    description_results = pd.DataFrame(description_dict)
    description_results.rename(
        index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"},
        inplace=True,
    )
    description_results = description_results.transpose()
    # calculate the standard deviation as the square root from the variance
    description_results["variance"] = description_results["variance"].astype(float)
    description_results["standard_deviation"] = np.sqrt(description_results["variance"])

    statistics_dataframe = pd.DataFrame(statistics_dict)
    statistics_dataframe.rename(
        index={
            0: "channel_group",
            1: "statistics_pair",
            2: "mann_whitney_u_stats",
            3: "p_val",
        },
        inplace=True,
    )
    statistics_dataframe = statistics_dataframe.transpose()

    return {
        "difference_dataframe": difference_dataframe,
        "group_data_to_plot": group_data_to_plot,
        "statistics_dataframe": statistics_dataframe,
        "description_results": description_results,
    }
