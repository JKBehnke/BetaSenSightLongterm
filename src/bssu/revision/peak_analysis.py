""" Analysis of FOOOF beta peak parameters for Revision MDS paper """

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
from scipy.stats import shapiro, friedmanchisquare, wilcoxon, ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder
import fooof
from fooof.plts.spectra import plot_spectrum

# Local Imports
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults
from ..utils import percept_helpers as percept_helpers


CHANNEL_GROUPS = ["ring", "segm_inter", "segm_intra"]
BETA_RANGES = ["beta", "low_beta", "high_beta"]

RESULTS_PATH = findfolders.get_local_path(folder="GroupResults")
FIGURES_PATH = findfolders.get_local_path(folder="GroupFigures")


def highest_beta_channels_fooof(fooof_spectrum: str, highest_beta_session: str, cohort: str):
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
    # fooof_group_result = loadResults.load_group_fooof_result(fooof_version=fooof_version)
    # fooof_group_result = loadResults.load_fooof_beta_ranks(
    #     fooof_spectrum=fooof_spectrum,
    #     fooof_version=fooof_version,
    #     all_or_one_chan="beta_ranks_all",
    #     all_or_one_longterm_ses="all_sessions",
    # )

    fooof_group_result = loadResults.select_fooof_data(
        dataset="bipolar_beta",
        cohort=cohort,
    )
    fooof_group_result = fooof_group_result["fooof_data"]

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
    sessions = ["postop", "fu3m", "fu12m", "fu18or24m"]

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

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index(drop=True)

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
            to_replace=["postop", "fu3m", "fu12m", "fu18or24m"], value=[0, 3, 12, 18]
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


def calculate_auc_beta_power(fooof_spectrum: str, highest_beta_session: str, around_cf: str, cohort: str):
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
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session,
        cohort=cohort,
    )
    # output is a dictionary with keys "ring", "segm_inter", "segm_intra"

    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = [0, 3, 12, 18]

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

    return {
        "group_dict": group_dict,
        "no_beta_peak_dict": no_beta_peak_dict,
        "no_low_beta_peak_dict": no_low_beta_peak_dict,
        "no_high_beta_peak_dict": no_high_beta_peak_dict,
    }


def peak_overview_table(fooof_spectrum: str, highest_beta_session: str, around_cf: str, cohort: str):
    """ """

    total_data = calculate_auc_beta_power(
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session,
        around_cf=around_cf,
        cohort=cohort,
    )

    ring_data = total_data["group_dict"]["ring"]
    no_peaks_ring = {}

    # only for the RING channel group: keep only the rows with NaN values in a certain column
    for b_range in BETA_RANGES:

        # Filter the dataframe for rows where "peak_frequency" is NaN
        nan_peak_frequency_df = ring_data[ring_data[f"{b_range}_center_frequency"].isna()]

        # Select only the "subject" and "session" columns
        no_peaks = nan_peak_frequency_df[
            ["subject_hemisphere", "session", "bipolar_channel", f"{b_range}_center_frequency"]
        ]

        no_peaks_ring[b_range] = no_peaks

    return no_peaks_ring


def analyze_peak_frequency_or_power_group_0(fooof_spectrum: str, highest_beta_session: str, peak_feature: str):
    """
    Analyze peak frequencies between two sessions only for group 0!
        - subjects with both recording sessions 0 and 3
        - first exclude all subjects who have no peak identified in one of the sessions
        - now all subjects should have a peak identified in both sessions
        - check for normal distribution of the data
        - perform a paired test between session 0 and session 3 (Wilcoxon or paired t-test) depending on normal distribution
        - correct for multiple comparison: FDR, Bonferroni or holm

    Input:
        - peak_feature: "peak_frequency" or "peak_power_auc_fixed_f_range" or "peak_power_auc_per_peak"
            if "peak_frequency" -> around_cf_at_each_session
            if "peak_power_auc_fixed_f_range" -> around_cf_at_fixed_session
            if "peak_power_auc_per_peak" -> around_cf_at_each_session

    Parameters:
        data (pd.DataFrame): DataFrame with columns "subject", "session", "peak_frequency".
        excluded_subjects (pd.DataFrame): DataFrame with "subject" and "session" identifying missing peaks.

    Returns:
        summary_df (pd.DataFrame): Summary of the analysis.
    """

    if peak_feature in ["peak_frequency", "peak_power_auc_per_peak"]:
        # load data
        total_data = calculate_auc_beta_power(
            fooof_spectrum=fooof_spectrum,
            highest_beta_session=highest_beta_session,
            around_cf="around_cf_at_each_session",  # individual peak CF per session
            cohort="group_0",
        )

    elif peak_feature == "peak_power_auc_fixed_f_range":
        # load data
        total_data = calculate_auc_beta_power(
            fooof_spectrum=fooof_spectrum,
            highest_beta_session=highest_beta_session,
            around_cf="around_cf_at_fixed_session",  # individual peak CF per session
            cohort="group_0",
        )

    if peak_feature == "peak_frequency":
        feature_column_name = "center_frequency"

    elif peak_feature in ["peak_power_auc_fixed_f_range", "peak_power_auc_per_peak"]:
        feature_column_name = "power_auc"

    ring_data = total_data["group_dict"]["ring"]

    result = pd.DataFrame()
    excluded_overview = {}
    raw_data = {}
    all_p_values = []  # Collect all p-values for multiple comparison correction
    beta_ranges = []  # Collect beta range names for reference

    excluded_stns = peak_overview_table(
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session,
        around_cf="around_cf_at_each_session",
        cohort="group_0",
    )

    for b_range in BETA_RANGES:
        excluded = excluded_stns[b_range]

        # Step 1: Exclude subjects with missing peaks
        excluded_set = set(zip(excluded["subject_hemisphere"], excluded["session"]))
        filtered_data = ring_data[~ring_data[["subject_hemisphere", "session"]].apply(tuple, axis=1).isin(excluded_set)]

        # Ensure subjects have peaks in both sessions
        subject_counts = filtered_data["subject_hemisphere"].value_counts()
        valid_subjects = subject_counts[subject_counts == 2].index
        filtered_data = filtered_data[filtered_data["subject_hemisphere"].isin(valid_subjects)]

        # add a column with b_range to excluded DF
        excluded_overview[b_range] = excluded

        # Step 2: Reshape data for paired testing
        paired_data = filtered_data.pivot(
            index="subject_hemisphere", columns="session", values=f"{b_range}_{feature_column_name}"
        )  # values=f"round_{b_range}_cf")
        paired_data.columns = [0, 3]

        raw_data[b_range] = paired_data

        # Step 3: Descriptive statistics
        differences = paired_data[0] - paired_data[3]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        q1 = np.percentile(differences, 25)
        median = np.median(differences)
        q3 = np.percentile(differences, 75)

        # Step 4: Check normality
        normality_stat, normality_p = shapiro(differences)
        normal_distribution = normality_p > 0.05

        # Step 5: Perform paired test
        if normal_distribution:
            test_stat, p_value = ttest_rel(paired_data[0], paired_data[3])
            test_name = "Paired t-test"
        else:
            test_stat, p_value = wilcoxon(paired_data[0], paired_data[3])
            test_name = "Paired Wilcoxon signed-rank test"

        # Collect p-values and beta ranges for correction
        all_p_values.append(p_value)
        beta_ranges.append(b_range)

        # Step 6: Create summary DataFrame
        summary_data = {
            "beta_range": [b_range],
            "Sample Size": [len(differences)],
            "Normal Distribution (p > 0.05)": [normal_distribution],
            "Mean Difference": [mean_diff],
            "Standard Deviation": [std_diff],
            "1st Quartile": [q1],
            "Median": [median],
            "3rd Quartile": [q3],
            "Test Name": [test_name],
            "Test Statistic": [test_stat],
            "P-value": [p_value],
            "valid_subjects": [valid_subjects],
        }

        summary_df = pd.DataFrame(summary_data)
        result = pd.concat([result, summary_df])

    # Step 7: Apply multiple comparison correction
    corrections = {}
    correction_methods = ["bonferroni", "holm", "fdr_bh"]

    for method in correction_methods:
        _, corrected_p_values, _, _ = multipletests(all_p_values, method=method)
        corrections[method] = corrected_p_values

    # Step 8: Add corrected p-values to the result DataFrame
    for method, corrected_p_values in corrections.items():
        result[f"P-value ({method})"] = corrected_p_values

    return result, excluded_overview, raw_data


def boxplot_peak_frequency_or_power_group_0(fooof_spectrum: str, highest_beta_session: str, peak_feature: str):
    """
    Plot boxplots per session with scatterplot for individual subjects connected by lines.

    Input:
        - peak_feature: "peak_frequency" or "peak_power_auc_per_peak" or "peak_power_auc_fixed_f_range"
            if "peak_frequency" -> around_cf_at_each_session
            if "peak_power_auc_per_peak" -> around_cf_at_each_session
            if "peak_power_auc_fixed_f_range" -> around_cf_at_fixed_session

    Parameters:
        data (pd.DataFrame): DataFrame with columns as session names (e.g., "0", "3")
                             and the index as subject IDs.


    """

    loaded_data = analyze_peak_frequency_or_power_group_0(
        fooof_spectrum=fooof_spectrum, highest_beta_session=highest_beta_session, peak_feature=peak_feature
    )
    data = loaded_data[2]

    for b_range in BETA_RANGES:
        range_data = data[b_range]

        # Reshape the dataframe for long-format plotting
        long_data = range_data.reset_index().melt(
            id_vars="subject_hemisphere", var_name="session", value_name=peak_feature
        )
        long_data.rename(columns={"index": "subject_hemisphere"}, inplace=True)

        # Ensure the session column is treated as categorical for proper ordering
        long_data["session"] = pd.Categorical(long_data["session"], ordered=True)

        # Plot the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplots for each session
        sns.boxplot(
            data=long_data,
            x="session",
            y=peak_feature,
            whis=[5, 95],
            width=0.5,
            color="white",
            # palette="pastel",
            showfliers=True,
            ax=ax,
        )

        # Overlay scatterplot with connections for each subject
        x_positions = [0, 1]
        for subject in range_data.index:

            plt.plot(
                x_positions,  # Sessions (columns of the original dataframe)
                range_data.loc[subject, [0, 3]],  # Peak frequencies for this subject
                marker="o",
                color="gray",
                alpha=0.4,
                linestyle="-",
                linewidth=1,
                markersize=9,
            )

        # Calculate and plot means for each session
        means = range_data.mean(axis=0)  # Mean for each session
        ax.scatter(x_positions, means, color="black", marker="+", s=100, label="Mean")

        # Add scatterplot for individual data points
        # sns.stripplot(data=long_data, x="session", y="peak_frequency", color="grey", alpha=0.2, size=10, jitter=False)

        # Customize the plot
        ax.set_title(f"Paired Comparison of {b_range} {peak_feature}: Group 0 (session 0 and 3/12)", fontsize=16)
        ax.set_xlabel("Session", fontsize=14)
        ax.set_ylabel(peak_feature, fontsize=14)

        if peak_feature == "peak_frequency":
            if b_range == "beta":
                ax.set_ylim(10, 38)

            elif b_range == "low_beta":
                ax.set_ylim(10, 23)

            elif b_range == "high_beta":
                ax.set_ylim(20, 38)

        elif peak_feature in ["peak_power_auc_per_peak", "peak_power_auc_fixed_f_range"]:
            ax.set_ylim(-3, 50)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        plt.tight_layout()

        # Sav figure
        percept_helpers.save_fig_png_and_svg(
            path=FIGURES_PATH,
            filename=f"revision_paired_comparison_{b_range}_{peak_feature}_group_0",
            figure=fig,
        )


def analyze_peak_frequency_or_power_three_sessions(
    fooof_spectrum: str, highest_beta_session: str, peak_feature: str, cohort: str, abs_or_rel: str
):
    """
    Analyze peak frequencies or power across three sessions for group 0.
        - Exclude subject hemispheres without peaks identified in all three sessions.
        - Check normal distribution of the data.
        - Perform repeated-measures ANOVA or Friedman test depending on normality.
        - Perform post-hoc Wilcoxon test (Friedman) or appropriate post-hoc test (ANOVA).
        - Apply multiple comparison corrections: FDR, Bonferroni, Holm.

    Parameters:
        fooof_spectrum (str): The input spectrum data.
        highest_beta_session (str): The session with the highest beta activity.
        peak_feature (str): The feature to analyze ("peak_frequency", "peak_power_auc_fixed_f_range", "peak_power_auc_per_peak").
        cohort (str): The cohort to analyze ("group_0", "group_1").
        abs_or_rel (str): The type of power to analyze ("absolute", "relative").

    Returns:
        summary_df (pd.DataFrame): Summary of the analysis.
    """

    # Load data based on peak feature
    if peak_feature in ["peak_frequency", "peak_power_auc_per_peak"]:
        total_data = calculate_auc_beta_power(
            fooof_spectrum=fooof_spectrum,
            highest_beta_session=highest_beta_session,
            around_cf="around_cf_at_each_session",
            cohort=cohort,
        )
    elif peak_feature == "peak_power_auc_fixed_f_range":
        total_data = calculate_auc_beta_power(
            fooof_spectrum=fooof_spectrum,
            highest_beta_session=highest_beta_session,
            around_cf="around_cf_at_fixed_session",
            cohort=cohort,
        )

    feature_column_name = "center_frequency" if peak_feature == "peak_frequency" else "power_auc"

    ring_data = total_data["group_dict"]["ring"]

    result = pd.DataFrame()
    excluded_overview = {}
    raw_data = {}
    all_p_values = {
        "beta": [],
        "low_beta": [],
        "high_beta": [],
    }  # Collect all post-hoc p-values for multiple comparison correction
    pairwise_comparisons = {"beta": [], "low_beta": [], "high_beta": []}  # To track pairwise tests for clarity

    descriptive_stats = []  # Collect descriptive statistics for each session

    excluded_stns = peak_overview_table(
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session,
        around_cf="around_cf_at_each_session",
        cohort=cohort,
    )

    for b_range in BETA_RANGES:
        excluded = excluded_stns[b_range]

        # Step 1: Exclude subjects with missing peaks in any session
        excluded_set = set(zip(excluded["subject_hemisphere"], excluded["session"]))
        filtered_data = ring_data[~ring_data[["subject_hemisphere", "session"]].apply(tuple, axis=1).isin(excluded_set)]

        # Ensure subjects have peaks in all three sessions
        subject_counts = filtered_data["subject_hemisphere"].value_counts()
        valid_subjects = subject_counts[subject_counts == 3].index
        filtered_data = filtered_data[filtered_data["subject_hemisphere"].isin(valid_subjects)]

        # Add a column with b_range to excluded DF
        excluded_overview[b_range] = excluded

        # Step 2: Reshape data for repeated-measures testing
        paired_data = filtered_data.pivot(
            index="subject_hemisphere", columns="session", values=f"{b_range}_{feature_column_name}"
        )

        if cohort == "group_0":
            paired_data.columns = [0, 1]

        else:
            paired_data.columns = [0, 1, 2]

        # Normalize power to "fu3m" session if "relative" power is selected
        if abs_or_rel == "relative":
            if cohort == "group_1" or cohort == "group_0":
                # Normalize values relative to session 1 ("fu3m")
                paired_data = paired_data.div(paired_data[1], axis=0)
            elif cohort == "group_2":
                # Normalize values relative to session 0 ("fu3m")
                paired_data = paired_data.div(paired_data[0], axis=0)

        raw_data[b_range] = paired_data

        # Step 3: Descriptive statistics
        for session in paired_data.columns:
            session_data = paired_data[session]
            descriptive_stats.append(
                {
                    "Frequency Range": b_range,
                    "Session": session,
                    "Mean": session_data.mean(),
                    "Median": session_data.median(),
                    "1st Quartile": session_data.quantile(0.25),
                    "3rd Quartile": session_data.quantile(0.75),
                    "Standard Deviation": session_data.std(ddof=1),
                    "Sample Size": session_data.count(),
                }
            )

        # Step 4: Check normality
        differences = paired_data.values.flatten()  # Flatten to test all observations together
        normality_stat, normality_p = shapiro(differences)
        normal_distribution = normality_p > 0.05

        # Step 5: Perform repeated-measures test
        if normal_distribution:
            # Perform repeated-measures ANOVA
            paired_data_long = paired_data.reset_index().melt(
                id_vars="subject_hemisphere", var_name="session", value_name="value"
            )
            anova = AnovaRM(paired_data_long, "value", "subject_hemisphere", within=["session"]).fit()
            test_stat = anova.anova_table["F Value"]["session"]
            p_value = anova.anova_table["Pr > F"]["session"]
            test_name = "Repeated-measures ANOVA"
        else:
            # Perform Friedman test
            test_stat, p_value = friedmanchisquare(paired_data[0], paired_data[1], paired_data[2])
            test_name = "Friedman test"

        # Step 6: Post-hoc testing
        if p_value < 0.05:  # Post-hoc only if the main test is significant
            if cohort == "group_0":
                comparisons = [(0, 1), (1, 0)]

            else:
                comparisons = [(0, 1), (0, 2), (1, 2)]

            for session1, session2 in comparisons:
                if test_name == "Repeated-measures ANOVA":
                    # Pairwise t-tests
                    post_stat, post_p = ttest_rel(paired_data[session1], paired_data[session2])
                else:
                    # Pairwise Wilcoxon tests
                    post_stat, post_p = wilcoxon(paired_data[session1], paired_data[session2])

                all_p_values[b_range].append(post_p)
                pairwise_comparisons[b_range].append(f"{b_range}: {session1} vs {session2}")

        # Step 7: Collect results
        summary_data = {
            "beta_range": [b_range],
            "Sample Size": [len(valid_subjects)],
            "Normal Distribution (p > 0.05)": [normal_distribution],
            "Test Name": [test_name],
            "Test Statistic": [test_stat],
            "P-value": [p_value],
        }

        summary_df = pd.DataFrame(summary_data)
        result = pd.concat([result, summary_df])

    # Step 7: Apply multiple comparison correction
    corrections = {}
    correction_methods = ["bonferroni", "holm", "fdr_bh"]

    post_hoc_results = []  # Initialize an empty list to store results for all frequency ranges

    for f_range in BETA_RANGES:
        if not all_p_values[f_range]:  # Check if there are p-values for this frequency range
            print(f"Skipping correction for {f_range}: no post-hoc p-values found.")
            continue  # Skip this frequency range if there are no p-values

        # Apply corrections for each method
        corrections[f_range] = {}
        for method in correction_methods:
            _, corrected_p_values, _, _ = multipletests(all_p_values[f_range], method=method)
            corrections[f_range][method] = corrected_p_values

        # Prepare results for this frequency range
        f_range_results = pd.DataFrame(
            {
                "Fq_range": [f_range] * len(pairwise_comparisons[f_range]),
                "Comparison": pairwise_comparisons[f_range],
                "Raw P-value": all_p_values[f_range],
            }
        )
        for method, corrected_p_values in corrections[f_range].items():
            f_range_results[f"Corrected P-value ({method})"] = corrected_p_values

        post_hoc_results.append(f_range_results)  # Append results for this frequency range

    # Combine all frequency range results into a single DataFrame
    if post_hoc_results:
        post_hoc_results = pd.concat(post_hoc_results, ignore_index=True)
    else:
        # Create an empty DataFrame if no results are available
        post_hoc_results = pd.DataFrame(
            columns=["Fq_range", "Comparison", "Raw P-value"]
            + [f"Corrected P-value ({method})" for method in correction_methods]
        )

    # Combine descriptive statistics into a DataFrame
    descriptive_stats_df = pd.DataFrame(descriptive_stats)

    return (
        result,
        post_hoc_results,
        excluded_overview,
        raw_data,
        all_p_values,
        descriptive_stats_df,
    )


def boxplot_peak_frequency_or_power_three_sessions(
    fooof_spectrum: str, highest_beta_session: str, peak_feature: str, cohort: str, abs_or_rel: str
):
    """
    Plot boxplots per session with scatterplot for individual subjects connected by lines for three sessions.

    Input:
        - peak_feature: "peak_frequency" or "peak_power_auc_per_peak" or "peak_power_auc_fixed_f_range"
            if "peak_frequency" -> around_cf_at_each_session
            if "peak_power_auc_per_peak" -> around_cf_at_each_session
            if "peak_power_auc_fixed_f_range" -> around_cf_at_fixed_session

        - abs_or_rel: "absolute" or "relative" values for peak power
            if "relative" -> relative to the 3MFU session within each subject_hemisphere

    Parameters:
        fooof_spectrum (str): Spectrum data source.
        highest_beta_session (str): Session with highest beta activity.
        peak_feature (str): Feature to analyze.
    """

    loaded_data = analyze_peak_frequency_or_power_three_sessions(
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session,
        peak_feature=peak_feature,
        cohort=cohort,
        abs_or_rel="absolute",
    )

    data = loaded_data[3]  # Extract raw data for plotting

    for b_range in BETA_RANGES:
        range_data = data[b_range]

        if abs_or_rel == "relative":
            # transform data to relative values to the 3MFU session within each subject_hemisphere
            if cohort == "group_1":
                # Normalize values relative to session 1 = "fu3m" within each subject_hemisphere
                range_data = range_data.div(range_data[1], axis=0)

            elif cohort == "group_2":
                # Normalize values relative to session 0 = "fu3m" within each subject_hemisphere
                range_data = range_data.div(range_data[0], axis=0)

            elif cohort == "group_0":
                # Normalize values relative to session 1 = "fu3m" within each subject_hemisphere
                range_data = range_data.div(range_data[1], axis=0)

        # Reshape the dataframe for long-format plotting
        long_data = range_data.reset_index().melt(
            id_vars="subject_hemisphere", var_name="session", value_name=peak_feature
        )

        # Ensure the session column is treated as categorical for proper ordering
        long_data["session"] = pd.Categorical(long_data["session"], categories=[0, 1, 2], ordered=True)

        # Plot the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create boxplots for each session
        sns.boxplot(
            data=long_data,
            x="session",
            y=peak_feature,
            whis=[5, 95],
            width=0.5,
            color="white",
            # palette="pastel",
            showfliers=True,
            ax=ax,
        )

        # Overlay scatterplot with connections for each subject
        x_positions = [0, 1, 2]

        if cohort == "group_1":
            x_positions = [0, 1]

        for subject in range_data.index:
            plt.plot(
                x_positions,  # Sessions (columns of the original dataframe)
                range_data.loc[subject, [0, 1, 2]],  # Peak frequencies for this subject
                marker="o",
                color="gray",
                alpha=0.3,
                linestyle="-",
                linewidth=1,
                markersize=9,
            )

        # Calculate and plot means for each session
        means = range_data.mean(axis=0)  # Mean for each session
        ax.scatter(x_positions, means, color="black", marker="+", s=100, label="Mean")

        # Customize the plot
        ax.set_title(f"Paired Comparison of {b_range} {peak_feature}: {cohort} (Three Sessions)", fontsize=16)
        ax.set_xlabel("Session", fontsize=14)
        ax.set_ylabel(peak_feature, fontsize=14)

        # Adjust y-axis limits based on the feature type
        if abs_or_rel == "absolute":
            if peak_feature == "peak_frequency":
                if b_range == "beta":
                    ax.set_ylim(10, 38)

                elif b_range == "low_beta":
                    ax.set_ylim(10, 23)

                elif b_range == "high_beta":
                    ax.set_ylim(20, 38)

            elif peak_feature in ["peak_power_auc_per_peak", "peak_power_auc_fixed_f_range"]:
                if b_range == "high_beta":
                    ax.set_ylim(-1, 30)
                # ax.set_ylim(-3, 50)

        elif abs_or_rel == "relative":
            ax.set_ylim(-0.2, 3)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        plt.tight_layout()

        # Save the figure
        percept_helpers.save_fig_png_and_svg(
            path=FIGURES_PATH,
            filename=f"revision_paired_comparison_{b_range}_{abs_or_rel}_{peak_feature}_three_sessions_{cohort}",
            figure=fig,
        )


################################### Peak shift analysis >2.5 or > 5 Hz ###################################


def analyze_peak_frequency_differences(cohort: str, beta_range: str, peak_shift: float):
    """
    Analyze peak frequency differences across three sessions and prepare for paired statistical test.

    Parameters:
        data (pd.DataFrame): A DataFrame with columns ['subject_hemisphere', 'session', 'peak_frequency'].

    Input:
        - peak_shift: The minimum shift in Hz to consider a peak as different. e.g. 2.5 or 5 Hz.
        - beta_range: The beta range to analyze (e.g., "beta", "low_beta", "high_beta").
        - cohort: The cohort to analyze (e.g., "group_1", "group_2").

    Returns:
        results (dict): A dictionary with keys:
            - "excluded_patients": A DataFrame specifying excluded patients and their NaN sessions.
            - "comparison_results": A DataFrame with binomial values for both comparisons.
    """

    # load data
    loaded_data = calculate_auc_beta_power(
        fooof_spectrum="periodic_spectrum",
        highest_beta_session="highest_fu3m",
        around_cf="around_cf_at_each_session",
        cohort=cohort,
    )

    data = loaded_data["group_dict"]["ring"]

    # Ensure the input DataFrame has the expected columns
    required_columns = ['subject_hemisphere', 'session', f'{beta_range}_center_frequency']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input data must contain columns: {required_columns}")

    # Pivot the data to wide format for comparisons
    pivot_data = data.pivot(
        index="subject_hemisphere", columns="session", values=f"{beta_range}_center_frequency"
    ).reset_index()
    pivot_data.columns.name = None  # Remove MultiIndex in columns

    # Rename session columns for clarity
    if cohort == "group_1":
        pivot_data.rename(columns={0: 'session1', 3: 'session2', 12: 'session3'}, inplace=True)

    elif cohort == "group_2":
        pivot_data.rename(columns={3: 'session1', 12: 'session2', 18: 'session3'}, inplace=True)

    elif cohort == "group_0":
        pivot_data.rename(columns={0: 'session1', 3: 'session2'}, inplace=True)

    # Step 1: Identify excluded patients
    if cohort in ["group_1", "group_2"]:
        pivot_data['exclude_reason'] = np.where(
            pivot_data[['session1', 'session2']].isna().all(axis=1)
            | pivot_data[['session2', 'session3']].isna().all(axis=1),
            "No peaks in both comparisons",
            None,
        )

    elif cohort == "group_0":
        pivot_data['exclude_reason'] = np.where(
            pivot_data[['session1', 'session2']].isna().all(axis=1),
            "No peaks in both comparisons",
            None,
        )
    excluded_patients = pivot_data[pivot_data['exclude_reason'].notnull()]

    # Filter data to exclude these patients
    valid_data = pivot_data[pivot_data['exclude_reason'].isnull()].copy()

    # Step 2: Analyze session 1 vs session 2
    valid_data['diff_1_2'] = abs(valid_data['session1'] - valid_data['session2'])
    valid_data['binomial_1_2'] = np.where(
        (valid_data['diff_1_2'] > peak_shift) | valid_data[['session1', 'session2']].isna().any(axis=1),
        0,
        1,
    )

    if cohort in ["group_1", "group_2"]:
        # Step 3: Analyze session 2 vs session 3
        valid_data['diff_2_3'] = abs(valid_data['session2'] - valid_data['session3'])
        valid_data['binomial_2_3'] = np.where(
            (valid_data['diff_2_3'] > peak_shift) | valid_data[['session2', 'session3']].isna().any(axis=1),
            0,
            1,
        )

    # Step 4: Identify patients to exclude from both comparisons
    invalid_subjects = excluded_patients['subject_hemisphere'].tolist()
    valid_data = valid_data[~valid_data['subject_hemisphere'].isin(invalid_subjects)]

    # Step 5: Prepare data for paired test
    if cohort in ["group_1", "group_2"]:
        comparison_results = valid_data[['subject_hemisphere', 'binomial_1_2', 'binomial_2_3']]

    elif cohort == "group_0":
        comparison_results = valid_data[['subject_hemisphere', 'binomial_1_2']]

    return {
        "excluded_patients": excluded_patients[['subject_hemisphere', 'exclude_reason']],
        "valid_data": valid_data,
        "comparison_results": comparison_results,
    }


def compare_binomial_proportions(cohort: str, peak_shift: float):
    """
    Compare binomial proportions between two periods using McNemar's test.

    Parameters:
        data (pd.DataFrame): A DataFrame with columns ['subject', 'binomial_1_2', 'binomial_2_3'].

    Returns:
        results (dict): A dictionary with test results and statistical description.
    """

    final_results = {}
    contingency_tables = {}
    description_DF = pd.DataFrame()

    for b_range in BETA_RANGES:

        loaded_data = analyze_peak_frequency_differences(cohort=cohort, beta_range=b_range, peak_shift=peak_shift)
        data = loaded_data["comparison_results"]

        # Ensure the input DataFrame has the expected columns
        required_columns = ['binomial_1_2', 'binomial_2_3']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # Step 1: Create a contingency table
        # comparison of the two columns: showing how many STN fall into category of the two columns
        # possible categories: 0-0 (shift in both periods); 1-1 (no shift in both periods); 0-1 (shift only in first period); 1-0 (shift only in second period)
        contingency_table = pd.crosstab(data['binomial_1_2'], data['binomial_2_3'])
        contingency_tables[b_range] = contingency_table

        # Step 2: Perform McNemar's test: evaluates the balance of changes: 0-1 vs 1-0 (shift-no shift vs no shift-shift)
        mcnemar_result = mcnemar(contingency_table, exact=True)
        mcnemar_stat, mcnemar_p = mcnemar_result.statistic, mcnemar_result.pvalue

        # Step 3: Descriptive statistics
        data['difference'] = data['binomial_1_2'] - data['binomial_2_3']
        mean_diff = data['difference'].mean()
        std_diff = data['difference'].std()
        median_diff = data['difference'].median()

        n_shift_session1_2 = (data['binomial_1_2'] == 0).sum()
        n_shift_session2_3 = (data['binomial_2_3'] == 0).sum()
        n_no_shift_session1_2 = (data['binomial_1_2'] == 1).sum()
        n_no_shift_session2_3 = (data['binomial_2_3'] == 1).sum()
        total_subjects = len(data)

        percentage_shift_session1_2 = (n_shift_session1_2 / total_subjects) * 100
        percentage_shift_session2_3 = (n_shift_session2_3 / total_subjects) * 100
        percentage_no_shift_session1_2 = (n_no_shift_session1_2 / total_subjects) * 100
        percentage_no_shift_session2_3 = (n_no_shift_session2_3 / total_subjects) * 100

        description_shifts_count = {
            "beta_range": [b_range],
            "n_shift_ses1_2": [n_shift_session1_2],
            "n_shift_ses2_3": [n_shift_session2_3],
            "n_no_shift_ses1_2": [n_no_shift_session1_2],
            "n_no_shift_ses2_3": [n_no_shift_session2_3],
            "total_subjects": [total_subjects],
            "perc_shift_ses1_2": [percentage_shift_session1_2],
            "perc_shift_ses2_3": [percentage_shift_session2_3],
            "perc_no_shift_ses1_2": [percentage_no_shift_session1_2],
            "perc_no_shift_ses2_3": [percentage_no_shift_session2_3],
        }
        # transform to dataframe
        description_shifts_count_df = pd.DataFrame(description_shifts_count)
        description_DF = pd.concat([description_DF, description_shifts_count_df])

        # Step 4: Paired t-test and Wilcoxon signed-rank test -> testing whether the median of differences between both columns is significantly different from 0
        # Q: Is there a significant difference in the paired distribution of the two columns?
        ttest_stat, ttest_p = ttest_rel(data['binomial_1_2'], data['binomial_2_3'])
        wilcoxon_stat, wilcoxon_p = wilcoxon(data['binomial_1_2'], data['binomial_2_3'])

        # Step 5: Perform a Wilcoxon signed-rank test on the differences themselves
        # tests whether differences between paired observations deviate significantly from zero, focuses specifically on the direction and magnitude of the differences
        # Q Are the differences between both periods predominantly positive or negative? -1 means shift-no shift, 1 means no shift-shift, 0 means same in both periods
        wilcoxon_stat_diff, wilcoxon_p_diff = wilcoxon(data['difference'])

        # Step 6: Compile results into a DataFrame
        results_data = [
            ["McNemar's Test", mcnemar_stat, mcnemar_p, None, None, None],
            ["Paired t-test", ttest_stat, ttest_p, mean_diff, std_diff, median_diff],
            ["Wilcoxon Test columns", wilcoxon_stat, wilcoxon_p, mean_diff, std_diff, median_diff],
            ["Wilcoxon Test differences", wilcoxon_stat_diff, wilcoxon_p_diff, mean_diff, std_diff, median_diff],
        ]

        results_df = pd.DataFrame(
            results_data,
            columns=["Test", "Statistic", "P-value", "Mean Difference", "Std Dev Difference", "Median Difference"],
        )

        final_results[b_range] = results_df

    return final_results, contingency_tables, description_DF


def plot_peak_frequency_with_binomial(cohort: str, peak_shift: float):
    """
    Plot boxplots and scatterplots for peak frequency with color-coded lines based on binomial stability.

    WATCH OUT: This function plots the data for 3 sessions analysis with peaks indentified at each session.
    But the binomial comparison is also done for subjects who are missing one peak at one session.

    Parameters:
        loaded_data (dict): Data from analyze_peak_frequency_or_power_three_sessions().
        comparison_results (pd.DataFrame): DataFrame with columns "binomial_1_2" and "binomial_2_3".
        peak_feature (str): Feature being analyzed.
        cohort (str): Cohort identifier.
        beta_ranges (list): List of beta ranges.
        save_path (str): Path to save the plots.
    """

    loaded_data = analyze_peak_frequency_or_power_three_sessions(
        fooof_spectrum="periodic_spectrum",
        highest_beta_session="highest_fu3m",
        peak_feature="peak_frequency",
        cohort=cohort,
    )

    data = loaded_data[3]  # Extract raw data for plotting

    for b_range in BETA_RANGES:
        range_data = data[b_range]

        # load the comparison results
        comp_result_data = analyze_peak_frequency_differences(cohort=cohort, beta_range=b_range, peak_shift=peak_shift)
        comparison_results = comp_result_data["comparison_results"]
        # range_data = comp_result_data["valid_data"]

        # rename columns session1, session2, session3 to integers 0, 1, 2
        # range_data.rename(columns={"session1": 0, "session2": 1, "session3": 2}, inplace=True)

        # Reshape the dataframe for long-format plotting
        long_data = range_data.reset_index().melt(
            id_vars="subject_hemisphere", var_name="session", value_name="peak_frequency"
        )

        # Ensure the session column is treated as categorical for proper ordering
        long_data["session"] = pd.Categorical(long_data["session"], categories=[0, 1, 2], ordered=True)

        # Plot the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create boxplots for each session
        sns.boxplot(
            data=long_data,
            x="session",
            y="peak_frequency",
            whis=[5, 95],
            width=0.5,
            # palette="pastel",
            color="white",
            showfliers=True,
            ax=ax,
        )

        # Overlay scatterplot with connections for each subject
        x_positions = [0, 1, 2]
        for subject in range_data.index:
            session_values = range_data.loc[subject, [0, 1, 2]].values

            # Get binomial values for the subject
            binomial_1_2 = comparison_results.loc[
                comparison_results["subject_hemisphere"] == subject, "binomial_1_2"
            ].values[0]
            binomial_2_3 = comparison_results.loc[
                comparison_results["subject_hemisphere"] == subject, "binomial_2_3"
            ].values[0]

            # Determine colors for lines based on binomial values
            color_1_2 = "red" if binomial_1_2 == 0 else "gray"
            color_2_3 = "red" if binomial_2_3 == 0 else "gray"

            # Plot the line between sessions 0 and 1
            plt.plot(
                x_positions[:2],  # Sessions 0 and 1
                session_values[:2],  # Values for sessions 0 and 1
                marker="o",
                color=color_1_2,
                alpha=0.3,
                linestyle="-",
                linewidth=1.5,
                markersize=9,
            )

            # Plot the line between sessions 1 and 2
            plt.plot(
                x_positions[1:],  # Sessions 1 and 2
                session_values[1:],  # Values for sessions 1 and 2
                marker="o",
                color=color_2_3,
                alpha=0.3,
                linestyle="-",
                linewidth=1.5,
                markersize=9,
            )

        # Calculate and plot means for each session
        means = range_data.mean(axis=0)  # Mean for each session
        ax.scatter(x_positions, means, color="black", marker="+", s=100, label="Mean")

        # Customize the plot
        ax.set_title(f"Paired Comparison of {b_range} peak_frequency: {cohort} (Three Sessions)", fontsize=16)
        ax.set_xlabel("Session", fontsize=14)
        ax.set_ylabel("peak_frequency", fontsize=14)

        # Adjust y-axis limits based on the feature type
        if b_range == "beta":
            ax.set_ylim(10, 38)
        elif b_range == "low_beta":
            ax.set_ylim(10, 23)
        elif b_range == "high_beta":
            ax.set_ylim(20, 38)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="best")

        # Save the figure
        percept_helpers.save_fig_png_and_svg(
            path=FIGURES_PATH,
            filename=f"revision_binomial_paired_comparison_{b_range}_peak_frequency_shift_{peak_shift}Hz_three_sessions_{cohort}",
            figure=fig,
        )


def plot_peak_frequency_with_binomial_at_least_one_peak_per_stn(cohort: str, peak_shift: float):
    """
    Plot boxplots and scatterplots for peak frequency with color-coded lines based on binomial stability.
    Includes handling and marking missing values (NaNs) with crosses.

    This is the complete data used for the peak shift analysis, including subjects with missing peaks.
    """

    range_data_dict = {}

    for b_range in BETA_RANGES:
        # range_data = data[b_range]

        comp_result_data = analyze_peak_frequency_differences(cohort=cohort, beta_range=b_range, peak_shift=peak_shift)
        range_data = comp_result_data["valid_data"]

        range_data.rename(columns={"session1": 0, "session2": 1, "session3": 2}, inplace=True)
        range_data.drop(columns=["exclude_reason", "diff_1_2", "diff_2_3"], inplace=True)
        # reset index
        range_data.reset_index(drop=True, inplace=True)

        # Separate rows with NaN values into a separate DataFrame
        nan_data = range_data[range_data.isnull().any(axis=1)]
        range_data_clean = range_data.dropna()

        # reset index
        nan_data.reset_index(drop=True, inplace=True)
        range_data_clean.reset_index(drop=True, inplace=True)

        # Load the comparison results for binomial analysis
        comparison_results = comp_result_data["comparison_results"]

        # Reshape the cleaned dataframe for long-format plotting
        long_data = range_data_clean.reset_index().melt(
            id_vars="subject_hemisphere", var_name="session", value_name="peak_frequency"
        )

        long_data["session"] = pd.Categorical(long_data["session"], categories=[0, 1, 2], ordered=True)

        range_data_dict[b_range] = long_data

        # Plot the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create boxplots for each session
        sns.boxplot(
            data=long_data,
            x="session",
            y="peak_frequency",
            whis=[5, 95],
            width=0.5,
            color="white",
            showfliers=True,
            ax=ax,
        )

        # Get the unique subject_hemisphere values from both dataframes
        valid_subjects_clean = range_data_clean.subject_hemisphere.unique()
        valid_nan_subjects = nan_data.subject_hemisphere.unique()

        # Filter the comparison_results dataframe
        comparison_results_filtered = comparison_results[
            comparison_results["subject_hemisphere"].isin(valid_subjects_clean)
        ]
        comparison_results_nan_filtered = comparison_results[
            comparison_results["subject_hemisphere"].isin(valid_nan_subjects)
        ]

        # Overlay scatterplot with connections for each subject in the cleaned data
        x_positions = [0, 1, 2]
        for s, subject in enumerate(valid_subjects_clean):
            session_values = range_data_clean.loc[s, [0, 1, 2]].values

            # Get binomial values for the subject
            binomial_1_2 = comparison_results_filtered.loc[
                comparison_results_filtered["subject_hemisphere"] == subject, "binomial_1_2"
            ].values[0]
            binomial_2_3 = comparison_results_filtered.loc[
                comparison_results_filtered["subject_hemisphere"] == subject, "binomial_2_3"
            ].values[0]

            # Determine colors for lines based on binomial values
            color_1_2 = "red" if binomial_1_2 == 0 else "gray"
            color_2_3 = "red" if binomial_2_3 == 0 else "gray"

            # Plot the line between sessions 0 and 1
            plt.plot(
                x_positions[:2],
                session_values[:2],
                marker="o",
                color=color_1_2,
                alpha=0.3,
                linestyle="-",
                linewidth=1.5,
                markersize=9,
            )

            # Plot the line between sessions 1 and 2
            plt.plot(
                x_positions[1:],
                session_values[1:],
                marker="o",
                color=color_2_3,
                alpha=0.3,
                linestyle="-",
                linewidth=1.5,
                markersize=9,
            )

        # Overlay data from nan_data (subjects with missing values), if available
        if not nan_data.empty:
            for s, subject in enumerate(valid_nan_subjects):
                session_values = nan_data.loc[s, [0, 1, 2]].values

                # Get binomial values for the subject
                binomial_1_2 = comparison_results_nan_filtered.loc[
                    comparison_results_nan_filtered["subject_hemisphere"] == subject, "binomial_1_2"
                ].values[0]
                binomial_2_3 = comparison_results_nan_filtered.loc[
                    comparison_results_nan_filtered["subject_hemisphere"] == subject, "binomial_2_3"
                ].values[0]

                # Determine colors for lines based on binomial values
                color_1_2 = "red" if binomial_1_2 == 0 else "gray"
                color_2_3 = "red" if binomial_2_3 == 0 else "gray"

                # Plot scatter and connecting lines for non-NaN values
                valid_sessions = [i for i, value in enumerate(session_values) if not np.isnan(value)]

                # Ensure lines are only plotted for consecutive valid sessions
                if len(valid_sessions) > 1:  # Only plot if there's more than one valid session
                    for i in range(len(valid_sessions) - 1):
                        # Check if the two valid sessions are consecutive
                        if valid_sessions[i + 1] == valid_sessions[i] + 1:
                            # Determine color based on binomial value
                            color = color_1_2 if valid_sessions[i] == 0 else color_2_3
                            plt.plot(
                                [x_positions[valid_sessions[i]], x_positions[valid_sessions[i + 1]]],  # X positions
                                [session_values[valid_sessions[i]], session_values[valid_sessions[i + 1]]],  # Y values
                                marker="o",
                                color=color,
                                alpha=0.5,
                                linestyle="-",
                                linewidth=1.5,
                                markersize=9,
                            )

                # Plot crosses for NaN values
                for i, value in enumerate(session_values):
                    if np.isnan(value):
                        # Scenario 1: NaN in the first column
                        if i == 0:
                            valid_session = 1  # Nearest valid column is the second column
                            if not np.isnan(session_values[valid_session]):
                                plt.scatter(
                                    x_positions[valid_session],
                                    session_values[valid_session],
                                    color="red",
                                    marker="x",
                                    s=100,
                                    label="Missing Value" if i == 0 else "",
                                )

                        # Scenario 2: NaN in the third column
                        elif i == 2:
                            valid_session = 1  # Nearest valid column is the second column
                            if not np.isnan(session_values[valid_session]):
                                plt.scatter(
                                    x_positions[valid_session],
                                    session_values[valid_session],
                                    color="red",
                                    marker="x",
                                    s=100,
                                    label="Missing Value" if i == 0 else "",
                                )

                        # Scenario 3: NaN in the second column
                        elif i == 1:
                            # Plot crosses at both the first and third columns
                            for valid_session in [0, 2]:
                                if not np.isnan(session_values[valid_session]):
                                    plt.scatter(
                                        x_positions[valid_session],
                                        session_values[valid_session],
                                        color="red",
                                        marker="x",
                                        s=100,
                                        label="Missing Value" if i == 0 else "",
                                    )

                # # Plot crosses for NaN values
                # for i, value in enumerate(session_values):
                #     if np.isnan(value):
                #         # Plot a cross for NaN values at the nearest valid session
                #         valid_session = i - 1 if i > 0 and not np.isnan(session_values[i - 1]) else i + 1
                #         if 0 <= valid_session < len(session_values) and not np.isnan(session_values[valid_session]):
                #             plt.scatter(
                #                 x_positions[valid_session],
                #                 session_values[valid_session],
                #                 color="red",
                #                 marker="x",
                #                 s=100,
                #                 label="Missing Value" if i == 0 else "",
                #             )

        # Calculate and plot means for each session
        numeric_columns = [0, 1, 2]  # Specify the session columns explicitly
        means = range_data_clean[numeric_columns].mean(axis=0)  # Include only numeric session columns

        # Ensure x_positions matches the length of means
        x_positions_adjusted = range(len(means))

        # Plot means
        ax.scatter(x_positions_adjusted, means, color="black", marker="+", s=100, label="Mean")

        # Customize the plot
        ax.set_title(f"Paired Comparison of {b_range} peak_frequency: {cohort} (Three Sessions)", fontsize=16)
        ax.set_xlabel("Session", fontsize=14)
        ax.set_ylabel("peak_frequency", fontsize=14)

        # Adjust y-axis limits based on the feature type
        if b_range == "beta":
            ax.set_ylim(10, 38)
        elif b_range == "low_beta":
            ax.set_ylim(10, 23)
        elif b_range == "high_beta":
            ax.set_ylim(20, 38)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="best")

        # Save the figure
        percept_helpers.save_fig_png_and_svg(
            path=FIGURES_PATH,
            filename=f"revision_binomial_with_missing_peaks_paired_comparison_{b_range}_peak_frequency_shift_{peak_shift}Hz_three_sessions_{cohort}",
            figure=fig,
        )

    return range_data_dict


########## THIS FUNCTION DOES NOT WORK..... ####################
########### peak frequency plot with binomial also for group 0 with two sessions only ####################


def plot_peak_frequency_with_binomial_also_group_0(cohort: str, peak_shift: float):
    """
    Plot boxplots and scatterplots for peak frequency with color-coded lines based on binomial stability.
    This function adapts to datasets with either two or three sessions.

    Parameters:
        cohort (str): Cohort identifier ("group_0" for two sessions, others for three sessions).
        peak_shift (float): Threshold for peak stability (binomial comparison).
    """

    # Load the main data
    if cohort == "group_0":
        loaded_data = analyze_peak_frequency_or_power_group_0(
            fooof_spectrum="periodic_spectrum",
            highest_beta_session="highest_fu3m",
            peak_feature="peak_frequency",
        )
        data = loaded_data[2]  # Extract raw data for plotting

    else:
        loaded_data = analyze_peak_frequency_or_power_three_sessions(
            fooof_spectrum="periodic_spectrum",
            highest_beta_session="highest_fu3m",
            peak_feature="peak_frequency",
            cohort=cohort,
        )
        data = loaded_data[3]  # Extract raw data for plotting

    for b_range in BETA_RANGES:
        range_data = data[b_range]

        if cohort == "group_0":
            range_data.rename(columns={"session1": 0, "session2": 1}, inplace=True)
            x_positions = [0, 1]  # Two sessions for group_0
        else:
            range_data.rename(columns={"session1": 0, "session2": 1, "session3": 2}, inplace=True)
            x_positions = [0, 1, 2]  # Three sessions for other cohorts

        # Load the comparison results
        comp_result_data = analyze_peak_frequency_differences(cohort=cohort, beta_range=b_range, peak_shift=peak_shift)
        comparison_results = comp_result_data["comparison_results"]

        # Determine session range based on the cohort
        session_range = [0, 1] if cohort == "group_0" else [0, 1, 2]

        # Reshape the dataframe for long-format plotting
        long_data = range_data.reset_index().melt(
            id_vars="subject_hemisphere", var_name="session", value_name="peak_frequency"
        )

        # Ensure the session column is treated as categorical for proper ordering
        long_data["session"] = pd.Categorical(long_data["session"], categories=session_range, ordered=True)

        # Plot the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplots for each session
        sns.boxplot(
            data=long_data,
            x="session",
            y="peak_frequency",
            whis=[5, 95],
            width=0.5,
            color="white",
            showfliers=True,
            ax=ax,
        )

        # Overlay scatterplot with connections for each subject
        # x_positions = session_range
        # for s, subject in enumerate(range_data.subject_hemisphere.unique()):
        #     session_values = range_data.loc[s, session_range].values

        #     # Get binomial values for the subject
        #     binomial_1_2 = comparison_results.loc[
        #         comparison_results["subject_hemisphere"] == subject, "binomial_1_2"
        #     ].values[0]

        #     binomial_2_3 = None
        #     if len(session_range) > 2:  # Check for three sessions
        #         binomial_2_3 = comparison_results.loc[
        #             comparison_results["subject_hemisphere"] == subject, "binomial_2_3"
        #         ].values[0]

        x_positions = session_range
        for subject in range_data.index:
            session_values = range_data.loc[subject, session_range].values

            # Get binomial values for the subject
            binomial_1_2 = comparison_results.loc[
                comparison_results["subject_hemisphere"] == subject, "binomial_1_2"
            ].values[0]

            binomial_2_3 = None
            if len(session_range) > 2:  # Check for three sessions
                binomial_2_3 = comparison_results.loc[
                    comparison_results["subject_hemisphere"] == subject, "binomial_2_3"
                ].values[0]

            # Determine colors for lines based on binomial values
            color_1_2 = "red" if binomial_1_2 == 0 else "gray"
            color_2_3 = "red" if binomial_2_3 == 0 else "gray" if binomial_2_3 is not None else None

            # Plot the line between sessions 0 and 1
            if 0 in range_data.columns and 1 in range_data.columns:
                # Plot the line between sessions 0 and 1
                plt.plot(
                    x_positions[:2],  # Sessions 0 and 1
                    session_values[:2],  # Values for sessions 0 and 1
                    marker="o",
                    color=color_1_2,
                    alpha=0.3,
                    linestyle="-",
                    linewidth=1.5,
                    markersize=9,
                )

            if len(session_range) > 2 and 1 in range_data.columns and 2 in range_data.columns:
                # Plot the line between sessions 1 and 2
                plt.plot(
                    x_positions[1:],  # Sessions 1 and 2
                    session_values[1:],  # Values for sessions 1 and 2
                    marker="o",
                    color=color_2_3,
                    alpha=0.3,
                    linestyle="-",
                    linewidth=1.5,
                    markersize=9,
                )

        # Calculate and plot means for each session
        means = range_data.mean(axis=0)  # Mean for each session
        ax.scatter(x_positions, means, color="black", marker="+", s=100, label="Mean")

        # Customize the plot
        session_label = "Two Sessions" if cohort == "group_0" else "Three Sessions"
        ax.set_title(f"Paired Comparison of {b_range} peak_frequency: {cohort} ({session_label})", fontsize=16)
        ax.set_xlabel("Session", fontsize=14)
        ax.set_ylabel("peak_frequency", fontsize=14)

        # Adjust y-axis limits based on the feature type
        if b_range == "beta":
            ax.set_ylim(10, 38)
        elif b_range == "low_beta":
            ax.set_ylim(10, 23)
        elif b_range == "high_beta":
            ax.set_ylim(20, 38)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="best")

        # Save the figure
        percept_helpers.save_fig_png_and_svg(
            path=FIGURES_PATH,
            filename=f"revision_binomial_paired_comparison_{b_range}_peak_frequency_shift_{peak_shift}Hz_{session_label}_{cohort}",
            figure=fig,
        )
