""" FOOOF Model """


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
import scipy
import fooof
from fooof.plts.spectra import plot_spectrum

# Local Imports
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
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


def get_input_w_wo_knee(message: str) -> str:
    """Get `w` or `wo` user input."""
    while True:
        user_input = input(f"{message} (w/wo)? ")
        if user_input.lower() in ["w", "wo"]:
            break
        print(
            f"Input must be `w` or `wo`. Got: {user_input}."
            " Please provide a valid input."
        )
    return user_input


def fooof_fit_tfr(incl_sub: list):
    """

    Input: 
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
      
    1) Load the Power Spectrum from main Class:
        - unfiltered
        - rawPSD (not normalized)
        - all channels: Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - condition: only ran m0s0 so far, if you also want to run m1s0, make sure you analyze data seperately!
        - save the features Power Spectrum and Frequencies as variable for each subject, hemisphere, session, channel combination


    2) First set and fit a FOOOF model without a knee -> within a frequency range from 5-95 Hz
        - peak_width_limits=(2, 20.0),
        - max_n_peaks=4,
        - min_peak_height=0.0,
        - peak_threshold=1.0,
        - aperiodic_mode="fixed", # fitting without knee component
        - verbose=True,

        plot a figure with the raw Power spectrum and the fitted model 

        POPUP: input asked (y/n) -> "Try new fit with knee?"

    3) if you choose "y": 
        - a new FOOOF model with aperiodic_mode="knee" will be fitted and plotted
        
        POPUP: input asked (w/wo) -> "Use fit with or without knee?"
    
        depending on your input the model with or without knee will be used
        and the chosen figure will be saved into figure folder of each subject:
        figure filename: fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_wo_knee.png

    
        if you choose "n":
        - the model without knee will be used 
    

    
    4) Extract following parameters and save as columns into DF 
        - 0: "subject_hemisphere", 
        - 1: "session", 
        - 2: "bipolar_channel", 
        - 3: "fooof_error", 
        - 4: "fooof_r_sq", 
        - 5: "fooof_exponent", 
        - 6: "fooof_offset", 
        - 7: "fooof_knee", 
        - 8: "fooof_number_peaks",
        - 9: "alpha_peak_CF_power_bandWidth",
        - 10: "low_beta_peak_CF_power_bandWidth",
        - 11: "high_beta_peak_CF_power_bandWidth",
        - 12: "beta_peak_CF_power_bandWidth",
        - 13: "gamma_peak_CF_power_bandWidth",
         
    5) save Dataframe into results folder of each subject
        - filename: "fooof_model_sub{subject}.json"

    """

    # define variables 
    hemispheres = ["Right", "Left"]
    sessions = ['postop', 'fu3m', 'fu12m', 'fu18m']
    channels = ['03', '13', '02', '12', '01', '23', 
                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                '1A2A', '1B2B', '1C2C']
    
    freq_range = [5, 95] # frequency range to fit FOOOF model


    ################### Load an unfiltered Power Spectrum with their frequencies for each STN ###################

    fooof_results = {}

    for subject in incl_sub:

        # get path to results folder of each subject
        local_figures_path = findfolders.get_local_path(folder="figures", sub=subject)
        local_results_path = findfolders.get_local_path(folder="results", sub=subject)


        for hemisphere in hemispheres:

            # get power spectrum and frequencies from each STN
            data_power_spectrum = mainAnalysis_class.MainClass(
                sub=subject,
                hemisphere=hemisphere,
                filter="unfiltered",
                result="PowerSpectrum",
                incl_session=sessions,
                pickChannels=channels,
                normalization=["rawPsd"],
                feature=["frequency", "time_sectors", "rawPsd", "SEM_rawPsd"]
            )

            for ses in sessions:

                try:
                    getattr(data_power_spectrum, ses)

                except AttributeError:
                    continue

                for chan in channels:
                    
                    # get the power spectra and frequencies from each channel
                    chan_data = getattr(data_power_spectrum, ses)
                    chan_data = getattr(chan_data, f"BIP_{chan}")
                    
                    power_spectrum = np.array(chan_data.rawPsd.data)
                    freqs = np.array(chan_data.frequency.data)

                    ############ SET PLOT LAYOUT ############
                    fig_wo_knee, ax_wo_knee = plt.subplots(2,1, figsize=(10,15))

                    # Plot the unfiltered Power spectrum in first ax
                    plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False,
                                    ax=ax_wo_knee[0])
                    

                    ############ SET FOOOF MODEL ############

                    model_without_knee = fooof.FOOOF(
                            peak_width_limits=(2, 20.0),
                            max_n_peaks=4,
                            min_peak_height=0.0,
                            peak_threshold=1.0,
                            aperiodic_mode="fixed", # fitting without knee component
                            verbose=True,
                        )
                    
                    # always fit a large Frequency band, later you can select Peaks within specific freq bands
                    model_without_knee.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                    # Plot an example power spectrum, with a model fit in second ax
                    # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                    model_without_knee.plot(ax=ax_wo_knee[1])

                    fig_wo_knee.suptitle(f"FOOOF model (w/o knee): \nsub {subject}, {hemisphere} hemisphere, {ses}, bipolar channel: {chan}",
                                         ha="center", fontsize=20)
                    
                    fig_wo_knee.tight_layout()


                    
                    # show the model fit without knee
                    plt.show(fig_wo_knee)

                    # input options: "y" or "n"
                    input_y_or_n = get_input_y_n("Try new fit with knee?") # interrups run and asks for input
                    
                    plt.close(fig_wo_knee)

                    # model with knee if input == y
                    if input_y_or_n == "y":

                        fig_with_knee, ax_with_knee = plt.subplots(2,1, figsize=(10,15))

                        # Plot the unfiltered Power spectrum in first ax
                        plot_spectrum(freqs, power_spectrum, log_freqs=False, log_powers=False,
                                        ax=ax_with_knee[0])

                        model_with_knee = fooof.FOOOF(
                            peak_width_limits=(2, 20.0),
                            max_n_peaks=4,
                            min_peak_height=0.0,
                            peak_threshold=1.0,
                            aperiodic_mode="knee", # fitting with knee component
                            verbose=True,
                        )

                        # always fit a large Frequency band, later you can select Peaks within specific freq bands
                        model_with_knee.fit(freqs=freqs, power_spectrum=power_spectrum, freq_range=freq_range)

                        # Plot an example power spectrum, with a model fit in second ax
                        # model.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'}, ax=ax[1])
                        model_with_knee.plot(ax=ax_with_knee[1])

                        fig_with_knee.suptitle(f"FOOOF model (with knee): \nsub {subject}, {hemisphere} hemisphere, {ses}, bipolar channel: {chan}",
                                               ha="center", fontsize=20)
                        
                        fig_with_knee.tight_layout()

                        
                        # show the model fit with knee
                        plt.show(fig_with_knee)

                        # input options: "w" or "wo"
                        input_w_or_wo_knee = get_input_w_wo_knee("Use fit with or without knee?")

                        plt.close(fig_with_knee)

                        # decide which model to use depending on the input
                        if input_w_or_wo_knee == "w":

                            model = model_with_knee
                            fig_with_knee.savefig(local_figures_path + f"\\fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{input_w_or_wo_knee}_knee.png")
                            knee_parameter = model.get_params('aperiodic_params', 'knee')

                        elif input_w_or_wo_knee == "wo":

                            model = model_without_knee
                            fig_wo_knee.savefig(local_figures_path + f"\\fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_{input_w_or_wo_knee}_knee.png")
                            knee_parameter = "wo_knee"
                    
                    # "n" as input of the first question will directly continue with model_without_knee
                    elif input_y_or_n == "n":

                        model = model_without_knee
                        knee_parameter = "wo_knee"
                        fig_wo_knee.savefig(local_figures_path + f"\\fooof_model_sub{subject}_{hemisphere}_{ses}_{chan}_wo_knee.png")
                        

                    # extract parameters from the chosen model
                    # model.print_results()

                    ############ SAVE APERIODIC PARAMETERS ############
                    # goodness of fit
                    err = model.get_params('error')
                    r_sq = model.r_squared_

                    # aperiodic components
                    exp = model.get_params('aperiodic_params', 'exponent')
                    offset = model.get_params('aperiodic_params', 'offset')

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
                    STN = "_".join([subject, hemisphere])

                    fooof_results[f"{subject}_{hemisphere}_{ses}_{chan}"] = [STN, ses, chan, 
                                                        err, r_sq, exp, offset, knee_parameter, 
                                                        number_peaks, alpha_peak, low_beta_peak, high_beta_peak, beta_peak, gamma_peak]
        # store results in a DataFrame
        fooof_results_df = pd.DataFrame(fooof_results)  
        fooof_results_df.rename(index={0: "subject_hemisphere", 
                                    1: "session", 
                                    2: "bipolar_channel", 
                                    3: "fooof_error", 
                                    4: "fooof_r_sq", 
                                    5: "fooof_exponent", 
                                    6: "fooof_offset", 
                                    7: "fooof_knee", 
                                    8: "fooof_number_peaks",
                                    9: "alpha_peak_CF_power_bandWidth",
                                    10: "low_beta_peak_CF_power_bandWidth",
                                    11: "high_beta_peak_CF_power_bandWidth",
                                    12: "beta_peak_CF_power_bandWidth",
                                    13: "gamma_peak_CF_power_bandWidth",
                                    }, 
                                inplace=True)
                        
        fooof_results_df = fooof_results_df.transpose()              

        # save DF in subject results folder
        fooof_results_df.to_json(os.path.join(local_results_path, f"fooof_model_sub{subject}.json"))

    	
   
    return {
        "fooof_results_df": fooof_results_df, 
        }
    

    
def fooof_peaks_per_session():

    """
    Load the group FOOOF json file as Dataframe: 
    "fooof_model_group_data.json" from the group result folder



    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    

    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    session_peak_dict = {}

    # load the json file as Dataframe
    fooof_group_result = loadResults.load_group_fooof_result()

    for ses in sessions:

        # get the dataframes for each session seperately
        fooof_session = fooof_group_result.loc[(fooof_group_result["session"]==ses)]

        # get total number of recordings (per STN all 15 recordings included) per session 
        total_number_all_channels_session = len(fooof_session)

        for freq in freq_bands:
            freq_list = []

            for item in fooof_session[f"{freq}_peak_CF_power_bandWidth"].values:
                # in the column "{freq}_peak_CF_power_bandWidth" each cell contains a list
                # only take rows with a list, if None is not in the list (so only take rows, if there was a Peak)
                if None not in item:
                    freq_list.append(item)

            freq_session_df = fooof_session.loc[fooof_session[f"{freq}_peak_CF_power_bandWidth"].isin(freq_list)]

            # count how many freq Peaks exist
            number_freq_peaks_session =  len(freq_session_df)

            # calculate % of channels with freq Peaks in this session
            percentage_freq_peaks_session = number_freq_peaks_session / total_number_all_channels_session

            session_peak_dict[f"{ses}_{freq}"] = [ses, freq, total_number_all_channels_session, number_freq_peaks_session, percentage_freq_peaks_session]
        
    # save the results in a dataframe
    session_peak_df = pd.DataFrame(session_peak_dict)
    session_peak_df.rename(index={
        0: "session",
        1: "frequency_band",
        2: "total_chans_number",
        3: "number_chans_with_peaks",
        4: "percentage_chans_with_peaks",
    }, inplace=True)
    session_peak_df = session_peak_df.transpose()

    # save Dataframe with data 
    session_peak_filepath = os.path.join(results_path, f"fooof_peaks_per_session.pickle")
    with open(session_peak_filepath, "wb") as file:
        pickle.dump(session_peak_df, file)
    
    print("file: ", 
          "fooof_peaks_per_session.pickle",
          "\nwritten in: ", results_path
          )
    
    return session_peak_df


def fooof_plot_peaks_per_session():

    """
    Load the file "fooof_peaks_per_session.pickle" from the group results folder

    Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band 
        - x = session
        - y = percentage_chans_with_peaks
        - label = freq_band


    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")

    # load the pickle file with the numbers and percentages of channels with peaks in all frequency bands
    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    # filter dataframe for each freq bands seperately
    alpha_peaks = peaks_per_session.loc[peaks_per_session.frequency_band == "alpha"]
    low_beta_peaks = peaks_per_session.loc[peaks_per_session.frequency_band == "low_beta"]
    high_beta_peaks = peaks_per_session.loc[peaks_per_session.frequency_band == "high_beta"]
    beta_peaks = peaks_per_session.loc[peaks_per_session.frequency_band == "beta"]
    gamma_peaks = peaks_per_session.loc[peaks_per_session.frequency_band == "gamma"]


    # Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band 
    fig = plt.figure()

    font = {"size": 14}

    plt.plot(alpha_peaks.session, alpha_peaks.percentage_chans_with_peaks, label="alpha")
    plt.plot(low_beta_peaks.session, low_beta_peaks.percentage_chans_with_peaks, label="low beta")
    plt.plot(high_beta_peaks.session, high_beta_peaks.percentage_chans_with_peaks, label="high beta")
    plt.plot(beta_peaks.session, beta_peaks.percentage_chans_with_peaks, label="beta")
    plt.plot(gamma_peaks.session, gamma_peaks.percentage_chans_with_peaks, label="gamma")

    plt.scatter(alpha_peaks.session, alpha_peaks.percentage_chans_with_peaks)
    plt.scatter(low_beta_peaks.session, low_beta_peaks.percentage_chans_with_peaks)
    plt.scatter(high_beta_peaks.session, high_beta_peaks.percentage_chans_with_peaks)
    plt.scatter(beta_peaks.session, beta_peaks.percentage_chans_with_peaks)
    plt.scatter(gamma_peaks.session, gamma_peaks.percentage_chans_with_peaks)

    plt.title("Relative amount of channels with Peaks", fontdict={"size": 19})

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("session", fontdict=font)
    plt.ylabel("channels with Peak \nrelative to all recorded channels", fontdict=font)
    fig.tight_layout()

    # save figure in group Figures folder
    fig.savefig(figures_path + "\\fooof_peaks_per_session.png", bbox_inches="tight")

    print("figure: ", 
          f"fooof_peaks_per_session.png",
          "\nwritten in: ", figures_path
          )
    

def fooof_low_vs_high_beta_ratio():

    """
    Load the file "fooof_peaks_per_session.pickle" from the group results folder

    Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band 
        - x = session
        - y = Peaks relative to all Peaks in beta band"
        - label = freq_band
    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")

    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    rel_low_vs_high_beta = {}

    for ses in sessions:

        session_df = peaks_per_session.loc[peaks_per_session.session==ses]

        low_beta_peaks = session_df.loc[session_df.frequency_band=="low_beta"]
        low_beta_peaks = low_beta_peaks.number_chans_with_peaks.values
        low_beta_peaks = low_beta_peaks[0]

        high_beta_peaks = session_df.loc[session_df.frequency_band=="high_beta"]
        high_beta_peaks = high_beta_peaks.number_chans_with_peaks.values
        high_beta_peaks = high_beta_peaks[0]

        # number of low + high beta peaks
        beta_peaks = low_beta_peaks + high_beta_peaks

        # calculate the relative amount of Peaks in low beta and high beta from all Peaks in beta band
        relative_low_beta = low_beta_peaks / beta_peaks
        relative_high_beta = high_beta_peaks / beta_peaks

        rel_low_vs_high_beta[f"{ses}"] = [ses, beta_peaks, low_beta_peaks, high_beta_peaks, relative_low_beta, relative_high_beta]


    # results transformed to a dataframe
    session_low_vs_high_peak_df = pd.DataFrame(rel_low_vs_high_beta)
    session_low_vs_high_peak_df.rename(index={
        0: "session",
        1: "beta_peaks",
        2: "low_beta_peaks",
        3: "high_beta_peaks",
        4: "relative_low_beta",
        5: "relative_high_beta"
    }, inplace=True)
    session_low_vs_high_peak_df = session_low_vs_high_peak_df.transpose()


    # Plot as lineplot
    fig = plt.figure()

    font = {"size": 14}

    plt.plot(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_low_beta, label="low beta")
    plt.plot(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta, label="high beta")

    plt.scatter(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_low_beta,)
    plt.scatter(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta)

    # alternative: stacked Barplot
    # plt.bar(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_low_beta, label="low beta")
    # plt.bar(session_low_vs_high_peak_df.session, session_low_vs_high_peak_df.relative_high_beta, label="high beta", bottom=session_low_vs_high_peak_df.relative_low_beta)


    plt.title("Relative amount of low beta vs high beta peaks", fontdict={"size": 19})

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("session", fontdict=font)
    plt.ylabel("Peaks relative to all Peaks in beta band", fontdict=font)
    fig.tight_layout()

    # save figure in group Figures folder
    fig.savefig(figures_path + "\\fooof_low_vs_high_beta_peaks_per_session.png", bbox_inches="tight")

    print("figure: ", 
          f"fooof_low_vs_high_beta_peaks_per_session.png",
          "\nwritten in: ", figures_path
          )


    return session_low_vs_high_peak_df


def fooof_highest_beta_peak_cf():

    """
    Load the group FOOOF json file as Dataframe: 
    "fooof_model_group_data.json" from the group result folder

    Plot a violinplot of the center frequencies of the highest Peaks within the beta band (13-35 Hz) at different sessions
        - x = session
        - y = "Peak center frequency \nin beta band (13-35 Hz)"
    
    Statistical Test between session groups
        - Mann-Whitney

    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")
    results_path = findfolders.get_local_path(folder="GroupResults")

    # load the json file as df
    fooof_result = loadResults.load_group_fooof_result()

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    beta_peak_parameters = {}

    for ses in sessions:

        if ses == "postop":
            numeric_session = 0 # violinplot only allowed integers, not strings for x axis
        
        elif ses == "fu3m":
            numeric_session = 3
        
        elif ses == "fu12m":
            numeric_session = 12
        
        elif ses == "fu18m":
            numeric_session = 18

        session_df = fooof_result.loc[fooof_result.session==ses]
        beta_peaks_wo_None = []

        # only get the rows with Peaks (drop all rows with None)
        for item in session_df.beta_peak_CF_power_bandWidth.values:
        
            if None not in item:
                beta_peaks_wo_None.append(item)

        beta_peak_ses_df = session_df.loc[session_df["beta_peak_CF_power_bandWidth"].isin(beta_peaks_wo_None)]

        # get only the center frequency from the column beta_peak_CF_power_bandWidth
        for i, item in enumerate(beta_peak_ses_df.beta_peak_CF_power_bandWidth.values):
            # item is a list of center frequency, power, band width of highest Peak in beta band

            beta_cf = item[0]
            beta_power = item[1]
            beta_band_width = item[2]

            beta_peak_parameters[f"{ses}_{i}"] = [numeric_session, beta_cf, beta_power, beta_band_width]


    # save the results in a dataframe
    beta_peak_parameters_df = pd.DataFrame(beta_peak_parameters)
    beta_peak_parameters_df.rename(index={
        0: "session",
        1: "beta_cf",
        2: "beta_power",
        3: "beta_band_width",
    }, inplace=True)
    beta_peak_parameters_df = beta_peak_parameters_df.transpose()


    ##################### PLOT VIOLINPLOT OF CENTER FREQUENCIES OF HIGHEST BETA PEAKS #####################

    fig=plt.figure()
    ax = fig.add_subplot()

    sns.violinplot(data=beta_peak_parameters_df, x="session", y="beta_cf", palette="pastel", inner="box", ax=ax)

    # statistical test: doesn't work if groups have different sample size
    num_sessions = [0.0, 3.0, 12.0, 18.0]
    pairs = list(combinations(num_sessions, 2))

    annotator = Annotator(ax, pairs, data=beta_peak_parameters_df, x='session', y='beta_cf')
    annotator.configure(test='Mann-Whitney', text_format='star') # or t-test_ind ??
    annotator.apply_and_annotate()

    sns.stripplot(
        data=beta_peak_parameters_df,
        x="session",
        y="beta_cf",
        ax=ax,
        size=6,
        color="black",
        alpha=0.2, # Transparency of dots
    )

    sns.despine(left=True, bottom=True) # get rid of figure frame

    plt.title("Highest Beta Peak Center Frequency")
    plt.ylabel("Peak center frequency \nin beta band (13-35 Hz)")
    plt.xlabel("session")

    fig.tight_layout()
    fig.savefig(figures_path + "\\fooof_highest_beta_peak_center_freq.png", bbox_inches="tight")

    print("figure: ", 
          "fooof_highest_beta_peak_center_freq.png",
          "\nwritten in: ", figures_path
          )

    ##################### DESCRIPTION OF EACH SESSION GROUP #####################
    # describe each group
    num_sessions = [0.0, 3.0, 12.0, 18.0]
    group_description = {}

    for ses in num_sessions:

        group = beta_peak_parameters_df.loc[beta_peak_parameters_df.session==ses]
        group = np.array(group.beta_cf.values)

        description = scipy.stats.describe(group)

        group_description[f"{ses}_months_postop"] = description


    description_results = pd.DataFrame(group_description)
    description_results.rename(index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"}, inplace=True)
    description_results = description_results.transpose()

    # save Dataframe with data 
    description_results_filepath = os.path.join(results_path, "fooof_center_freq_session_description_highest_beta_peak.pickle")
    with open(description_results_filepath, "wb") as file:
        pickle.dump(description_results, file)
    
    print("file: ", 
          "fooof_center_freq_session_description_highest_beta_peak.pickle",
          "\nwritten in: ", results_path
          )
    

    return {
        "beta_peak_parameters_df": beta_peak_parameters_df,
        "description_results": description_results

    }
























            









