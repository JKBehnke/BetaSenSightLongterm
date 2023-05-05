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

    For all electrodes at all sessions seperately:
    Get the number of Peaks and the % of channels with Peaks within one electrode at one session 

    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    

    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    session_peak_dict = {}

    # load the json file as Dataframe
    fooof_group_result = loadResults.load_group_fooof_result()

    # list of unique STNs
    stn_unique = list(fooof_group_result.subject_hemisphere.unique())

    for stn in stn_unique:
        # get dataframe of one stn 
        fooof_stn = fooof_group_result.loc[fooof_group_result.subject_hemisphere==stn]

        for ses in sessions:

            # check if session exists for current STN
            if ses not in fooof_stn.session.values:
                continue

            # get the dataframes for each session seperately
            fooof_session = fooof_stn.loc[(fooof_stn["session"]==ses)]

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

                session_peak_dict[f"{stn}_{ses}_{freq}"] = [stn, ses, freq, total_number_all_channels_session, number_freq_peaks_session, percentage_freq_peaks_session]
            
    # save the results in a dataframe
    session_peak_df = pd.DataFrame(session_peak_dict)
    session_peak_df.rename(index={
        0: "subject_hemisphere",
        1: "session",
        2: "frequency_band",
        3: "total_chans_number",
        4: "number_chans_with_peaks",
        5: "percentage_chans_with_peaks",
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
    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]


    # load the pickle file with the numbers and percentages of channels with peaks in all frequency bands
    peaks_per_session = loadResults.load_fooof_peaks_per_session()

    perc_chans_with_peaks_data = {}


    # calculate the Mean and standard deviation across STNs from each frequency band at every session
    for ses in sessions:

        # get the dataframes for each session seperately
        fooof_session = peaks_per_session.loc[(peaks_per_session["session"]==ses)]

        for freq in freq_bands:

            # get dataframes for each frequency
            freq_session_df = fooof_session.loc[fooof_session.frequency_band == freq]

            mean_percentage_chans_with_peaks = np.mean(freq_session_df.percentage_chans_with_peaks.values)
            std_percentage_chans_with_peaks = np.std(freq_session_df.percentage_chans_with_peaks.values)
            sample_size = len(freq_session_df.percentage_chans_with_peaks.values)

            perc_chans_with_peaks_data[f"{ses}_{freq}"] = [ses, freq, mean_percentage_chans_with_peaks, std_percentage_chans_with_peaks, sample_size]

            #df_copy = freq_df.copy()
            # new column with mean-std and mean+std
            # df_copy["mean-std"] = df_copy.mean_percentage_chans_with_peaks.values - df_copy.std_percentage_chans_with_peaks.values
            # df_copy["mean+std"] = df_copy.mean_percentage_chans_with_peaks.values + df_copy.std_percentage_chans_with_peaks.values


    # save the mean and std values in a dataframe
    perc_chans_with_peaks_df = pd.DataFrame(perc_chans_with_peaks_data)
    perc_chans_with_peaks_df.rename(index={
        0: "session",
        1: "frequency_band",
        2: "mean_percentage_chans_with_peaks",
        3: "std_percentage_chans_with_peaks",
        4: "sample_size"
    }, inplace=True)
    perc_chans_with_peaks_df = perc_chans_with_peaks_df.transpose()


    ###################### Plot a lineplot for the amount of channels with Peaks per session with lines for each freq band ######################
    fig = plt.figure()
    font = {"size": 14}

    for freq in freq_bands:

        freq_df = perc_chans_with_peaks_df.loc[perc_chans_with_peaks_df.frequency_band == freq]
      
        if freq=="alpha":
            color="sandybrown"
        
        elif freq=="beta":
            color="darkcyan"
        
        elif freq=="low_beta":
            color="turquoise"

        elif freq=="high_beta":
            color="cornflowerblue"
        
        elif freq=="gamma":
            color="plum"

        plt.plot(freq_df.session, freq_df.mean_percentage_chans_with_peaks, label=freq, color=color)
        # plt.fill_between(df_copy.session, 
        #                  df_copy["mean-std"],
        #                  df_copy["mean+std"],
        #                  color="gainsboro", alpha=0.5)
        
        plt.scatter(freq_df.session, freq_df.mean_percentage_chans_with_peaks, color=color)
        #plt.errorbar(freq_df.session, freq_df.mean_percentage_chans_with_peaks, yerr=freq_df.std_percentage_chans_with_peaks, fmt="o", color=color)


    plt.title("BSSU channels with Peaks", fontdict={"size": 19})

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("session", fontdict=font)
    plt.ylabel("amount of channels with Peaks \nrelative to all channels per electrode", fontdict=font)
    fig.tight_layout()

    # save figure in group Figures folder
    fig.savefig(figures_path + "\\fooof_peaks_per_session.png", bbox_inches="tight")

    print("figure: ", 
          f"fooof_peaks_per_session.png",
          "\nwritten in: ", figures_path
          )


    return perc_chans_with_peaks_df




def fooof_peaks_in_freq_band_stats():

    """
    
    """

    figures_path = findfolders.get_local_path(folder="GroupFigures")
    freq_bands = ["alpha", "low_beta", "high_beta", "beta", "gamma"]


    # load the pickle file with the numbers and percentages of channels with peaks in all frequency bands
    peaks_per_session = loadResults.load_fooof_peaks_per_session()


    for f, freq in enumerate(freq_bands):

        freq_df = peaks_per_session.loc[peaks_per_session.frequency_band == freq]

        # replace session names by integers because of seaborn plot
        freq_df = freq_df.replace(to_replace="postop", value=0)
        freq_df = freq_df.replace(to_replace="fu3m", value=3)
        freq_df = freq_df.replace(to_replace="fu12m", value=12)
        freq_df = freq_df.replace(to_replace="fu18m", value=18)
    
        fig=plt.figure()
        ax=fig.add_subplot()
        font = {"size": 14}

        sns.violinplot(data=freq_df, x="session", y="percentage_chans_with_peaks", palette="pastel", inner="box", ax=ax)

        sns.stripplot(
            data=freq_df,
            x="session",
            y="percentage_chans_with_peaks",
            ax=ax,
            size=6,
            color="black",
            alpha=0.2, # Transparency of dots
        )

        sns.despine(left=True, bottom=True) # get rid of figure frame


        # statistical test: doesn't work if groups have different sample size
        num_sessions = [0, 3, 12, 18]
        pairs = list(combinations(num_sessions, 2))

        annotator = Annotator(ax, pairs, data=freq_df, x='session', y='percentage_chans_with_peaks')
        annotator.configure(test='Mann-Whitney', text_format='star') # or ANOVA first to check if there is any difference between all groups
        annotator.apply_and_annotate()

        plt.title(f"BSSU channels with Peaks in {freq}", fontdict={"size": 19})
        plt.ylabel(f"amount of channels with Peaks \n rel. to all channels per electrode", fontdict=font)
        plt.ylim(-0.25, 2.5)
        plt.xlabel("session", fontdict=font)

        fig.tight_layout()

        # save figure in group Figures folder
        fig.savefig(figures_path + f"\\fooof_{freq}_peaks_per_session_violinplot.png", bbox_inches="tight")

    return annotator




    

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

    fig = plt.figure()
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


def fooof_rank_beta_peak_power():

    """
    load the results file: "fooof_model_group_data.json" with the function load_group_fooof_result()

    1) split the beta peak array into three seperate columns:
        - beta_center_frequency
        - beta_peak_power
        - beta_band_width
    
    2) for each stn - session - channel group combination: 
        - add a column with ranks of beta peak power 

    """

    ################# VARIABLES #################
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    channel_groups = ["ring", "segm_inter", "segm_intra"]

    results_path = findfolders.get_local_path(folder="GroupResults")

    # load the fooof result
    fooof_group_result = loadResults.load_group_fooof_result()

    ################# only take the beta peaks and split the array into seperate columns #################
    # define split array function
    split_array = lambda x: pd.Series(x)

    # new columns with single parameters per cell
    fooof_group_result_copy = fooof_group_result.copy()
    fooof_group_result_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = fooof_group_result_copy["beta_peak_CF_power_bandWidth"].apply(split_array)
    fooof_group_result_copy = fooof_group_result_copy.drop(columns=["alpha_peak_CF_power_bandWidth", "low_beta_peak_CF_power_bandWidth", "high_beta_peak_CF_power_bandWidth", "gamma_peak_CF_power_bandWidth"])


    #################  RANK CHANNELS BY THEIR BETA PEAK POWER #################
    stn_unique = list(fooof_group_result_copy.subject_hemisphere.unique())

    rank_beta_power_dataframe = pd.DataFrame() 

        
    for stn in stn_unique:

        stn_dataframe = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for group in channel_groups:

            if group == "ring":
                channel_list = ["01", "12", "23"]
            
            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

            # get only the channels within a channel group
            stn_group_dataframe = stn_dataframe.loc[stn_dataframe["bipolar_channel"].isin(channel_list)]

            
            # rank channels by their beta peak power for each session
            for ses in sessions:

                # check if session exists for this stn
                if ses not in stn_group_dataframe.session.values:
                    continue

                # select only session rows
                stn_group_ses_df = stn_group_dataframe.loc[stn_group_dataframe.session == ses]
                

                # rank the beta peak power values for this session
                rank_by_power = stn_group_ses_df.beta_peak_power.rank(ascending=False) # highest rank = 1.0
                stn_group_ses_df_copy = stn_group_ses_df.copy()
                stn_group_ses_df_copy["rank_beta_power"] = rank_by_power

                # add to big dataframe
                rank_beta_power_dataframe = pd.concat([rank_beta_power_dataframe, stn_group_ses_df_copy])
    

    # save dataframe as pickle
    rank_beta_power_dataframe_filepath = os.path.join(results_path, "fooof_rank_beta_power_dataframe.pickle")
    with open(rank_beta_power_dataframe_filepath, "wb") as file:
        pickle.dump(rank_beta_power_dataframe, file)
    
    print("file: ", 
          "fooof_rank_beta_power_dataframe.pickle",
          "\nwritten in: ", results_path
          )
    

    return rank_beta_power_dataframe


def fooof_rank1_baseline_beta_peak(
        session_baseline:str,
):

    """

    Input:
        - session_baseline = str, e.g. "postop", "fu3m"

    Load the file "fooof_rank_beta_power_dataframe.pickle"
    written by fooof_rank_beta_peak_power()

    1) for each stn-session-channel_group: 
        - select the channel with highest beta peak in {session_baseline}
        - extract the channel name, peak power and center frequency
    
    2) for every following fu session:
        - select only the channel from {session_baseline} with the highest peak 
        - normalize the beta peak power and peak center frequency accordingly: 
            peak_power_fu / peak_power_session_baseline
            peak_cf_fu / peak_cf_session_baseline
        
    3) add normalized parameters to a new column and concatenate all DF rows together to one

    """

    # load the dataframe with ranks of beta peak power
    rank_beta_power_dataframe = loadResults.load_fooof_rank_beta_peak_power()

    stn_unique = list(rank_beta_power_dataframe.subject_hemisphere.unique())
    sessions_after_postop = ["fu3m", "fu12m", "fu18m"]
    sessions_after_fu3m = ["fu12m", "fu18m"]
    channel_groups = ["ring", "segm_inter", "segm_intra"]

    normalized_peak_to_baseline_session = pd.DataFrame()

    for stn in stn_unique:

        stn_dataframe = rank_beta_power_dataframe.loc[rank_beta_power_dataframe.subject_hemisphere == stn]

        for group in channel_groups:

            if group == "ring":
                channel_list = ["01", "12", "23"]
            
            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

            # get only the channels within a channel group
            stn_group_dataframe = stn_dataframe.loc[stn_dataframe["bipolar_channel"].isin(channel_list)]

            ################### POSTOP ###################
            if session_baseline not in stn_group_dataframe.session.values:
                continue

            baseline_dataframe = stn_group_dataframe.loc[stn_group_dataframe.session == session_baseline]

            # check if there was a peak
            if 1.0 not in baseline_dataframe.rank_beta_power.values:
                continue

            # select the channel with rank == 1.0
            baseline_rank1 = baseline_dataframe.loc[baseline_dataframe.rank_beta_power == 1.0]
            baseline_rank1_channel = baseline_rank1.bipolar_channel.values[0] # channel name rank 1
            baseline_rank1_peak_power = baseline_rank1.beta_peak_power.values[0] # beta peak power rank 1
            normalized_baseline_peak_power = baseline_rank1_peak_power - baseline_rank1_peak_power # always 0

            baseline_rank1_peak_cf = baseline_rank1.beta_center_frequency.values[0] # beta peak cf rank 1
            normalized_baseline_peak_cf = baseline_rank1_peak_cf - baseline_rank1_peak_cf # always 0

            # new column: normalized peak power
            baseline_rank1_copy = baseline_rank1.copy()
            baseline_rank1_copy[f"peak_power_rel_to_{session_baseline}"] = normalized_baseline_peak_power

            # new column: normalized peak cf
            baseline_rank1_copy[f"peak_cf_rel_to_{session_baseline}"] = normalized_baseline_peak_cf

            # save to collected DF
            normalized_peak_to_baseline_session = pd.concat([normalized_peak_to_baseline_session, baseline_rank1_copy])

            ################### FOLLOW UP SESSIONS ###################
            # check for which sessions apart from postop exist and get the rows for the same channel at different sessions
            if session_baseline == "postop":
                sessions_after_baseline = sessions_after_postop
            
            elif session_baseline == "fu3m":
                sessions_after_baseline = sessions_after_fu3m

            for fu_ses in sessions_after_baseline:

                # check if ses exists
                if fu_ses not in stn_group_dataframe.session.values:
                    continue

                fu_dataframe = stn_group_dataframe.loc[stn_group_dataframe.session == fu_ses]

                # select the rank 1 channel from postop
                channel_selection = fu_dataframe.loc[fu_dataframe.bipolar_channel == baseline_rank1_channel]
                fu_peak_power = channel_selection.beta_peak_power.values[0]
                fu_peak_cf = channel_selection.beta_center_frequency.values[0]

                # normalize by peak power from postop
                normalized_peak_power = fu_peak_power - baseline_rank1_peak_power

                # normalize by peak cf from postop
                normalized_peak_cf = fu_peak_cf - baseline_rank1_peak_cf

                # new column: normalized peak power -> NaN value, if no peak 
                channel_selection_copy = channel_selection.copy()
                channel_selection_copy[f"peak_power_rel_to_{session_baseline}"] = normalized_peak_power
                channel_selection_copy[f"peak_cf_rel_to_{session_baseline}"] = normalized_peak_cf

                # save to collected DF
                normalized_peak_to_baseline_session = pd.concat([normalized_peak_to_baseline_session, channel_selection_copy])
    
    # replace session names by integers
    normalized_peak_to_baseline_session["session"] = normalized_peak_to_baseline_session["session"].replace({"postop":0, "fu3m":3, "fu12m":12, "fu18m":18})

    return normalized_peak_to_baseline_session


def fooof_plot_highest_beta_peak_normalized_to_baseline(
        session_baseline:str,
        peak_parameter:str,
        normalized_to_session_baseline:str
):
    """
    Input:
        - session_baseline = str, "postop" or "fu3m"
        - peak_parameter = str, "power" or "center_frequency"
        - normalized_to_session_baseline = str "yes" or "no"

    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")


    # load the dataframe with normalized peak values
    normalized_peak_to_baseline_session = fooof_rank1_baseline_beta_peak(session_baseline=session_baseline)

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    results_df = pd.DataFrame()

    # plot for each channel group seperately
    for group in channel_groups:

        if group == "ring":
            channel_list = ["01", "12", "23"]
        
        elif group == "segm_inter":
            channel_list = ["1A2A", "1B2B", "1C2C"]

        elif group == "segm_intra":
            channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]

        # get only the channels within a channel group
        group_dataframe = normalized_peak_to_baseline_session.loc[normalized_peak_to_baseline_session["bipolar_channel"].isin(channel_list)]

        ###################### choose the right column to plot  ######################
        if peak_parameter == "power":
            
            if normalized_to_session_baseline == "yes":
                y_parameter = f"peak_power_rel_to_{session_baseline}"
                y_label = f"peak power \nrelative to peak power at {session_baseline}"
            
            elif normalized_to_session_baseline == "no":
                y_parameter = "beta_peak_power"
                y_label = f"peak power"

            
        elif peak_parameter == "center_frequency":

            if normalized_to_session_baseline == "yes":
                y_parameter = f"peak_cf_rel_to_{session_baseline}"
                y_label = f"peak center frequency \nrelative to center frequency at {session_baseline}"
            
            if normalized_to_session_baseline == "no":
                y_parameter = "beta_center_frequency"
                y_label = "peak center frequency"
         
        ###################### plot violinplot and scatter ######################
        fig = plt.figure()
        ax = fig.add_subplot()

        sns.violinplot(data=group_dataframe, x="session", y=y_parameter, ax=ax, palette="pastel")

        sns.stripplot(
                data=group_dataframe,
                x="session",
                y=y_parameter,
                size=6,
                alpha=0.4,
                hue="subject_hemisphere",
                palette="mako",
                ax=ax
            )


        # statistical test: doesn't work if groups have different sample size
        if session_baseline == "postop":
            pairs = list(combinations([0, 3, 12, 18], 2))
            num_sessions = [0, 3, 12, 18]
        
        elif session_baseline == "fu3m":
            pairs = list(combinations([3, 12, 18], 2))
            num_sessions = [3, 12, 18]

        # pairs = list(combinations(num_sessions, 2))

        annotator = Annotator(ax, pairs, data=group_dataframe, x='session', y=y_parameter)
        annotator.configure(test='Mann-Whitney', text_format='star') # or t-test_ind ??
        annotator.apply_and_annotate()

        plt.title(f"BSSu channels in {group} group with highest peak power in beta band (13-35 Hz) \nduring {session_baseline} recording")
        plt.ylabel(y_label)
        plt.xlabel("session")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()
        fig.savefig(figures_path + f"\\MINUS_fooof_highest_beta_peak_{group}_from_{session_baseline}_{peak_parameter}_normalized_{normalized_to_session_baseline}.png", bbox_inches="tight")

        print("figure: ", 
            f"MINUS_fooof_highest_beta_peak_{group}_from_{session_baseline}_{peak_parameter}_normalized_{normalized_to_session_baseline}.png",
            "\nwritten in: ", figures_path
            )


        ##################### DESCRIPTION OF EACH SESSION GROUP #####################
        # describe each group
        group_description = {}

        for ses in num_sessions:

            session_group = group_dataframe.loc[group_dataframe.session==ses]
            session_group = np.array(session_group[y_parameter].values)

            description = scipy.stats.describe(session_group)

            group_description[f"{ses}"] = description


        description_results = pd.DataFrame(group_description)
        description_results.rename(index={0: "number_observations", 1: "min_and_max", 2: "mean", 3: "variance", 4: "skewness", 5: "kurtosis"}, inplace=True)
        description_results = description_results.transpose()
        description_results_copy = description_results.copy()
        description_results_copy["channel_group"] = group

        results_df = pd.concat([results_df, description_results_copy])

    
    return results_df


def fooof_count_rank1_or_2(
        session_baseline:str,
):
    """
    Input: 
        - session_baseline=str, "postop" or "fu3m"
    """

    results_path = findfolders.get_local_path(folder="GroupResults")

    normalized_peak_to_baseline = fooof_rank1_baseline_beta_peak(session_baseline=session_baseline)


    if session_baseline == "postop":
        session = [0, 3, 12, 18]

    elif session_baseline == "fu3m":
        session = [3, 12, 18]

    channel_groups = ["ring", "segm_inter", "segm_intra"]

    count_dict = {}

    for ses in session:

        session_dataframe = normalized_peak_to_baseline.loc[normalized_peak_to_baseline.session == ses]

        for group in channel_groups:

            if group == "ring":
                channel_list = ["01", "12", "23"]
            
            elif group == "segm_inter":
                channel_list = ["1A2A", "1B2B", "1C2C"]

            elif group == "segm_intra":
                channel_list = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"]
            
            # get only the channels within a channel group
            group_dataframe = session_dataframe.loc[session_dataframe["bipolar_channel"].isin(channel_list)]

            # check if there are rows in the dataframe
            if group_dataframe["rank_beta_power"].count() == 0:
                total_count = 0
            
            else:
                # get percentage how often the rank 1 postop channels stays rank 1 in 3, 12 and 18 MFU
                total_count = group_dataframe["rank_beta_power"].count() # number of channels from one session and within a channel group 

            # number how often rank 1.0 or rank 2.0
            if 1.0 not in group_dataframe.rank_beta_power.values:
                rank_1_count = 0
            
            else:
                rank_1_count = group_dataframe["rank_beta_power"].value_counts()[1.0]

            if 2.0 not in group_dataframe.rank_beta_power.values:
                rank_2_count = 0

            else:
                rank_2_count = group_dataframe["rank_beta_power"].value_counts()[2.0]

            # percentage how often channel stays rank 1 or 2
            rank_1_percentage = rank_1_count / total_count
            rank_1_or_2_percentage = (rank_1_count + rank_2_count) / total_count # segm_inter und ring groups only have 3 ranks so not very much info...


            # save in dict
            count_dict[f"{ses}_{group}"] = [ses, group, total_count, rank_1_count, rank_2_count, rank_1_percentage, rank_1_or_2_percentage]


    count_dataframe = pd.DataFrame(count_dict)
    count_dataframe.rename(index={
        0:"session",
        1:"channel_group",
        2:"total_count_of_channels",
        3:"count_rank_1",
        4:"count_rank_2",
        5:"percentage_rank_1",
        6:"percentage_rank_1_or_2",
    }, inplace=True)
    count_dataframe = count_dataframe.transpose()

    return count_dataframe













































            









