""" Power Spectra within Channels Plot """

######### PUBLIC PACKAGES #########
import numpy as np
import os
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import scipy
from scipy import stats


######### PRIVATE PACKAGES #########
from ..classes import mainAnalysis_class as mainAnalysis_class
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults


def PowerSpectra_perChannel(sub: str, 
                            hemisphere: str, 
                            incl_session: list, 
                            signalFilter: str, 
                            normalization: list,
                            ):

    """
    Plots all Power Spectra per Channel of multiple timepoints from a single subject hemisphere.

    Input:
        - sub: str e.g. "029"
        - hemisphere: str e.g. "Right"
        - incl_session: list e.g. ["postop", "fu3m", "fu12m", "fu18m"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - feature: list of features you want to extract from the json file
            "PowerSpectrum": 
                ["frequency", "time_sectors", 
                "rawPsd", "SEM_rawPsd", 
                "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]

    """

    # path to subject folder figures
    subject_figures_path = find_folders.get_local_path(folder="figures", sub=sub)

    # Lists for for-looping
    Ring = ['BIP_03', 'BIP_13', 'BIP_02', 'BIP_12', 'BIP_01', 'BIP_23']
    SegmIntra = ['BIP_1A1B', 'BIP_1B1C', 'BIP_1A1C', 'BIP_2A2B', 'BIP_2B2C', 'BIP_2A2C']
    SegmInter = ['BIP_1A2A', 'BIP_1B2B', 'BIP_1C2C']

    groupChannels = ["Ring", "SegmIntra", "SegmInter"]


    ######### load the PSD data from the classes #########
    # define variable feature depending on normalization
    feature = ["frequency"]

    if "rawPsd" in normalization:
        feature.append("rawPsd")
        feature.append("SEM_rawPsd")

    if "normPsdToTotalSum" in normalization:
        feature.append("normPsdToTotalSum") 
        feature.append("SEM_normPsdToTotalSum")
    
    if "normPsdToSum1_100Hz" in normalization:
        feature.append("normPsdToSumPsd1to100Hz") 
        feature.append("SEM_normPsdToSumPsd1to100Hz")
    
    if "normPsdToSum40_90Hz" in normalization:
        feature.append("normPsdToSum40to90Hz")
        feature.append("SEM_normPsdToSum40to90Hz")
    

    # load the PSD data 
    data = mainAnalysis_class.MainClass(
        sub=sub,
        hemisphere=hemisphere,
        filter=signalFilter,
        result="PowerSpectrum",
        incl_session=incl_session,
        pickChannels=['03', '13', '02', '12', '01', '23', 
                        '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                        '1A2A', '1B2B', '1C2C'],
        normalization=normalization,
        freqBands=None,
        feature=feature
    )

    # 5 colors used for the cycle of matplotlib 
    cycler_colors = cycler("color", ["black", "blue", "lime", "red", "yellow"])
    plt.rc('axes', prop_cycle=cycler_colors)

    # loop through variants of absolute or relative PSD
    for norm in normalization:

        # one figure for each normalization variant
        fig = plt.figure(figsize= (50, 20), layout="tight")

        # loop through Ring, SegmIntra, SegmInter each will create a row (rows n=3)
        for group in groupChannels:

            # eval(group)  takes from the string "Ring" the variable Ring, so we can loop through the list Ring 
            for c, chan in enumerate(eval(group)):

                # subplot layout: one row for each group, max. 6 columns, if 6 BIP channels, index from left to right 1-16
                if group == "Ring":
                    plt.subplot(3, 6, c+1, label=chan) # indeces 1-6 (first row)
                
                elif group == "SegmIntra":
                    plt.subplot(3, 6, c+7, label=chan,) # indeces 7-12 (second row)
                
                elif group == "SegmInter":
                    plt.subplot(3, 6, c+13, label=chan) # indices 13-15 (third row) - only 3 Channels in SegmInter

                else: 
                    print("group must be Ring, SegmIntra or SegmInter")

                # get the single array from one line like this
                # data.postop.BIP_03.rawPSD.data
                for ses in incl_session:
                    PowerSpectrum = getattr(data, ses)
                    PowerSpectrum = getattr(PowerSpectrum, chan)

                    f = PowerSpectrum.frequency.data
                    psd = {}
                    sem = {}

                    # psd and sem must be an array, it won´t work as a list
                    if norm == "rawPsd":
                        psd[norm] = np.array(PowerSpectrum.rawPsd.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_rawPsd.data)
                
                    elif norm == "normPsdToTotalSum":
                        psd[norm] = np.array(PowerSpectrum.normPsdToTotalSum.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToTotalSum.data)
                    
                    elif norm == "normPsdToSum1_100Hz":
                        psd[norm] = np.array(PowerSpectrum.normPsdToSumPsd1to100Hz.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToSumPsd1to100Hz.data)

                    elif norm == "normPsdToSum40_90Hz":
                        psd[norm] = np.array(PowerSpectrum.normPsdToSum40to90Hz.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToSum40to90Hz.data)
                    
                    # plot each Power spectrum for each channel seperately, different lines and colors for each session
                    plt.title(f"{chan}", fontdict={"size": 50})
                    
                    plt.plot(f, psd[norm], label = f"{ses}")
                    plt.fill_between(f, psd[norm]-sem[norm], psd[norm]+sem[norm], color="lightgray", alpha=0.5)

                    # add lines for freq Bands
                    plt.axvline(x=8, color='black', linestyle='--')
                    plt.axvline(x=13, color='black', linestyle='--')
                    plt.axvline(x=20, color='black', linestyle='--')
                    plt.axvline(x=35, color='black', linestyle='--')

                    plt.xlabel("Frequency [Hz]", fontdict={"size": 30})
                    plt.xlim(2, 50)

                    # different y limit depending on absolute or relative PSD
                    if norm == "rawPsd":
                        plt.ylim(0, 3) 
                
                    elif norm == "normPsdToTotalSum":
                        plt.ylim(0, 17)
                    
                    elif norm == "normPsdToSum1_100Hz":
                        plt.ylim(0, 17)

                    elif norm == "normPsdToSum40_90Hz":
                        plt.ylim(0, 150)

                    # different y label depending on absolute or relative PSD
                    y_label = {}
                    if norm == "rawPsd":
                        y_label[norm] = "absolute PSD [uV^2/Hz+-SEM]"
                
                    elif norm == "normPsdToTotalSum":
                        y_label[norm] = "rel. PSD to total sum[%]+-SEM"
                    
                    elif norm == "normPsdToSum1_100Hz":
                        y_label[norm] = "PSD to sum 1-100 Hz[%]+-SEM"

                    elif norm == "normPsdToSum40_90Hz":
                        y_label[norm] = "PSD to sum 40-90 Hz[%]+-SEM"


                    plt.ylabel(y_label[norm], fontdict={"size": 30})
                    plt.xticks(fontsize= 20), plt.yticks(fontsize= 20)
                    plt.legend(loc= 'upper right', edgecolor="black", fontsize=20)


        # Title of each plot per normalization variant
        fig.suptitle(f"Power Spectra sub-{sub}, {hemisphere} hemisphere, {signalFilter}, {norm}", fontsize=55, y=1.02)

        # adjust Layout
        fig.subplots_adjust(wspace=40, hspace=60)
        # plt.tight_layout(pad=10, w_pad=10, h_pad=10)

        plt.show()
        # Save figure to subject result path
        fig.savefig(os.path.join(subject_figures_path, f"PowerSpectraPerChannel_sub{sub}_{hemisphere}_{norm}_{signalFilter}.png"), 
                    bbox_inches = "tight") # bbox_inches makes sure that the title won´t be cut off




def PowerSpectra_perChannelGroup(sub: str, 
                            hemisphere: str, 
                            incl_session: list, 
                            signalFilter: str, 
                            normalization: list,
                            ):

    """
    Plots all Power Spectra per Channel Group of multiple timepoints from a single subject hemisphere.

    Input:
        - sub: str e.g. "029"
        - hemisphere: str e.g. "Right"
        - incl_session: list e.g. ["postop", "fu3m", "fu12m", "fu18m"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - feature: list of features you want to extract from the json file
            "PowerSpectrum": 
                ["frequency", "time_sectors", 
                "rawPsd", "SEM_rawPsd", 
                "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]

    """

    # path to subject folder figures
    subject_figures_path = find_folders.get_local_path(folder="figures", sub=sub)

    # Lists for for-looping
    Ring = ['BIP_03', 'BIP_02', 'BIP_13', 'BIP_12', 'BIP_01', 'BIP_23']
    SegmIntra = ['BIP_1A1B', 'BIP_1B1C', 'BIP_1A1C', 'BIP_2A2B', 'BIP_2B2C', 'BIP_2A2C']
    SegmInter = ['BIP_1A2A', 'BIP_1B2B', 'BIP_1C2C']

    groupChannels = ["Ring", "SegmIntra", "SegmInter"]


    ######### load the PSD data from the classes #########
    # define variable feature depending on normalization
    feature = ["frequency"]

    if "rawPsd" in normalization:
        feature.append("rawPsd")
        feature.append("SEM_rawPsd")

    if "normPsdToTotalSum" in normalization:
        feature.append("normPsdToTotalSum") 
        feature.append("SEM_normPsdToTotalSum")
    
    if "normPsdToSum1_100Hz" in normalization:
        feature.append("normPsdToSumPsd1to100Hz") 
        feature.append("SEM_normPsdToSumPsd1to100Hz")
    
    if "normPsdToSum40_90Hz" in normalization:
        feature.append("normPsdToSum40to90Hz")
        feature.append("SEM_normPsdToSum40to90Hz")
    

    # load the PSD data 
    data = mainAnalysis_class.MainClass(
        sub=sub,
        hemisphere=hemisphere,
        filter=signalFilter,
        result="PowerSpectrum",
        incl_session=incl_session,
        pickChannels=['03', '02', '13', '12', '01', '23',
                        '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                        '1A2A', '1B2B', '1C2C'],
        normalization=normalization,
        freqBands=None,
        feature=feature
    )

    # 5 colors used for the cycle of matplotlib 
    cycler_colors = cycler("color", ["tab:blue", "tab:pink", "tab:brown", "tab:green", "tab:olive", "tab:cyan"])
    plt.rc('axes', prop_cycle=cycler_colors)

    # loop through variants of absolute or relative PSD
    for norm in normalization:

        # one figure for each normalization variant
        fig = plt.figure(figsize= (30, 30), layout="tight")

        for g, group in enumerate(groupChannels): # 0,1,2
      
            for s, ses in enumerate(incl_session): # 0,1,2,3
            #for g, group in enumerate(groupChannels):

                if ses == "postop":
                    plt.subplot(4,3,s+g+1, label=f"{group}_{ses}") # e.g. 0+0+1, 0+1+1, 0+2+1 = row1
                
                elif ses == "fu3m":
                    plt.subplot(4,3,s+g+3, label=f"{group}_{ses}") # e.g. 1+0+3, 1+1+3, 1+2+3 = row1
                
                elif ses == "fu12m":
                    plt.subplot(4,3,s+g+5, label=f"{group}_{ses}") # e.g. 2+0+5, 2+1+5, 2+2+5 = row1
                
                elif ses == "fu18m":
                    plt.subplot(4,3,s+g+7, label=f"{group}_{ses}") # e.g. 3+0+7, 3+1+7, 3+2+7 = row1
                
                else: 
                    print("group must be Ring, SegmIntra or SegmInter")

                # for each group, get all channels
                for c, chan in enumerate(eval(group)):
                    # get f, psd and sem from data
                    PowerSpectrum = getattr(data, ses)
                    PowerSpectrum = getattr(PowerSpectrum, chan)

                    f = PowerSpectrum.frequency.data
                    psd = {}
                    sem = {}

                    # psd and sem must be an array, it won´t work as a list
                    if norm == "rawPsd":
                        psd[norm] = np.array(PowerSpectrum.rawPsd.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_rawPsd.data)
                
                    elif norm == "normPsdToTotalSum":
                        psd[norm] = np.array(PowerSpectrum.normPsdToTotalSum.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToTotalSum.data)
                    
                    elif norm == "normPsdToSum1_100Hz":
                        psd[norm] = np.array(PowerSpectrum.normPsdToSumPsd1to100Hz.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToSumPsd1to100Hz.data)

                    elif norm == "normPsdToSum40_90Hz":
                        psd[norm] = np.array(PowerSpectrum.normPsdToSum40to90Hz.data)
                        sem[norm] = np.array(PowerSpectrum.SEM_normPsdToSum40to90Hz.data)

                    # plot each Power spectrum for each channel in the same group_ses subplot
                    plt.title(f"{group}_{ses}", fontdict={"size": 40})
                   
                    plt.plot(f, psd[norm], label = f"{chan}", linewidth= 3)
                    plt.fill_between(f, psd[norm]-sem[norm], psd[norm]+sem[norm], color="lightgray", alpha=0.5)

                    # add lines for freq Bands
                    plt.axvline(x=8, color='tab:gray', linestyle='--', linewidth= 3)
                    plt.axvline(x=13, color='tab:gray', linestyle='--', linewidth= 3)
                    plt.axvline(x=20, color='tab:gray', linestyle='--', linewidth= 3)
                    plt.axvline(x=35, color='tab:gray', linestyle='--', linewidth= 3)

                    plt.xlabel("Frequency [Hz]", fontdict={"size": 30})
                    plt.xlim(2, 50)

                    # different y limit depending on absolute or relative PSD
                    if norm == "rawPsd":
                        plt.ylim(0, 3) 
                
                    elif norm == "normPsdToTotalSum":
                        plt.ylim(0, 17)
                    
                    elif norm == "normPsdToSum1_100Hz":
                        plt.ylim(0, 17)

                    elif norm == "normPsdToSum40_90Hz":
                        plt.ylim(0, 150)


                    # different y label depending on absolute or relative PSD
                    y_label = {}
                    if norm == "rawPsd":
                        y_label[norm] = "absolute PSD [uV^2/Hz+-SEM]"
                
                    elif norm == "normPsdToTotalSum":
                        y_label[norm] = "rel. PSD to total sum[%]+-SEM"
                    
                    elif norm == "normPsdToSum1_100Hz":
                        y_label[norm] = "PSD to sum 1-100 Hz[%]+-SEM"

                    elif norm == "normPsdToSum40_90Hz":
                        y_label[norm] = "PSD to sum 40-90 Hz[%]+-SEM"


                    plt.ylabel(y_label[norm], fontdict={"size": 30})
                    plt.xticks(fontsize= 20), plt.yticks(fontsize= 20)
                    
                    # Plot the legend only for the first row "postop"
                    plt.legend(loc= 'upper right', edgecolor="black", fontsize=20)

                    # if ses == "fu3m":
                    #     plt.legend().remove()
            
                    # elif ses == "fu12m":
                    #     plt.legend().remove()
                    
                    # elif ses == "fu18m":
                    #     plt.legend().remove()




        # Title of each plot per normalization variant
        fig.suptitle(f"Power Spectra sub-{sub}, {hemisphere} hemisphere, {signalFilter}, {norm}", fontsize=55, y=1.02)

        # adjust Layout
        fig.subplots_adjust(wspace=40, hspace=60)
        # plt.tight_layout(pad=10, w_pad=10, h_pad=10)

        plt.show()

        # Save figure to subject result path
        fig.savefig(os.path.join(subject_figures_path, f"PowerSpectraPerChannelGroup_sub{sub}_{hemisphere}_{norm}_{signalFilter}.png"), 
                    bbox_inches = "tight") # bbox_inches makes sure that the title won´t be cut off



def power_spectra_grand_average_per_session(
        incl_sub: list,
        channel_group: str,
        signalFilter: str,
        normalization: str
):

    """

    Input:
        - incl_sub: list, e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - channel_group: str, e.g. "SegmInter", "SegmIntra", "Ring"
        - signalFilter: str, e.g. "band-pass" or "unfiltered"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    """

    # path to subject folder figures
    figures_path = find_folders.get_local_path(folder="GroupFigures")

    if channel_group == "SegmIntra":
        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    elif channel_group == "SegmInter":
        channels = ['1A2A', '1B2B', '1C2C']

    elif channel_group == "Ring":
        channels = ['01', '12', '23']

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    hemispheres = ["Right", "Left"]

        
    single_channels_dict = {}

    for sub in incl_sub:

        for hem in hemispheres:

            # load all sessions and selected channel data per STN
            stn_power_spectra = mainAnalysis_class.MainClass(
                    sub=sub,
                    hemisphere=hem,
                    filter=signalFilter,
                    result="PowerSpectrum",
                    incl_session=["postop", "fu3m", "fu12m", "fu18m"],
                    pickChannels=channels,
                    normalization=["rawPsd"],
                    feature=["frequency", "time_sectors", 
                            "rawPsd", "SEM_rawPsd",
                            "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                            "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                            "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]
                )
            
            for ses in sessions:
                
                # check which sessions exist
                try:
                    getattr(stn_power_spectra, ses)

                except AttributeError:
                    continue

                for chan in channels: 
                        
                    # get the power spectra and frequencies from each channel
                    chan_data = getattr(stn_power_spectra, ses)
                    chan_data = getattr(chan_data, f"BIP_{chan}")
                    
                    if normalization == "normPsdToSum40to90Hz": 
                        power_spectrum = np.array(chan_data.normPsdToSum40to90Hz.data)
                    
                    elif normalization == "rawPsd": 
                        power_spectrum = np.array(chan_data.rawPsd.data)
                    
                    elif normalization == "normPsdToTotalSum": 
                        power_spectrum = np.array(chan_data.normPsdToTotalSum.data)
                    
                    elif normalization == "normPsdToSum1_100Hz": 
                        power_spectrum = np.array(chan_data.normPsdToSumPsd1to100Hz.data)
                    
                    freqs = np.array(chan_data.frequency.data)

                    # save all channels of an STN in a dict
                    single_channels_dict[f"{sub}_{hem}_{ses}_{chan}"] = [sub, hem, ses, chan, power_spectrum, freqs]

    # Dataframe with all single channels and their power_spectra + frequencies
    single_channels_df = pd.DataFrame(single_channels_dict)
    single_channels_df.rename(index={
        0: "subject",
        1: "hemisphere",
        2: "session",
        3: "bipolar_channel",
        4: "power_spectrum",
        5: "frequencies"
    }, inplace=True)
    single_channels_df = single_channels_df.transpose()

    # join sub, hem columns together -> stn
    single_channels_df["stn"] = single_channels_df[['subject', 'hemisphere']].agg('_'.join, axis=1)
    single_channels_df.drop(columns=['subject', 'hemisphere'], inplace=True)


    ############# PLOT THE GRAND AVERAGE POWER SPECTRUM PER SESSION #############

    # Plot all power spectra in one figure, one color for each session

    # 4 colors used for the cycle of matplotlib 
    cycler_colors = cycler("color", ["turquoise", "sandybrown", "plum", "cornflowerblue"])
    plt.rc('axes', prop_cycle=cycler_colors)

    fig = plt.figure(layout="tight")

    average_spectra = {}

    for ses in sessions:

        session_df = single_channels_df.loc[single_channels_df.session==ses]

        frequencies = session_df.frequencies.values[0]

        power_spectrum_session_grand_average = np.mean(session_df.power_spectrum.values)
        standard_deviation_session = np.std(session_df.power_spectrum.values)
        sem_session = stats.sem(session_df.power_spectrum.values)

        # save and return 
        average_spectra[f"{ses}"] = [ses, frequencies, power_spectrum_session_grand_average, standard_deviation_session, sem_session,
                                     len(session_df.power_spectrum.values)]
        
        # Plot the grand average power spectrum per session
        plt.plot(frequencies, power_spectrum_session_grand_average, label=ses, linewidth=3)

        # plot the standard deviation as shaded grey area
        plt.fill_between(frequencies, 
                    power_spectrum_session_grand_average-standard_deviation_session,
                    power_spectrum_session_grand_average+standard_deviation_session,
                    color="gainsboro", alpha=0.5)


    plt.title(f"Grand average power spectra across {channel_group} channels", fontdict={"size": 18})
    plt.legend(loc= 'upper right', fontsize=14)

    # add lines for freq Bands
    plt.axvline(x=8, color='dimgrey', linestyle='--')
    plt.axvline(x=13, color='dimgrey', linestyle='--')
    plt.axvline(x=20, color='dimgrey', linestyle='--')
    plt.axvline(x=35, color='dimgrey', linestyle='--')

    plt.xlabel("Frequency [Hz]", fontdict={"size": 14})
    plt.xlim(1, 60)

    if normalization == "normPsdToSum40to90Hz":

        plt.ylabel("average PSD rel. to sum 40-90 Hz [%] +- std", fontdict={"size": 14})
        plt.ylim(-2, 140)
    
    elif normalization == "rawPsd":

        plt.ylabel("average PSD [uV^2/Hz] +- std", fontdict={"size": 14})
        plt.ylim(-0.05, 3)
    
    elif normalization == "normPsdToTotalSum":

        plt.ylabel("average PSD rel. to total sum [%] +- std", fontdict={"size": 14})
        plt.ylim(-0.05, 14)
    
    elif normalization == "normPsdToSum1_100Hz":

        plt.ylabel("average PSD rel. to 1-100 Hz [%] +- std", fontdict={"size": 14})
        plt.ylim(-0.05, 14)


    fig.tight_layout()

    fig.savefig(os.path.join(figures_path, f"grand_average_power_spectra_{channel_group}_{normalization}_{signalFilter}.png"),
                bbox_inches = "tight")

    print("figure: ", 
          f"grand_average_power_spectra_{channel_group}_{normalization}_{signalFilter}.png",
          "\nwritten in: ", figures_path
          )
    
    average_spectra_df = pd.DataFrame(average_spectra)
    average_spectra_df.rename(index={
        0: "session",
        1: "frequencies",
        2: "power_spectrum_grand_average",
        3: "standard_deviation",
        4: "sem",
        5: "sample_size"
    }, inplace=True)
    average_spectra_df = average_spectra_df.transpose()

    
    return average_spectra_df


def grand_average_FOOOF_spectra(
        std_or_sem:str,
        spectrum_to_plot:str,
        highest_beta_chans_only:str
):
    
    """
    Plot an average power spectrum of FOOOF modeled spectra per session for each channel group.

    Input: 
        - std_or_sem: "std" or "sem" plotting SEM or standard deviation
        - spectrum_to_plot: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - highest_beta_chans_only: "yes" -> only plotting average of channels with highest beta in their channel group per session and STN


        save figure: "grand_average_FOOOF_{all_or_one}_{group}_{spectrum_to_plot}_{std_or_sem}.png"
    """

    figures_path = find_folders.get_local_path(folder="GroupFigures")

    # load the fooof group result
    if highest_beta_chans_only == "yes":
        # load the dataframe with only one channel with highest beta per channel group, session and STN
        fooof_group_result = loadResults.load_fooof_highest_beta_channels(fooof_spectrum=spectrum_to_plot)

    else:
        fooof_group_result = loadResults.load_group_fooof_result()

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    channel_group = ["ring", "segm_inter", "segm_intra"]

    ############# PLOT THE GRAND AVERAGE POWER SPECTRUM PER SESSION #############

    # Plot all power spectra in one figure, one color for each session
    for group in channel_group:

        if group == "ring":
            channels = ['01', '12', '23']
        
        elif group == "segm_inter":
            channels = ["1A2A", "1B2B", "1C2C"]
        
        elif group == "segm_intra":
            channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

        group_df = fooof_group_result.loc[fooof_group_result["bipolar_channel"].isin(channels)]

        # 4 colors used for the cycle of matplotlib 
        cycler_colors = cycler("color", ["turquoise", "sandybrown", "plum", "cornflowerblue"])
        plt.rc('axes', prop_cycle=cycler_colors)

        fig = plt.figure(layout="tight")

        average_spectra = {}

        for ses in sessions:

            session_df = group_df.loc[group_df.session==ses]

            frequencies = np.arange(1, 96) # 1-95 Hz, 1 Hz resolution

            # choose what to plot:
            if spectrum_to_plot == "periodic_spectrum":
                column_name = "fooof_power_spectrum"
                y_label = f"average Power [uV^2/Hz] +- {std_or_sem}"
            
            elif spectrum_to_plot == "periodic_plus_aperiodic":
                column_name = "periodic_plus_aperiodic_power_log"
                y_label = f"average log(Power) +- {std_or_sem}"
            
            elif spectrum_to_plot == "periodic_flat":
                column_name = "fooof_periodic_flat"
                y_label = f"average Power of periodic component +- {std_or_sem}"

            # transform all arrays with 95 values into a new dataframe with 95 columns = 1 column for each position (1-95)
            df_transposed = pd.DataFrame(session_df[column_name].tolist())

            power_spectrum_session_grand_average = df_transposed.apply(lambda x: np.mean(x), axis=0) # mean along columns: for each position (1-95) -> outcome=pd.Series with 95 values

            # CALCULATE STANDARD DEVIATION AND SEM for each position 
            standard_deviation_session = df_transposed.apply(lambda x: np.std(x), axis=0)
            sem_session = df_transposed.apply(lambda x: stats.sem(x), axis=0)

            # save and return 
            average_spectra[f"{ses}_{group}"] = [ses, group, frequencies, power_spectrum_session_grand_average, standard_deviation_session, sem_session,
                                            len(session_df.fooof_power_spectrum.values)]
            
            # Plot the grand average power spectrum per session
            plt.plot(frequencies, power_spectrum_session_grand_average, label=ses, linewidth=3)

            # choose plotting sem or std
            if std_or_sem == "sem":
                shadowed = sem_session
            
            if std_or_sem == "std":
                shadowed = standard_deviation_session

            # plot the standard deviation as shaded grey area
            plt.fill_between(frequencies, 
                        power_spectrum_session_grand_average-shadowed,
                        power_spectrum_session_grand_average+shadowed,
                        color="gainsboro", alpha=0.5)

        # TITLE
        if highest_beta_chans_only == "yes":
            plt.title(f"Grand average FOOOF power spectra of \nonly highest beta channels of the {group} group", fontdict={"size": 18})
            all_or_one = "highest_beta_channels"

        else:
            plt.title(f"Grand average FOOOF power spectra across all {group} channels", fontdict={"size": 18})
            all_or_one = "all_channels"
        
        # plot settings
        plt.legend(loc= 'upper right', fontsize=14)

        # add lines for freq Bands
        plt.axvline(x=8, color='dimgrey', linestyle='--')
        plt.axvline(x=13, color='dimgrey', linestyle='--')
        plt.axvline(x=20, color='dimgrey', linestyle='--')
        plt.axvline(x=35, color='dimgrey', linestyle='--')

        plt.xlabel("Frequency [Hz]", fontdict={"size": 14})
        # plt.xlim(1, 95)

        plt.ylabel(y_label, fontdict={"size": 14})
        #plt.ylim(-0.05, 3)

        fig.tight_layout()

        fig.savefig(os.path.join(figures_path, f"grand_average_FOOOF_{all_or_one}_{group}_{spectrum_to_plot}_{std_or_sem}.png"),
                bbox_inches = "tight")

    print("figure: ", 
          f"grand_average_FOOOF_{all_or_one}_{group}_{spectrum_to_plot}_{std_or_sem}.png",
          "\nwritten in: ", figures_path
          )
    
    average_spectra_df = pd.DataFrame(average_spectra)
    average_spectra_df.rename(index={
        0: "session",
        1: "channel_group",
        2: "frequencies",
        3: "power_spectrum_grand_average",
        4: "standard_deviation",
        5: "sem",
        6: "sample_size"
    }, inplace=True)
    average_spectra_df = average_spectra_df.transpose()

    
    return average_spectra_df
    



    
def FOOOF_spectra_per_channel(
        incl_sub:str
):
    
    """
    Plot an average power spectrum of FOOOF modeled spectra per session for each channel group.

    Input: 
        - incl_sub: str e.g. "all" or "029_Right"
        save figure: "grand_average_FOOOF_{all_or_one}_{group}_{spectrum_to_plot}_{std_or_sem}.png"
    """

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    channel_group = ["ring", "segm_inter", "segm_intra"]

    ring = ['01', '12', '23', '02', '03', '13']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    # load FOOOF group result
    fooof_group_result = loadResults.load_group_fooof_result()

    if incl_sub == "all":

        stn_list = list(fooof_group_result.subject_hemisphere.unique())
    
    else:

        stn_list = [incl_sub]

    # one figure for each subject
    for stn in stn_list:

        sub_hem = stn.split("_")
        sub = sub_hem[0]

        subject_figures_path = find_folders.get_local_path(folder="figures", sub=sub)

        stn_df = fooof_group_result.loc[fooof_group_result.subject_hemisphere == stn]

        # 5 colors used for the cycle of matplotlib 
        cycler_colors = cycler("color", ["turquoise", "sandybrown", "plum", "cornflowerblue", "tab:grey", "yellowgreen"])
        plt.rc('axes', prop_cycle=cycler_colors)

        # one figure for each normalization variant
        fig = plt.figure(figsize= (50, 20), layout="tight")

        # loop through Ring, SegmIntra, SegmInter each will create a row (rows n=3)
        for group in channel_group:

            # eval(group)  takes from the string "Ring" the variable Ring, so we can loop through the list Ring 
            for c, chan in enumerate(eval(group)):

                # subplot layout: one row for each group, max. 6 columns, if 6 BIP channels, index from left to right 1-16
                if group == "ring":
                    plt.subplot(3, 6, c+1, label=chan) # indeces 1-6 (first row)
                
                elif group == "segm_intra":
                    plt.subplot(3, 6, c+7, label=chan,) # indeces 7-12 (second row)
                
                elif group == "segm_inter":
                    plt.subplot(3, 6, c+13, label=chan) # indices 13-15 (third row) - only 3 Channels in SegmInter

                else: 
                    print("group must be Ring, SegmIntra or SegmInter")
                

                for ses in sessions:

                    # check if session exists
                    if ses not in stn_df.session.values:
                        continue

                    # get the correct row of the stn dataframe
                    chan_ses_df = stn_df.loc[stn_df.bipolar_channel == chan]
                    chan_ses_df = chan_ses_df.loc[chan_ses_df.session == ses]

                    power_spectrum = chan_ses_df.fooof_power_spectrum.values[0]
                    frequencies = np.arange(1, 96) # 1-95 Hz, 1 Hz resolution

                    # plot each Power spectrum for each channel seperately, different lines and colors for each session
                    plt.title(f"bipolar channel: {chan}", fontdict={"size": 50})
                    
                    plt.plot(frequencies, power_spectrum, label = f"{ses}", linewidth=4)

                    # add lines for freq Bands
                    # plt.axvline(x=8, color='black', linestyle='--')
                    # plt.axvline(x=13, color='black', linestyle='--')
                    # plt.axvline(x=20, color='black', linestyle='--')
                    # plt.axvline(x=35, color='black', linestyle='--')
                    x1 = 13
                    x2 = 35
                    plt.axvspan(x1, x2, color="whitesmoke")

                    plt.xlabel("Frequency [Hz]", fontdict={"size": 30})
                    plt.xlim(2, 50)
                    plt.ylabel("Power [µV°2/Hz]", fontdict={"size": 30})
                    plt.ylim(-0.02, 3.5)
                    plt.xticks(fontsize= 20), plt.yticks(fontsize= 20)
                    plt.legend(loc= 'upper right', edgecolor="black", fontsize=20)
                    plt.grid(False)

        # Title of each plot per normalization variant
        fig.suptitle(f"FOOOF Power Spectra sub-{stn} hemisphere", fontsize=55, y=1.02)

        # adjust Layout
        fig.subplots_adjust(wspace=40, hspace=60)
        # plt.tight_layout(pad=10, w_pad=10, h_pad=10)

        fig.savefig(os.path.join(subject_figures_path, f"sub_{stn}_fooof_power_spectra_per_channel.png"), 
                        bbox_inches = "tight")
        fig.savefig(os.path.join(subject_figures_path, f"sub_{stn}_fooof_power_spectra_per_channel.svg"), 
                        bbox_inches = "tight", format="svg")



            





