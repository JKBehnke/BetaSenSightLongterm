""" BSSu PSD Fast Fourier """


import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import numpy as np

import scipy
from scipy.signal import spectrogram, hann, butter, filtfilt, freqz

import sklearn
from sklearn.preprocessing import normalize

import json
import os
import mne

# PyPerceive Imports
import PerceiveImport.classes.main_class as mainclass
import PerceiveImport.methods.find_folders as findfolders




# mapping = dictionary of all possible channel names as keys, new channel names (Retune standard) as values
mapping = {
    'LFP_Stn_0_3_RIGHT_RING':"LFP_R_03_STN_MT",
    'LFP_Stn_1_3_RIGHT_RING':"LFP_R_13_STN_MT",
    'LFP_Stn_0_2_RIGHT_RING':"LFP_R_02_STN_MT",
    'LFP_Stn_1_2_RIGHT_RING':"LFP_R_12_STN_MT",
    'LFP_Stn_0_1_RIGHT_RING':"LFP_R_01_STN_MT",
    'LFP_Stn_2_3_RIGHT_RING':"LFP_R_23_STN_MT",
    'LFP_Stn_0_3_LEFT_RING':"LFP_L_03_STN_MT",
    'LFP_Stn_1_3_LEFT_RING':"LFP_L_13_STN_MT",
    'LFP_Stn_0_2_LEFT_RING':"LFP_L_02_STN_MT",
    'LFP_Stn_1_2_LEFT_RING':"LFP_L_12_STN_MT",
    'LFP_Stn_0_1_LEFT_RING':"LFP_L_01_STN_MT",
    'LFP_Stn_2_3_LEFT_RING':"LFP_L_23_STN_MT",
    'LFP_Stn_1_A_1_B_RIGHT_SEGMENT':"LFP_R_1A1B_STN_MT",
    'LFP_Stn_1_B_1_C_RIGHT_SEGMENT':"LFP_R_1B1C_STN_MT",
    'LFP_Stn_1_A_1_C_RIGHT_SEGMENT':"LFP_R_1A1C_STN_MT",
    'LFP_Stn_2_A_2_B_RIGHT_SEGMENT':"LFP_R_2A2B_STN_MT",
    'LFP_Stn_2_B_2_C_RIGHT_SEGMENT':"LFP_R_2B2C_STN_MT",
    'LFP_Stn_2_A_2_C_RIGHT_SEGMENT':"LFP_R_2A2C_STN_MT",
    'LFP_Stn_1_A_1_B_LEFT_SEGMENT':"LFP_L_1A1B_STN_MT",
    'LFP_Stn_1_B_1_C_LEFT_SEGMENT':"LFP_L_1B1C_STN_MT",
    'LFP_Stn_1_A_1_C_LEFT_SEGMENT':"LFP_L_1A1C_STN_MT",
    'LFP_Stn_2_A_2_B_LEFT_SEGMENT':"LFP_L_2A2B_STN_MT",
    'LFP_Stn_2_B_2_C_LEFT_SEGMENT':"LFP_L_2B2C_STN_MT",
    'LFP_Stn_2_A_2_C_LEFT_SEGMENT':"LFP_L_2A2C_STN_MT",
    'LFP_Stn_1_A_2_A_RIGHT_SEGMENT':"LFP_R_1A2A_STN_MT",
    'LFP_Stn_1_B_2_B_RIGHT_SEGMENT':"LFP_R_1B2B_STN_MT",
    'LFP_Stn_1_C_2_C_RIGHT_SEGMENT':"LFP_R_1C2C_STN_MT",
    'LFP_Stn_1_A_2_A_LEFT_SEGMENT':"LFP_L_1A2A_STN_MT",
    'LFP_Stn_1_B_2_B_LEFT_SEGMENT':"LFP_L_1B2B_STN_MT",
    'LFP_Stn_1_C_2_C_LEFT_SEGMENT':"LFP_L_1C2C_STN_MT",
    "LFP_Stn_R_03":"LFP_R_03_STN_MT",
    "LFP_Stn_R_13":"LFP_R_13_STN_MT",
    "LFP_Stn_R_02":"LFP_R_02_STN_MT",
    "LFP_Stn_R_12":"LFP_R_12_STN_MT",
    "LFP_Stn_R_01":"LFP_R_01_STN_MT",
    "LFP_Stn_R_23":"LFP_R_23_STN_MT",
    "LFP_Stn_L_03":"LFP_L_03_STN_MT",
    "LFP_Stn_L_13":"LFP_L_13_STN_MT",
    "LFP_Stn_L_02":"LFP_L_02_STN_MT",
    "LFP_Stn_L_12":"LFP_L_12_STN_MT",
    "LFP_Stn_L_01":"LFP_L_01_STN_MT",
    "LFP_Stn_L_23":"LFP_L_23_STN_MT",
    'LFP_Stn_R_1A1B':"LFP_R_1A1B_STN_MT",
    'LFP_Stn_R_1B1C':"LFP_R_1B1C_STN_MT",
    'LFP_Stn_R_1A1C':"LFP_R_1A1C_STN_MT",
    'LFP_Stn_R_2A2B':"LFP_R_2A2B_STN_MT",
    'LFP_Stn_R_2B2C':"LFP_R_2B2C_STN_MT",
    'LFP_Stn_R_2A2C':"LFP_R_2A2C_STN_MT",
    'LFP_Stn_L_1A1B':"LFP_L_1A1B_STN_MT",
    'LFP_Stn_L_1B1C':"LFP_L_1B1C_STN_MT",
    'LFP_Stn_L_1A1C':"LFP_L_1A1C_STN_MT",
    'LFP_Stn_L_2A2B':"LFP_L_2A2B_STN_MT",
    'LFP_Stn_L_2B2C':"LFP_L_2B2C_STN_MT",
    'LFP_Stn_L_2A2C':"LFP_L_2A2C_STN_MT",
    'LFP_Stn_R_1A2A':"LFP_R_1A2A_STN_MT", 
    'LFP_Stn_R_1B2B':"LFP_R_1B2B_STN_MT", 
    'LFP_Stn_R_1C2C':"LFP_R_1C2C_STN_MT",
    'LFP_Stn_L_1A2A':"LFP_L_1A2A_STN_MT", 
    'LFP_Stn_L_1B2B':"LFP_L_1B2B_STN_MT", 
    'LFP_Stn_L_1C2C':"LFP_L_1C2C_STN_MT",
    
}



def spectrogram_Psd(incl_sub: str, incl_session: list, incl_condition: list, incl_contact: list, pickChannels: list, hemisphere: str, normalization: str):
    """

    Input: 
        - incl_sub: str e.g. "024"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - incl_contact: a list of contacts to include ["RingR", "SegmIntraR", "SegmInterR", "RingL", "SegmIntraL", "SegmInterL"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"

    
    1) load data from mainclass.PerceiveData using the input values.

    2) Rename and pick channels
    
    2) band-pass filter by a Butterworth Filter of fifth order (5-95 Hz).
    
    3) Calculate the raw psd values of every channel for each timepoint by using scipy.sinal.scpectrogram.
        - Compute a spectrogram with consecutive Fourier transforms.
        - hanning window (scipy.signal.hann):
            - sampling frequency: 250 Hz
            - window samples: 250
            - sym=False
            - noverlap: 0.5 (50% overlap of windows)

        output variables:
        - f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
        - time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (21 time sectors)
        - Sxx = 126 frequency rows (arrays with 21 PSD values [µV^2/Hz] of each time sector, 21 time sector columns
    
    4) Normalization variants: calculate different normalized PSD values 
        - normalized to total sum of PSD from each power spectrum
        - normalized to sum of PSD from 1-100 Hz
        - normalized to sum of PSD from 40-90 Hz


    Depending on normalization variation: 
    
    5) For each frequency band alpha (8-12 Hz), low beta (13-20 Hz), high beta (21-35 Hz), beta (13-35 Hz), gamma (40-90 Hz) the highest Peak values (frequency and psd) will be seleted and saved in a DataFrame.

    6) The raw or noramlized PSD values will be plotted and the figure will be saved as:
        f"\sub{incl_sub}_{hemisphere}_normalizedPsdToTotalSum_seperateTimepoints_{pickChannels}.png"
    
    7) All frequencies and relative psd values, as well as the values for the highest PEAK in each frequency band will be returned as a Dataframe in a dictionary: 
    
    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "normPsdToTotalSumDataFrame":normToTotalSumPsdDataFrame,
        "normPsdToSum1_100Hz": normToSum1_100Hz,
        "normPsdToSum40_90Hz":normToSum40_90Hz,
        "psdAverage_dict": psdAverage_dict,
        "highestPeakRawPSD": highestPeakRawPsdDF,
    }

    """

    # sns.set()
    # plt.style.use('seaborn-whitegrid')  

    mainclass_sub = mainclass.PerceiveData(
        sub = incl_sub, 
        incl_modalities= ["survey"],
        incl_session = incl_session,
        incl_condition = incl_condition,
        incl_task = ["rest"],
        incl_contact=incl_contact
        )

    
    figures_path = findfolders.get_local_path(folder="figures", sub=incl_sub)
    results_path = findfolders.get_local_path(folder="results", sub=incl_sub)

    # add error correction for sub and task??
    
    f_rawPsd_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject
    f_normPsdToTotalSum_dict = {}
    f_normPsdToSum1to100Hz_dict = {}
    f_normPsdToSum40to90Hz_dict = {}
    psdAverage_dict = {}
    highest_peak_dict = {}

    # set layout for figures: using the object-oriented interface
    fig, axes = plt.subplots(len(incl_session), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)
    

    # Create a list of 15 colors and add it to the cycle of matplotlib 
    cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
    plt.rc('axes', prop_cycle=cycler_colors)


    for t, tp in enumerate(incl_session):
        # t is indexing time_points, tp are the time_points

        for c, cond in enumerate(incl_condition):

            for cont, contact in enumerate(incl_contact): 
                # tk is indexing task, task is the input task

                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(mainclass_sub.survey, tp) is None:
                    continue

            
                # apply loop over channels
                temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                
                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(temp_data, cond) is None:
                    continue
            
                # try:
                #     temp_data = getattr(temp_data, cond)
                #     temp_data = temp_data.rest.data[tasks[tk]]
                
                # except AttributeError:
                #     continue

                temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                temp_data = getattr(temp_data.rest, contact)
                temp_data = temp_data.data[incl_contact[cont]] # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
    

                #################### CREATE A BUTTERWORTH FILTER #################### 

                # set filter parameters for band-pass filter
                filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
                frequency_cutoff_low = 5 # 5Hz high-pass filter
                frequency_cutoff_high = 95 # 95 Hz low-pass filter
                fs = temp_data.info['sfreq'] # sample frequency: 250 Hz

                # create the filter
                b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
 

                #################### RENAME CHANNELS ####################
                # all channel names of one loaded file (one session, one task)
                ch_names_original = temp_data.info.ch_names

                # select only relevant keys and values from the mapping dictionary to rename channels
                mappingSelected = dict((key, mapping[key]) for key in ch_names_original if key in mapping)

                # rename channels using mne and the new selected mapping dictionary
                mne.rename_channels(info=temp_data.info, mapping=mappingSelected, allow_duplicates=False)

                # get new channel names
                ch_names_renamed = temp_data.info.ch_names


                #################### PICK CHANNELS ####################
                include_channelList = [] # this will be a list with all channel names selected
                exclude_channelList = []

                for n, names in enumerate(ch_names_renamed):
                    
                    # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
                    for picked in pickChannels:
                        if picked in names:
                            include_channelList.append(names)


                    # exclude all bipolar 0-3 channels, because they do not give much information
                    # if "03" in names:
                    #     exclude_channelList.append(names)
                    
                # Error Checking: 
                if len(include_channelList) == 0:
                    continue

                # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
                ch_names_indices = mne.pick_channels(ch_names_renamed, include=include_channelList)

                
                for i, ch in enumerate(ch_names_renamed):
                    
                    # only get picked channels
                    if i not in ch_names_indices:
                        continue

                    #################### FILTER ####################

                    # filter the signal by using the above defined butterworth filter
                    filtered = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 

                    #################### PERFORM FOURIER TRANSFORMATION AND CALCULATE POWER SPECTRAL DENSITY ####################

                    window = 250 # with sfreq 250 frequencies will be from 0 to 125 Hz, 125Hz = Nyquist = fs/2
                    noverlap = 0.5 # 50% overlap of windows

                    window = hann(window, sym=False)

                    # compute spectrogram with Fourier Transforms
                    
                    f,time_sectors,Sxx = scipy.signal.spectrogram(x=filtered, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
                    # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
                    # time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (in total 21 time sectors)
                    # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
                    # Sxx = 126 frequency rows, 21 time sector columns

                    # average all 21 Power spectra of all time sectors 
                    average_Sxx = np.mean(Sxx, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                                   

                    #################### CALCULATE THE STANDARD ERROR OF MEAN ####################
                    # SEM = standard deviation / square root of sample size
                    Sxx_std = np.std(Sxx, axis=1) # standard deviation of each frequency row
                    semRawPsd = Sxx_std / np.sqrt(Sxx.shape[1]) # sample size = 21 time vectors -> sem with 126 values

                    # store frequency, time vectors and psd values in a dictionary, together with session timepoint and channel
                    f_rawPsd_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, average_Sxx, semRawPsd] 
                

                    #################### NORMALIZE PSD IN MULTIPLE WAYS ####################
                    
                    #################### NORMALIZE PSD TO TOTAL SUM OF THE POWER SPECTRUM (ALL FREQUENCIES) ####################

                    normToTotalSum_psd = (average_Sxx/np.sum(average_Sxx))*100 # in percentage               

                    # calculate the SEM of psd values 
                    semNormToTotalSum_psd = (semRawPsd/np.sum(average_Sxx))*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToTotalSum_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normToTotalSum_psd, semNormToTotalSum_psd]


                    #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 1-100 Hz  ####################
                   
                    # get raw psd values from 1 to 100 Hz by indexing the numpy arrays f and px
                    rawPsd_1to100Hz = average_Sxx[1:100]

                    # sum of rawPSD between 1 and 100 Hz
                    psdSum1to100Hz = rawPsd_1to100Hz.sum()

                    # raw psd divided by sum of psd between 1 and 100 Hz
                    normPsdToSum1to100Hz = (average_Sxx/psdSum1to100Hz)*100

                    # calculate the SEM of psd values 
                    semNormPsdToSum1to100Hz = (semRawPsd/psdSum1to100Hz)*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToSum1to100Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum1to100Hz, semNormPsdToSum1to100Hz]


                    #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 40-90 Hz  ####################
                   
                    # get raw psd values from 40 to 90 Hz (gerundet) by indexing the numpy arrays f and px
                    rawPsd_40to90Hz = average_Sxx[40:90] 

                    # sum of rawPSD between 40 and 90 Hz
                    psdSum40to90Hz = rawPsd_40to90Hz.sum()

                    # raw psd divided by sum of psd between 40 and 90 Hz
                    normPsdToSum40to90Hz = (average_Sxx/psdSum40to90Hz)*100
                   
                    # calculate the SEM of psd values 
                    semNormPsdToSum40to90Hz = (semRawPsd/psdSum40to90Hz)*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToSum40to90Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum40to90Hz, semNormPsdToSum40to90Hz]


                    #################### PEAK DETECTION ####################

                    # depending on what normalization or raw was chosen: define variables for psd, sem and ylabel accordingly
                    if normalization == "rawPsd":
                        chosenPsd = average_Sxx
                        chosenSem = semRawPsd
                        chosen_ylabel = "uV^2/Hz +- SEM"
                    
                    elif normalization == "normPsdToTotalSum":
                        chosenPsd = normToTotalSum_psd
                        chosenSem = semNormToTotalSum_psd
                        chosen_ylabel = "rel. PSD to total sum (%) +- SEM"

                    elif normalization == "normPsdToSum1_100Hz":
                        chosenPsd = normPsdToSum1to100Hz
                        chosenSem = semNormPsdToSum1to100Hz
                        chosen_ylabel = "rel. PSD to sum 1-100 Hz (%) +- SEM"

                    elif normalization == "normPsdToSum40_90Hz":
                        chosenPsd = normPsdToSum40to90Hz
                        chosenSem = semNormPsdToSum40to90Hz
                        chosen_ylabel = "rel. PSD to sum 40-90 Hz (%) +- SEM"
                    
                    else:
                        chosenPsd = average_Sxx
                        chosenSem = semRawPsd
                        chosen_ylabel = "uV^2/Hz +- SEM"
                    # else statement is necessary to ensure the definition variable is not only locally 
                        

                    #################### PSD AVERAGE OF EACH FREQUENCY BAND DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                    
                    # create booleans for each frequency-range for alpha, low beta, high beta, beta and gamma
                    alpha_frequency = (f >= 8) & (f <= 12) # alpha_range will output a boolean of True values within the alpha range
                    lowBeta_frequency = (f >= 13) & (f <= 20)
                    highBeta_frequency = (f >= 21) & (f <= 35)
                    beta_frequency = (f >= 13) & (f <= 35)
                    narrowGamma_frequency = (f >= 40) & (f <= 90)

                    # make a list with all boolean masks of each frequency, so I can loop through
                    range_allFrequencies = [alpha_frequency, lowBeta_frequency, highBeta_frequency, beta_frequency, narrowGamma_frequency]

                    # loop through frequency ranges and get all psd values of each frequency band
                    for count, boolean in enumerate(range_allFrequencies):

                        frequency = []
                        if count == 0:
                            frequency = "alpha"
                        elif count == 1:
                            frequency = "lowBeta"
                        elif count == 2:
                            frequency = "highBeta"
                        elif count == 3:
                            frequency = "beta"
                        elif count == 4:
                            frequency = "narrowGamma"
                        


                        # get all frequencies and chosen psd values within each frequency range
                        # frequencyInFreqBand = f[range_allFrequencies[count]] # all frequencies within a frequency band
                        psdInFreqBand = chosenPsd[range_allFrequencies[count]] # all psd values within a frequency band

                        psdAverage = np.mean(psdInFreqBand)

                        # store averaged psd values of each frequency band in a dictionary
                        psdAverage_dict[f'{tp}_{ch}_psdAverage_{frequency}'] = [tp, ch, frequency, psdAverage]



                    #################### PEAK DETECTION PSD DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                    # find all peaks: peaks is a tuple -> peaks[0] = index of frequency?, peaks[1] = dictionary with keys("peaks_height") 
                    peaks = scipy.signal.find_peaks(chosenPsd, height=0.1) # height: peaks only above 0.1 will be recognized

                    # Error checking: if no peaks found, continue
                    if len(peaks) == 0:
                        continue

                    peaks_height = peaks[1]["peak_heights"] # np.array of y-value of peaks = power
                    peaks_pos = f[peaks[0]] # np.array of indeces on x-axis of peaks = frequency

                    # set the x-range for each frequency band
                    alpha_range = (peaks_pos >= 8) & (peaks_pos <= 12) # alpha_range will output a boolean of True values within the alpha range
                    lowBeta_range = (peaks_pos >= 13) & (peaks_pos <= 20)
                    highBeta_range = (peaks_pos >= 21) & (peaks_pos <= 35)
                    beta_range = (peaks_pos >= 13) & (peaks_pos <= 35)
                    narrowGamma_range = (peaks_pos >= 40) & (peaks_pos <= 90)

                    # make a list with all boolean masks of each frequency, so I can loop through
                    frequency_ranges = [alpha_range, lowBeta_range, highBeta_range, beta_range, narrowGamma_range]

                    # loop through frequency ranges and get the highest peak of each frequency band
                    for count, boolean in enumerate(frequency_ranges):

                        frequency = []
                        if count == 0:
                            frequency = "alpha"
                        elif count == 1:
                            frequency = "lowBeta"
                        elif count == 2:
                            frequency = "highBeta"
                        elif count == 3:
                            frequency = "beta"
                        elif count == 4:
                            frequency = "narrowGamma"
                        
                        # get all peak positions and heights within each frequency range
                        peaksinfreq_pos = peaks_pos[frequency_ranges[count]]
                        peaksinfreq_height = peaks_height[frequency_ranges[count]]

                        # Error checking: check first, if there is a peak in the frequency range
                        if len(peaksinfreq_height) == 0:
                            continue

                        # select only the highest peak within the alpha range
                        highest_peak_height = peaksinfreq_height.max()

                        ######## calculate psd average of +- 2 Hz from highest Peak ########
                        # 1) find psd values from -2Hz until + 2Hz from highest Peak by slicing and indexing the numpy array of all chosen psd values
                        peakIndex = np.where(chosenPsd == highest_peak_height) # np.where output is a tuple: index, dtype
                        peakIndexValue = peakIndex[0].item() # only take the index value of the highest Peak psd value in all chosen psd

                        # 2) go -2 and +3 indeces 
                        indexlowCutt = peakIndexValue-2
                        indexhighCutt = peakIndexValue+3   # +3 because the ending index is left out when slicing a numpy array

                        # 3) slice the numpy array of all chosen psd values, only get values from -2 until +2 Hz from highest Peak
                        psdArray5HzRangeAroundPeak = chosenPsd[indexlowCutt:indexhighCutt] # array only of psd values -2 until +2Hz around Peak = 5 values

                        # 4) Average of 5Hz Array
                        highest_peak_height_5Hzaverage = np.mean(psdArray5HzRangeAroundPeak)                       



                        # get the index of the highest peak y value to get the corresponding peak position x
                        ix = np.where(peaksinfreq_height == highest_peak_height)
                        highest_peak_pos = peaksinfreq_pos[ix].item()

                        # plot only the highest peak within each frequency band
                        axes[t].scatter(highest_peak_pos, highest_peak_height, color="k", s=15, marker='D')

                        # store highest peak values of each frequency band in a dictionary
                        highest_peak_dict[f'{tp}_{ch}_highestPEAK_{frequency}'] = [tp, ch, frequency, highest_peak_pos, highest_peak_height, highest_peak_height_5Hzaverage]




                    #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

                    # the title of each plot is set to the timepoint e.g. "postop"
                    axes[t].set_title(tp, fontsize=20) 

                    # get y-axis label and limits
                    # axes[t].get_ylabel()
                    # axes[t].get_ylim()

                    # .plot() method for creating the plot, axes[0] refers to the first plot, the plot is set on the appropriate object axes[t]
                    axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}")  # or np.log10(px) 
                    # colors of each line in different color, defined at the beginning
                    # axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}", color=colors[i])

                    # make a shadowed line of the sem
                    axes[t].fill_between(f, chosenPsd-chosenSem, chosenPsd+chosenSem, color='lightgray', alpha=0.5)



    #################### PLOT SETTINGS ####################

    plt.subplots_adjust(wspace=0, hspace=0)

    font = {"size": 16}

    for ax in axes: 
        # ax.legend(loc= 'upper right') # Legend will be in upper right corner
        ax.grid() # show grid
        ax.set(xlim=[4, 40]) # no ylim for rawPSD and normalization to sum 40-90 Hz
        # ax.set(xlim=[-5, 60] ,ylim=[0,7]) for normalizations to total sum or to sum 1-100Hz set ylim to zoom in
        ax.set_xlabel("Frequency", fontdict=font)
        ax.set_ylabel(chosen_ylabel, fontdict=font)
        ax.axvline(x=8, color='black', linestyle='--')
        ax.axvline(x=13, color='black', linestyle='--')
        ax.axvline(x=20, color='black', linestyle='--')
        ax.axvline(x=35, color='black', linestyle='--')
    
    # remove x ticks and labels from all but the bottom subplot
    for ax in axes.flat[:-1]:
        ax.set(xlabel='')
        
    #fig.legend(title="bipolar channels", loc= 'upper right', bbox_to_anchor=(0.5,1.05), fancybox=True, shadow=True, fontsize="small")

    ###### LEGEND ######
    legend = axes[0].legend(loc= 'lower right', edgecolor="black", bbox_to_anchor=(1.5, -0.1)) # only show the first subplot´s legend 
    # frame the legend with black edges amd white background color 
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()

    plt.show()
    fig.savefig(figures_path + f"\\PSDspectrogram_sub{incl_sub}_{hemisphere}_{normalization}_{pickChannels}.png")
    

    #################### WRITE DATAFRAMES TO STORE VALUES ####################
    # write raw PSD Dataframe
    rawPSDDataFrame = pd.DataFrame(f_rawPsd_dict)
    rawPSDDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "averagedPSD", 5: "SEM_rawPsd"}, inplace=True) # rename the rows
    rawPSDDataFrame = rawPSDDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to total Sum
    normPsdToTotalSumDataFrame = pd.DataFrame(f_normPsdToTotalSum_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToTotalSumDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToTotalSum", 5: "SEM_normPsdToTotalSum"}, inplace=True) # rename the rows
    normPsdToTotalSumDataFrame = normPsdToTotalSumDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum1to100HzDataFrame = pd.DataFrame(f_normPsdToSum1to100Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum1to100HzDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSumPsd1to100Hz", 5: "SEM_normPsdToSumPsd1to100Hz"}, inplace=True) # rename the rows
    normPsdToSum1to100HzDataFrame = normPsdToSum1to100HzDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum40to90DataFrame = pd.DataFrame(f_normPsdToSum40to90Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum40to90DataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSum40to90Hz", 5: "SEM_normPsdToSum40to90Hz"}, inplace=True) # rename the rows
    normPsdToSum40to90DataFrame = normPsdToSum40to90DataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(psdAverage_dict) # Dataframe with 5 rows and columns for each single power spectrum
    psdAverageDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: f"averaged{normalization}"}, inplace=True) # rename the rows
    psdAverageDF = psdAverageDF.transpose() # Dataframe with 4 columns and rows for each single power spectrum


    # write DataFrame of frequency and psd values of the highest peak in each frequency band
    highestPEAKDF = pd.DataFrame(highest_peak_dict) # Dataframe with 5 rows and columns for each single power spectrum
    highestPEAKDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: "PEAK_frequency", 4:f"PEAK_{normalization}", 5: "highest_peak_height_5HzAverage"}, inplace=True) # rename the rows
    highestPEAKDF = highestPEAKDF.transpose() # Dataframe with 6 columns and rows for each single power spectrum


    # save Dataframes as csv in the results folder
    
    rawPSDDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMrawPSD_{hemisphere}"), sep=",")
    normPsdToTotalSumDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToTotalSum_{hemisphere}"), sep=",")
    normPsdToSum1to100HzDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_1to100Hz_{hemisphere}"), sep=",")
    normPsdToSum40to90DataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_40to90Hz_{hemisphere}"), sep=",")
    psdAverageDF.to_csv(os.path.join(results_path,f"SPECTROGRAMpsdAverageFrequencyBands_{normalization}_{hemisphere}"), sep=",")
    highestPEAKDF.to_csv(os.path.join(results_path,f"SPECTROGRAM_highestPEAK_FrequencyBands_{normalization}_{hemisphere}"), sep=",")


    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "normPsdToTotalSumDataFrame":normPsdToTotalSumDataFrame,
        "normPsdToSum1to100HzDataFrame":normPsdToSum1to100HzDataFrame,
        "normPsdToSum40to90HzDataFrame":normPsdToSum40to90DataFrame,
        f"averaged{normalization}": psdAverageDF,
        f"highestPEAK{normalization}": highestPEAKDF,
    }






# #################### PERFORM FOURIER TRANSFORMATION AND CALCULATE POWER SPECTRAL DENSITY ####################

#                     window = 250 # with sfreq 250 frequencies will be from 0 to 125 Hz, 125Hz = Nyquist = fs/2
#                     noverlap = 0.5 # 50% overlap of windows

#                     window = hann(window, sym=False)

#                     # compute spectrogram with Fourier Transforms
                    
#                     f,time_sectors,Sxx = scipy.signal.spectrogram(x=filtered, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
#                     # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
#                     # time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (in total 21 time sectors)
#                     # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
#                     # Sxx = 126 frequency rows, 21 time sector columns

#                     # average all 21 Power spectra of all time sectors 
#                     average_Sxx = np.mean(Sxx, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                    
#                     # store frequency, time vectors and psd values in a dictionary, together with session timepoint and channel
#                     f_rawPsd_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, average_Sxx]                




#                     #################### CALCULATE THE CONFIDENCE INTERVAL ####################

#                     # calculate the SEM of psd values and store sem of each channel in dictionary
#                     semRawPsd = np.std(average_Sxx)/np.sqrt(len(average_Sxx)) # SEM = standard deviation / square root of sample size



def absoluteSpectrogram_Psd_noFilter(incl_sub: str, incl_session: list, incl_condition: list, incl_contact: list, pickChannels: list, hemisphere: str, normalization: str):
    """

    Input: 
        - incl_sub: str e.g. "024"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - incl_contact: a list of contacts to include ["RingR", "SegmIntraR", "SegmInterR", "RingL", "SegmIntraL", "SegmInterL"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"

    
    1) load data from mainclass.PerceiveData using the input values.

    2) Rename and pick channels
    
    2) no  filter
    
    3) Calculate the raw psd values of every channel for each timepoint by using scipy.sinal.scpectrogram.
        - Compute a spectrogram with consecutive Fourier transforms.
        - hanning window (scipy.signal.hann):
            - sampling frequency: 250 Hz
            - window samples: 250
            - sym=False
            - noverlap: 0.5 (50% overlap of windows)

        output variables:
        - f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
        - time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (21 time sectors)
        - Sxx = 126 frequency rows (arrays with 21 PSD values [µV^2/Hz] of each time sector, 21 time sector columns
    
    4) no relative values


    Depending on normalization variation: 
    
    5) For each frequency band alpha (8-12 Hz), low beta (13-20 Hz), high beta (21-35 Hz), beta (13-35 Hz), gamma (40-90 Hz) the highest Peak values (frequency and psd) will be seleted and saved in a DataFrame.

    6) The raw or noramlized PSD values will be plotted and the figure will be saved as:
        f"\sub{incl_sub}_{hemisphere}_normalizedPsdToTotalSum_seperateTimepoints_{pickChannels}.png"
    
    7) All frequencies and psd values, as well as the values for the highest PEAK in each frequency band will be returned as a Dataframe in a dictionary: 
    
    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "psdAverage_dict": psdAverage_dict,
        "highestPeakRawPSD": highestPeakRawPsdDF,
    }

    """

    # sns.set()
    # plt.style.use('seaborn-whitegrid')  

    mainclass_sub = mainclass.PerceiveData(
        sub = incl_sub, 
        incl_modalities= ["survey"],
        incl_session = incl_session,
        incl_condition = incl_condition,
        incl_task = ["rest"],
        incl_contact=incl_contact
        )

    
    figures_path = findfolders.get_local_path(folder="figures", sub=incl_sub)
    results_path = findfolders.get_local_path(folder="results", sub=incl_sub)

    # add error correction for sub and task??
    
    f_rawPsd_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject
    psdAverage_dict = {}
    highest_peak_dict = {}

    # set layout for figures: using the object-oriented interface
    fig, axes = plt.subplots(len(incl_session), 1, figsize=(10, 15)) # subplot(rows, columns, panel number), figsize(width,height)
    

    # Create a list of 15 colors and add it to the cycle of matplotlib 
    cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
    plt.rc('axes', prop_cycle=cycler_colors)


    for t, tp in enumerate(incl_session):
        # t is indexing time_points, tp are the time_points

        for c, cond in enumerate(incl_condition):

            for cont, contact in enumerate(incl_contact): 
                # tk is indexing task, task is the input task

                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(mainclass_sub.survey, tp) is None:
                    continue

            
                # apply loop over channels
                temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                
                # avoid Attribute Error, continue if attribute doesn´t exist
                if getattr(temp_data, cond) is None:
                    continue
            
                # try:
                #     temp_data = getattr(temp_data, cond)
                #     temp_data = temp_data.rest.data[tasks[tk]]
                
                # except AttributeError:
                #     continue

                temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                temp_data = getattr(temp_data.rest, contact)
                temp_data = temp_data.data[incl_contact[cont]] # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
    

                #################### CREATE A BUTTERWORTH FILTER #################### 

                # set filter parameters for band-pass filter
                filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
                frequency_cutoff_low = 5 # 5Hz high-pass filter
                frequency_cutoff_high = 95 # 95 Hz low-pass filter
                fs = temp_data.info['sfreq'] # sample frequency: 250 Hz

                # create the filter
                b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
 

                #################### RENAME CHANNELS ####################
                # all channel names of one loaded file (one session, one task)
                ch_names_original = temp_data.info.ch_names

                # select only relevant keys and values from the mapping dictionary to rename channels
                mappingSelected = dict((key, mapping[key]) for key in ch_names_original if key in mapping)

                # rename channels using mne and the new selected mapping dictionary
                mne.rename_channels(info=temp_data.info, mapping=mappingSelected, allow_duplicates=False)

                # get new channel names
                ch_names_renamed = temp_data.info.ch_names


                #################### PICK CHANNELS ####################
                include_channelList = [] # this will be a list with all channel names selected
                exclude_channelList = []

                for n, names in enumerate(ch_names_renamed):
                    
                    # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
                    for picked in pickChannels:
                        if picked in names:
                            include_channelList.append(names)


                    # exclude all bipolar 0-3 channels, because they do not give much information
                    # if "03" in names:
                    #     exclude_channelList.append(names)
                    
                # Error Checking: 
                if len(include_channelList) == 0:
                    continue

                # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
                ch_names_indices = mne.pick_channels(ch_names_renamed, include=include_channelList)

                
                for i, ch in enumerate(ch_names_renamed):
                    
                    # only get picked channels
                    if i not in ch_names_indices:
                        continue

                    #################### FILTER ####################

                    # filter the signal by using the above defined butterworth filter
                    filtered = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 

                    #################### PERFORM FOURIER TRANSFORMATION AND CALCULATE POWER SPECTRAL DENSITY ####################

                    window = 250 # with sfreq 250 frequencies will be from 0 to 125 Hz, 125Hz = Nyquist = fs/2
                    noverlap = 0.5 # 50% overlap of windows

                    window = hann(window, sym=False)

                    # compute spectrogram with Fourier Transforms
                    
                    f,time_sectors,Sxx = scipy.signal.spectrogram(x=filtered, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
                    # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
                    # time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (in total 21 time sectors)
                    # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
                    # Sxx = 126 frequency rows, 21 time sector columns

                    # average all 21 Power spectra of all time sectors 
                    average_Sxx = np.mean(Sxx, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                                   

                    #################### CALCULATE THE STANDARD ERROR OF MEAN ####################
                    # SEM = standard deviation / square root of sample size
                    Sxx_std = np.std(Sxx, axis=1) # standard deviation of each frequency row
                    semRawPsd = Sxx_std / np.sqrt(Sxx.shape[1]) # sample size = 21 time vectors -> sem with 126 values

                    # store frequency, time vectors and psd values in a dictionary, together with session timepoint and channel
                    f_rawPsd_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, average_Sxx, semRawPsd] 
                

                    #################### NORMALIZE PSD IN MULTIPLE WAYS ####################
                    
                    #################### NORMALIZE PSD TO TOTAL SUM OF THE POWER SPECTRUM (ALL FREQUENCIES) ####################

                    normToTotalSum_psd = (average_Sxx/np.sum(average_Sxx))*100 # in percentage               

                    # calculate the SEM of psd values 
                    semNormToTotalSum_psd = (semRawPsd/np.sum(average_Sxx))*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToTotalSum_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normToTotalSum_psd, semNormToTotalSum_psd]


                    #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 1-100 Hz  ####################
                   
                    # get raw psd values from 1 to 100 Hz by indexing the numpy arrays f and px
                    rawPsd_1to100Hz = average_Sxx[1:100]

                    # sum of rawPSD between 1 and 100 Hz
                    psdSum1to100Hz = rawPsd_1to100Hz.sum()

                    # raw psd divided by sum of psd between 1 and 100 Hz
                    normPsdToSum1to100Hz = (average_Sxx/psdSum1to100Hz)*100

                    # calculate the SEM of psd values 
                    semNormPsdToSum1to100Hz = (semRawPsd/psdSum1to100Hz)*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToSum1to100Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum1to100Hz, semNormPsdToSum1to100Hz]


                    #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 40-90 Hz  ####################
                   
                    # get raw psd values from 40 to 90 Hz (gerundet) by indexing the numpy arrays f and px
                    rawPsd_40to90Hz = average_Sxx[40:90] 

                    # sum of rawPSD between 40 and 90 Hz
                    psdSum40to90Hz = rawPsd_40to90Hz.sum()

                    # raw psd divided by sum of psd between 40 and 90 Hz
                    normPsdToSum40to90Hz = (average_Sxx/psdSum40to90Hz)*100
                   
                    # calculate the SEM of psd values 
                    semNormPsdToSum40to90Hz = (semRawPsd/psdSum40to90Hz)*100

                    # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                    f_normPsdToSum40to90Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum40to90Hz, semNormPsdToSum40to90Hz]


                    #################### PEAK DETECTION ####################

                    # depending on what normalization or raw was chosen: define variables for psd, sem and ylabel accordingly
                    if normalization == "rawPsd":
                        chosenPsd = average_Sxx
                        chosenSem = semRawPsd
                        chosen_ylabel = "uV^2/Hz +- SEM"
                    
                    elif normalization == "normPsdToTotalSum":
                        chosenPsd = normToTotalSum_psd
                        chosenSem = semNormToTotalSum_psd
                        chosen_ylabel = "rel. PSD to total sum (%) +- SEM"

                    elif normalization == "normPsdToSum1_100Hz":
                        chosenPsd = normPsdToSum1to100Hz
                        chosenSem = semNormPsdToSum1to100Hz
                        chosen_ylabel = "rel. PSD to sum 1-100 Hz (%) +- SEM"

                    elif normalization == "normPsdToSum40_90Hz":
                        chosenPsd = normPsdToSum40to90Hz
                        chosenSem = semNormPsdToSum40to90Hz
                        chosen_ylabel = "rel. PSD to sum 40-90 Hz (%) +- SEM"
                    
                    else:
                        chosenPsd = average_Sxx
                        chosenSem = semRawPsd
                        chosen_ylabel = "uV^2/Hz +- SEM"
                    # else statement is necessary to ensure the definition variable is not only locally 
                        

                    #################### PSD AVERAGE OF EACH FREQUENCY BAND DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                    
                    # create booleans for each frequency-range for alpha, low beta, high beta, beta and gamma
                    alpha_frequency = (f >= 8) & (f <= 12) # alpha_range will output a boolean of True values within the alpha range
                    lowBeta_frequency = (f >= 13) & (f <= 20)
                    highBeta_frequency = (f >= 21) & (f <= 35)
                    beta_frequency = (f >= 13) & (f <= 35)
                    narrowGamma_frequency = (f >= 40) & (f <= 90)

                    # make a list with all boolean masks of each frequency, so I can loop through
                    range_allFrequencies = [alpha_frequency, lowBeta_frequency, highBeta_frequency, beta_frequency, narrowGamma_frequency]

                    # loop through frequency ranges and get all psd values of each frequency band
                    for count, boolean in enumerate(range_allFrequencies):

                        frequency = []
                        if count == 0:
                            frequency = "alpha"
                        elif count == 1:
                            frequency = "lowBeta"
                        elif count == 2:
                            frequency = "highBeta"
                        elif count == 3:
                            frequency = "beta"
                        elif count == 4:
                            frequency = "narrowGamma"
                        


                        # get all frequencies and chosen psd values within each frequency range
                        # frequencyInFreqBand = f[range_allFrequencies[count]] # all frequencies within a frequency band
                        psdInFreqBand = chosenPsd[range_allFrequencies[count]] # all psd values within a frequency band

                        psdAverage = np.mean(psdInFreqBand)

                        # store averaged psd values of each frequency band in a dictionary
                        psdAverage_dict[f'{tp}_{ch}_psdAverage_{frequency}'] = [tp, ch, frequency, psdAverage]



                    #################### PEAK DETECTION PSD DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                    # find all peaks: peaks is a tuple -> peaks[0] = index of frequency?, peaks[1] = dictionary with keys("peaks_height") 
                    peaks = scipy.signal.find_peaks(chosenPsd, height=0.1) # height: peaks only above 0.1 will be recognized

                    # Error checking: if no peaks found, continue
                    if len(peaks) == 0:
                        continue

                    peaks_height = peaks[1]["peak_heights"] # np.array of y-value of peaks = power
                    peaks_pos = f[peaks[0]] # np.array of indeces on x-axis of peaks = frequency

                    # set the x-range for each frequency band
                    alpha_range = (peaks_pos >= 8) & (peaks_pos <= 12) # alpha_range will output a boolean of True values within the alpha range
                    lowBeta_range = (peaks_pos >= 13) & (peaks_pos <= 20)
                    highBeta_range = (peaks_pos >= 21) & (peaks_pos <= 35)
                    beta_range = (peaks_pos >= 13) & (peaks_pos <= 35)
                    narrowGamma_range = (peaks_pos >= 40) & (peaks_pos <= 90)

                    # make a list with all boolean masks of each frequency, so I can loop through
                    frequency_ranges = [alpha_range, lowBeta_range, highBeta_range, beta_range, narrowGamma_range]

                    # loop through frequency ranges and get the highest peak of each frequency band
                    for count, boolean in enumerate(frequency_ranges):

                        frequency = []
                        if count == 0:
                            frequency = "alpha"
                        elif count == 1:
                            frequency = "lowBeta"
                        elif count == 2:
                            frequency = "highBeta"
                        elif count == 3:
                            frequency = "beta"
                        elif count == 4:
                            frequency = "narrowGamma"
                        
                        # get all peak positions and heights within each frequency range
                        peaksinfreq_pos = peaks_pos[frequency_ranges[count]]
                        peaksinfreq_height = peaks_height[frequency_ranges[count]]

                        # Error checking: check first, if there is a peak in the frequency range
                        if len(peaksinfreq_height) == 0:
                            continue

                        # select only the highest peak within the alpha range
                        highest_peak_height = peaksinfreq_height.max()

                        ######## calculate psd average of +- 2 Hz from highest Peak ########
                        # 1) find psd values from -2Hz until + 2Hz from highest Peak by slicing and indexing the numpy array of all chosen psd values
                        peakIndex = np.where(chosenPsd == highest_peak_height) # np.where output is a tuple: index, dtype
                        peakIndexValue = peakIndex[0].item() # only take the index value of the highest Peak psd value in all chosen psd

                        # 2) go -2 and +3 indeces 
                        indexlowCutt = peakIndexValue-2
                        indexhighCutt = peakIndexValue+3   # +3 because the ending index is left out when slicing a numpy array

                        # 3) slice the numpy array of all chosen psd values, only get values from -2 until +2 Hz from highest Peak
                        psdArray5HzRangeAroundPeak = chosenPsd[indexlowCutt:indexhighCutt] # array only of psd values -2 until +2Hz around Peak = 5 values

                        # 4) Average of 5Hz Array
                        highest_peak_height_5Hzaverage = np.mean(psdArray5HzRangeAroundPeak)                       



                        # get the index of the highest peak y value to get the corresponding peak position x
                        ix = np.where(peaksinfreq_height == highest_peak_height)
                        highest_peak_pos = peaksinfreq_pos[ix].item()

                        # plot only the highest peak within each frequency band
                        axes[t].scatter(highest_peak_pos, highest_peak_height, color="k", s=15, marker='D')

                        # store highest peak values of each frequency band in a dictionary
                        highest_peak_dict[f'{tp}_{ch}_highestPEAK_{frequency}'] = [tp, ch, frequency, highest_peak_pos, highest_peak_height, highest_peak_height_5Hzaverage]




                    #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

                    # the title of each plot is set to the timepoint e.g. "postop"
                    axes[t].set_title(tp, fontsize=20) 

                    # get y-axis label and limits
                    # axes[t].get_ylabel()
                    # axes[t].get_ylim()

                    # .plot() method for creating the plot, axes[0] refers to the first plot, the plot is set on the appropriate object axes[t]
                    axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}")  # or np.log10(px) 
                    # colors of each line in different color, defined at the beginning
                    # axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}", color=colors[i])

                    # make a shadowed line of the sem
                    axes[t].fill_between(f, chosenPsd-chosenSem, chosenPsd+chosenSem, color='lightgray', alpha=0.5)



    #################### PLOT SETTINGS ####################

    plt.subplots_adjust(wspace=0, hspace=0)

    font = {"size": 16}

    for ax in axes: 
        # ax.legend(loc= 'upper right') # Legend will be in upper right corner
        ax.grid() # show grid
        ax.set(xlim=[4, 40]) # no ylim for rawPSD and normalization to sum 40-90 Hz
        # ax.set(xlim=[-5, 60] ,ylim=[0,7]) for normalizations to total sum or to sum 1-100Hz set ylim to zoom in
        ax.set_xlabel("Frequency", fontdict=font)
        ax.set_ylabel(chosen_ylabel, fontdict=font)
        ax.axvline(x=8, color='black', linestyle='--')
        ax.axvline(x=13, color='black', linestyle='--')
        ax.axvline(x=20, color='black', linestyle='--')
        ax.axvline(x=35, color='black', linestyle='--')
    
    # remove x ticks and labels from all but the bottom subplot
    for ax in axes.flat[:-1]:
        ax.set(xlabel='')
        
    #fig.legend(title="bipolar channels", loc= 'upper right', bbox_to_anchor=(0.5,1.05), fancybox=True, shadow=True, fontsize="small")

    ###### LEGEND ######
    legend = axes[0].legend(loc= 'lower right', edgecolor="black", bbox_to_anchor=(1.5, -0.1)) # only show the first subplot´s legend 
    # frame the legend with black edges amd white background color 
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()

    plt.show()
    fig.savefig(figures_path + f"\\PSDspectrogram_sub{incl_sub}_{hemisphere}_{normalization}_{pickChannels}.png")
    

    #################### WRITE DATAFRAMES TO STORE VALUES ####################
    # write raw PSD Dataframe
    rawPSDDataFrame = pd.DataFrame(f_rawPsd_dict)
    rawPSDDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "averagedPSD", 5: "SEM_rawPsd"}, inplace=True) # rename the rows
    rawPSDDataFrame = rawPSDDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to total Sum
    normPsdToTotalSumDataFrame = pd.DataFrame(f_normPsdToTotalSum_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToTotalSumDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToTotalSum", 5: "SEM_normPsdToTotalSum"}, inplace=True) # rename the rows
    normPsdToTotalSumDataFrame = normPsdToTotalSumDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum1to100HzDataFrame = pd.DataFrame(f_normPsdToSum1to100Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum1to100HzDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSumPsd1to100Hz", 5: "SEM_normPsdToSumPsd1to100Hz"}, inplace=True) # rename the rows
    normPsdToSum1to100HzDataFrame = normPsdToSum1to100HzDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum40to90DataFrame = pd.DataFrame(f_normPsdToSum40to90Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum40to90DataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSum40to90Hz", 5: "SEM_normPsdToSum40to90Hz"}, inplace=True) # rename the rows
    normPsdToSum40to90DataFrame = normPsdToSum40to90DataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(psdAverage_dict) # Dataframe with 5 rows and columns for each single power spectrum
    psdAverageDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: f"averaged{normalization}"}, inplace=True) # rename the rows
    psdAverageDF = psdAverageDF.transpose() # Dataframe with 4 columns and rows for each single power spectrum


    # write DataFrame of frequency and psd values of the highest peak in each frequency band
    highestPEAKDF = pd.DataFrame(highest_peak_dict) # Dataframe with 5 rows and columns for each single power spectrum
    highestPEAKDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: "PEAK_frequency", 4:f"PEAK_{normalization}", 5: "highest_peak_height_5HzAverage"}, inplace=True) # rename the rows
    highestPEAKDF = highestPEAKDF.transpose() # Dataframe with 6 columns and rows for each single power spectrum


    # save Dataframes as csv in the results folder
    
    rawPSDDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMrawPSD_{hemisphere}"), sep=",")
    normPsdToTotalSumDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToTotalSum_{hemisphere}"), sep=",")
    normPsdToSum1to100HzDataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_1to100Hz_{hemisphere}"), sep=",")
    normPsdToSum40to90DataFrame.to_csv(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_40to90Hz_{hemisphere}"), sep=",")
    psdAverageDF.to_csv(os.path.join(results_path,f"SPECTROGRAMpsdAverageFrequencyBands_{normalization}_{hemisphere}"), sep=",")
    highestPEAKDF.to_csv(os.path.join(results_path,f"SPECTROGRAM_highestPEAK_FrequencyBands_{normalization}_{hemisphere}"), sep=",")


    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "normPsdToTotalSumDataFrame":normPsdToTotalSumDataFrame,
        "normPsdToSum1to100HzDataFrame":normPsdToSum1to100HzDataFrame,
        "normPsdToSum40to90HzDataFrame":normPsdToSum40to90DataFrame,
        f"averaged{normalization}": psdAverageDF,
        f"highestPEAK{normalization}": highestPEAKDF,
    }

