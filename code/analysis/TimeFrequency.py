""" Time Frequency Plot """


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

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



def time_frequency(incl_sub: str, incl_session: list, incl_condition: list, incl_contact: list, pickChannels: list, hemisphere: str):
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
       
    
    1) load data from mainclass.PerceiveData using the input values.
    
    2) band-pass filter by a Butterworth Filter of fifth order (5-95 Hz).

    3) Plot Time Frequency plot for each Channel of each session.
    
  
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
    

    # set layout for figures: using the object-oriented interface
    cols = ['Session {}'.format(col) for col in incl_session]
    rows = ['Channel {}'.format(row) for row in pickChannels]

    fig, axes = plt.subplots(len(pickChannels), len(incl_session), figsize=(15, 15)) # subplot(rows, columns, panel number)
    
    plt.setp(axes.flat, xlabel='Frequency', ylabel='Time')

    pad = 5 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need 
    # to make some room. These numbers are are manually tweaked. 
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    
 

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
    
                print("DATA", temp_data)

                #################### CREATE A BUTTERWORTH FILTER ####################

                # sample frequency: 250 Hz
                fs = temp_data.info['sfreq'] 

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

                # ch_names = [ch_names_renamed[idx] for idx in ch_names_indices] # new list of picked channel names based on the indeces 


                # create a time frequency plot per channel
                for i, ch in enumerate(ch_names_renamed):
                    
                    # only get picked channels
                    if i not in ch_names_indices:
                        continue

                    #################### FILTER ####################

                    # filter the signal by using the above defined butterworth filter
                    filtered = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) # .get_data()

                    # plot Time Frequency plot
                    noverlap = 0.5

                    # plot in subplot row=channel, column=timepoint
                    axes[i, t].specgram(x = filtered, Fs = fs, noverlap = noverlap, cmap = 'viridis', vmin = -25, vmax = 10)
                
    plt.show()
    
    fig.savefig(figures_path + f"\TimeFrequency_sub{incl_sub}_{hemisphere}_{pickChannels}.png")
    
