""" Load result files from results folder"""


import os
import pandas as pd
import pickle
import json

from .. utils import find_folders as find_folders


def load_PSDjson(sub: str, result: str, hemisphere: str, filter: str):

    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - result = str "PowerSpectrum", "PSDaverageFrequencyBands", "PeakParameters"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded CSV file as a Dataframe 

    """


    
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)


    # create Filename out of input 
    filename = ""
    filebase = ""

    if result == "PowerSpectrum":
        filebase = "SPECTROGRAMPSD"
    
    elif result == "PSDaverageFrequencyBands":
        filebase = f"SPECTROGRAMpsdAverageFrequencyBands"
    
    elif result == "PeakParameters":
        filebase = f"SPECTROGRAM_highestPEAK_FrequencyBands"

    
    
    hem = f"_{hemisphere}"
    filt = f"_{filter}.json"
    

    string_list = [filebase, hem, filt]
    filename = "".join(string_list)


    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )

    with open(os.path.join(local_results_path, filename)) as file:
        data = json.load(file)

    return data


def load_BIPChannelGroups_ALL(freqBand:str, normalization:str, signalFilter:str):

    """
    Loads pickle file from Group Results folder
    filename example: "BIPChannelGroups_ALL_{freqBand}_{normalization}_{signalFilter}.pickle"

    """
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert freqBand in [ "beta", "highBeta", "lowBeta"], f'Result ({freqBand}) INCORRECT' 

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    norm = f"_{normalization}"
    filt = f"_{signalFilter}.pickle"

    
    string_list = ["BIPChannelGroups_ALL_", freqBand, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ",filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_BIPChannelGroups_psdRanks_relToRank1(freqBand:str, normalization:str, signalFilter:str):

    """
    Loads pickle file from Group Results folder
    filename: "BIPChannelGroups_psdRanks_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"

    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: list e.g. ["beta", "highBeta", "lowBeta"]


    """
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert freqBand in [ "beta", "highBeta", "lowBeta"], f'Result ({freqBand}) INCORRECT' 

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    norm = f"_{normalization}"
    filt = f"_{signalFilter}.pickle"

    
    string_list = ["BIPChannelGroups_psdRanks_relToRank1_", freqBand, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ",filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data




def load_PSDresultCSV(sub: str, psdMethod: str, normalization: str, hemisphere: str, filter: str):

    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - psdMethod = str "Welch" or "Spectrogram"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded CSV file as a Dataframe 

    """


    
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)


    # create Filename out of input 
    filename = ""

    if psdMethod == "Spectrogram":
        method = "SPECTROGRAM"
    
    elif psdMethod == "Welch":
        method = ""
    
    norm = normalization
    hem = f"_{hemisphere}"
    filt = f"_{filter}"
    

    string_list = [method, norm, hem, filt]
    filename = "".join(string_list)


    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )


    df = pd.read_csv(os.path.join(local_results_path, filename), sep= ",")

    return df


def load_freqBandsCSV(sub: str, parameters: str, normalization: str, hemisphere: str, filter: str):

    """
    Reads result CSV file of the initial SPECTROGRAM method

    Input:
        - subject = str, e.g. "024"
        - parameters = str, "Peak" or "PSDaverage"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"
        - filter = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded CSV file as a Dataframe 

    """


    
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="results", sub=sub)


    # create Filename out of input 
    filename = ""

    if parameters == "Peak":
        values = "_highestPEAK_FrequencyBands_"
    
    elif parameters == "PSDaverage":
        values = "psdAverageFrequencyBands_"

    else:
        print("define parameters correctly: as 'Peak' or 'PSDaverage'.") 

    norm = normalization
    hem = f"_{hemisphere}"
    filt = f"_{filter}"
    

    string_list = ["SPECTROGRAM", values, norm, hem, filt]
    filename = "".join(string_list)


    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )


    df = pd.read_csv(os.path.join(local_results_path, filename))

    return df


def load_BIPchannelGroupsPickle(result: str,  channelGroup: list, normalization: str, filterSignal: str):

    """
    Reads pickle file written with functions in BIP_channelGroups.py -> filename e.g. BIPpsdAverage_Ring_{normalization}_{signalFilter}.pickle

    Input:
        - result = str "psdAverage", "peak"
        - channelGroup = list, ["Ring", "SegmInter", "SegmIntra"]
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded pickle file as a Dataframe 

    """

    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in [ "psdAverage", "peak"], f'Result ({result}) INCORRECT' 

    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")
    data = {}

    # create Filename out of input for each channel Group
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    for g in channelGroup:
        group = f"_{g}"

        string_list = ["BIP", result, group, norm, filt]
        filename = "".join(string_list)
        print("pickle file: ",filename, "\nloaded from: ", local_results_path)

        filepath = os.path.join(local_results_path, filename)
       


        # Error if filename doesn´t end with .mat
        # assert filename[-4:] == '.csv', (
        #     f'filename no .csv INCORRECT extension: {filename}'
        # )

    
        with open(filepath, "rb") as file:
            data[g] = pickle.load(file)

    return data



def load_BIPchannelGroup_sessionPickle(result: str,  freqBand: str, normalization: str, filterSignal: str):

    """
    Reads pickle file written with function Rank_BIPRingSegmGroups() in BIPchannelGroups_ranks.py
    -> filename: BIPranksChannelGroup_session_dict_{result}_{freqBand}_{normalization}_{filterSignal}.pickle 

    Input:
        - result = str "psdAverage", "peak"
        - freqBand = str, e.g. "beta"
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded pickle file as a Dataframe 

    """

    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in [ "psdAverage", "peak"], f'Result ({result}) INCORRECT' 

    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    freq = f"_{freqBand}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    
    string_list = ["BIPranksChannelGroup_session_dict_", result, freq, norm, filt]
    filename = "".join(string_list)
    print("pickle file loaded: ",filename, "\nloaded from: ", local_results_path)

    filepath = os.path.join(local_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    return data


def load_BIPpermutationComparisonsPickle(result: str,  freqBand: str, normalization: str, filterSignal: str):

    """
    Reads pickle file written with function Rank_BIPRingSegmGroups() in BIPchannelGroups_ranks.py 

    Input:
        - result = str "psdAverage", "peak"
        - freqBand = str, e.g. "beta"
        - normalization =  str "rawPsd" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - filterSignal = str "band-pass" or "unfiltered"


    Returns: 
        - data: loaded pickle file as a Dataframe 

    """

    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert result in [ "psdAverage", "peak"], f'Result ({result}) INCORRECT' 

    

    # find the path to the results folder of a subject
    local_results_path = find_folders.get_local_path(folder="GroupResults")

    # create Filename out of input for each channel Group
    # example: BIPranksPermutation_dict_peak_beta_rawPsd_band-pass.pickle

    comparison = ["Postop_Postop", "Postop_Fu3m", "Postop_Fu12m", "Postop_Fu18m", 
                   "Fu3m_Postop", "Fu3m_Fu3m", "Fu3m_Fu12m", "Fu3m_Fu18m", 
                   "Fu12m_Postop", "Fu12m_Fu3m", "Fu12m_Fu12m", "Fu12m_Fu18m",
                   "Fu18m_Postop", "Fu18m_Fu3m", "Fu18m_Fu12m", "Fu18m_Fu18m"]

    res = f"_{result}"
    freq = f"_{freqBand}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    data = {}

    for c in comparison:
        string_list = ["BIPpermutationDF_", c, res, freq, norm, filt]
        filename = "".join(string_list)

        filepath = os.path.join(local_results_path, filename)

        with open(filepath, "rb") as file:
            data[c] = pickle.load(file)

        print("pickle file loaded: ",filename, "\nloaded from: ", local_results_path)


    return data



def load_BIPpermutation_ranks_result(
        data2permute:str,
        filterSignal: str,
        normalization: str,
        freqBand: str,
):

    """
    Reads pickle file written with function PermutationTest_BIPchannelGroups() in Permutation_rankings.py 
        filename: "Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"

    Input:
        - data2permute: str e.g. "psdAverage",  "peak"
        - filterSignal: str e.g. "band-pass"
        - normalization: str e.g. "rawPsd"
        - freqBand: str e.g. "beta"


    Returns: 
        - data: loaded pickle file as a Dataframe 

    
    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"Permutation_BIP_{data2permute}_{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data



def load_BestClinicalStimulation_excel():

    """
    Reads Excel file from the results folder: BestClinicalStimulation.xlsx
    loaded file = dictionary
        - all sheets are loaded as different keys 

    Input:
        

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    data_path = find_folders.get_local_path(folder="data")
    
    filename = "BestClinicalStimulation.xlsx"
    filepath = os.path.join(data_path, filename)

    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=None) # all sheets are loaded
    print("Excel file loaded: ",filename, "\nloaded from: ", data_path)


    return data



def load_monoRef_weightedPsdCoordinateDistance_pickle(
        sub: str,
        hemisphere: str,
        freqBand: str,
        normalization: str,
        filterSignal: str,


):

    """
    Reads Pickle file from the subjects results folder: 
        - sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle
    
    
    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values



    Input:
        

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    hem = f"_{hemisphere}"
    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["sub", sub, hem, "_monoRef_weightedPsdByCoordinateDistance_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", sub_results_path)

    return data


def load_monoRef_only_segmental_weight_psd_by_distance(
        sub: str,
        hemisphere: str,
        freqBand: str,
        normalization: str,
        filterSignal: str,
):

    """
    Reads Pickle file from the subjects results folder: 
        - sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle
    
    
    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    filename = f"sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", sub_results_path)

    return data




def load_Group_monoRef_only_segmental_weight_psd_by_distance(
        freqBand: str,
        normalization: str,
        filterSignal: str,
):

    """
    Reads Pickle file from the results folder: 
        - "group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{signalFilter}.pickle"
    
    
    loaded file is a dictionary with keys:
        - f"{session}_bipolar_Dataframe": bipolar channels, averaged PSD in freq band, coordinates z- and xy-axis of mean point between two contacts
        - f"{session}_monopolar_Dataframe": 10 monopolar contacts, averaged monopolar PSD in freq band, rank of averagedPsd values

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    filename = f"group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{filterSignal}.pickle"

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", results_path)

    return data



def load_monopol_rel_psd_from0To8_pickle(
        freqBand: str,
        normalization: str,
        filterSignal: str,
        ):

    """
    Reads Pickle file from the subjects results folder: 
        - "monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle"
    

    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["monopol_rel_psd_from0To8_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", results_path)

    return data



def load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
        freqBand: str,
        normalization: str,
        filterSignal: str,
        ):

    """
    Reads Pickle file from the subjects results folder: 
        - "GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"
    

    Input:
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    norm = f"_{normalization}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_", freqBand, norm, filt]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", results_path)

    return data




def load_monoRef_JLB_pickle(
        sub: str,
        hemisphere: str,
        normalization: str,
        filterSignal: str,


):

    """
    Reads Pickle file from the subjects results folder: 
        - sub{incl_sub}_{hemisphere}_MonoRef_JLB_result_{normalization}_band-pass.pickle"
    
    
    loaded file is a dictionary with keys:
        - "BIP_psdAverage"
        - "BIP_directionalPercentage"
        - "monopolar_psdAverage"
        - "monopolar_psdRank"


    Input:
        

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder of a subject
    sub_results_path = find_folders.get_local_path(folder="results", sub=sub)

    hem = f"_{hemisphere}"
    filt = f"_{filterSignal}.pickle"

    string_list = ["sub", sub, hem, "_MonoRef_JLB_result_", normalization, filt]
    filename = "".join(string_list)

    filepath = os.path.join(sub_results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", sub_results_path)

    return data



def load_ClinicalActiveVsInactive(
        freqBand:str,
        attribute:str,
        singleContacts_or_average:str
):
    
    """
    Loading Dataframe with clinically active and inactive contacts

    file: "ClinicalActiveVsNonactiveContacts_{attribute}_{freqBand}_{singleContacts_or_average}.pickle

    Input:
        - freqBand: str e.g. "beta"
        - attribute: str e.g. "rank", "relativeToRank1_psd"
        - singleContacts_or_average: str e.g. "singleContacts", "averageContacts"

    """

    # find the path to the results folder of a subject
    results_path = find_folders.get_local_path(folder="GroupResults")

    r_or_psd = f"_{attribute}"
    freq = f"_{freqBand}"
    single_or_average = f"_{singleContacts_or_average}.pickle"

    if attribute == "rank":
        r_or_psd = ""

    string_list = ["ClinicalActiveVsNonactiveContacts", r_or_psd, freq, single_or_average]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", results_path)

    return data





def load_SSD_results_pickle(
        f_band:str

):

    """
    Reads Pickle file from the group results folder: 
        - "SSD_results_Dataframe_{f_band}.pickle"
    
    
    loaded file is a Dataframe with columns:

        - subject
        - hemisphere
        - session
        - recording_group (e.g. RingR)
        - bipolarChannel
        - ssd_filtered_timedomain (array)
        - ssd_pattern (weights of a channel contributing to the first component)
        - ssd_eigvals


    Input:
        - f_band = str, e.g. "beta", "highBeta", "lowBeta"
        

    Returns: 
        - data: loaded pickle file as a Dataframe 

    """
    

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    freq = f"_{f_band}.pickle"

    string_list = ["SSD_results_Dataframe", freq]
    filename = "".join(string_list)

    filepath = os.path.join(results_path, filename)

    with open(filepath, "rb") as file:
        data = pickle.load(file)

    print("pickle file loaded: ",filename, "\nloaded from: ", results_path)

    return data


def load_fooof_json(subject: str):

    """
    Load the file: "fooof_model_sub{subject}.json"
    from each subject result folder

    """
   
    # find the path to the results folder
    results_path_sub = find_folders.get_local_path(folder="results", sub=subject)

    # create filename
    filename = f"fooof_model_sub{subject}.json"

    # load the json file
    with open(os.path.join(results_path_sub, filename)) as file:
        json_data = json.load(file)

    fooof_result_df = pd.DataFrame(json_data)
    
    return fooof_result_df


def load_group_fooof_result():

    """
    Load the file: "fooof_model_group_data.json"
    from the group result folder

    """
   
    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_model_group_data.json"

    # load the json file
    with open(os.path.join(results_path, filename)) as file:
        json_data = json.load(file)

    fooof_result_df = pd.DataFrame(json_data)
    
    return fooof_result_df



def load_fooof_peaks_per_session():

    """
    Load the file: "fooof_peaks_per_session.pickle"
    from the group result folder

    """
   
    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_peaks_per_session.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data

def load_fooof_rank_beta_peak_power():

    """
    Load the file: "fooof_rank_beta_power_dataframe.pickle"
    from the group result folder
    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"fooof_rank_beta_power_dataframe.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data

def load_fooof_beta_ranks(
        fooof_spectrum:str,
        all_or_one_chan: str
):

    """
    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - all_or_one_chan: str "highest_beta" or "beta_ranks_all"


    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data


def load_power_spectra_session_comparison(
        incl_channels:str,
        signalFilter:str,
        normalization:str
):

    """
    Load the file: f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle"
    from the group result folder

    Input:
        - incl_channels: str, e.g. "SegmInter", "SegmIntra", "Ring"
        - signalFilter: str, e.g. "band-pass" or "unfiltered"
        - normalization: str, e.g. "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    Return:
        - dictionary with keys: 
        ['postop_fu3m_df', 'postop_fu12m_df', 'postop_fu18m_df', 'fu3m_fu12m_df', 'fu3m_fu18m_df', 'fu12m_fu18m_df']

        - each key value is a dataframe with the power spectra of STNs with recordings at both sessions

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data


def load_fooof_monopolar_weighted_psd(
        fooof_spectrum:str,
        segmental:str
):

    """
    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - segmental: "yes"              -> only using segmental channels to weight monopolar psd



    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    if segmental=="yes":
        bipolar_chans = "only_segmental_"
    
    else:
        bipolar_chans = "segments_and_rings_"

    # create filename
    filename = f"fooof_monoRef_{bipolar_chans}weight_beta_psd_by_distance_{fooof_spectrum}.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data


def load_fooof_permutation_bip_beta_ranks():

    """
    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - segmental: "yes"              -> only using segmental channels to weight monopolar psd



    Load the file: f"{all_or_one_chan}_channels_fooof_{fooof_spectrum}.pickle"
    from the group result folder

    """

    # find the path to the results folder
    results_path = find_folders.get_local_path(folder="GroupResults")

    # create filename
    filename = f"permutation_beta_ranks_fooof_spectra.pickle"

    filepath = os.path.join(results_path, filename)

    # load the pickle file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data




# def load_MonoRef_GroupCSV(normalization: str, hemisphere: str):

#     """
#     Reads monopolar reference result of all subjects in one CSV from Johannes' method

#     Input:
#         - subject = str, e.g. "024"
#         - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
#         - hemisphere = str, "Right" or "Left"


#     Returns: loading csv files into a dictionary
#         - psd average (columns: session, frequency_band, channel, averagedPSD)
#         - percentage of psd per directions (columns: session, frequency_band, direction, percentagePSD_perDirection)
#         - monopolar Reference Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)
#         - monopolar Ranks Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)

#     """

#     # find the path to the results folder of a subject
#     project_path = os.getcwd()

#     while project_path[-21:] != 'Longterm_beta_project':
#         project_path = os.path.dirname(project_path)

#     results_path = os.path.join(project_path, 'results')
#     sys.path.append(results_path)

#     # change directory to code path
#     os.chdir(results_path)
#     local_results_path = find_folder.get_local_path(folder="results", sub=sub)


#     psdAverageDataframe = pd.read_csv(os.path.join(local_results_path, f"averagedPSD_{normalization}_{hemisphere}"))
#     psdPercentagePerDirection = pd.read_csv(os.path.join(local_results_path, f"percentagePsdDirection_{normalization}_{hemisphere}"))
#     monopolRefDF = pd.read_csv(os.path.join(local_results_path, f"monopolarReference_{normalization}_{hemisphere}"))
#     monopolRefDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     monopolRankDF =  pd.read_csv(os.path.join(local_results_path, f"monopolarRanks_{normalization}_{hemisphere}"))
#     monopolRankDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     # FirstRankChannel_PSD_DF = pd.read_csv(os.path.join(local_results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_beta = pd.read_csv(os.path.join(local_results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_beta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
    



#     return {
#         "psdAverageDataframe": psdAverageDataframe, 
#         "psdPercentagePerDirection": psdPercentagePerDirection, 
#         "monopolRefDF": monopolRefDF, 
#         "monopolRankDF": monopolRankDF,
#         # "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
#         # "PostopBaseline_beta": postopBaseline_beta,
#         # "PostopBaseline_lowBeta": postopBaseline_lowBeta,
#         # "PostopBaseline_highBeta": postopBaseline_highBeta,
#         # "Fu3mBaseline_beta": fu3mBaseline_beta,
#         # "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
#         # "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
        
#         }



# def load_MonoRef_JLBresultCSV(sub: str, normalization: str, hemisphere: str):

#     """
#     Reads monopolar reference result from Johannes' method CSV file

#     Input:
#         - subject = str, e.g. "024"
#         - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
#         - hemisphere = str, "Right" or "Left"


#     Returns: loading csv files into a dictionary
#         - psd average (columns: session, frequency_band, channel, averagedPSD)
#         - percentage of psd per directions (columns: session, frequency_band, direction, percentagePSD_perDirection)
#         - monopolar Reference Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)
#         - monopolar Ranks Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)

#     """
    
    
#     # Error check: 
#     # Error if sub str is not exactly 3 letters e.g. 024
#     assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

#     # find the path to the results folder of a subject
#     local_results_path = find_folder.get_local_path(folder="results", sub=sub)


#     psdAverageDataframe = pd.read_csv(os.path.join(local_results_path, f"averagedPSD_{normalization}_{hemisphere}"))
#     psdPercentagePerDirection = pd.read_csv(os.path.join(local_results_path, f"percentagePsdDirection_{normalization}_{hemisphere}"))
#     monopolRefDF = pd.read_csv(os.path.join(local_results_path, f"monopolarReference_{normalization}_{hemisphere}"))
#     monopolRefDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     monopolRankDF =  pd.read_csv(os.path.join(local_results_path, f"monopolarRanks_{normalization}_{hemisphere}"))
#     monopolRankDF.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True)

#     # FirstRankChannel_PSD_DF = pd.read_csv(os.path.join(local_results_path,f"FirstRankChannel_PSD_DF{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_beta = pd.read_csv(os.path.join(local_results_path,f"postopBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # postopBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"postopBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_beta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_beta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_lowBeta{normalization}_{hemisphere}"), sep=",")
#     # fu3mBaseline_highBeta= pd.read_csv(os.path.join(local_results_path,f"fu3mBaseline_highBeta{normalization}_{hemisphere}"), sep=",")
    



#     return {
#         "psdAverageDataframe": psdAverageDataframe, 
#         "psdPercentagePerDirection": psdPercentagePerDirection, 
#         "monopolRefDF": monopolRefDF, 
#         "monopolRankDF": monopolRankDF,
#         # "FirstRankChannel_PSD_DF":FirstRankChannel_PSD_DF,
#         # "PostopBaseline_beta": postopBaseline_beta,
#         # "PostopBaseline_lowBeta": postopBaseline_lowBeta,
#         # "PostopBaseline_highBeta": postopBaseline_highBeta,
#         # "Fu3mBaseline_beta": fu3mBaseline_beta,
#         # "Fu3mBaseline_lowBeta": fu3mBaseline_lowBeta,
#         # "Fu3mBaseline_highBeta": fu3mBaseline_highBeta,
        
#         }



















