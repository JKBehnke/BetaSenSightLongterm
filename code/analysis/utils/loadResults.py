""" Load result files from results folder"""


import os
import pandas as pd
import pickle
import json

import analysis.utils.find_folders as find_folders


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
    Reads pickle file written with functions in BIP_channelGroups.py 

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

    comparison = ["Postop_Fu3m", "Fu3m_Fu12m", "Fu12m_Fu18m"]

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



















