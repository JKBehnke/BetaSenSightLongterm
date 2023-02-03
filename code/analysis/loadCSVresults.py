""" Load .csv files from results folder"""


import os
import pandas as pd
import PerceiveImport.methods.find_folders as find_folder


def load_PSDresultCSV(sub: str, psdMethod: str, normalization: str, hemisphere: str):

    """
    Reads result CSV file

    Input:
        - subject = str, e.g. "024"
        - psdMethod = str "Welch" or "Spectrogram"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"


    Returns: 
        - data: loaded CSV file as a Dataframe 

    """


    
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

    # find the path to the results folder of a subject
    local_results_path = find_folder.get_local_path(folder="results", sub=sub)


    # create Filename out of input 
    filename = ""

    if psdMethod == "Spectrogram":
        method = "SPECTROGRAM"
    
    elif psdMethod == "Welch":
        method = ""
    
    norm = normalization
    side = f"_{hemisphere}"
    

    string_list = [method, norm, side]
    filename = "".join(string_list)


    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )


    df = pd.read_csv(os.path.join(local_results_path, filename))

    return df




def load_MonoRef_JLBresultCSV(sub: str, normalization: str, hemisphere: str):

    """
    Reads monopolar reference result from Johannes' method CSV file

    Input:
        - subject = str, e.g. "024"
        - normalization =  str "rawPSD" or "normPsdToTotalSum" or "normPsdToSum1_100Hz" or "normPsdToSum40_90Hz"
        - hemisphere = str, "Right" or "Left"


    Returns: loading csv files into a dictionary
        - psd average (columns: session, frequency_band, channel, averagedPSD)
        - percentage of psd per directions (columns: session, frequency_band, direction, percentagePSD_perDirection)
        - monopolar Reference Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)
        - monopolar Ranks Dataframe (columns: for every session and frequency band lowBeta, highBeta, beta)

    """
    
    
    # Error check: 
    # Error if sub str is not exactly 3 letters e.g. 024
    assert len(sub) == 3, f'Subject string ({sub}) INCORRECT' 
    

    # find the path to the results folder of a subject
    local_results_path = find_folder.get_local_path(folder="results", sub=sub)

    # Error if filename doesn´t end with .mat
    # assert filename[-4:] == '.csv', (
    #     f'filename no .csv INCORRECT extension: {filename}'
    # )

    psdAverageDataframe = pd.read_csv(os.path.join(local_results_path, f"averagedPSD_{normalization}_{hemisphere}"))
    psdPercentagePerDirection = pd.read_csv(os.path.join(local_results_path, f"percentagePsdDirection_{normalization}_{hemisphere}"))
    monopolRefDF = pd.read_csv(os.path.join(local_results_path, f"monopolarReference_{normalization}_{hemisphere}"))
    monopolRankDF =  pd.read_csv(os.path.join(local_results_path, f"monopolarRanks_{normalization}_{hemisphere}"))

    return {
        "psdAverageDataframe": psdAverageDataframe, 
        "psdPercentagePerDirection": psdPercentagePerDirection, 
        "monopolRefDF": monopolRefDF, 
        "monopolRankDF": monopolRankDF}











