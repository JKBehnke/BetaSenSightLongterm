""" load rotated coordinates"""



import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy

import scipy.io as sio


######### PRIVATE PACKAGES #########
import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults



def load_mni_coordinates(incl_sub:list):
    """
    Find data folder and load per subject each ea_reconstruction.mat file and extract x,y,z coordinates
        - reco_native 
        - reco_scrf
        - reco_mni 

    Write an Excel file: SenSightElectrode_coordinates.xlsx into imagingData folder

    watchout !! sub017 and sub034 only have native coordinates !! scrf and native coordinates are missing
    
    """


    hemispheres = ["Right", "Left"]

    current_path = os.getcwd()
    while current_path[-8:] != 'Research':
        current_path = os.path.dirname(current_path)


    reco_native_concat = pd.DataFrame()
    reco_scrf_concat = pd.DataFrame()
    reco_mni_concat = pd.DataFrame()

    for sub in incl_sub:
        
        # directory to data folder with mni coordinates
        data_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', f"sub-{sub}")
        filename = os.path.join(data_path, "ea_reconstruction.mat")

        # load .mat file
        mni_file = sio.loadmat(filename)

        print(sub)

        reco_native = mni_file["reco"][0][0][1]
        reco_scrf = mni_file["reco"][0][0][2]
        reco_mni = mni_file["reco"][0][0][3]


        # get the correct array from the .mat file per hemisphere: positive x = Right, negative x = Left
        for h, hem in enumerate(hemispheres):
            
            reco_native_hem = reco_native[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_native_DF = pd.DataFrame(reco_native_hem, columns=["reco_native_x", "reco_native_y", "reco_native_z"])
            reco_native_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}") # insert column with subject_hemisphere on first position
            reco_native_concat = pd.concat([reco_native_concat, reco_native_DF], ignore_index=True)
        
            reco_scrf_hem = reco_scrf[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_scrf_DF = pd.DataFrame(reco_scrf_hem, columns=["reco_scrf_x", "reco_scrf_y", "reco_scrf_z"])
            reco_scrf_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}")
            reco_scrf_concat = pd.concat([reco_scrf_concat, reco_scrf_DF], ignore_index=True)

            reco_mni_hem = reco_mni[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_mni_DF = pd.DataFrame(reco_mni_hem, columns=["reco_mni_x", "reco_mni_y", "reco_mni_z"])
            reco_mni_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}")
            reco_mni_concat = pd.concat([reco_mni_concat, reco_mni_DF], ignore_index=True)
    

    # store each Dataframe in seperate sheets of an Excel file
    Excel_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', 'SenSightElectrode_coordinates.xlsx')

    # create an Excel writer, so that different sheets are written within the same excel file
    with pd.ExcelWriter(Excel_path) as writer:
        
        reco_native_concat.to_excel(writer, sheet_name="reco_native")
        reco_scrf_concat.to_excel(writer, sheet_name="reco_scrf")
        reco_mni_concat.to_excel(writer, sheet_name="reco_mni")


    return {
        "reco_native_concat": reco_native_concat,
        "reco_scrf_concat": reco_scrf_concat,
        "reco_mni_concat": reco_mni_concat
    }

        
