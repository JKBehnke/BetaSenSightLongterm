""" Load files from data folder"""


import os
import pandas as pd
import numpy as np
import pickle
import json
import h5py
from pathlib import Path
import mne
import mne_bids
from mne_bids import (
    BIDSPath,
    inspect_dataset,
    mark_channels
)

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.bssu.utils import find_folders
from src.bssu.extern import tmsi_poly5reader 



def load_patient_metadata_externalized():

    """
    Input:
        
        
    Load the file: movement_artifacts_from_raw_time_series_band-pass.pickle  # always band-pass because filtered signal is easier to find movement artifacts
    from the group result folder

    """

    # find the path to the results folder
    path = find_folders.get_monopolar_project_path(folder="data")

    # create filename
    filename = "patient_metadata.xlsx"

    filepath = os.path.join(path, filename)

    # load the file
    data = pd.read_excel(filepath, keep_default_na=True, sheet_name="patient_metadata") # all sheets are loaded
    print("Excel file loaded: ",filename, "\nloaded from: ", path)

    return data


def load_excel_data(
        filename:str
):
    """
    Input:
        - filename: "patient_metadata", "movement_artefacts"
    
    """

    patient_metadata_sheet = ["patient_metadata", "movement_artefacts"]

    # find the path to the results folder
    path = find_folders.get_monopolar_project_path(folder="data")

    # create filename
    f_name = f"{filename}.xlsx"
    
    if filename in patient_metadata_sheet:
        sheet_name = "patient_metadata"
    

    filepath = os.path.join(path, f_name)

    # load the file
    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=sheet_name) 
    print("Excel file loaded: ",f_name, "\nloaded from: ", path)

    return data



def load_externalized_Poly5_files(
        sub: str
):
    """
    Input:
        - sub: str e.g. "24"

    filepath: '/Users/jenniferbehnke/Dropbox/work/ResearchProjects/Monopolar_power_estimation/data/externalized_lfp/
    -> subject path depending on the externalized patient ID

    load the correct Poly5 file of the input subject
    - externalized LFP
    - Med Off
    - Stim Off
    - Rest 

    
    """

    subject_folder_path = find_folders.get_monopolar_project_path(
        folder="data_sub",
        sub= sub
    )

    # check if there is a .Poly5 file
    files = os.listdir(subject_folder_path)
    for f in files:
        if f.endswith(".Poly5"):
            filename = f
    
    filepath = os.path.join(subject_folder_path, filename)

    # load the Poly5 file
    raw_file = tmsi_poly5reader.Poly5Reader(filepath)
    raw_file = raw_file.read_data_MNE()
    raw_file.load_data()

    return raw_file


def load_BIDS_externalized_vhdr_files(
        sub: str
):
    """

    BIDS_root: '/Users/jenniferbehnke/OneDrive - Charité - Universitätsmedizin Berlin/BIDS_01_Berlin_Neurophys/rawdata/'
    -> subject path depending on the externalized patient ID

    load the correct vhdr file of the input subject
   
    BIDS structure: 
    - ECoG + LFP: sub-EL... > "ses-EcogLfpMedOff01" > "ieeg" > filename containing Rest, StimOff, run-1, endswith .vhdr
    - only LFP: sub-L... > "ses-LfpMedOff01"  > "ieeg" > filename containing Rest, StimOff, run-1, endswith .vhdr

        EL session = "EcogLfpMedOff01"
        L session = "LfpMedOff01"
        
        task = "Rest"
        aquisition = "StimOff"
        run = "1"
        datatype = "ieeg"
        extension = ".vhdr"
        suffix = "ieeg"

    
    """

    # get the BIDS key from the subject
    local_path = find_folders.get_monopolar_project_path(folder="data")
    patient_metadata = pd.read_excel(os.path.join(local_path, "patient_metadata.xlsx"),
                                        keep_default_na=True, sheet_name="patient_metadata")

    # change column "patient_ID" to strings
    patient_metadata["patient_ID"] = patient_metadata.patient_ID.astype(str)
    sub_BIDS_ID = patient_metadata.loc[patient_metadata.patient_ID == sub] # row of subject

    # check if the subject has a BIDS key
    if pd.isna(sub_BIDS_ID.BIDS_key.values[0]):
        print(f"The subject {sub} has no BIDS key yet.")
        return "no BIDS key"
    
    # only if there is a BIDS key.
    else:
        sub_BIDS_ID = sub_BIDS_ID.BIDS_key.values[0]

        raw_data_folder = find_folders.get_onedrive_path_externalized_bids(folder="rawdata")
        bids_root = raw_data_folder
        bids_path = BIDSPath(root=bids_root)


        ########## UPDATE BIDS PATH ##########
        # check if BIDS session directory contains "Dys"
        sessions = os.listdir(os.path.join(raw_data_folder, f"sub-{sub_BIDS_ID}"))
        dys_list = []
        for s in sessions:
            if "MedOffDys" in s:
                dys_list.append("Dys")
            
        if len(dys_list) == 0:
            dys = ""
            dopa = ""
        
        else:
            dys = "Dys"
            if sub_BIDS_ID == "EL016":
                dopa = "DopaPre"
            else:
                dopa = "Dopa00"
        
        # check if the BIDS key has a sub-EL or sub-L folder 
        session = f"LfpMedOff{dys}01"

        if "EL" in sub_BIDS_ID:
            session = f"EcogLfpMedOff{dys}01"    
        
        task = "Rest"
        acquisition = f"StimOff{dopa}"

        run = "1"
        if sub_BIDS_ID == "L014":
            run = "2"
        
        datatype = "ieeg"
        extension = ".vhdr"
        suffix = "ieeg"

        bids_path.update(
            subject = sub_BIDS_ID,
            session = session,
            task = task,
            acquisition = acquisition,
            run = run,
            datatype = datatype,
            extension = extension,
            suffix = suffix,
        )
        
        #inspect_dataset(bids_path, l_freq=5.0, h_freq=95.0)

        data = mne_bids.read_raw_bids(bids_path=bids_path) # datatype: mne.io.brainvision.brainvision.RawBrainVision
        # to work with the data, load the data
        data.load_data()

        return data
        


def load_externalized_pickle(
        filename:str
):
    
    """
    Pickle files in the group results folder of the monopolar estimation project
    Input: 
        - filename: str, must be in 
            ["externalized_preprocessed_data",
            "externalized_recording_info_original",
            "mne_objects_cropped_2_min",
            "externalized_preprocessed_data_artefact_free",
            "externalized_power_spectra_250Hz_artefact_free",
            "externalized_contacts_common_reference",
            "fooof_externalized_group",
            "fooof_externalized_group_notch-filtered",
            "fooof_externalized_beta_ranks_all_contacts",
            "fooof_externalized_beta_ranks_directional_contacts"
            ]
    """

    group_results_path = find_folders.get_monopolar_project_path(folder="GroupResults")

    # create filename and filepath
    pickle_filename = f"{filename}.pickle"
    filepath = os.path.join(group_results_path, pickle_filename)

    # load the file
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data





    


