""" Load files from data folder"""


import os
import pandas as pd
import pickle
import json
from pathlib import Path
import mne_bids
from mne_bids import BIDSPath

from .. utils import find_folders as find_folders


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


def load_externalized_Poly5_files(
        sub: str
):
    """

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
    
    return os.path.join(subject_folder_path, filename)


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
        # check if the BIDS key has a sub-EL or sub-L folder 
        if "EL" in sub_BIDS_ID:
            session = "EcogLfpMedOff01"
        
        else:
            session = "LfpMedOff01"
        
        task = "Rest"
        acquisition = "StimOff"
        run = "1"
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
        
        data = mne_bids.read_raw_bids(bids_path=bids_path)

        return data
        
       

        
    


