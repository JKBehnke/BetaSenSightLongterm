""" Analyze the change of beta peak power and center frequency in the highest beta channels through linear mixed effects models """

import os
import sys
import numpy as np



# create a path to the BetaSenSightLongterm folder 
# and a path to the code folder within the BetaSenSightLongterm Repo
BetaSenSightLongterm_path = os.getcwd()
while BetaSenSightLongterm_path[-16:] != 'ResearchProjects':
    BetaSenSightLongterm_path = os.path.dirname(BetaSenSightLongterm_path)

# directory to PyPerceive code folder
PyPerceive_path = os.path.join(BetaSenSightLongterm_path,'PyPerceive_project', 'code', 'PyPerceive', 'code')
sys.path.append(PyPerceive_path)

# # change directory to PyPerceive code path within BetaSenSightLongterm Repo
os.chdir(PyPerceive_path)
os.getcwd()


from PerceiveImport.classes import (
    main_class, modality_class, metadata_class,
    session_class, condition_class, task_class,
    contact_class, run_class
)
import PerceiveImport.methods.load_rawfile as load_rawfile
import PerceiveImport.methods.find_folders as find_folders
import PerceiveImport.methods.metadata_helpers as metaHelpers


#######################     USE THIS DIRECTORY FOR WORKING WITH FOLDERS INSIDE OF CODE FOLDER OF BETASENSIGHTLONGTERM REPO  #######################


# create a path to the BetaSenSightLongterm folder 
# and a path to the code folder within the BetaSenSightLongterm Repo
current_path = os.getcwd()
while current_path[-16:] != 'ResearchProjects':
    current_path = os.path.dirname(current_path)

# directory to code folder
code_path = os.path.join(current_path, 'BetaSenSightLongterm','code', 'BetaSenSightLongterm')
sys.path.append(code_path)

# # change directory to code path within BetaSenSightLongterm Repo
os.chdir(code_path)
os.getcwd()

import src.bssu.tfr.fooof_peak_analysis as fooof_peaks


######### run all options of LME #########
# selection of highest beta channels at 3MFU

data_to_fit = ["beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc" ]
around_cf = ["around_cf_at_each_session", "around_cf_at_fixed_session"]
shape_of_model = ["straight", "curved", "asymptotic"]

for analysis in data_to_fit:

    if analysis == "beta_power_auc":

        for cf in around_cf:

            for shape in shape_of_model:

                mixedlm_highest_beta_channels = fooof_peaks.fooof_mixedlm_highest_beta_channels(
                    fooof_spectrum="periodic_spectrum",
                    highest_beta_session="highest_fu3m",
                    data_to_fit=analysis,
                    around_cf=cf,
                    incl_sessions=[0,3,12,18],
                    shape_of_model=shape
                )
    
    else:

        for shape in shape_of_model:

                mixedlm_highest_beta_channels = fooof_peaks.fooof_mixedlm_highest_beta_channels(
                    fooof_spectrum="periodic_spectrum",
                    highest_beta_session="highest_fu3m",
                    data_to_fit=analysis,
                    around_cf="around_cf_at_fixed_session", # this doesn´t affect the result
                    incl_sessions=[0,3,12,18],
                    shape_of_model=shape
                )




