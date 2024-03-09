""" Movement artifact cleaning before computing power spectra """

# before running .py file make sure you are using bssu environment: 
# Ctrl+Shift+P -> Select Interpreter -> choose bssu

import os
import sys
import numpy as np
import importlib
from importlib import reload 


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

import src.bssu.tfr.movement_artifact_cleaning as move_artifacts


# run the plotting raw time series function to plot all raw data and select movement artifacts
time_series = move_artifacts.plot_raw_time_series(
    incl_sub=["024"],
    incl_session=["fu3m","fu18m"],
    incl_condition=["m0s0"],
    filter="band-pass"
)

#artifact_removal_time_series = move_artifacts.clean_time_series_move_artifact()

# plot_spectra = move_artifacts.plot_clean_power_spectra(signal_filter="band-pass")




