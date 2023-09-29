""" Read and preprocess externalized LFPs"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import mne
import pickle

# internal Imports
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults



results_path = find_folders.get_local_path(folder="GroupResults")
figures_path = find_folders.get_local_path(folder="GroupFigures")


#patient_metadata = loadResults.load_patient_metadata_externalized()
