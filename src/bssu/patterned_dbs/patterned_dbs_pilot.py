""" Patterned DBS Pilot"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import py_perceive
from PerceiveImport.classes import main_class

# internal Imports
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import patterned_dbs_helpers as helpers

GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")
