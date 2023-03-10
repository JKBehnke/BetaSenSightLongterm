import os
import numpy as np

def find_project_folder():
    """
    find_project_folder is a function to find the folder "PyPerceive_Project" on your local computer

    Return: tuple[str, str] -> project_path, data_path
    to only use one str use _ -> example: project_folder, _ = find_project_folder()
    """

    # from the cwd get path to PyPerceive_Project (=Git Repository)
    project_path = os.getcwd()
    while project_path[-21:] != 'Longterm_beta_project':
        project_path = os.path.dirname(project_path)
    
    results_path = os.path.join(project_path, 'results')
    
    return project_path, results_path


def get_onedrive_path(
    folder: str = 'onedrive', sub: str = None
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """

    folder_options = [
        'onedrive', 'sourcedata'
        ]

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path) # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower())
            ]

    path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder


    # add the folder DATA-Test to the path and from there open the folders depending on input folder
    datapath = os.path.join(path, 'Percept_Data_structured')
    if folder == 'onedrive': 
        return datapath

    elif folder == 'sourcedata':
        return os.path.join(datapath, 'sourcedata')

    # elif folder == 'results': # must be data or figures
    #     return os.path.join(datapath, 'results')
    
    # elif folder == "raw_perceive": # containing all relevant perceive .mat files
    #     return os.path.join(datapath, "sourcedata", f"sub-{sub}", "raw_perceive")


def get_local_path(folder: str, sub: str = None):
    """
    find_project_folder is a function to find the folder "Longterm_beta_project" on your local computer

    Input:
        - folder: str
            'Research': path to Research folder
            'Longterm_beta_project': path to Project folder
            'GroupResults': path to results folder, without going in subject level
            'results': subject folder of results
            'GroupFigures': path to figures folder, without going in subject level 
            'figures': figure folder of results
        
        - sub: str, e.g. "029"
    

    """

    folder_options = [
        'Research', 'Longterm_beta_project', 'GroupResults', 'results', 'GroupFigures', 'figures', 'data'
        ]

    # Error checking, if folder input is in folder options
    #if folder.lower() not in folder_options:
        # raise ValueError(
        #     f'given folder: {folder} is incorrect, '
        #     f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path) # path is now leading to Users/username

    # get the Research folder and add it to the path

    path = os.path.join(path, 'Research') # path is now leading to Research folder


    # add the folder to the path and from there open the folders depending on input folder
    if folder == 'Research':
        return path

    elif folder == 'Longterm_beta_project': 
        return os.path.join(path, "Longterm_beta_project")
    
    elif folder == "GroupResults":
        return os.path.join(path, "Longterm_beta_project", "results")

    elif folder == 'results':
        return os.path.join(path, "Longterm_beta_project", "results", f"sub-{sub}")
    
    elif folder == 'GroupFigures':
        return os.path.join(path, "Longterm_beta_project", "figures")

    elif folder == 'figures':
        return os.path.join(path, "Longterm_beta_project", "figures", f"sub-{sub}")
    
    elif folder == 'data':
        return os.path.join(path, "Longterm_beta_project", "data")
