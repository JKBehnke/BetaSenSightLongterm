""" import local packages """
import os
import sys


def import_pyPerceive():
    """
    
    """

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

    print("PyPerceive import was successful")


def chdir_to_local_repo_to_import(
        
):
    """
    changing directory to BetaSenSightLongterm Repo
    
    """

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

    print(f"directory changed to local Repository to import local functions: {os.getcwd}")


