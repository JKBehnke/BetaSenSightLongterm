""" Highest Ranked monopolar Channel PSD analysis """


import os
import pandas as pd


# internal Imports
import analysis.utils.find_folders as find_folders


def compareHighestRankedPsd(incl_sub:list, incl_hemisphere:list, normalization:str,):

    """
    Input:

        - incl_sub: ["019", "024", "025", "026", "029", "030"]
        - incl_hemisphere: list e.g. ["Right", "Left"]
        - normalization: str "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    """

    frequencyBand = ["beta", "lowBeta", "highBeta"]

    # dictionary to fill in for each subject
    subject_dictionary = {}
    postopBaseline_dict = {}
    fu3mBaseline_dict = {}

    ################### Load csv files from results folder for each subject ###################
    for sub, subject in enumerate(incl_sub):
        for hem, hemisphere in enumerate(incl_hemisphere):

            # get path to results folder of each subject
            local_results_path = find_folders.get_local_path(folder="results", sub=subject)

            # Get all csv filenames from each subject
            filenames = [
                f"postopBaseline_beta{normalization}_{hemisphere}",
                f"postopBaseline_lowBeta{normalization}_{hemisphere}",
                f"postopBaseline_highBeta{normalization}_{hemisphere}", 
                f"fu3mBaseline_beta{normalization}_{hemisphere}",
                f"fu3mBaseline_lowBeta{normalization}_{hemisphere}",
                f"fu3mBaseline_highBeta{normalization}_{hemisphere}"
                         ]
            
            # Error Checking: check if files exist
            for f, file in enumerate(filenames):
                try: 
                    os.path.isdir(os.path.join(local_results_path, file))
                
                except OSError:
                    continue

            # get the csv results for each postop or fu3m Baseline highest Rank of each frequency band (beta, lowBeta, highBeta)
            postopBaseline_beta = pd.read_csv(os.path.join(local_results_path,filenames[0]), sep=",")
            postopBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path,filenames[1]), sep=",")
            postopBaseline_highBeta= pd.read_csv(os.path.join(local_results_path, filenames[2]), sep=",")
            fu3mBaseline_beta= pd.read_csv(os.path.join(local_results_path, filenames[3]), sep=",")
            fu3mBaseline_lowBeta= pd.read_csv(os.path.join(local_results_path, filenames[4]), sep=",")
            fu3mBaseline_highBeta= pd.read_csv(os.path.join(local_results_path, filenames[5]), sep=",")
        
            
            # store all Dataframes in a subject dictionary: values beta, lowBeta, highBeta respectively
            subject_dictionary[f"sub{subject}_{hemisphere}_postopBaseline"] = [postopBaseline_beta, postopBaseline_lowBeta, postopBaseline_highBeta]
            subject_dictionary[f"sub{subject}_{hemisphere}_fu3mBaseline"] = [fu3mBaseline_beta, fu3mBaseline_lowBeta, fu3mBaseline_highBeta]


            # get one dataframe for each combination across all subjects
        
            # loop over every frequency band
            for ind, freq in enumerate(frequencyBand):

                ######## POSTOP BASELINE ########
                postopBase = subject_dictionary[f"sub{subject}_{hemisphere}_postopBaseline"][ind]
                postopBase = postopBase.loc[:, (f"monoRef_postop_{freq}", f"monoRef_fu3m_{freq}", f"monoRef_fu12m_{freq}")] # only select postop, fu3m, fu12m columns for now
                # later also collect fu18m, and add Error continue, if sessions don´t exist
                postopBase.rename(index={0:f"{subject}_{hemisphere}"}, inplace=True) # rename Index to subject
            
                # postopBase = Dataframe of one subject, one hemisphere, postopBase highest channel, columns: 3 for sessions postop, fu3m, fu12m
                postopBaseline_dict[f"postopBaseline_{freq}_{subject}_{hemisphere}"] = postopBase


                ######## FU3M BASELINE ########
                fu3mBase = subject_dictionary[f"sub{subject}_{hemisphere}_fu3mBaseline"][ind]
                fu3mBase = fu3mBase.loc[:, (f"monoRef_postop_{freq}", f"monoRef_fu3m_{freq}", f"monoRef_fu12m_{freq}")] # only select postop, fu3m, fu12m columns for now
                # later also collect fu18m, and add Error continue, if sessions don´t exist
                fu3mBase.rename(index={0:f"{subject}_{hemisphere}"}, inplace=True) # rename Index to subject
            
                # fu3mBase = Dataframe of one subject, one hemisphere, fu3mBase highest channel, columns: 3 for sessions postop, fu3m, fu12m
                fu3mBaseline_dict[f"fu3mBaseline_{freq}_{subject}_{hemisphere}"] = fu3mBase


            # get Dataframes for each frequency band seperately 
            ######## POSTOP BASELINE ########
            postopBaseline_beta = {k: v for k, v in postopBaseline_dict.items() if "_beta" in k} # select all keys and values that contain substring beta
            postopBaseline_beta_All = pd.concat(postopBaseline_beta.values()) # concatenate all values, which are Dataframe rows for each subject

            postopBaseline_lowBeta = {k: v for k, v in postopBaseline_dict.items() if "_lowBeta" in k} # select all keys and values that contain substring beta
            postopBaseline_lowBeta_All = pd.concat(postopBaseline_lowBeta.values()) # concatenate all values, which are Dataframe rows for each subject

            postopBaseline_highBeta = {k: v for k, v in postopBaseline_dict.items() if "_highBeta" in k} # select all keys and values that contain substring beta
            postopBaseline_highBeta_All = pd.concat(postopBaseline_highBeta.values()) # concatenate all values, which are Dataframe rows for each subject


            ######## FU3M BASELINE ########
            fu3mBaseline_beta = {k: v for k, v in fu3mBaseline_dict.items() if "_beta" in k} # select all keys and values that contain substring beta
            fu3mBaseline_beta_All = pd.concat(fu3mBaseline_beta.values()) # concatenate all values, which are Dataframe rows for each subject

            fu3mBaseline_lowBeta = {k: v for k, v in fu3mBaseline_dict.items() if "_lowBeta" in k} # select all keys and values that contain substring beta
            fu3mBaseline_lowBeta_All = pd.concat(fu3mBaseline_lowBeta.values()) # concatenate all values, which are Dataframe rows for each subject

            fu3mBaseline_highBeta = {k: v for k, v in fu3mBaseline_dict.items() if "_highBeta" in k} # select all keys and values that contain substring beta
            fu3mBaseline_highBeta_All = pd.concat(fu3mBaseline_highBeta.values()) # concatenate all values, which are Dataframe rows for each subject

            


    
    return {
        "subject_dictionary": subject_dictionary,
        "postopBaseline_beta_All": postopBaseline_beta_All,
        "postopBaseline_lowBeta_All": postopBaseline_lowBeta_All,
        "postopBaseline_highBeta_All": postopBaseline_highBeta_All,
        "fu3mBaseline_beta_All": fu3mBaseline_beta_All,
        "fu3mBaseline_lowBeta_All": fu3mBaseline_lowBeta_All,
        "fu3mBaseline_highBeta_All": fu3mBaseline_highBeta_All
    }
   
