""" Group all PSD monopolar averages and ranks """



import os
import pandas as pd


# PyPerceive Imports
import PerceiveImport.methods.find_folders as findfolders


def group_monopolarPSDaverageAndRanks(incl_sub:list, incl_hemisphere:list, normalization:str,):

    """

    This function concatenates the result DataFrames of each subject into a group DataFrame

    Input:

        - incl_sub: ["019", "024", "025", "026", "029", "030"]
        - incl_hemisphere: list e.g. ["Right", "Left"]
        - normalization: str "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    ToDo: correct findfolders so file will be saved in results folder without giving a subject input
    """

    frequencyBand = ["beta", "lowBeta", "highBeta"]
    
    # path to results folder
    cwd = os.getcwd()
    while os.path.dirname(cwd)[-5:] != 'Users':
        cwd= os.path.dirname(cwd)
    
    results_path_general = os.path.join(cwd, "Research", "Longterm_beta_project", "results")

    # dictionary to fill in for each subject
    subject_dictionary = {}
    Rank_dict = {}
    monopolPsd_dict = {}

    ################### Load csv files from results folder for each subject ###################
    for sub, subject in enumerate(incl_sub):
        for hem, hemisphere in enumerate(incl_hemisphere):

            # get path to results folder of each subject
            local_results_path = findfolders.get_local_path(folder="results", sub=subject)

            # Get all csv filenames from each subject
            filenames = [
                f"monopolarReference_{normalization}_{hemisphere}",
                f"monopolarRanks_{normalization}_{hemisphere}"
                         ]
            
            # Error Checking: check if files exist
            for f, file in enumerate(filenames):
                try: 
                    os.path.isdir(os.path.join(local_results_path, file))
                
                except OSError:
                    continue

            # get the csv results for all monopolar PSD averages and Ranks (JLB)
            monopolPsd = pd.read_csv(os.path.join(local_results_path,filenames[0]), sep=",")
            monopolPsd.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True) # rename the column to Monopolar contact

            monopolRanks= pd.read_csv(os.path.join(local_results_path,filenames[1]), sep=",")
            monopolRanks.rename(columns={"Unnamed: 0": "Monopolar_contact"}, inplace=True) 

            # store all Dataframes in a subject dictionary
            subject_dictionary[f"sub{subject}_{hemisphere}_monopolPsd"] = monopolPsd
            subject_dictionary[f"sub{subject}_{hemisphere}_monopolRanks"] = monopolRanks
            

            # loop over every frequency band and filter Dataframes according to Rank or Psd of every frequency band
            for ind, freq in enumerate(frequencyBand):

                ######## MONOPOLAR PSD AVERAGE ########
                # boolean for each frequency band columns
                frequencyBand_booleanPSD = monopolPsd.filter(like=f"_{freq}")

                # first column = Monopolar_contact
                contact_columnPsd = monopolPsd.iloc[:,0]

                # concatenate the contact column to the filtered Dataframe 
                filteredPSD_DF = pd.concat([contact_columnPsd, frequencyBand_booleanPSD], axis=1)
                filteredPSD_DF.insert(0, "subject_hemisphere", f"sub{subject}_{hemisphere}", True) # insert new column as first column with subject and hemisphere details

                # store the filtered Dataframe of each frequency in monopolPsd_dictionary
                monopolPsd_dict[f"monopolPsd_{subject}_{hemisphere}_{freq}"] = filteredPSD_DF


                ######## MONOPOLAR RANKS ########
                 # boolean for each frequency band column 
                frequencyBand_booleanRanks = monopolRanks.filter(like=f"_{freq}")

                # first column = Monopolar_contact
                contact_columnRanks = monopolRanks.iloc[:,0]

                # concatenate the contact column to the filtered Dataframe 
                filteredRank_DF = pd.concat([contact_columnRanks, frequencyBand_booleanRanks], axis=1)
                filteredRank_DF.insert(0, "subject_hemisphere", f"sub{subject}_{hemisphere}", True) # insert new column as first column with subject and hemisphere details

                # store the filtered Dataframe of each frequency in monopolRanks
                Rank_dict[f"monopolRanks_{subject}_{hemisphere}_{freq}"] = filteredRank_DF


    # now concatenate all Dataframes from all subjects and hemispheres for each PSD or Rank frequency band combination
    
    ######## MONOPOLAR PSD AVERAGE ########
    # select only the Dataframes of each frequency band from the monopolar psd dictionary
    beta_sel_PSD = {k: v for k, v in monopolPsd_dict.items() if "_beta" in k}
    lowBeta_sel_PSD = {k: v for k, v in monopolPsd_dict.items() if "_lowBeta" in k}
    highBeta_sel_PSD = {k: v for k, v in monopolPsd_dict.items() if "_highBeta" in k}

    # concatenate all subject Dataframes of each frequency band
    monopolPSD_beta_ALL = pd.concat(beta_sel_PSD.values(), keys=beta_sel_PSD.keys(), ignore_index=True)
    monopolPSD_lowBeta_ALL = pd.concat(lowBeta_sel_PSD.values(), keys=lowBeta_sel_PSD.keys(), ignore_index=True)
    monopolPSD_highBeta_ALL = pd.concat(highBeta_sel_PSD.values(), keys=highBeta_sel_PSD.keys(), ignore_index=True)


    ######## MONOPOLAR RANKS ########
    # select only the Dataframes of each frequency band from the monopolar ranks dictionary
    beta_sel_Rank = {k: v for k, v in Rank_dict.items() if "_beta" in k}
    lowBeta_sel_Rank = {k: v for k, v in Rank_dict.items() if "_lowBeta" in k}
    highBeta_sel_Rank = {k: v for k, v in Rank_dict.items() if "_highBeta" in k}

    # concatenate all subject Dataframes of each frequency band
    monopolRanks_beta_ALL = pd.concat(beta_sel_Rank.values(), keys=beta_sel_Rank.keys(), ignore_index=True)
    monopolRanks_lowBeta_ALL = pd.concat(lowBeta_sel_Rank.values(), keys=lowBeta_sel_Rank.keys(), ignore_index=True)
    monopolRanks_highBeta_ALL = pd.concat(highBeta_sel_Rank.values(), keys=highBeta_sel_Rank.keys(), ignore_index=True)


    # save Dataframes as csv in the results folder
    monopolPSD_beta_ALL.to_csv(os.path.join(results_path_general,f"monopolPSD_beta_ALL{normalization}"), sep=",")
    monopolPSD_lowBeta_ALL.to_csv(os.path.join(results_path_general,f"monopolPSD_lowBeta_ALL{normalization}"), sep=",")
    monopolPSD_highBeta_ALL.to_csv(os.path.join(results_path_general,f"monopolPSD_highBeta_ALL{normalization}"), sep=",")
    monopolRanks_beta_ALL.to_csv(os.path.join(results_path_general,f"monopolRanks_beta_ALL{normalization}"), sep=",")
    monopolRanks_lowBeta_ALL.to_csv(os.path.join(results_path_general,f"monopolRanks_lowBeta_ALL{normalization}"), sep=",")
    monopolRanks_highBeta_ALL.to_csv(os.path.join(results_path_general,f"monopolRanks_highBeta_ALL{normalization}"), sep=",")
    



    return {
       "subject_dictionary": subject_dictionary,
       "monopolPSD_beta_ALL": monopolPSD_beta_ALL,
       "monopolPSD_lowBeta_ALL": monopolPSD_lowBeta_ALL,
       "monopolPSD_highBeta_ALL": monopolPSD_highBeta_ALL,
       "monopolRanks_beta_ALL": monopolRanks_beta_ALL,
       "monopolRanks_lowBeta_ALL": monopolRanks_lowBeta_ALL,
       "monopolRanks_highBeta_ALL": monopolRanks_highBeta_ALL

    }