""" Write clean Group Dataframes and save as Pickle """

import os
import pickle

import pandas as pd
import json
import numpy as np

from ..  tfr import feats_ssd as feats_ssd
######### PRIVATE PACKAGES #########
from .. classes import mainAnalysis_class as mainAnalysis_class
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults
# PyPerceive Imports
# import py_perceive
from PerceiveImport.classes import main_class


def write_BIPChannelGroups_ALLpsd(
        incl_sub: list, 
        signalFilter: str,
        normalization: str,
        freqBand: str
):

    """
    Read from Dataclasses and write Group Dataframe containing: 
        - session
        - subject_hemisphere
        - absoluteOrRelativePSD
        - PSDinfreq
        - recording_montage
        - recording_montage_group (circular, segm_intralevel or segm_interlevel)
        - recording_montage_specificGroup (circular_ring_ring, circular_ring_segm_long, circular_ring_segm_short, circular_segm_segm, 
                                            segm_intralevel_1, segm_intralevel_2, segm_interlevel)
    
    Input:
        - sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: list e.g. ["beta", "highBeta", "lowBeta"]
    
    Output: 
        - saving Dataframe as .pickle file
        filename: "BIPChannelGroups_ALL_{freqBand}_{normalization}_{signalFilter}.pickle"
    
    """


    results_path = find_folders.get_local_path(folder="GroupResults")

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    circular = ["03", "13", "02", "12", "01", "23"]
    segm_intralevel = ["1A1B", "1B1C", "1A1C", "2A2B", "2B2C", "2A2C"]
    segm_interlevel = ["1A2A", "1B2B", "1C2C"]

    circular_ring_ring = ["03"]
    circular_ring_segm_long = ["02", "13"]
    circular_ring_segm_short = ["01", "23"]
    circular_segm_segm = ["12"]

    segm_intralevel_1 = ["1A1B", "1B1C", "1A1C"]
    segm_intralevel_2 = ["2A2B", "2B2C", "2A2C"]



    psdAverage_dataframe = pd.DataFrame()

    ##################### LOAD DATA for all subject hemispheres #####################
    for sub in incl_sub:

        for hem in hemispheres:

            # load the data from each subject hemisphere
            mainClass_data = mainAnalysis_class.MainClass(
                sub = sub,
                hemisphere = hem,
                filter = signalFilter,
                result = "PSDaverageFrequencyBands",
                incl_session = sessions,
                pickChannels = ['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization = [normalization],
                freqBands = [freqBand],
                feature= ["averagedPSD"]
            )
    

            # for each timepoint and frequency band seperately, get the Dataframes of the correct normalization and frequency band
            for ses in sessions:

                # get the Dataframe
                if ses == "postop":
                    # Error check: check if session exists as an attribute of one subject_hemisphere object
                    try: 
                        mainClass_data.postop
                    
                    except AttributeError:
                        continue
                        
                    session_DF = mainClass_data.postop.Result_DF # select for session
                    
                # get the Dataframe
                if ses == "fu3m":
                    # Error check: check if session exists as an attribute of one subject_hemisphere object
                    try: 
                        mainClass_data.fu3m
                    
                    except AttributeError:
                        continue
                        
                    session_DF = mainClass_data.fu3m.Result_DF # select for session
                    
                # get the Dataframe
                if ses == "fu12m":
                    # Error check: check if session exists as an attribute of one subject_hemisphere object
                    try: 
                        mainClass_data.fu12m
                    
                    except AttributeError:
                        continue
                        
                    session_DF = mainClass_data.fu12m.Result_DF # select for session
                    
                # get the Dataframe
                if ses == "fu18m":
                    # Error check: check if session exists as an attribute of one subject_hemisphere object
                    try: 
                        mainClass_data.fu18m
                    
                    except AttributeError:
                        continue
                        
                    session_DF = mainClass_data.fu18m.Result_DF # select for session
                    
                ############# divide each Dataframe in 3 groups: Ring, SegmIntra, SegmInter #############
                session_DF_copy = session_DF.copy()

                # add column subject_hemisphere
                session_DF_copy["subject_hemisphere"] = f"{sub}_{hem}"

                # add column recording_montage
                for index, row in session_DF_copy.iterrows():
                    bipolarChannel = row["bipolarChannel"].split("_")
                    recording_montage = bipolarChannel[2] # just take 03, 02, etc from bipolarChannel column

                    session_DF_copy.loc[index, "recording_montage"] = recording_montage
                

                # rename column averagedPSD to beta_psd
                session_DF_copy = session_DF_copy.rename(columns={"averagedPSD": f"{freqBand}_psd"})

                # only get rows with correct normalization and frequency band 
                session_DF_copy = session_DF_copy[session_DF_copy.absoluteOrRelativePSD == normalization]
                session_DF_copy = session_DF_copy[session_DF_copy.frequencyBand == freqBand]

                # drop columns bipolarChannel, 
                session_DF_copy = session_DF_copy.drop(columns=["bipolarChannel", "frequencyBand"])

                # add column recording_montage_group
                for index, row in session_DF_copy.iterrows():
                    if row["recording_montage"] in circular:
                        session_DF_copy.loc[index, "recording_montage_group"] = "circular"
                    
                    elif row["recording_montage"] in segm_intralevel:
                        session_DF_copy.loc[index, "recording_montage_group"] = "segm_intralevel"
                    
                    elif row["recording_montage"] in segm_interlevel:
                        session_DF_copy.loc[index, "recording_montage_group"] = "segm_interlevel"
                
                # add column recording_montage_specificGroup
                for index, row in session_DF_copy.iterrows():
                    if row["recording_montage"] in circular_ring_ring:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "circular_ring_ring"
                    
                    elif row["recording_montage"] in circular_segm_segm:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "circular_segm_segm"
                    
                    elif row["recording_montage"] in circular_ring_segm_short:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "circular_ring_segm_short"
                    
                    elif row["recording_montage"] in circular_ring_segm_long:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "circular_ring_segm_long"
                    
                    elif row["recording_montage"] in segm_intralevel_1:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "segm_intralevel_1"
                    
                    elif row["recording_montage"] in segm_intralevel_2:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "segm_intralevel_2"
                    
                    elif row["recording_montage"] in segm_interlevel:
                        session_DF_copy.loc[index, "recording_montage_specificGroup"] = "segm_interlevel"
                    


                # concatenate all dataframes together
                psdAverage_dataframe = pd.concat([psdAverage_dataframe, session_DF_copy], ignore_index=True)

    ### save the Dataframes with pickle 
    BIPChannelGroups_ALLpsd_filepath = os.path.join(results_path, f"BIPChannelGroups_ALL_{freqBand}_{normalization}_{signalFilter}.pickle")
    with open(BIPChannelGroups_ALLpsd_filepath, "wb") as file:
        pickle.dump(psdAverage_dataframe, file)

    return {
        "psdAverage_dataframe": psdAverage_dataframe

    }





def write_BIPChannelGroups_psdRanks_relToRank1(
        signalFilter: str,
        normalization: str,
        freqBand: str
):

    """
    Read from existing pickle file "BIPChannelGroups_ALL_{freqBand}_{normalization}_{signalFilter}.pickle" 
        - session
        - subject_hemisphere
        - absoluteOrRelativePSD
        - PSDinfreq
        - recording_montage
        - recording_montage_group (circular, segm_intralevel or segm_interlevel)
        - recording_montage_specificGroup (circular_ring_ring, circular_ring_segm_long, circular_ring_segm_short, circular_segm_segm, 
                                            segm_intralevel_1, segm_intralevel_2, segm_interlevel)
    
    Input:
        - sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: list e.g. ["beta", "highBeta", "lowBeta"]

    1) divides Dataframe into different recording montage groups: "circular", "segm_intralevel", "segm_interlevel"

    2) filters in a loop: 
        - unique subject hemispheres = single STN DF
        - unique session from each single STN DF

    3) adds column "rank" with ranks of freq_psd within each filtered DF
        - ranks within Rings (01, 12, 23), SegmIntra and SegmInter seperately

    3) adds column "relativePSD_to_{freqBand}_Rank1" 
        - calculates relative PSD to beta rank 1 within each group (rank1 PSD = 1.0)
        - adds relative PSD value to each row accordingly 
    
    Output: 
        - saving Dataframe as .pickle file
        filename: "BIPChannelGroups_psdRanks_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle"
    
    """


    results_path = find_folders.get_local_path(folder="GroupResults")


    recording_montage_group = ["circular", "segm_intralevel", "segm_interlevel"]

    psdRank_and_relative_dataframe = pd.DataFrame()

    ##################### LOAD DATA for all subject hemispheres #####################

    loaded_Dataframe = loadResults.load_BIPChannelGroups_ALL(
        freqBand=freqBand,
        normalization=normalization,
        signalFilter=signalFilter
        )
    
    sub_hem_keys = list(loaded_Dataframe["subject_hemisphere"].unique())
    
     # Edit all Dataframes from 3 recording_montage_groups: add column with ranks and additionally relative PSD to rank1 
    for group in recording_montage_group:

        #first select DF containing all rows from one STN
        for STN in sub_hem_keys: 
            singleSTN_dataframe = loaded_Dataframe[loaded_Dataframe["subject_hemisphere"] == STN]

            # then select for only one existing session
            sessions_per_STN = list(singleSTN_dataframe["session"].unique())

            for ses in sessions_per_STN:
                STN_session_dataframe = singleSTN_dataframe[singleSTN_dataframe["session"] == ses]
    
                #now from each recording montage group filter only for one group 
               
                if group == "circular":

                    dataframe_to_rank =  STN_session_dataframe[STN_session_dataframe["recording_montage_group"] == "circular"] # containing 6 circular BIP channels
                    
                    # drop the BIP channels which contacts are not adjacent 
                    dataframe_to_rank = dataframe_to_rank[dataframe_to_rank.recording_montage.str.contains("03|13|02") == False]
                
                elif group == "segm_intralevel":

                    dataframe_to_rank =  STN_session_dataframe[STN_session_dataframe["recording_montage_group"] == "segm_intralevel"] # containing 6 circular BIP channels
                    
                elif group == "segm_interlevel":

                    dataframe_to_rank =  STN_session_dataframe[STN_session_dataframe["recording_montage_group"] == "segm_interlevel"] # containing 6 circular BIP channels
                    

                # add column rank to DF 
                if freqBand == "beta":
                    rank_column = dataframe_to_rank.beta_psd.rank(ascending=False) # rank highest psdAverage = 1.0
                
                elif freqBand == "highBeta":
                    rank_column = dataframe_to_rank.highBeta_psd.rank(ascending=False) # rank highest psdAverage = 1.0
                
                elif freqBand == "lowBeta":
                    rank_column = dataframe_to_rank.lowBeta_psd.rank(ascending=False) # rank highest psdAverage = 1.0
                
                # add column with ranks to DF
                copyDF = dataframe_to_rank.copy() # has to be copied, otherwise can´t add a new column
                copyDF["rank"] = rank_column # add new column "rank" to the copied DF

                # add column with PSD relative to rank 1 PSD to DF

                # if rank = 1.0 define averagedPSD value of the same row as beta_rank_1
                beta_rank_1 = copyDF[copyDF["rank"] == 1.0] # taking the row containing 1.0 in rank
                beta_rank_1 = beta_rank_1[f"{freqBand}_psd"].values[0] # just taking psdAverage of rank 1.0

                copyDF[f"relativePSD_to_{freqBand}_Rank1"] = copyDF.apply(lambda row: row[f"{freqBand}_psd"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1

                
                # concatenate all dataframes together
                psdRank_and_relative_dataframe = pd.concat([psdRank_and_relative_dataframe, copyDF], ignore_index=True)
                

    ### save the Dataframes with pickle 
    BIPChannelGroups_psdRanks_relToRank1_filepath = os.path.join(results_path, f"BIPChannelGroups_psdRanks_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle")
    with open(BIPChannelGroups_psdRanks_relToRank1_filepath, "wb") as file:
        pickle.dump(psdRank_and_relative_dataframe, file)
    
    print("file: ", 
          f"BIPChannelGroups_psdRanks_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle",
          "\nwritten in: ", results_path)

    return {
        "psdRank_and_relative_dataframe": psdRank_and_relative_dataframe

    }




def write_GroupMonopolar_weightedPsdCoordinateDistance_relToRank1(
        incl_sub: list,
        signalFilter: str,
        normalization: str,
        freqBand: str
):

    """
    Read from existing pickle file "sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle"
    coumns:
        - index (=Contact)
        - coord_z
        - coord_xy
        - subject_hemisphere
        - session
        - averaged_monopolar_PSD_beta
        - rank
    
    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    1) Load for each subject, hemisphere and session single Dataframes from the given pickle file
    
    2) from index -> add new column with contacts "contact"

    2) add new column:  "relativePSD_to_{freqBand}_Rank1"
        - calculates relative PSD to beta rank 1 within each group (rank1 PSD = 1.0)
        - adds relative PSD value to each row accordingly 
    
    3) concatenate all dataframes from single subject_hemispheres at single sessions together

  
    Output: 
        - saving Dataframe as .pickle file
        filename: GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle
    
    """


    results_path = find_folders.get_local_path(folder="GroupResults")

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]


    relToRank1_dataframe = pd.DataFrame()

    for sub in incl_sub:

        for hem in hemispheres:

            # load the pickle file "sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle 
            # from each subject folder
            sub_hem_result = loadResults.load_monoRef_weightedPsdCoordinateDistance_pickle(
                sub=sub,
                hemisphere=hem,
                freqBand=freqBand,
                normalization=normalization,
                filterSignal=signalFilter
            )

            for ses in sessions:

                # check if session exists for a subject
                sub_session_keys = list(sub_hem_result.keys()) # list of fu3m_monopolar_Dataframe, fu12m_bipolar_Dataframe
                combined_sub_session_keys = "_".join(sub_session_keys)

                if ses not in combined_sub_session_keys:
                    continue
                

                # load the Dataframe of a single session from the given subject hemisphere
                session_DF = sub_hem_result[f"{ses}_monopolar_Dataframe"]

                # add a column "contact" from the indeces values
                session_DF = session_DF.rename_axis("contact").reset_index()
                session_DF_copy = session_DF.copy()

                # add column with PSD relative to rank 1 PSD to DF
                # if rank = 1.0 define averagedPSD value of the same row as beta_rank_1
                beta_rank_1 = session_DF_copy[session_DF_copy["rank"] == 1.0] # taking the row containing 1.0 in rank
                beta_rank_1 = beta_rank_1[f"averaged_monopolar_PSD_{freqBand}"].values[0] # just taking psdAverage of rank 1.0

                session_DF_copy[f"relativePSD_to_{freqBand}_Rank1"] = session_DF_copy.apply(lambda row: row[f"averaged_monopolar_PSD_{freqBand}"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1

                
                # concatenate all dataframes together
                relToRank1_dataframe = pd.concat([relToRank1_dataframe, session_DF_copy], ignore_index=True)
    
    ### save the Dataframes with pickle 
    relToRank1_dataframe_filepath = os.path.join(results_path, f"GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle")
    with open(relToRank1_dataframe_filepath, "wb") as file:
        pickle.dump(relToRank1_dataframe, file)
    
    print("file: ", 
          f"GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_{freqBand}_{normalization}_{signalFilter}.pickle",
          "\nwritten in: ", results_path)

    return relToRank1_dataframe



def write_Group_monoRef_only_segmental_weight_psd_by_distance(
        incl_sub: list,
        signalFilter: str,
        normalization: str,
        freqBand: str
):

    """
    Read from existing pickle file "sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"
    
    coumns:
        - coord_z
        - coord_xy
        - subject_hemisphere
        - session
        - estimated_monopolar_psd_beta
        - contact
        - rank
    
    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "029", "030", "031", "032", "033", "037", "038", "041", "045"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    1) Load for each subject, hemisphere and session single Dataframes from the given pickle file
    
    2) from index -> add new column with contacts "contact"

    2) add new column:  "relativePSD_to_{freqBand}_Rank1"
        - calculates relative PSD to beta rank 1 within each group (rank1 PSD = 1.0)
        - adds relative PSD value to each row accordingly 
    
    3) concatenate all dataframes from single subject_hemispheres at single sessions together

  
    Output: 
        - saving Dataframe as .pickle file in results path
        filename: group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{signalFilter}.pickle
    
    """


    results_path = find_folders.get_local_path(folder="GroupResults")

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]


    relToRank1_dataframe = pd.DataFrame()

    for sub in incl_sub:

        for hem in hemispheres:

            # load the pickle file "sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle 
            # from each subject folder
            sub_hem_result = loadResults.load_monoRef_only_segmental_weight_psd_by_distance(
                sub=sub,
                hemisphere=hem,
                freqBand=freqBand,
                normalization=normalization,
                filterSignal=signalFilter
            )

            for ses in sessions:

                # check if session exists for a subject
                sub_session_keys = list(sub_hem_result.keys()) # list of fu3m_monopolar_Dataframe, fu12m_bipolar_Dataframe
                combined_sub_session_keys = "_".join(sub_session_keys)

                if ses not in combined_sub_session_keys:
                    continue
                

                # load the Dataframe of a single session from the given subject hemisphere
                session_DF = sub_hem_result[f"{ses}_monopolar_Dataframe"]

                # reset index
                session_DF = session_DF.reset_index()
                session_DF_copy = session_DF.copy()

                # add column with PSD relative to rank 1 PSD to DF
                # if rank = 1.0 define averagedPSD value of the same row as beta_rank_1
                beta_rank_1 = session_DF_copy[session_DF_copy["rank"] == 1.0] # taking the row containing 1.0 in rank
                beta_rank_1 = beta_rank_1[f"estimated_monopolar_psd_{freqBand}"].values[0] # just taking psdAverage of rank 1.0

                session_DF_copy[f"relativePSD_to_{freqBand}_Rank1"] = session_DF_copy.apply(lambda row: row[f"estimated_monopolar_psd_{freqBand}"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1

                
                # concatenate all dataframes together
                relToRank1_dataframe = pd.concat([relToRank1_dataframe, session_DF_copy], ignore_index=True)
    
    ### save the Dataframes with pickle 
    relToRank1_dataframe_filepath = os.path.join(results_path, f"group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{signalFilter}.pickle")
    with open(relToRank1_dataframe_filepath, "wb") as file:
        pickle.dump(relToRank1_dataframe, file)
    
    print("file: ", 
          f"group_monoRef_only_segmental_weight_psd_by_distance_{freqBand}_{normalization}_{signalFilter}.pickle",
          "\nwritten in: ", results_path)

    return relToRank1_dataframe



def write_monopol_rel_psd_from0To8(
        incl_sub: list,
        signalFilter: str,
        normalization: str,
        freqBand: str
):

    """
    Read from existing pickle file "sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle"
    coumns:
        - index (=Contact)
        - coord_z
        - coord_xy
        - subject_hemisphere
        - session
        - averaged_monopolar_PSD_beta
        - rank
    
    Input:
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - signalFilter: str "unfiltered", "band-pass"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - freqBand: str e.g. "beta", "highBeta", "lowBeta"

    1) Load for each subject, hemisphere and session single Dataframes from the given pickle file
    
    2) from index -> add new column with contacts "contact"

    2) add new column:  "relativePSD_to_{freqBand}_Rank1"
        - calculates relative PSD to beta rank 1 within each group (rank1 PSD = 1.0)
        - adds relative PSD value to each row accordingly 
    
    3) concatenate all dataframes from single subject_hemispheres at single sessions together

    4) take out contacts 1 and 2, rank only 8 contacts

    5) calculate the relative Beta values, so the contact with the highest value = 1.0 and the contact with lowest value = 0
        - subtract the PSD value of rank 8 contact from all contact values, so value of lowest ranked contact = 0
        - subsequently, divide all contact values by the value of contact ranked 1

    6) add another column with PSD noramlized to mean and standard deviation
        - (psd - mean) / standard deviation


    7) Load clinical stimulation parameters from Excel Sheet "BestClinicalContacts"
        and columns to the dataframe accordingly:
        - currentPolarity
        - clinicalUse

        
    Output: 
        - saving Dataframe as .pickle file
        filename: monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle
    
    """


    results_path = find_folders.get_local_path(folder="GroupResults")

    hemispheres = ["Right", "Left"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    contacts_8 = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]


    relToRank1_dataframe = pd.DataFrame()

    for sub in incl_sub:

        for hem in hemispheres:

            # load the pickle file "sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle 
            # from each subject folder
            sub_hem_result = loadResults.load_monoRef_weightedPsdCoordinateDistance_pickle(
                sub=sub,
                hemisphere=hem,
                freqBand=freqBand,
                normalization=normalization,
                filterSignal=signalFilter
            )

            for ses in sessions:

                # check if session exists for a subject
                sub_session_keys = list(sub_hem_result.keys()) # list of fu3m_monopolar_Dataframe, fu12m_bipolar_Dataframe
                combined_sub_session_keys = "_".join(sub_session_keys)

                if ses not in combined_sub_session_keys:
                    continue
                

                # load the Dataframe of a single session from the given subject hemisphere
                session_DF = sub_hem_result[f"{ses}_monopolar_Dataframe"]

                # add a column "contact" from the indeces values
                session_DF = session_DF.rename_axis("contact").reset_index()
                session_DF_copy = session_DF.copy()

                # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
                session_DF_copy = session_DF_copy[session_DF_copy["contact"].isin(contacts_8)]
                session_DF_copy["Rank8contacts"] = session_DF_copy["averaged_monopolar_PSD_beta"].rank(ascending=False)
                session_DF_copy.drop(["rank"], axis=1, inplace=True)

                #################### add column with PSD relative to rank 1 and 8 PSD ####################
                # value of rank 8 
                beta_rank_8 = session_DF_copy[session_DF_copy["Rank8contacts"] == 8.0]
                beta_rank_8 = beta_rank_8[f"averaged_monopolar_PSD_{freqBand}"].values[0] # just taking psdAverage of rank 8.0

                # value of rank 1 after subtracting the value of rank 8  
                beta_rank_1 = session_DF_copy[session_DF_copy["Rank8contacts"] == 1.0] # taking the row containing 1.0 in Rank8contacts
                beta_rank_1 = beta_rank_1[f"averaged_monopolar_PSD_{freqBand}"].values[0] # just taking psdAverage of rank 1.0
                beta_rank_1 = beta_rank_1 - beta_rank_8 # this is necessary to get value 1.0 after dividng the subtracted PSD value of rank 1 by itself

                # in each row add in new column: (psd-beta_rank_8)/beta_rank1
                session_DF_copy[f"relativePSD_{freqBand}_from_0_to_1"] = session_DF_copy.apply(lambda row: (row[f"averaged_monopolar_PSD_{freqBand}"] - beta_rank_8) / beta_rank_1, axis=1) 

                #################### add new column with relPSD standardized to mean and std ####################
                mean = session_DF_copy[f"averaged_monopolar_PSD_{freqBand}"].mean()
                std = session_DF_copy[f"averaged_monopolar_PSD_{freqBand}"].std()

                # in each row add in new column: (psd-beta_rank_8)/beta_rank1
                session_DF_copy[f"relativePSD_{freqBand}_to_mean_std"] = session_DF_copy.apply(lambda row: (row[f"averaged_monopolar_PSD_{freqBand}"] - mean) / std, axis=1) 

                
                # concatenate all dataframes together
                relToRank1_dataframe = pd.concat([relToRank1_dataframe, session_DF_copy], ignore_index=True)


    #################### LOAD CLINICAL STIMULATION PARAMETERS #####################

    bestClinicalStim_file = loadResults.load_BestClinicalStimulation_excel()

    # get sheet with best clinical contacts
    BestClinicalContacts = bestClinicalStim_file["BestClinicalContacts"]



    ##################### FILTER THE MONOBETA8RANKS_DF: clinically ACTIVE contacts #####################
    activeMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():

        activeContacts = str(BestClinicalContacts.CathodalContact.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        activeContacts_list = activeContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]
        currentPolarity = BestClinicalContacts.currentPolarity.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = relToRank1_dataframe.loc[(relToRank1_dataframe["subject_hemisphere"]==sub_hem) & (relToRank1_dataframe["session"]==session) & (relToRank1_dataframe["contact"].isin(activeContacts_list))]
        # add a column and fill the cell with the current Polarity
        sub_hem_ses_rows_copy = sub_hem_ses_rows.copy()
        sub_hem_ses_rows_copy["currentPolarity"] = currentPolarity
        
        # concatenate single rows to new Dataframe
        activeMonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, sub_hem_ses_rows_copy], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "active"
    activeMonoBeta8Ranks["clinicalUse"] = "active"


    ##################### FILTER THE MONOBETA8RANKS_DF: clinically INACTIVE contacts #####################
    inactiveMonoBeta8Ranks = pd.DataFrame()

    for idx, row in BestClinicalContacts.iterrows():

        inactiveContacts = str(BestClinicalContacts.InactiveContacts.values[idx]) # e.g. "2A_2B_2C_3"
        # split active Contacts into list with single contact strings
        inactiveContacts_list = inactiveContacts.split("_") # e.g. ["2A", "2B", "2C", "3"]

        sub_hem = BestClinicalContacts.subject_hemisphere.values[idx]
        session = BestClinicalContacts.session.values[idx]

        # get rows with equal sub_hem and session and with monopolarChannel in the list of activeContacts
        sub_hem_ses_rows = relToRank1_dataframe.loc[(relToRank1_dataframe["subject_hemisphere"]==sub_hem) & (relToRank1_dataframe["session"]==session) & (relToRank1_dataframe["contact"].isin(inactiveContacts_list))]
        # add a column and fill the cell with the current Polarity
        sub_hem_ses_rows_copy = sub_hem_ses_rows.copy()
        sub_hem_ses_rows_copy["currentPolarity"] = "0"

        # concatenate single rows to new Dataframe
        inactiveMonoBeta8Ranks = pd.concat([inactiveMonoBeta8Ranks, sub_hem_ses_rows_copy], ignore_index=True)

    # add a column "clinicalUse" to the Dataframe and fill with "non_active"
    inactiveMonoBeta8Ranks["clinicalUse"] = "inactive"



    ##################### CONCATENATE BOTH DATAFRAMES: CLINICALLY ACTIVE and INACTIVE CONTACTS #####################
    active_and_inactive_MonoBeta8Ranks = pd.concat([activeMonoBeta8Ranks, inactiveMonoBeta8Ranks], ignore_index=True)


    ### save the Dataframes with pickle 
    active_and_inactive_MonoBeta8Ranks_filepath = os.path.join(results_path, f"monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle")
    with open(active_and_inactive_MonoBeta8Ranks_filepath, "wb") as file:
        pickle.dump(active_and_inactive_MonoBeta8Ranks, file)
    
    print("file: ", 
          f"monopol_rel_psd_from0To8_{freqBand}_{normalization}_{signalFilter}.pickle",
          "\nwritten in: ", results_path)

    return active_and_inactive_MonoBeta8Ranks




def write_SSD_filtered_groupChannels(
        incl_sub=list,
        f_band=str
):
    """
    Loads unfiltered raw time domain data with PyPerceive

    Input:
        - incl_sub = ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - f_range = tuple e.g. beta: 13, 35
    
    1) Load the BSSu rest Time domain data of each existing subject, hemisphere, session, and recording group (e.g. right: "RingR", "SegmIntraR", "SegmInterR")
        - in the Ring Group only choose bipolar recording montages with same distances between contacts (01, 12, 23) --> caveat: probably still not comparable because segmented contacts have higher impedance than circular contacts

    2) apply the SSD filter by using feats_ssd.get_SSD_component() function from Jeroen (based on Gunnar Waterstraat's meet Toolbox)
        for each recording group seperately!

        Input
            - data_2d: 2d array of channels x time domain data
            - fband_interest: tuple f_range e.g. 13, 35 (beta band)
            - s_rate: 250 Hz usually for BSSu
            - use_freqBand_filtered=True (uses only the filtered frequency band, not the flanks)
            - return_comp_n=0 -> only the first component is returned, which is the most important component
   
    3) Dataframe returned and saved: columns 
        - subject
        - hemisphere
        - session
        - recording_group (e.g. RingR)
        - bipolarChannel
        - ssd_filtered_timedomain (array)
        - ssd_pattern (weights of a channel contributing to the first component)
        - ssd_eigvals
    
    Dataframe saved in results folder: 
        - filename: "SSD_results_Dataframe_{f_band}.pickle"
    
    TODO: 
        - change ch_names to short name, only contacts e.g. 12
        - maybe add a new column with more specific recording montage
        - add new functionality to filter SSD for all 15 bipolar channels?? useful??

    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    # load the PSD data with classes
    hemisphere =["Right", "Left"]
    incl_session=["postop", "fu3m", "fu12m", "fu18m"]
    incl_cond=["m0s0"]

    incl_contact={}
    rec_group_input_array={}
    SSD_results_storage = {}

    # set the frequency range depending on the f_band of interest
    if f_band == "beta":
        f_range = [13,35]

    elif f_band == "lowBeta":
        f_range = [13,20]
    
    elif f_band == "highBeta":
        f_range = [21,35]
    


    for sub in incl_sub:

        sub_results_path = find_folders.get_local_path(folder="figures", sub=sub) 

        for hem in hemisphere:
            if hem == "Right":
                incl_contact["Right"] = ["RingR", "SegmIntraR", "SegmInterR"]

            elif hem == "Left":
                incl_contact["Left"] = ["RingL", "SegmIntraL", "SegmInterL"]

            mainclass_sub = main_class.PerceiveData(
                sub = sub, 
                incl_modalities= ["survey"],
                incl_session = incl_session,
                incl_condition = incl_cond,
                incl_task = ["rest"],
                incl_contact=incl_contact[f"{hem}"]
                )

            for ses in incl_session:
                for cond in incl_cond:
                    for recording_group in incl_contact[f"{hem}"]:

                        # check if session exists 
                        try: 
                            getattr(mainclass_sub.survey, ses)
                        except AttributeError:
                            continue

                        timedomain_data = getattr(mainclass_sub.survey, ses)
                        

                        # check if condition exists
                        try:
                            getattr(timedomain_data, cond)
                        except AttributeError:
                            continue

                        timedomain_data = getattr(timedomain_data, cond)
                        timedomain_data = getattr(timedomain_data.rest, recording_group) # for each rec group in incl_contact
                        timedomain_data = timedomain_data.run1.data

                        # channel names (list) and sampling frequency
                        ch_names = timedomain_data.ch_names
                        sfreq = timedomain_data.info["sfreq"]

                        # Time domain arrays of all channels in a recording group                
                        rec_group_data = timedomain_data.get_data() # 2D array of all rec channels in a rec group

                        # for Ring contacts only choose channels with adjacent contacts (01,12,23)
                        if (recording_group == "RingR" or recording_group == "RingL"):

                            rec_group_data = rec_group_data[[3,4,5], :]
                            ch_names = ch_names[3:6]


                        rec_group_input_array[f"{sub}_{hem}_{ses}_{cond}_{recording_group}"] = rec_group_data

                        # apply SSD filter on time domain of each recording group
                        (ssd_filt_data, ssd_pattern, ssd_eigvals 
                                    ) = feats_ssd.get_SSD_component( 
                                        data_2d=rec_group_data, 
                                        fband_interest=f_range, 
                                        s_rate=sfreq, 
                                        use_freqBand_filtered=True, 
                                        return_comp_n=0, ) 
                        
                        # store SSD outcome in Dataframe
                        for c, chan in enumerate(ch_names):
                            SSD_results_storage[f"{sub}_{hem}_{ses}_{chan}"] = [sub, hem, ses, recording_group, chan, ssd_filt_data, ssd_pattern[0][c], ssd_eigvals[c]]
                            # ssd_filt_data is ssd filtered time domain of a channel
                            # ssd_pattern[0] is weight of a channel, contributing to FIRST component 
                    

    SSD_results_Dataframe = pd.DataFrame(SSD_results_storage)
    SSD_results_Dataframe.rename(index={0:"subject", 1:"hemisphere", 2:"session", 3:"recording_group", 4:"bipolarChannel", 5:"ssd_filtered_timedomain", 6:"ssd_pattern", 7:"ssd_eigvals"}, inplace=True)
    SSD_results_Dataframe = SSD_results_Dataframe.transpose()

    ### save the Dataframes with pickle 
    SSD_results_Dataframe_filepath = os.path.join(results_path, f"SSD_results_Dataframe_{f_band}.pickle")
    with open(SSD_results_Dataframe_filepath, "wb") as file:
        pickle.dump(SSD_results_Dataframe, file)
    
    print("file: ", 
          f"SSD_results_Dataframe_{f_band}.pickle",
          "\nwritten in: ", results_path)

    return SSD_results_Dataframe


def write_fooof_group_json(incl_sub: list):

    """
    Load the file: "fooof_model_sub{subject}.json"
    from each subject result folder

    Input
        - incl_sub: list e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038", "041", "060"]]

    """
    # results folder for group results
    results_path_group = find_folders.get_local_path(folder="GroupResults")

    group_fooof_dataframe = pd.DataFrame()

    for sub in incl_sub:

        # find the path to the results folder
        results_path_sub = find_folders.get_local_path(folder="results", sub=sub)

        # create filename
        filename = f"fooof_model_sub{sub}.json"

        # check if file exists
        files_in_folder = os.listdir(results_path_sub) # list of all files in the sub results folder

        if filename not in files_in_folder:
            print(f"no file: fooof_model_sub{sub}.json in sub-{sub} results folder")
            continue

        elif filename in files_in_folder:
            # load the json file
            with open(os.path.join(results_path_sub, filename)) as file:
                data = json.load(file)
                data = pd.DataFrame(data)
            
            # concatenate all Dataframes together
            group_fooof_dataframe = pd.concat([group_fooof_dataframe, data], ignore_index=True)
    
    # save the group Dataframe into group results folder
    group_fooof_dataframe.to_json(os.path.join(results_path_group, f"fooof_model_group_data.json"))

    return group_fooof_dataframe


def highest_beta_channels_fooof(
        fooof_spectrum:str
):
    """
    Load the file "fooof_model_group_data.json"
    from the group result folder

    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

    1) calculate beta average for each channel and rank within 1 stn, 1 session and 1 channel group
    
    2) rank beta averages and only select the channels with rank 1.0 

    Output highest_beta_df
        - containing all stns, all sessions, all channels with rank 1.0 within their channel group
    
    """

    results_path = find_folders.get_local_path(folder="GroupResults")

    # load the group dataframe
    fooof_group_result = loadResults.load_group_fooof_result()

    # create new column: first duplicate column fooof power spectrum, then apply calculation to each row -> average of indices [13:36] so averaging the beta range
    fooof_group_result_copy = fooof_group_result.copy()

    if fooof_spectrum == "periodic_spectrum":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_power_spectrum"]
    
    elif fooof_spectrum == "periodic_plus_aperiodic":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["periodic_plus_aperiodic_power_log"]

    elif fooof_spectrum == "periodic_flat":
        fooof_group_result_copy["beta_average"] = fooof_group_result_copy["fooof_periodic_flat"]
    
    
    fooof_group_result_copy["beta_average"] = fooof_group_result_copy["beta_average"].apply(lambda row: np.mean(row[13:36]))


    ################################ WRITE DATAFRAME ONLY WITH HIGHEST BETA CHANNELS PER STN | SESSION | CHANNEL_GROUP ################################
    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    stn_unique = fooof_group_result_copy.subject_hemisphere.unique().tolist()

    highest_beta_df = pd.DataFrame()
    beta_ranks_all_channels = pd.DataFrame()

    for stn in stn_unique:

        stn_df = fooof_group_result_copy.loc[fooof_group_result_copy.subject_hemisphere == stn]

        for ses in sessions:

            # check if session exists
            if ses not in stn_df.session.values:
                continue

            else:
                stn_ses_df = stn_df.loc[stn_df.session == ses] # df of only 1 stn and 1 session

            for group in channel_group:

                if group == "ring":
                    channels = ['01', '12', '23']
                    
                elif group == "segm_inter":
                    channels = ["1A2A", "1B2B", "1C2C"]
                
                elif group == "segm_intra":
                    channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                group_comp_df = stn_ses_df.loc[stn_ses_df["bipolar_channel"].isin(channels)].reset_index() # df of only 1 stn, 1 session and 1 channel group

                # rank beta average of channels within one channel group
                group_comp_df_copy = group_comp_df.copy()
                group_comp_df_copy["beta_rank"] = group_comp_df_copy["beta_average"].rank(ascending=False) 

                beta_ranks = group_comp_df_copy.copy()

                # only keep the row with beta rank 1.0
                group_comp_df_copy = group_comp_df_copy.loc[group_comp_df_copy.beta_rank == 1.0]

                # save to ranked_beta_df
                beta_ranks_all_channels = pd.concat([beta_ranks_all_channels, beta_ranks])
                highest_beta_df = pd.concat([highest_beta_df, group_comp_df_copy])
    
    # save DF as pickle file
    highest_beta_df_filepath = os.path.join(results_path, f"highest_beta_channels_fooof_{fooof_spectrum}.pickle")
    with open(highest_beta_df_filepath, "wb") as file:
        pickle.dump(highest_beta_df, file)

    print("file: ", 
          f"highest_beta_channels_fooof_{fooof_spectrum}.pickle",
          "\nwritten in: ", results_path
          )
    
    # save DF as pickle file
    beta_ranks_all_channels_filepath = os.path.join(results_path, f"beta_ranks_all_channels_fooof_{fooof_spectrum}.pickle")
    with open(beta_ranks_all_channels_filepath, "wb") as file:
        pickle.dump(beta_ranks_all_channels, file)

    print("file: ", 
          f"beta_ranks_all_channels_fooof_{fooof_spectrum}.pickle",
          "\nwritten in: ", results_path
          )


    return {
        "highest_beta_df": highest_beta_df,
        "beta_ranks_all_channels": beta_ranks_all_channels
    }


def write_ses_comparison_power_spectra(
        incl_sub: list,
        incl_channels: str,
        signalFilter: str,
        normalization: str
):
    """

    Input:
        - incl_sub: list, e.g. ["017", "019", "021", "024", "025", "026", "028", "029", "030", "031", "032", "033", "038"]
        - incl_channels: str, e.g. "SegmInter", "SegmIntra", "Ring"
        - signalFilter: str, e.g. "band-pass" or "unfiltered"
        - normalization: str, e.g. "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
    
    1) Load the power spectra of each subject hemisphere (=STN) via classes
        - save each power spectrum for each session and each bipolar channel
        - save in Dataframe: single_channels_df

    2) for each session comparison select only the STNs with recordings at both sessions
        - comparisons: ["postop_fu3m", "postop_fu12m", "postop_fu18m", 
                        "fu3m_fu12m", "fu3m_fu18m", "fu12m_fu18m"]
        - create Dataframe with only STNs with both sessions: comparison_df
    
    3) store all comparison_df in a dictionary and save as file: "power_spectra_{signalFilter}_{incl_channels}_session_comparisons.pickle"
        - keys of the dictionary: 
        ["postop_fu3m_df", "postop_fu12m_df", "postop_fu18m_df", "fu3m_fu12m_df", "fu3m_fu18m_df", "fu12m_fu18m_df"]

    """

    # results folder for group results
    results_path = find_folders.get_local_path(folder="GroupResults")

    # variables
    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    hemispheres = ["Right", "Left"]


    if normalization == "rawPsd":
        feature_normalization = ["frequency", "time_sectors", "rawPsd", "SEM_rawPsd"]
    
    elif normalization == "normPsdToTotalSum":
        feature_normalization = ["frequency", "time_sectors", "normPsdToTotalSum", "SEM_normPsdToTotalSum"]
    
    elif normalization == "normPsdToSum1_100Hz":
        feature_normalization = ["frequency", "time_sectors", "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz"]
    
    elif normalization == "normPsdToSum40_90Hz":
        feature_normalization = ["frequency", "time_sectors", "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]
    


    if incl_channels == "SegmInter":
        channels = ["1A2A", "1B2B", "1C2C"]
    
    elif incl_channels == "SegmIntra":
        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
    
    elif incl_channels == "Ring":
        channels = ['01', '12', '23'] # taking the grand average of these channels probably not a good idea bc different impedances of contacts

    single_channels_dict = {}

    for sub in incl_sub:

        for hem in hemispheres:

            # load all sessions and selected channel data per STN
            stn_power_spectra = mainAnalysis_class.MainClass(
                    sub=sub,
                    hemisphere=hem,
                    filter=signalFilter,
                    result="PowerSpectrum",
                    incl_session=["postop", "fu3m", "fu12m", "fu18m"],
                    pickChannels=channels,
                    normalization=[normalization],
                    feature=feature_normalization
                )
            
            for ses in sessions:
                
                # check which sessions exist
                try:
                    getattr(stn_power_spectra, ses)

                except AttributeError:
                    continue

                for chan in channels: 
                        
                    # get the power spectra and frequencies from each channel
                    chan_data = getattr(stn_power_spectra, ses)
                    chan_data = getattr(chan_data, f"BIP_{chan}")
                    
                    if normalization == "rawPsd":
                        power_spectrum = np.array(chan_data.rawPsd.data)
    
                    elif normalization == "normPsdToTotalSum":
                        power_spectrum = np.array(chan_data.normPsdToTotalSum.data)
                    
                    elif normalization == "normPsdToSum1_100Hz":
                        power_spectrum = np.array(chan_data.normPsdToSumPsd1to100Hz.data)
                    
                    elif normalization == "normPsdToSum40_90Hz":
                        power_spectrum = np.array(chan_data.normPsdToSum40to90Hz.data)
                    
                    freqs = np.array(chan_data.frequency.data)

                    # save all channels of an STN in a dict
                    single_channels_dict[f"{sub}_{hem}_{ses}_{chan}"] = [sub, hem, ses, chan, power_spectrum, freqs, normalization]

    # Dataframe with all single channels and their power_spectra + frequencies
    single_channels_df = pd.DataFrame(single_channels_dict)
    single_channels_df.rename(index={
        0: "subject",
        1: "hemisphere",
        2: "session",
        3: "bipolar_channel",
        4: "power_spectrum",
        5: "frequencies",
        6: "normalization"
    }, inplace=True)
    single_channels_df = single_channels_df.transpose()


    # join sub, hem columns together -> stn
    single_channels_df["stn"] = single_channels_df[['subject', 'hemisphere']].agg('_'.join, axis=1)
    single_channels_df.drop(columns=['subject', 'hemisphere'], inplace=True)

    #################  AVERAGE OF ALL POWER SPECTRA WITHIN ONE STN  #################
    # averaged_across_stn_dict = {}

    # stn_unique = list(single_channels_df.stn.unique())

    # for stn in stn_unique:

    #     # filter the df and only get rows with stn
    #     stn_df = single_channels_df.loc[(single_channels_df["stn"]==stn)]

    #     for ses in sessions:

    #         # check if session exists 
    #         if ses not in stn_df.session.values:
    #             continue

    #         stn_session_df = stn_df.loc[(stn_df["session"]==ses)]

    #         # save one vector with frequencies (all the same)
    #         freqs = stn_session_df.freqencies.values[0]

    #         # calculate the grand average of all selected channels of one STN
    #         across_chans_average = np.mean(stn_session_df.power_spectrum.values)

    #         # save in average dict
    #         averaged_across_stn_dict[f"{stn}_{ses}_averaged"] = [stn, ses, across_chans_average, freqs]


    # averaged_across_stn_df = pd.DataFrame(averaged_across_stn_dict)
    # averaged_across_stn_df.rename(index={
    #     0: "stn",
    #     1: "session",
    #     2: f"power_spectrum_average_{incl_channels}",
    #     3: "frequencies"
    # }, inplace=True)

    # averaged_across_stn_df = averaged_across_stn_df.transpose()


    #################  WRITE DATAFRAMES WITH SESSION COMPARISONS INCLUDING ONLY STNs AVAILABLE FOR BOTH SESSIONS  #################

    # Dataframes for each session comparison, one DF per session for all comparisons
    compare_sessions = ["postop_fu3m", "postop_fu12m", "postop_fu18m", 
                        "fu3m_fu12m", "fu3m_fu18m", "fu12m_fu18m"]
    
    comparisons_storage = {}
    
    for comparison in compare_sessions:

        two_sessions = comparison.split("_")
        session_1 = two_sessions[0]
        session_2 = two_sessions[1]

        # Dataframe per session
        session_1_df = single_channels_df.loc[(single_channels_df["session"]==session_1)]
        session_2_df = single_channels_df.loc[(single_channels_df["session"]==session_2)]

        # list of STNs per session
        session_1_stns = list(session_1_df.stn.unique())
        session_2_stns = list(session_2_df.stn.unique())

        # list of STNs included in both sessions
        STN_list = list(set(session_1_stns) & set(session_2_stns))
        STN_list.sort()

        comparison_df = pd.DataFrame()

        for stn in STN_list:

            session_1_compared_to_2 = session_1_df.loc[session_1_df["stn"]==stn]
            session_2_compared_to_1 = session_2_df.loc[session_2_df["stn"]==stn]
            
            comparison_df = pd.concat([comparison_df, session_1_compared_to_2, session_2_compared_to_1])
            

        comparisons_storage[f"{comparison}_df"] = comparison_df


    # save dictionary as pickle file
    comparisons_storage_filepath = os.path.join(results_path, f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle")
    with open(comparisons_storage_filepath, "wb") as file:
        pickle.dump(comparisons_storage, file)

    print("file: ", 
          f"power_spectra_{signalFilter}_{incl_channels}_{normalization}_session_comparisons.pickle",
          "\nwritten in: ", results_path
          )


    return {
        "single_channels_df":single_channels_df,
        # "averaged_across_stn_df":averaged_across_stn_df,
        "comparisons_storage": comparisons_storage,

    }













        
    
















