""" Write clean Group Dataframes and save as Pickle """

import os
import pickle

import pandas as pd

from ..  tfr import feats_ssd as feats_ssd
######### PRIVATE PACKAGES #########
from .. classes import mainAnalysis_class as mainAnalysis_class
from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults
# PyPerceive Imports
import py_perceive


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

            mainclass_sub = mainclass.PerceiveData(
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


        
    















