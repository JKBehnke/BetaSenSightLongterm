""" load rotated coordinates"""



import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy

import scipy.io as sio


######### PRIVATE PACKAGES #########
import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults

from analysis.classes import mainAnalysis_class



def load_mni_coordinates(incl_sub:list):
    """
    Find data folder and load per subject each ea_reconstruction.mat file and extract x,y,z coordinates
        - reco_native 
        - reco_scrf
        - reco_mni 

    Write an Excel file: SenSightElectrode_coordinates.xlsx into imagingData folder

    watchout !! sub017 and sub034 only have native coordinates !! scrf and native coordinates are missing
    
    """


    hemispheres = ["Right", "Left"]

    current_path = os.getcwd()
    while current_path[-8:] != 'Research':
        current_path = os.path.dirname(current_path)


    reco_native_concat = pd.DataFrame()
    reco_scrf_concat = pd.DataFrame()
    reco_mni_concat = pd.DataFrame()

    for sub in incl_sub:
        
        # directory to data folder with mni coordinates
        data_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', f"sub-{sub}")
        filename = os.path.join(data_path, "ea_reconstruction.mat")

        # load .mat file
        mni_file = sio.loadmat(filename)

        print(sub)

        reco_native = mni_file["reco"][0][0][1]
        reco_scrf = mni_file["reco"][0][0][2]
        reco_mni = mni_file["reco"][0][0][3]


        # get the correct array from the .mat file per hemisphere: positive x = Right, negative x = Left
        for h, hem in enumerate(hemispheres):
            
            reco_native_hem = reco_native[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_native_DF = pd.DataFrame(reco_native_hem, columns=["reco_native_x", "reco_native_y", "reco_native_z"])
            reco_native_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}") # insert column with subject_hemisphere on first position
            reco_native_concat = pd.concat([reco_native_concat, reco_native_DF], ignore_index=True)
        
            reco_scrf_hem = reco_scrf[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_scrf_DF = pd.DataFrame(reco_scrf_hem, columns=["reco_scrf_x", "reco_scrf_y", "reco_scrf_z"])
            reco_scrf_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}")
            reco_scrf_concat = pd.concat([reco_scrf_concat, reco_scrf_DF], ignore_index=True)

            reco_mni_hem = reco_mni[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_mni_DF = pd.DataFrame(reco_mni_hem, columns=["reco_mni_x", "reco_mni_y", "reco_mni_z"])
            reco_mni_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}")
            reco_mni_concat = pd.concat([reco_mni_concat, reco_mni_DF], ignore_index=True)
    

    # store each Dataframe in seperate sheets of an Excel file
    Excel_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', 'SenSightElectrode_coordinates.xlsx')

    # create an Excel writer, so that different sheets are written within the same excel file
    with pd.ExcelWriter(Excel_path) as writer:
        
        reco_native_concat.to_excel(writer, sheet_name="reco_native")
        reco_scrf_concat.to_excel(writer, sheet_name="reco_scrf")
        reco_mni_concat.to_excel(writer, sheet_name="reco_mni")


    return {
        "reco_native_concat": reco_native_concat,
        "reco_scrf_concat": reco_scrf_concat,
        "reco_mni_concat": reco_mni_concat
    }





def calculate_mean_coordinates_bipolarRecordings(incl_sub:list, coordinates:str):
    """
    Input:
        - incl_sub: list of subjects to include
        - coordinates: str, "native", "scrf", "mni"

    Find data folder and load per subject each ea_reconstruction.mat file and extract x,y,z coordinates
        - reco_native 
        - reco_scrf
        - reco_mni 

    Write an Excel file: SenSightElectrode_coordinates.xlsx into imagingData folder

    watchout !! sub017 and sub034 only have native coordinates !! scrf and native coordinates are missing
    
    """



    hemispheres = ["Right", "Left"]


    ##################### Load the psdAverage_Dataframe #####################
    psdAverage_dataframe = pd.DataFrame()

    for sub in incl_sub:

        for hem in hemispheres:

            # load data from classes
            mainClass_data = mainAnalysis_class.MainClass(
                sub=sub,
                hemisphere=hem,
                filter="band-pass",
                result="PSDaverageFrequencyBands",
                incl_session=["fu3m"],
                pickChannels=['03', '13', '02', '12', '01', '23', 
                                '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                '1A2A', '1B2B', '1C2C'],
                normalization=["rawPsd"],
                freqBands=["beta", "lowBeta", "highBeta"],
                feature=["averagedPSD"]
            )

            # get the fu3m Dataframe
            fu3m_DF = mainClass_data.fu3m.Result_DF
            fu3m_DF_copy = fu3m_DF.copy()

            # add columnn subject_hemisphere
            fu3m_DF_copy["subject_hemisphere"] = f"{sub}_{hem}"

            # add column recording_montage
            for index, row in fu3m_DF_copy.iterrows():
                bipolarChannel = row["bipolarChannel"].split("_")
                recording_montage = bipolarChannel[2] # just take 03, 02, etc from bipolarChannel column

                # add 03 in new column "recording_montage"
                fu3m_DF_copy.loc[index, "recording_montage"] = recording_montage
            
            # only get rows with rawPsd and beta 
            fu3m_DF_copy = fu3m_DF_copy[fu3m_DF_copy.absoluteOrRelativePSD == "rawPsd"]
            fu3m_DF_copy = fu3m_DF_copy[fu3m_DF_copy.frequencyBand == "beta"]

            # rename column averagedPSD to beta_psd
            fu3m_DF_copy = fu3m_DF_copy.rename(columns={"averagedPSD": "beta_psd"})
            
            # drop columns bipolarChannel, 
            fu3m_DF_copy = fu3m_DF_copy.drop(columns=["bipolarChannel", "frequencyBand", "absoluteOrRelativePSD"])

            # concatenate all dataframes together
            psdAverage_dataframe = pd.concat([psdAverage_dataframe, fu3m_DF_copy], ignore_index=True)

            # new column with subject, hemisphere and recording info, necessary for merging later
            psdAverage_dataframe["subject_hemisphere_recording"] = psdAverage_dataframe[["subject_hemisphere", "recording_montage"]].agg('_'.join, axis=1)



    ##################### Load the coordinates #####################

    if coordinates == "native":
        index_coord = 1
    
    elif coordinates == "scrf":
        index_coord = 2
    
    elif coordinates == "mni":
        index_coord = 3


    reco_concat = pd.DataFrame()


    current_path = os.getcwd()
    while current_path[-8:] != 'Research':
        current_path = os.path.dirname(current_path)

    for sub in incl_sub:
        
        # directory to data folder with mni coordinates
        data_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', f"sub-{sub}")
        filename = os.path.join(data_path, "ea_reconstruction.mat")

        # load .mat file
        mni_file = sio.loadmat(filename)

        reco_coord = mni_file["reco"][0][0][index_coord]


        for h, hem in enumerate(hemispheres):
            
            reco_hem = reco_coord[0][0][0][0][h] # 0 = Right, 1 = Left
            reco_DF = pd.DataFrame(reco_hem, columns=[f"reco_{coordinates}_x", f"reco_{coordinates}_y", f"reco_{coordinates}_z"])
            reco_DF.insert(0, "subject_hemisphere", f"{sub}_{hem}") # insert column with subject_hemisphere on first position
            reco_concat = pd.concat([reco_concat, reco_DF], ignore_index=True)


    # add electrode_contacts to the dataframe
    sub_hem_unique = list(reco_concat.subject_hemisphere.unique())
    repeat = len(sub_hem_unique) # add contacts 0, 1A, 1B etc to dataframe repeatedly depending on how many sub_hem in dataframe

    contact_list = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
    reco_concat["electrode_contact"] = contact_list * repeat # now there is a column with contact information

    recording_montage_list = ["0_3", "1_3", "0_2", "1_2", "0_1", "2_3", "1A_1B", "1B_1C", "1A_1C", "2A_2B", "2B_2C", "2A_2C", "1A_2A", "1B_2B", "1C_2C"]

    # calculate the mean coordinate between contacts per recording montage

        
    x_val = {}
    y_val = {}
    z_val = {}
    x_mean_coordinate = {}
    y_mean_coordinate = {}
    z_mean_coordinate = {}

    for sub_hem in sub_hem_unique:

        # only take DF of one electrode
        single_electrode = reco_concat[reco_concat.subject_hemisphere == sub_hem]

        for c in contact_list:
            
            # for every contact extract the x, y, z coordinates
            single_contact = single_electrode[single_electrode.electrode_contact == c]

            if coordinates == "native":
                x_val[f"{sub_hem}_{c}"] = float(single_contact.reco_native_x.values)
                y_val[f"{sub_hem}_{c}"] = float(single_contact.reco_native_y.values)
                z_val[f"{sub_hem}_{c}"] = float(single_contact.reco_native_z.values)

            elif coordinates == "scrf":
                x_val[f"{sub_hem}_{c}"] = float(single_contact.reco_scrf_x.values)
                y_val[f"{sub_hem}_{c}"] = float(single_contact.reco_scrf_y.values)
                z_val[f"{sub_hem}_{c}"] = float(single_contact.reco_scrf_z.values)    
            
            elif coordinates == "mni":
                x_val[f"{sub_hem}_{c}"] = float(single_contact.reco_mni_x.values)
                y_val[f"{sub_hem}_{c}"] = float(single_contact.reco_mni_y.values)
                z_val[f"{sub_hem}_{c}"] = float(single_contact.reco_mni_z.values)
            


        # calculate mean x, y, z coordinates for 1 and 2
        x_val[f"{sub_hem}_1"] = np.mean([x_val[f"{sub_hem}_1A"], x_val[f"{sub_hem}_1B"], x_val[f"{sub_hem}_1C"]])
        y_val[f"{sub_hem}_1"] = np.mean([y_val[f"{sub_hem}_1A"], y_val[f"{sub_hem}_1B"], y_val[f"{sub_hem}_1C"]])
        z_val[f"{sub_hem}_1"] = np.mean([z_val[f"{sub_hem}_1A"], z_val[f"{sub_hem}_1B"], z_val[f"{sub_hem}_1C"]])

        x_val[f"{sub_hem}_2"] = np.mean([x_val[f"{sub_hem}_2A"], x_val[f"{sub_hem}_2B"], x_val[f"{sub_hem}_2C"]])
        y_val[f"{sub_hem}_2"] = np.mean([y_val[f"{sub_hem}_2A"], y_val[f"{sub_hem}_2B"], y_val[f"{sub_hem}_2C"]])
        z_val[f"{sub_hem}_2"] = np.mean([z_val[f"{sub_hem}_2A"], z_val[f"{sub_hem}_2B"], z_val[f"{sub_hem}_2C"]])


    for sub_hem in sub_hem_unique:
        for recording in recording_montage_list:

            rec = recording.split("_") # rec[0] is first contact, rec[1] is second contact
            
            x_mean_coordinate[f"{sub_hem}_{recording}"] = np.mean([x_val[f"{sub_hem}_{rec[0]}"], x_val[f"{sub_hem}_{rec[1]}"]])
            y_mean_coordinate[f"{sub_hem}_{recording}"] = np.mean([y_val[f"{sub_hem}_{rec[0]}"], y_val[f"{sub_hem}_{rec[1]}"]])
            z_mean_coordinate[f"{sub_hem}_{recording}"] = np.mean([z_val[f"{sub_hem}_{rec[0]}"], z_val[f"{sub_hem}_{rec[1]}"]])

            
    x_mean_df = pd.DataFrame.from_dict(x_mean_coordinate, orient="index", columns=[f"{coordinates}_mean_coord_x"])
    y_mean_df = pd.DataFrame.from_dict(y_mean_coordinate, orient="index", columns=[f"{coordinates}_mean_coord_y"])
    z_mean_df = pd.DataFrame.from_dict(z_mean_coordinate, orient="index", columns=[f"{coordinates}_mean_coord_z"])

    mean_xyz_coord = pd.concat([x_mean_df, y_mean_df, z_mean_df], axis=1) # dataframe with 3 columns

    # now extract from the index (017_Right_0_3) sub_hem and recording montage and store in a seperate 
    sub_hem_rec = mean_xyz_coord.index.tolist() # list of indeces with sub,hem,recording info
    sub_hem_rec_tocolumn = []

    for i, string in enumerate(sub_hem_rec):

        sub_hem_rec_split = string.split("_") # 017_Right_0_3, split into 4 parts

        sub_hem_str = '_'.join([sub_hem_rec_split[0], sub_hem_rec_split[1]]) # 017_Right
        rec_str = ''.join([sub_hem_rec_split[2], sub_hem_rec_split[3]]) # 03
        sub_hem_rec_string = '_'.join([sub_hem_str, rec_str]) # 017_Right_03
        sub_hem_rec_tocolumn.append(sub_hem_rec_string)


    # add columns subject_hemisphere and recording montage to dataframe
    mean_xyz_coord["subject_hemisphere_recording"] = sub_hem_rec_tocolumn  


    # merge 2 dataframes: mean coordinates and psd together

    merged_Dataframe =  mean_xyz_coord.merge(psdAverage_dataframe, left_on="subject_hemisphere_recording", right_on="subject_hemisphere_recording")


    # write Dataframe to new Excel file
    Excel_path = os.path.join(current_path, 'Longterm_beta_project','data', 'imagingData', f'SenSightElectrode_MEANcoordinates_{coordinates}.xlsx')

    # create an Excel writer, so that different sheets are written within the same excel file
    with pd.ExcelWriter(Excel_path) as writer:
        
        merged_Dataframe.to_excel(writer, sheet_name=f"mean_reco_{coordinates}")


    return {
        "merged_Dataframe":merged_Dataframe
    }

















        
