""" monopolar Referencing: Robert approach """


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import plotly
import plotly.graph_objs as go

import pickle

# utility functions
from .. utils import loadResults as loadResults
from .. utils import find_folders as find_folders





def monoRef_weightPsdBetaAverageByCoordinateDistance(
    sub: str, 
    hemisphere: str, 
    filterSignal: str,
    normalization: str,
    freqBand: str,
    incl_sessions: list
):

    """
    
    Input:
        - sub: str e.g. "029"
        - hemisphere: str e.g. "Right", "Left"
        - filterSignal: str e.g. "band-pass"
        - normalization: str, e.g. "rawPsd"
        - freqBand: str, e.g. "beta"
        - incl_sessions: list e.g. ["postop", "fu3m", "fu12m", "fu18m"]

    1) Load the .json file SPECTROGRAMpsdAverageFrequencyBands_{hemisphere}_{filterSignal}.json

    2) 




    """

    results_paths = find_folders.get_local_path(folder="results", sub=sub)


    #####################  defining the coordinates of monopolar contacts #####################
    # rcosθ+(rsinθ)i
    # z coordinates of the vertical axis
    # xy coordinates of the polar plane around the percept device

    d = 2
    r = 0.65 # change this radius as you wish - needs to be optimised
    contact_coordinates = {'0': [d*0.0,0+0*1j],
                        '1': [d*1.0,0+0*1j],
                        '2': [d*2.0,0+0*1j],
                        '3': [d*3.0,0+0*1j],
                        '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
                        '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}
    
    # contact_coordinates = tuple z-coord + xy-coord

    ##################### lets plot the monopolar contact coordinates! #####################
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    plot_data = []

    for contact in contact_coordinates.keys():
        zs = contact_coordinates[contact][0]
        
        y = contact_coordinates[contact][1]    
        xs = np.real(y)
        ys = np.imag(y)  

        trace = go.Scatter3d(
            x=np.array([xs]),  
            y=np.array([ys]),  
            z=np.array([zs]),  
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.8,
            }
        )
        plot_data.append(trace)

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    #data = [trace]

    plot_figure = go.Figure(data=plot_data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)


    #####################  Loading the Data #####################

    originalPsdAverageDataframe = loadResults.load_PSDjson(
        sub=sub,
        result="PSDaverageFrequencyBands",
        hemisphere=hemisphere,
        filter=filterSignal
    )

    # transform dict into Dataframe and clean up, to only get the values of interest
    originalPsdAverageDataframe = pd.DataFrame(originalPsdAverageDataframe)
    originalPsdAverageDataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.frequencyBand==freqBand]
    originalPsdAverageDataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.absoluteOrRelativePSD==normalization]


    session_data = {}

    # loop over sessions
    for ses in incl_sessions:

        session_Dataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.session==ses]

        # copying session_Dataframe to add new columns
        session_Dataframe_coord = session_Dataframe.copy()

        session_Dataframe_coord["subject_hemisphere"] = f"{sub}_{hemisphere}"        
                
        ##################### looping over bipolar channels, get average coordinates between 2 contacts #####################
        for idx in session_Dataframe.index:
            
            # extracting contact names
            bipolar_channel = session_Dataframe.loc[idx,'bipolarChannel']
            bipolar_channel = bipolar_channel.split('_')[2] # splits channel Names: e.g. LFP_R_03_STN_MT, only keep 03
            
            # extracting individual monopolar contact names from bipolar channels
            if len(bipolar_channel)==4: # if len ==4 e.g. 1A2A
                bipolar_channel_1 = bipolar_channel[:2] # 
                bipolar_channel_2 = bipolar_channel[2:]        

            elif len(bipolar_channel)==2:
                bipolar_channel_1 = bipolar_channel[0]
                bipolar_channel_2 = bipolar_channel[1]
            

 
            
            # storing monopolar contact names for bipolar contacts
            # e.g. channel 1A2A: contact1 = 1A, contact2 = 2A
            session_Dataframe_coord.loc[idx,'contact1'] = bipolar_channel_1 
            session_Dataframe_coord.loc[idx,'contact2'] = bipolar_channel_2

            
            # extracting coordinates of each monopolar contact from dictionary contact_coordinates
            coords1 = contact_coordinates[bipolar_channel_1]
            coords2 = contact_coordinates[bipolar_channel_2]

            # computing mean distance between monopolar contacts to get the bipolar average coordinate
            # coords1, e.g. contact 1A -> tuple of (z-coordinates, xy-coordinates)
            z_av = np.mean([coords1[0],coords2[0]]) # average of z-coord from contact1 and contact2
            xy_av = np.mean([coords1[1],coords2[1]]) # average of xy-coord from contact1 and contact2


            
            # storing new coordinates of bipolar contacts
            session_Dataframe_coord.loc[idx,'coord_z'] = z_av # mean of z-coordinates from contact 1 and 2
            session_Dataframe_coord.loc[idx,'coord_xy'] = xy_av # mean of xy-coordinates from contact 1 and 2


        # store copied and modified session_Dataframe into session dictionary
        session_data[f"{ses}_bipolar_Dataframe"]=session_Dataframe_coord


        ##################### New Dataframe for calculated beta psd average of each monopolar contact from all averaged coordinates #####################

        # Create Dataframe with the coordinates of 10 contact coordinates: 0, 1, 2, 3, 1A, 1B, 1C, 2A, 2B, 2C
        mono_data = pd.DataFrame(contact_coordinates).T
        mono_data.columns = ['coord_z','coord_xy'] # columns with z- and xy-coordinates of each contact


        # copy mono_data dataframe to add new columns
        mono_data_psdAverage = mono_data.copy()

        mono_data_psdAverage["subject_hemisphere"] = f"{sub}_{hemisphere}"
        mono_data_psdAverage["session"] = f"{ses}"

        
        # loop over all monopolar contacts and get average beta
        # error that some will have zero distance!!

        for contact in contact_coordinates.keys():
            
            # extracting coordinates for mono polar contacts
            coord_z = mono_data.loc[contact,'coord_z']
            coord_xy = mono_data.loc[contact,'coord_xy']
            
            # loop over all bipolar contacts and compute distance to monopolar contact
            all_dists = [] # list of all distances from all averaged bipolar coordinates to one monopolar coordinate

            for bipolar_channel in session_Dataframe_coord.index:
                
                # finding difference from the monopolar contact to each bipolar mean coordinates
                diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
                diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])
                
                # compute euclidean distance based on both directions
                # Pythagoras: a^2 + b^2 = c^2
                # a=difference of z-coord
                # b=difference of xy-coord
                # c=distance of interest 
                dist = np.sqrt(diff_z**2 + diff_xy**2)
                
                #append the distance
                all_dists.append(dist)

            # collect all distances in numpy array    
            all_dists = np.array(all_dists)
            
            # compute similarity from distances 
            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x doesn´t reach 0
            
            # normalise the similarity measures (to not weight central contacts more)
            similarity = similarity/np.sum(similarity)
            # explanation: contacts 0 and 3 only have 1 adjacent contact, while contacts 1 and 2 have 2 adjacent contacts
            # normalizing the similarities by their total sum (e.g. contact 0: dividing by sum similarity 01+02+03)
            # normalized similarity is now comparable between different contacts -> sum of normalized similarities should equal 1
            
            # weighting the beta of bipolar contacts by their similarity to the monopolar contact
            weighted_beta = session_Dataframe_coord['averagedPSD'].values *  similarity #(1/all_dists) 


            # storing the weighted beta for the mono polar contact
            mono_data_psdAverage.loc[contact,f'averaged_monopolar_PSD_{freqBand}'] = np.sum(weighted_beta) # sum of all 15 weighted psdAverages = one monopolar contact psdAverage

        

        # ranking the weighted monopolar psd    
        mono_data_psdAverage["rank"] = mono_data_psdAverage[f"averaged_monopolar_PSD_{freqBand}"].rank(ascending=False) # rank highest psdAverage as 1.0

        # store copied and modified mono data into session dictionary
        session_data[f"{ses}_monopolar_Dataframe"]=mono_data_psdAverage


    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    session_data_filepath = os.path.join(results_paths, f"sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(session_data_filepath, "wb") as file:
        pickle.dump(session_data, file)

    print(f"New file: sub{sub}_{hemisphere}_monoRef_weightedPsdByCoordinateDistance_{freqBand}_{normalization}_{filterSignal}.pickle",
            f"\nwritten in: {results_paths}" )




    

    return {
        "session_data":session_data,
    }




def monoRef_only_segmental_weight_psd_by_distance(
    sub: str, 
    hemisphere: str, 
    filterSignal: str,
    normalization: str,
    freqBand: str,
):

    """
    
    Input:
        - sub: str e.g. "029"
        - hemisphere: str e.g. "Right", "Left"
        - filterSignal: str e.g. "band-pass"
        - normalization: str, e.g. "rawPsd"
        - freqBand: str, e.g. "beta"
        - incl_sessions: list e.g. ["postop", "fu3m", "fu12m", "fu18m"]

    1) define imaginary coordinates only for segmental contacts
        - plot the imaginary contact coordinates using plotly

    
    
    2) Load the .json file SPECTROGRAMpsdAverageFrequencyBands_{hemisphere}_{filterSignal}.json and edit dataframe
        
        - check which sessions exist for this patient
        - only for segmental bipolar channels: add columns
            subject_hemisphere
            contact1
            contact2
            bip_chan
            coord_z = mean coordinate between contact 1 and 2
            coord_xy = mean coordinate between contact 1 and 2
            channel_group
        
        - delete all rows with Ring bipolar channels using drop NaN 

        save Dataframe for each session: session_data[f"{ses}_bipolar_Dataframe"]
            

    3) Calculate for each segmental contact the estimated PSD

        - new dataframe per session with segmental contacts as index

        - calculate an array with all euclidean distances based on both directions in z- and xy-axis

            diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
            diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])
            
            dist = np.sqrt(diff_z**2 + diff_xy**2)
        
        - compute similarity from distances

            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x does't reach 0

        - normalization of similarity is not necessary when only using segmental bipolar recordings
        -> each contact should have the same similarities

        - weight the recorded psd in a frequency band: 

            for each bipolar segmental channel: 
            weighted_beta = averaged PSD * similarity
        
        -> monopolar estimated psd of one segmental contact = np.sum(weighted_beta)
        
        
    4) save the dictionary sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

        - in results_path of the subject
        - keys of dictionary: 

            f"{ses}_bipolar_Dataframe" with bipolar content: contact1 and contact2 with coordinates and psd average of bipolar channels

            f"{ses}_monopolar_Dataframe" with monopolar content: contact, estimated_monopol_psd_{freqBand}, rank

            

    TODO: ask Rob again, is normalization of similarity in this case with only segmental contacts not necessary?

    """

    results_paths = find_folders.get_local_path(folder="results", sub=sub)

    #####################  defining the coordinates of monopolar contacts #####################
    # rcosθ+(rsinθ)i
    # z coordinates of the vertical axis
    # xy coordinates of the polar plane around the percept device

    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    incl_sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    d = 2
    r = 0.65 # change this radius as you wish - needs to be optimised
    contact_coordinates = {
                        '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
                        '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}

    # contact_coordinates = tuple z-coord + xy-coord

    ##################### lets plot the monopolar contact coordinates! #####################
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    plot_data = []

    for contact in contact_coordinates.keys():
        zs = contact_coordinates[contact][0]
        
        y = contact_coordinates[contact][1]    
        xs = np.real(y)
        ys = np.imag(y)  

        trace = go.Scatter3d(
            x=np.array([xs]),  
            y=np.array([ys]),  
            z=np.array([zs]),  
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.8,
            }
        )
        plot_data.append(trace)
    
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    #data = [trace]

    plot_figure = go.Figure(data=plot_data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)

    #####################  Loading the Data #####################

    originalPsdAverageDataframe = loadResults.load_PSDjson(
        sub=sub,
        result="PSDaverageFrequencyBands",
        hemisphere=hemisphere,
        filter=filterSignal
    )

    # transform dict into Dataframe and clean up, to only get the values of interest
    originalPsdAverageDataframe = pd.DataFrame(originalPsdAverageDataframe)
    originalPsdAverageDataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.frequencyBand==freqBand]
    originalPsdAverageDataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.absoluteOrRelativePSD==normalization]


    session_data = {}

    # loop over sessions
    for ses in incl_sessions:

        # check if session exists
        if ses not in originalPsdAverageDataframe.session.values:
            continue 

        session_Dataframe = originalPsdAverageDataframe[originalPsdAverageDataframe.session==ses]

        # copying session_Dataframe to add new columns
        session_Dataframe_coord = session_Dataframe.copy()

        session_Dataframe_coord["subject_hemisphere"] = f"{sub}_{hemisphere}"        
                
        ##################### looping over bipolar channels, get average coordinates between 2 contacts #####################
        for idx in session_Dataframe.index:
            
            # extracting contact names
            bipolar_channel = session_Dataframe.loc[idx,'bipolarChannel']
            bipolar_channel = bipolar_channel.split('_')[2] # splits channel Names: e.g. LFP_R_03_STN_MT, only keep 03
            
            # extracting individual monopolar contact names from bipolar channels
            if len(bipolar_channel)==4: # if len ==4 e.g. 1A2A
                bipolar_channel_1 = bipolar_channel[:2] # 1A
                bipolar_channel_2 = bipolar_channel[2:] # 2A
                channel_group = "segments"    

            elif len(bipolar_channel)==2:
                # Ring bipolar channels will have NaN -> will be dropped later
                continue
            
            # storing monopolar contact names for bipolar contacts
            # e.g. channel 1A2A: contact1 = 1A, contact2 = 2A
            session_Dataframe_coord.loc[idx,'contact1'] = bipolar_channel_1 
            session_Dataframe_coord.loc[idx,'contact2'] = bipolar_channel_2
            session_Dataframe_coord.loc[idx, 'bip_chan'] = f"{bipolar_channel_1}{bipolar_channel_2}"

            # extracting coordinates of each monopolar contact from dictionary contact_coordinates
            coords1 = contact_coordinates[bipolar_channel_1]
            coords2 = contact_coordinates[bipolar_channel_2]

            # computing mean distance between monopolar contacts to get the bipolar average coordinate
            # coords1, e.g. contact 1A -> tuple of (z-coordinates, xy-coordinates)
            z_av = np.mean([coords1[0],coords2[0]]) # average of z-coord from contact1 and contact2
            xy_av = np.mean([coords1[1],coords2[1]]) # average of xy-coord from contact1 and contact2

            # storing new coordinates of bipolar contacts
            session_Dataframe_coord.loc[idx,'coord_z'] = z_av # mean of z-coordinates from contact 1 and 2
            session_Dataframe_coord.loc[idx,'coord_xy'] = xy_av # mean of xy-coordinates from contact 1 and 2

            # column with Ring, Segm
            session_Dataframe_coord.loc[idx,'channel_group'] = channel_group


        # drop rows with NaN, so all Ring bipolar channels are taken out
        session_Dataframe_coord = session_Dataframe_coord.dropna(axis="index")

        # store copied and modified session_Dataframe into session dictionary
        session_data[f"{ses}_bipolar_Dataframe"]=session_Dataframe_coord


        ##################### New Dataframe for calculated beta psd average of each monopolar contact from all averaged coordinates #####################

        # Create Dataframe with the coordinates of 10 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
        mono_data = pd.DataFrame(contact_coordinates).T
        mono_data.columns = ['coord_z','coord_xy'] # columns with z- and xy-coordinates of each contact


        # copy mono_data dataframe to add new columns
        mono_data_psdAverage = mono_data.copy()

        mono_data_psdAverage["subject_hemisphere"] = f"{sub}_{hemisphere}"
        mono_data_psdAverage["session"] = f"{ses}"

        # only keep segmental contacts
        mono_data_psdAverage= mono_data_psdAverage.loc[mono_data_psdAverage.index.isin(segmental_contacts)]

        
        # loop over all segmental contacts and get average beta
        # error that some will have zero distance!!

        # filter only segmental channels
        #segm_session_Dataframe = session_Dataframe_coord.loc[session_Dataframe_coord.channel_group=="segments"]

        for contact in segmental_contacts:
            
            # extracting coordinates for mono polar contacts
            coord_z = mono_data.loc[contact,'coord_z']
            coord_xy = mono_data.loc[contact,'coord_xy']
            
            # loop over all bipolar contacts and compute distance to monopolar contact
            all_dists = [] # list of all distances from all averaged bipolar coordinates to one monopolar coordinate

            for bipolar_channel in session_Dataframe_coord.index:
                
                # finding difference from the monopolar contact to each bipolar mean coordinates
                diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
                diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])
                
                # compute euclidean distance based on both directions
                # Pythagoras: a^2 + b^2 = c^2
                # a=difference of z-coord
                # b=difference of xy-coord
                # c=distance of interest 
                dist = np.sqrt(diff_z**2 + diff_xy**2)
                
                #append the distance
                all_dists.append(dist)

            # collect all distances in numpy array    
            all_dists = np.array(all_dists)
            
            # compute similarity from distances 
            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x doesn´t reach 0
            
            # NO NORMALISATION OF SIMILARITY NECESSARY -> each contact has the same similarities
            # normalise the similarity measures (to not weight central contacts more)
            # similarity = similarity/np.sum(similarity)
            # explanation: contacts 0 and 3 only have 1 adjacent contact, while contacts 1 and 2 have 2 adjacent contacts
            # normalizing the similarities by their total sum (e.g. contact 0: dividing by sum similarity 01+02+03)
            # normalized similarity is now comparable between different contacts -> sum of normalized similarities should equal 1
            
            # weighting the beta of bipolar contacts by their similarity to the monopolar contact
            weighted_beta = session_Dataframe_coord['averagedPSD'].values *  similarity # two arrays with same length = 9 bip_chans

            # storing the weighted beta for the mono polar contact
            mono_data_psdAverage.loc[contact,f'estimated_monopolar_psd_{freqBand}'] = np.sum(weighted_beta) # sum of all 9 weighted psdAverages = one monopolar contact psdAverage

            # add column with contact
            mono_data_psdAverage.loc[contact, 'contact'] = contact

        
        # ranking the weighted monopolar psd    
        mono_data_psdAverage["rank"] = mono_data_psdAverage[f"estimated_monopolar_psd_{freqBand}"].rank(ascending=False) # rank highest psdAverage as 1.0

        # store copied and modified mono data into session dictionary
        session_data[f"{ses}_monopolar_Dataframe"]=mono_data_psdAverage



    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    session_data_filepath = os.path.join(results_paths, f"sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle")
    with open(session_data_filepath, "wb") as file:
        pickle.dump(session_data, file)

    print(f"New file: sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle",
            f"\nwritten in: {results_paths}" )
    

    return session_data



# def fooof_monoRef_weight_psd_by_distance(
#     fooof_spectrum:str,
#     only_segmental:str
    
# ):

#     """
    
#     Input:
#         - fooof_spectrum: 
#             "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
#             "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
#             "periodic_flat"             -> model._peak_fit
        
#         - only_segmental: str e.g. "yes" or "no"


#     1) define imaginary coordinates only for segmental contacts
#         - plot the imaginary contact coordinates using plotly

    
    
#     2) Load the fooof dataframe and edit dataframe
        
#         - check which sessions exist for this patient
#         - only for segmental bipolar channels: add columns
#             subject_hemisphere
#             contact1
#             contact2
#             bip_chan
#             coord_z = mean coordinate between contact 1 and 2
#             coord_xy = mean coordinate between contact 1 and 2
#             channel_group
        
#         - delete all rows with Ring bipolar channels using drop NaN 

#         save Dataframe for each session: session_data[f"{ses}_bipolar_Dataframe"]
            

#     3) Calculate for each segmental contact the estimated PSD

#         - new dataframe per session with segmental contacts as index

#         - calculate an array with all euclidean distances based on both directions in z- and xy-axis

#             diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
#             diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])
            
#             dist = np.sqrt(diff_z**2 + diff_xy**2)
        
#         - compute similarity from distances

#             similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x does't reach 0

#         - normalization of similarity is not necessary when only using segmental bipolar recordings
#         -> each contact should have the same similarities

#         - weight the recorded psd in a frequency band: 

#             for each bipolar segmental channel: 
#             weighted_beta = averaged PSD * similarity
        
#         -> monopolar estimated psd of one segmental contact = np.sum(weighted_beta)
        
        
#     4) save the dictionary sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

#         - in results_path of the subject
#         - keys of dictionary: 

#             f"{ses}_bipolar_Dataframe" with bipolar content: contact1 and contact2 with coordinates and psd average of bipolar channels

#             f"{ses}_monopolar_Dataframe" with monopolar content: contact, estimated_monopol_psd_{freqBand}, rank

            

#     TODO: ask Rob again, is normalization of similarity in this case with only segmental contacts not necessary?

#     """

#     results_paths = find_folders.get_local_path(folder="GroupResults")


#     #####################  defining the coordinates of monopolar contacts #####################
#     # rcosθ+(rsinθ)i
#     # z coordinates of the vertical axis
#     # xy coordinates of the polar plane around the percept device

#     #segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
#     #segmental_channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
#     incl_sessions = ["postop", "fu3m", "fu12m", "fu18m"]


#     d = 2
#     r = 0.65 # change this radius as you wish - needs to be optimised

#     if only_segmental == "yes":
#         contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
#         channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
#         contact_coordinates = {
#                             '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
#                             '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
#                             '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
#                             '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
#                             '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
#                             '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}

#         # contact_coordinates = tuple z-coord + xy-coord

#     elif only_segmental == "no":
#         contacts = ["0", "3", "1A", "1B", "1C", "2A", "2B", "2C"]
#         channels = ["01", "12", "23", "1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
#         contact_coordinates = {
#                         '0': [d*0.0,0+0*1j],
#                         '1': [d*1.0,0+0*1j],
#                         '2': [d*2.0,0+0*1j],
#                         '3': [d*3.0,0+0*1j],
#                         '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
#                         '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
#                         '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
#                         '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
#                         '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
#                         '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}
    
#     # contact_coordinates = tuple z-coord + xy-coord


#     ##################### lets plot the monopolar contact coordinates! #####################
#     # Configure Plotly to be rendered inline in the notebook.
#     plotly.offline.init_notebook_mode()

#     # Configure the trace.
#     plot_data = []

#     for contact in contact_coordinates.keys():
#         zs = contact_coordinates[contact][0]
        
#         y = contact_coordinates[contact][1]    
#         xs = np.real(y)
#         ys = np.imag(y)  

#         trace = go.Scatter3d(
#             x=np.array([xs]),  
#             y=np.array([ys]),  
#             z=np.array([zs]),  
#             mode='markers',
#             marker={
#                 'size': 10,
#                 'opacity': 0.8,
#             }
#         )
#         plot_data.append(trace)

#     # Configure the layout.
#     layout = go.Layout(
#         margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
#     )

#     #data = [trace]

#     plot_figure = go.Figure(data=plot_data, layout=layout)

#     # Render the plot.
#     plotly.offline.iplot(plot_figure)


#     #####################  Loading the Data #####################
#     beta_average_DF = loadResults.load_fooof_beta_ranks(
#         fooof_spectrum=fooof_spectrum,
#         all_or_one_chan="beta_ranks_all"
#     )

#     # only take rows of channels of interest
#     beta_average_DF = beta_average_DF.loc[beta_average_DF.bipolar_channel.isin(channels)]

#     session_data = {}
#     # loop over sessions
#     for ses in incl_sessions:

#         # check if session exists
#         if ses not in beta_average_DF.session.values:
#             continue 

#         session_Dataframe = beta_average_DF[beta_average_DF.session==ses]
#         # copying session_Dataframe to add new columns
#         session_Dataframe_coord = session_Dataframe.copy()
#         session_Dataframe_coord = session_Dataframe_coord.reset_index()
#         session_Dataframe_coord = session_Dataframe_coord.drop(columns=["index", "level_0"])

#         ##################### looping over bipolar channels, get average coordinates between 2 contacts #####################
#         for idx in session_Dataframe_coord.index:
            
#             # extracting contact names
#             bipolar_channel = session_Dataframe_coord.loc[idx,'bipolar_channel'] # e.g. 1A2A
            
#             # extracting individual monopolar contact names from bipolar channels
#             if len(bipolar_channel)==4: # if len ==4 e.g. 1A2A
#                 contact_1 = bipolar_channel[:2] # 1A
#                 contact_2 = bipolar_channel[2:] # 2A
#                 #channel_group = "segments"    

#             elif len(bipolar_channel)==2:

#                 if only_segmental == "yes":
#                     continue
#                 else:
#                     contact_1 = bipolar_channel[0] # 0
#                     contact_2 = bipolar_channel[1] # 1   
#                     #channel_group = "ring"   
            
#             # storing monopolar contact names for bipolar contacts
#             # e.g. channel 1A2A: contact1 = 1A, contact2 = 2A
#             session_Dataframe_coord.loc[idx,'contact1'] = contact_1 
#             session_Dataframe_coord.loc[idx,'contact2'] = contact_2

#             # extracting coordinates of each monopolar contact from dictionary contact_coordinates
#             coords1 = contact_coordinates[contact_1]
#             coords2 = contact_coordinates[contact_2]

#             # computing mean distance between monopolar contacts to get the bipolar average coordinate
#             # coords1, e.g. contact 1A -> tuple of (z-coordinates, xy-coordinates)
#             z_av = np.mean([coords1[0],coords2[0]]) # average of z-coord from contact1 and contact2
#             xy_av = np.mean([coords1[1],coords2[1]]) # average of xy-coord from contact1 and contact2

#             # storing new coordinates of bipolar contacts
#             session_Dataframe_coord.loc[idx,'coord_z'] = z_av # mean of z-coordinates from contact 1 and 2
#             session_Dataframe_coord.loc[idx,'coord_xy'] = xy_av # mean of xy-coordinates from contact 1 and 2


#         # store copied and modified session_Dataframe into session dictionary
#         session_data[f"{ses}_bipolar_Dataframe"]=session_Dataframe_coord




#     ##################### Calculate beta psd average of each monopolar contact from all averaged coordinates #####################

#     for ses in incl_sessions:

#         session_data[f"{ses}_monopolar_Dataframe"] = pd.DataFrame()

#         ses_dataframe = session_data[f"{ses}_bipolar_Dataframe"]

#         stn_list = list(ses_dataframe.subject_hemisphere.unique())

#         for stn in stn_list:

#             # only select bipolar channels of this stn and session
#             stn_ses_bipolar = ses_dataframe.loc[ses_dataframe.subject_hemisphere == stn]
#             stn_ses_bipolar = stn_ses_bipolar.reset_index()

#             # Create Dataframe with the coordinates of 6 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
#             mono_data = pd.DataFrame(contact_coordinates).T
#             mono_data.columns = ['coord_z','coord_xy'] # columns with z- and xy-coordinates of each contact

#             # copy mono_data dataframe to add new columns
#             mono_data_copy = mono_data.copy()

#             mono_data_copy["session"] = f"{ses}"
#             mono_data_copy["subject_hemisphere"] = f"{stn}"

#             for contact in contacts:

#                 # extracting coordinates for mono polar contacts
#                 coord_z = mono_data.loc[contact,'coord_z']
#                 coord_xy = mono_data.loc[contact,'coord_xy']
                
#                 # loop over all bipolar channels and compute distance to monopolar contact
#                 all_dists = [] # list of all distances from all averaged bipolar coordinates to one monopolar coordinate

#                 for bipolar_channel in stn_ses_bipolar.index:

#                     # finding difference from the monopolar contact to each bipolar mean coordinates
#                     diff_z = abs(coord_z - stn_ses_bipolar.loc[bipolar_channel, 'coord_z'])
#                     diff_xy = abs(coord_xy - stn_ses_bipolar.loc[bipolar_channel, 'coord_xy'])
                    
#                     # compute euclidean distance based on both directions
#                     # Pythagoras: a^2 + b^2 = c^2
#                     # a=difference of z-coord
#                     # b=difference of xy-coord
#                     # c=distance of interest 
#                     dist = np.sqrt(diff_z**2 + diff_xy**2)
                    
#                     #append the distance
#                     all_dists.append(dist)
                
#                 # collect all distances in numpy array    
#                 all_dists = np.array(all_dists)
                
#                 # compute similarity from distances 
#                 similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x doesn´t reach 0

#                 # weighting the beta of bipolar contacts by their similarity to the monopolar contact
#                 weighted_beta = stn_ses_bipolar['beta_average'].values *  similarity # two arrays with same length = 9 bip_chans

#                 # storing the weighted beta for the mono polar contact
#                 mono_data_copy.loc[contact,'estimated_monopolar_beta_psd'] = np.sum(weighted_beta) # sum of all 9 weighted psdAverages = one monopolar contact psdAverage

#                 # add column with contact
#                 mono_data_copy.loc[contact, 'contact'] = contact

#             # ranking the weighted monopolar psd
#             mono_data_copy["rank"] = mono_data_copy["estimated_monopolar_beta_psd"].rank(ascending=False) # rank highest psdAverage as 1.0

#             session_data[f"{ses}_monopolar_Dataframe"] = pd.concat([session_data[f"{ses}_monopolar_Dataframe"], mono_data_copy])

#     if only_segmental == "yes":
#         filename = "only_segmental_"
    
#     else: 
#         filename = "segments_and_rings_"
        

#     # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
#     session_data_filepath = os.path.join(results_paths, f"fooof_monoRef_{filename}weight_beta_psd_by_distance_{fooof_spectrum}.pickle")
#     with open(session_data_filepath, "wb") as file:
#         pickle.dump(session_data, file)

#     print(f"New file: fooof_monoRef_{filename}weight_beta_psd_by_distance_{fooof_spectrum}.pickle",
#             f"\nwritten in: {results_paths}" )
    

#     return session_data                





    


def fooof_monoRef_weight_psd_by_distance_segm_or_ring(
    fooof_spectrum:str,
    only_segmental:str,
    similarity_calculation:str
):

    """
    
    Input:
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - only_segmental: str e.g. "yes" or "no"

        - similarity_calculation: "inverse_distance", "exp_neg_distance"


    1) define imaginary coordinates for segmental contacts only or for all contacts
        - plot the imaginary contact coordinates using plotly

    
    
    2) Load the fooof dataframe and edit dataframe
        
        - check which sessions exist for this patient
        - only for segmental bipolar channels: add columns
            subject_hemisphere
            contact1
            contact2
            bip_chan
            coord_z = mean coordinate between contact 1 and 2
            coord_xy = mean coordinate between contact 1 and 2
            channel_group
        
        - delete all rows with Ring bipolar channels using drop NaN 

        save Dataframe for each session: session_data[f"{ses}_bipolar_Dataframe"]
            

    3) Calculate for each segmental contact the estimated PSD

        - new dataframe per session with segmental contacts as index

        - calculate an array with all euclidean distances based on both directions in z- and xy-axis

            diff_z = abs(coord_z - session_Dataframe_coord.loc[bipolar_channel, 'coord_z'])
            diff_xy = abs(coord_xy - session_Dataframe_coord.loc[bipolar_channel, 'coord_xy'])
            
            dist = np.sqrt(diff_z**2 + diff_xy**2)
        
        - compute similarity from distances

            similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x does't reach 0

        - normalization of similarity is not necessary when only using segmental bipolar recordings
        -> each contact should have the same similarities

        - weight the recorded psd in a frequency band: 

            for each bipolar segmental channel: 
            weighted_beta = averaged PSD * similarity
        
        -> monopolar estimated psd of one segmental contact = np.sum(weighted_beta)
        
        
    4) save the dictionary sub{sub}_{hemisphere}_monoRef_only_segmental_weight_psd_by_distance{freqBand}_{normalization}_{filterSignal}.pickle"

        - in results_path of the subject
        - keys of dictionary: 

            f"{ses}_bipolar_Dataframe" with bipolar content: contact1 and contact2 with coordinates and psd average of bipolar channels

            f"{ses}_monopolar_Dataframe" with monopolar content: contact, estimated_monopol_psd_{freqBand}, rank

            

    TODO: ask Rob again, is normalization of similarity in this case with only segmental contacts not necessary?

    """

    results_paths = find_folders.get_local_path(folder="GroupResults")

    #####################  defining the coordinates of monopolar contacts #####################
    # rcosθ+(rsinθ)i
    # z coordinates of the vertical axis
    # xy coordinates of the polar plane around the percept device

    #segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
    #segmental_channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
    incl_sessions = ["postop", "fu3m", "fu12m", "fu18m"]


    d = 2 # SenSight B33005: 0.5mm spacing between electrodes, 1.5mm electrode length
    # so 2mm from center of one contact to center of next contact
    
    r = 0.65 # change this radius as you wish - needs to be optimised

    if only_segmental == "yes":
        contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
        channels = ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
        contact_coordinates = {
                            '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
                            '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                            '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
                            '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
                            '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                            '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}

        # contact_coordinates = tuple z-coord + xy-coord

    elif only_segmental == "no":
        contacts = ["0", "3"]
        channels = ["01", "12", "23", "1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
        contact_coordinates = {
                        '0': [d*0.0,0+0*1j],
                        '1': [d*1.0,0+0*1j],
                        '2': [d*2.0,0+0*1j],
                        '3': [d*3.0,0+0*1j],
                        '1A':[d*1.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '1B':[d*1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '1C':[d*1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
                        '2A':[d*2.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '2B':[d*2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '2C':[d*2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}
    
    # contact_coordinates = tuple z-coord + xy-coord


    ##################### lets plot the monopolar contact coordinates! #####################
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    plot_data = []

    for contact in contact_coordinates.keys():
        zs = contact_coordinates[contact][0]
        
        y = contact_coordinates[contact][1]    
        xs = np.real(y)
        ys = np.imag(y)  

        trace = go.Scatter3d(
            x=np.array([xs]),  
            y=np.array([ys]),  
            z=np.array([zs]),  
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.8,
            }
        )
        plot_data.append(trace)

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    #data = [trace]

    plot_figure = go.Figure(data=plot_data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)


    #####################  Loading the Data #####################
    beta_average_DF = loadResults.load_fooof_beta_ranks(
        fooof_spectrum=fooof_spectrum,
        all_or_one_chan="beta_ranks_all"
    )

    # only take rows of channels of interest
    beta_average_DF = beta_average_DF.loc[beta_average_DF.bipolar_channel.isin(channels)]

    session_data = {}
    # loop over sessions
    for ses in incl_sessions:

        # check if session exists
        if ses not in beta_average_DF.session.values:
            continue 

        session_Dataframe = beta_average_DF[beta_average_DF.session==ses]
        # copying session_Dataframe to add new columns
        session_Dataframe_coord = session_Dataframe.copy()
        session_Dataframe_coord = session_Dataframe_coord.reset_index()
        session_Dataframe_coord = session_Dataframe_coord.drop(columns=["index", "level_0"])

        ##################### looping over bipolar channels, get average coordinates between 2 contacts #####################
        for idx in session_Dataframe_coord.index:
            
            # extracting contact names
            bipolar_channel = session_Dataframe_coord.loc[idx,'bipolar_channel'] # e.g. 1A2A
            
            # extracting individual monopolar contact names from bipolar channels
            if len(bipolar_channel)==4: # if len ==4 e.g. 1A2A
                contact_1 = bipolar_channel[:2] # 1A
                contact_2 = bipolar_channel[2:] # 2A
                #channel_group = "segments"    

            elif len(bipolar_channel)==2:

                if only_segmental == "yes":
                    continue
                else:
                    contact_1 = bipolar_channel[0] # 0
                    contact_2 = bipolar_channel[1] # 1   
                    #channel_group = "ring"   
            
            # storing monopolar contact names for bipolar contacts
            # e.g. channel 1A2A: contact1 = 1A, contact2 = 2A
            session_Dataframe_coord.loc[idx,'contact1'] = contact_1 
            session_Dataframe_coord.loc[idx,'contact2'] = contact_2

            # extracting coordinates of each monopolar contact from dictionary contact_coordinates
            coords1 = contact_coordinates[contact_1]
            coords2 = contact_coordinates[contact_2]

            # computing mean distance between monopolar contacts to get the bipolar average coordinate
            # coords1, e.g. contact 1A -> tuple of (z-coordinates, xy-coordinates)
            z_av = np.mean([coords1[0],coords2[0]]) # average of z-coord from contact1 and contact2
            xy_av = np.mean([coords1[1],coords2[1]]) # average of xy-coord from contact1 and contact2

            # storing new coordinates of bipolar contacts
            session_Dataframe_coord.loc[idx,'coord_z'] = z_av # mean of z-coordinates from contact 1 and 2
            session_Dataframe_coord.loc[idx,'coord_xy'] = xy_av # mean of xy-coordinates from contact 1 and 2


        # store copied and modified session_Dataframe into session dictionary
        session_data[f"{ses}_bipolar_Dataframe"]=session_Dataframe_coord




    ##################### Calculate beta psd average of each monopolar contact from all averaged coordinates #####################

    for ses in incl_sessions:

        session_data[f"{ses}_monopolar_Dataframe"] = pd.DataFrame()

        ses_dataframe = session_data[f"{ses}_bipolar_Dataframe"]

        stn_list = list(ses_dataframe.subject_hemisphere.unique())

        for stn in stn_list:

            # only select bipolar channels of this stn and session
            stn_ses_bipolar = ses_dataframe.loc[ses_dataframe.subject_hemisphere == stn]
            stn_ses_bipolar = stn_ses_bipolar.reset_index()

            # Create Dataframe with the coordinates of 6 contact coordinates: 1A, 1B, 1C, 2A, 2B, 2C
            mono_data = pd.DataFrame(contact_coordinates).T
            mono_data.columns = ['coord_z','coord_xy'] # columns with z- and xy-coordinates of each contact

            # copy mono_data dataframe to add new columns
            mono_data_copy = mono_data.copy()

            mono_data_copy["session"] = f"{ses}"
            mono_data_copy["subject_hemisphere"] = f"{stn}"

            for contact in contacts:

                # extracting coordinates for mono polar contacts
                coord_z = mono_data.loc[contact,'coord_z']
                coord_xy = mono_data.loc[contact,'coord_xy']
                
                # loop over all bipolar channels and compute distance to monopolar contact
                all_dists = [] # list of all distances from all averaged bipolar coordinates to one monopolar coordinate

                for bipolar_channel in stn_ses_bipolar.index:

                    # finding difference from the monopolar contact to each bipolar mean coordinates
                    diff_z = abs(coord_z - stn_ses_bipolar.loc[bipolar_channel, 'coord_z'])
                    diff_xy = abs(coord_xy - stn_ses_bipolar.loc[bipolar_channel, 'coord_xy'])
                    
                    # compute euclidean distance based on both directions
                    # Pythagoras: a^2 + b^2 = c^2
                    # a=difference of z-coord
                    # b=difference of xy-coord
                    # c=distance of interest 
                    dist = np.sqrt(diff_z**2 + diff_xy**2)
                    
                    #append the distance
                    all_dists.append(dist)
                
                # collect all distances in numpy array    
                all_dists = np.array(all_dists)
                
                # compute similarity from distances 
                if similarity_calculation == "inverse_distance":
                    similarity = 1/all_dists

                elif similarity_calculation == "exp_neg_distance":
                    similarity = np.exp(-all_dists) # alternative to 1/x, but exp^-x doesn´t reach 0
                

                # weighting the beta of bipolar contacts by their similarity to the monopolar contact
                weighted_beta = stn_ses_bipolar['beta_average'].values * similarity # two arrays with same length = 9 bip_chans

                # storing the weighted beta for the mono polar contact
                mono_data_copy.loc[contact,'estimated_monopolar_beta_psd'] = np.sum(weighted_beta) # sum of all 9 weighted psdAverages = one monopolar contact psdAverage

                # add column with contact
                mono_data_copy.loc[contact, 'contact'] = contact

            # ranking the weighted monopolar psd
            mono_data_copy["rank"] = mono_data_copy["estimated_monopolar_beta_psd"].rank(ascending=False) # rank highest psdAverage as 1.0

            session_data[f"{ses}_monopolar_Dataframe"] = pd.concat([session_data[f"{ses}_monopolar_Dataframe"], mono_data_copy])

    if only_segmental == "yes":
        filename = "only_segmental_"
    
    else: 
        filename = "segments_and_rings_"
        

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    session_data_filepath = os.path.join(results_paths, f"fooof_monoRef_{filename}weight_beta_psd_by_{similarity_calculation}_{fooof_spectrum}.pickle")
    with open(session_data_filepath, "wb") as file:
        pickle.dump(session_data, file)

    print(f"New file: fooof_monoRef_{filename}weight_beta_psd_by_{similarity_calculation}_{fooof_spectrum}.pickle",
            f"\nwritten in: {results_paths}" )
    

    return session_data  
           




def fooof_monoRef_weight_psd_by_distance_all_contacts(
        similarity_calculation:str
):
    """

    Input: 

        - similarity_calculation: "inverse_distance", "exp_neg_distance"

    merging the monopolar estimated beta power from segmented contacts only from segmental channels 
    and the ring contacts (0 and 3) from all 13 bipolar channels 
    
    """

    results_paths = find_folders.get_local_path(folder="GroupResults")

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]

    # load the dataframes from segmented and ring contacts
    segmented_data = fooof_monoRef_weight_psd_by_distance_segm_or_ring(
        fooof_spectrum = "periodic_spectrum",
        only_segmental="yes",
        similarity_calculation=similarity_calculation
    )

    ring_data = fooof_monoRef_weight_psd_by_distance_segm_or_ring(
        fooof_spectrum = "periodic_spectrum",
        only_segmental="no",
        similarity_calculation=similarity_calculation
    )

    # clean up the dataframes

    merged_data = pd.DataFrame()

    for ses in sessions:

        segmented_clean_data = segmented_data[f"{ses}_monopolar_Dataframe"] # DF of only one session
        segmented_clean_data = segmented_clean_data.dropna()

        ring_clean_data = ring_data[f"{ses}_monopolar_Dataframe"] # DF of only one session
        ring_clean_data = ring_clean_data.dropna()

        # merge into complete dataframe
        merged_data = pd.concat([merged_data, segmented_clean_data, ring_clean_data], ignore_index=True)

    all_ranked_data = pd.DataFrame()
    # rank again within each electrode
    electrodes_list = list(merged_data.subject_hemisphere.unique())
    electrodes_list.sort()

    for electrode in electrodes_list:

        electrode_data = merged_data.loc[merged_data.subject_hemisphere == electrode]

        for ses in sessions:

            if ses not in electrode_data.session.values:
                continue

            electrode_session_data = electrode_data.loc[electrode_data.session == ses]

            # rank estimated monopolar beta of all 8 contacts
            electrode_session_copy = electrode_session_data.copy()
            electrode_session_copy["rank_8"] = electrode_session_copy["estimated_monopolar_beta_psd"].rank(ascending=False)
            electrode_session_copy = electrode_session_copy.drop(columns=["rank"])
            electrode_session_copy = electrode_session_copy.reset_index()

            # add column with relative beta power to beta rank 1 power
            beta_rank_1 = electrode_session_copy[electrode_session_copy["rank_8"] == 1.0]
            beta_rank_1 = beta_rank_1["estimated_monopolar_beta_psd"].values[0] # just taking psdAverage of rank 1.0

            electrode_session_copy["beta_psd_rel_to_rank1"] = electrode_session_copy.apply(lambda row: row["estimated_monopolar_beta_psd"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1


            # add column with relative beta power to beta rank1 and rank8, so values ranging from 0 to 1
            # value of rank 8 
            beta_rank_8 = electrode_session_copy[electrode_session_copy["rank_8"] == 8.0]
            beta_rank_8 = beta_rank_8["estimated_monopolar_beta_psd"].values[0] # just taking psdAverage of rank 8.0

            beta_rank_1 = beta_rank_1 - beta_rank_8 # this is necessary to get value 1.0 after dividng the subtracted PSD value of rank 1 by itself

            # in each row add in new column: (psd-beta_rank_8)/beta_rank1
            electrode_session_copy["beta_psd_rel_range_0_to_1"] = electrode_session_copy.apply(lambda row: (row["estimated_monopolar_beta_psd"] - beta_rank_8) / beta_rank_1, axis=1) 

            # save
            all_ranked_data = pd.concat([all_ranked_data, electrode_session_copy], ignore_index=True)

    # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
    all_ranked_datapath = os.path.join(results_paths, f"fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}.pickle")
    with open(all_ranked_datapath, "wb") as file:
        pickle.dump(all_ranked_data, file)

    print(f"New file: fooof_monoRef_all_contacts_weight_beta_psd_by_{similarity_calculation}.pickle",
            f"\nwritten in: {results_paths}" )
        

    return all_ranked_data














    