""" monopolar Referencing: Robert approach """


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import plotly
import plotly.graph_objs as go

# utility functions
import analysis.utils.loadResults as loadResults
import analysis.utils.find_folders as find_folders




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

    


    #####################  defining the coordinates of monopolar contacts #####################
    # rcosθ+(rsinθ)i
    # z coordinates of the vertical axis
    # xy coordinates of the polar plane around the percept device

    r = 1 # change this radius as you wish - needs to be optimised
    contact_coordinates = {'0': [0.0,0+0*1j],
                        '1': [1.0,0+0*1j],
                        '2': [2.0,0+0*1j],
                        '3': [3.0,0+0*1j],
                        '1A':[1.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '1B':[1.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '1C':[1.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)],
                        '2A':[2.0,r*np.cos(0)+r*1j*np.sin(0)],
                        '2B':[2.0,r*np.cos(2*np.pi/3)+r*1j*np.sin(2*np.pi/3)],
                        '2C':[2.0,r*np.cos(4*np.pi/3)+r*1j*np.sin(4*np.pi/3)]}
    

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
            
            

        # store copied and modified mono data into session dictionary
        session_data[f"{ses}_monopolar_Dataframe"]=mono_data_psdAverage


        # save session_data dictionary with bipolar and monopolar psd average Dataframes as pickle files
        session_data_filepath = os.path.join(results_path, )




    

    return {
        "session_data":session_data,
        "sessionDF": session_Dataframe
        
    }
















    