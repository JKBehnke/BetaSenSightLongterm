""" Group all PSD monopolar averages and ranks """



import os
import pandas as pd
import itertools
from scipy import stats
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle

# local Imports
from .. utils import find_folders as findfolders
from .. utils import loadResults as loadResults



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




def monopol_psd_correlations_sessions(
        freqBand:str,
        ranks_or_relPsd:str,
        mean_or_median:str
        
):

    """
    Load file: GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_beta_rawPsd_band-pass.pickle
    containing DF with all monopolar PSD estimates in a frequency band, their ranks along an electrode and their PSD relative to the highest PSD of an electrode.


    Input:
        - freqBand: str, e.g. "beta"
        - ranks_or_relPsd: str, e.g. "ranks", "relPsd"
        - mean_or_median: str, e.g. "mean", "median"

    1) After loading the data, only select the contacts 0, 1A, 1B, 1C, 2A, 2B, 2C and 3
        - rank again from 1-8 -> column "Rank8contacts"
        - calculate the relative PSD normalized to the highest PSD within an electrode -> column "relativePSD_to_beta_Rank1from8"
    
    2) Use scipy.stats.spearmanr to correlate each STN electrode at two sessions
        - choose between ranks or rel PSD normalized to the highest PSD per electrode
        - STN = one hemisphere of one subject
        - pairs of sessions: 
            [('postop', 'postop'),
            ('postop', 'fu3m'),
            ('postop', 'fu12m'),
            ('postop', 'fu18m'),
            ('fu3m', 'postop'),
            ('fu3m', 'fu3m'),
            ('fu3m', 'fu12m'),
            ('fu3m', 'fu18m'),
            ('fu12m', 'postop'),
            ('fu12m', 'fu3m'),
            ('fu12m', 'fu12m'),
            ('fu12m', 'fu18m'),
            ('fu18m', 'postop'),
            ('fu18m', 'fu3m'),
            ('fu18m', 'fu12m'),
            ('fu18m', 'fu18m')]
    
    3) save values in results_DF with columns:
        - session_1
        - session_2
        - subject_hemisphere
        - spearman_r (r value from -1 to 1, 0=no correlation; 1=positive correlation, ???)
        - pval
    
    4) Calculate the Mean or Median of all STN correlation r values per session combination

    5) Restructure the column of means or medians to 4x4 matrices with floats

    6) Plot a Heatmap using plotly visualizing the mean or medians of all session combinations


    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    contacts = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
    pairs = list(itertools.product(sessions, sessions))
    comparison = ["postop_postop", "postop_fu3m", "postop_fu12m", "postop_fu18m", 
                  "fu3m_fu3m", "fu3m_fu12m", "fu3m_fu18m",
                  "fu12m_fu12m", "fu12m_fu18m", "fu18m_fu18m"]

    # load the monolopar beta psd for for each electrode at different timepoints
    data_weightedByCoordinates = loadResults.load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
        freqBand=freqBand,
        normalization="rawPsd",
        filterSignal="band-pass")
    
    # first check, which STNs and sessions exist in data 
    sub_hem_keys = list(data_weightedByCoordinates.subject_hemisphere.unique())

    weightedByCoordinate_Dataframe = pd.DataFrame() # concat all Dataframes from all sub, hem, sessions
    sample_size_dict = {}

    ################## CHOOSE ONLY 8 CONTACTS AND RANK AGAIN ##################
    for STN in sub_hem_keys:

        # select only one STN
        STN_data = data_weightedByCoordinates[data_weightedByCoordinates.subject_hemisphere == STN]

        for ses in sessions:

            # first check, if session exists in STN data
            if ses not in STN_data.session.values:
                continue
            
            # get the dataframe per session
            STN_session_data = STN_data[STN_data.session == ses]

            # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
            STN_session_data = STN_session_data[STN_session_data["contact"].isin(contacts)]
            STN_session_data["Rank8contacts"] = STN_session_data["averaged_monopolar_PSD_beta"].rank(ascending=False) # ranks 1-8
            STN_session_data_copy = STN_session_data.copy()
            STN_session_data_copy.drop(["rank"], axis=1, inplace=True)

            # calculate the relative PSD to the highest PSD of the 8 remaining contacts
            beta_rank_1 = STN_session_data_copy[STN_session_data_copy["Rank8contacts"] == 1.0] # taking the row containing 1.0 in rank
            beta_rank_1 = beta_rank_1[f"averaged_monopolar_PSD_{freqBand}"].values[0] # just taking psdAverage of rank 1.0

            STN_session_data_copy[f"relativePSD_to_{freqBand}_Rank1from8"] = STN_session_data_copy.apply(lambda row: row[f"averaged_monopolar_PSD_{freqBand}"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1
            STN_session_data_copy.drop([f"relativePSD_to_{freqBand}_Rank1"], axis=1, inplace=True)
            # session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)

            weightedByCoordinate_Dataframe = pd.concat([weightedByCoordinate_Dataframe, STN_session_data_copy], ignore_index=True)


    ################## CORRELATE RANKS OR REL PSD TO HIGHEST PSD BETWEEN ALL SESSION COMBINATIONS ##################
    sub_hem_keys = list(weightedByCoordinate_Dataframe.subject_hemisphere.unique()) # n=13 subjects

    # from the list of all existing sub_hem STNs, get only the STNs with existing sessions 1 + 2 
    session_pair_STNs = {}

    # find session pairs to correlate: check for each pair if subject exists
    for p, pair in enumerate(pairs): # pair e.g. ["postop", "fu3m"]

        # define session 1 and session 2 to correlate
        session_1 = pair[0] # e.g. "postop"
        session_2 = pair[1] # e.g. "fu3m"

        session_1_df = weightedByCoordinate_Dataframe.loc[(weightedByCoordinate_Dataframe.session == session_1)]
        session_2_df = weightedByCoordinate_Dataframe.loc[(weightedByCoordinate_Dataframe.session == session_2)]

        for STN in sub_hem_keys:

            # only run, if sub_hem STN exists in both session Dataframes
            if STN not in session_1_df.subject_hemisphere.values:
                continue

            elif STN not in session_2_df.subject_hemisphere.values:
                continue
            
            # get the rows with the current STN of both sessions: 1 and 2
            STN_session_1 = session_1_df.loc[(session_1_df.subject_hemisphere == STN)]
            STN_session_2 = session_2_df.loc[(session_2_df.subject_hemisphere == STN)]

            if ranks_or_relPsd == "ranks":

                # correlate the ranks of session 1 and session 2 of each STN 
                spearman_correlation = stats.spearmanr(STN_session_1.Rank8contacts.values, STN_session_2.Rank8contacts.values)
            
            elif ranks_or_relPsd == "relPsd":

                # correlate the rel Psd values normalized to highest PSD per electrode of session 1 and session 2 of each STN 
                spearman_correlation = stats.spearmanr(STN_session_1.relativePSD_to_beta_Rank1from8.values, STN_session_2.relativePSD_to_beta_Rank1from8.values)
            
            # dictionary to store the spearman r values and pval per STN and session combination
            session_pair_STNs[f"{session_1}_{session_2}_{STN}"] = [session_1, session_2, f"{session_1}_{session_2}", STN, spearman_correlation.statistic, spearman_correlation.pvalue]

    # save the dictionary as a Dataframe
    results_DF = pd.DataFrame(session_pair_STNs)
    results_DF.rename(index={0: "session_1", 1: "session_2", 2: "session_comparison", 3: "subject_hemisphere", 4: f"spearman_r", 5: f"pval"}, inplace=True)
    results_DF = results_DF.transpose()

    # get sample size
    for s_comp in comparison:

        s_comp_df = results_DF.loc[results_DF.session_comparison == s_comp]
        s_comp_count = s_comp_df["session_comparison"].count()

        sample_size_dict[f"{s_comp}"] = [s_comp, s_comp_count]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(index={0: "session_comparison", 1: "sample_size"}, inplace=True)
    sample_size_df = sample_size_df.transpose()

    ################## CALCULATE THE MEAN OR MEDIAN OF ALL SPEARMAN R CORRELATION VALUES OF EACH SESSION COMBINATION ##################
    spearman_m = {}

    # calculate the MEAN or median of each session pair
    for p, pair in enumerate(pairs):

        # define session 1 and session 2 to correlate
        session_1 = pair[0]
        session_2 = pair[1]

        pairs_df = results_DF.loc[(results_DF.session_1 == session_1)]
        pairs_df = pairs_df.loc[(pairs_df.session_2 == session_2)]

        if mean_or_median == "mean":
            m_spearmanr = pairs_df.spearman_r.mean() 
            m_pval = pairs_df.pval.mean()
        
        elif mean_or_median == "median":
            m_spearmanr = pairs_df.spearman_r.median() 
            m_pval = pairs_df.pval.median()

        spearman_m[f"{session_1}_{session_2}_spearman_m"] = [session_1, session_2, m_spearmanr, m_pval]

    # write a Dataframe with the mean or median spearman values per session combination
    spearman_m_df = pd.DataFrame(spearman_m)
    spearman_m_df.rename(index={0: "session_1", 1: "session_2", 2: f"{mean_or_median}_spearmanr", 3: f"{mean_or_median}_pval"}, inplace=True)
    spearman_m_df = spearman_m_df.transpose()

    ################## PLOT A HEAT MAP OF SPEARMAN CORRELATION MEAN OR MEDIAN VALUES PER SESSION COMBINATION ##################

    # transform spearman mean or median values to floats and 4x4 matrices
    if mean_or_median == "mean":
        spearmanr_to_plot = spearman_m_df.mean_spearmanr.values.astype(float)
        # reshape the mean of spearman r values into 4x4 matrix
        spearmanr_to_plot = spearmanr_to_plot.reshape(4,4)

    elif mean_or_median == "median":
        spearmanr_to_plot = spearman_m_df.median_spearmanr.values.astype(float)
        # reshape the medians of spearman r values into 4x4 matrix
        spearmanr_to_plot = spearmanr_to_plot.reshape(4,4)
    
    # # plot a heatmap
    # fig = px.imshow(spearmanr_to_plot,
    #             labels=dict(x="session 1", y="session 2", color=f"spearman correlation {mean_or_median}"),
    #             x=['postop', 'fu3m', 'fu12m', 'fu18m'],
    #             y=['postop', 'fu3m', 'fu12m', 'fu18m'],
    #             title=f"Correlation per electrode of {freqBand} {ranks_or_relPsd}",
    #             text_auto=True
    #            )
    
    # fig.update_xaxes(side="top")
    # fig.update_layout(title={
    #     'y':0.98,
    #     'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'})
    # fig.show()
    # fig.write_image(os.path.join(figures_path, f"Monopol_session_correlations_heatmap_{freqBand}_{ranks_or_relPsd}_{mean_or_median}.png"))
    
    # plot a heatmap
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(spearmanr_to_plot, cmap=plt.cm.YlOrRd)
    # other color options: GnBu, YlOrRd, YlGn, Greys, Blues, PuBuGn, YlGnBu

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(spearmanr_to_plot.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(spearmanr_to_plot.shape[0])+0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)
    ax.set_yticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)

    
    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label(f"spearman correlation {mean_or_median}")

    # Add the cell values to the heatmap
    for i in range(spearmanr_to_plot.shape[0]):
        for j in range(spearmanr_to_plot.shape[1]):
            plt.text(j + 0.5, i + 0.5, str("{: .2f}".format(spearmanr_to_plot[i, j])), ha='center', va='center') # only show 2 numbers after the comma of a float

    # Add a title
    plt.title(f"correlation of {freqBand} {ranks_or_relPsd}")

    fig.tight_layout()
    fig.savefig(figures_path + f"\\monopol_session_correlations_heatmap_{freqBand}_{ranks_or_relPsd}_{mean_or_median}.png", bbox_inches="tight")




    return {
        "results_DF":results_DF,
        "spearman_m_df":spearman_m_df,
        "sample_size": sample_size_df
    }


def fooof_monopol_psd_spearman_betw_sessions(
        mean_or_median:str,
        only_segmental:str,
        values_to_correlate:str        
        
):

    """
    Load file: 
    containing DF with all monopolar PSD estimates in a frequency band, their ranks along an electrode and their PSD relative to the highest PSD of an electrode.


    Input:
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            
        
        - mean_or_median: str, e.g. "mean", "median"
        - only_segmental:str, "yes" -> will only included segmental contacts
        - values_to_correlate:str  "not_normalized", "rel_to_rank_1", "rel_range_0_to_1" (only "not_normalized" can be used for only segmental, because the relative values were calucalted with ring contacts included)


    1) After loading the data, only select the contacts 0, 1A, 1B, 1C, 2A, 2B, 2C and 3
        - rank again from 1-8 -> column "Rank8contacts"
    
    2) Use scipy.stats.spearmanr to correlate each STN electrode at two sessions
        - choose between ranks or rel PSD normalized to the highest PSD per electrode
        - STN = one hemisphere of one subject
        - pairs of sessions: 
            [('postop', 'postop'),
            ('postop', 'fu3m'),
            ('postop', 'fu12m'),
            ('postop', 'fu18m'),
            ('fu3m', 'postop'),
            ('fu3m', 'fu3m'),
            ('fu3m', 'fu12m'),
            ('fu3m', 'fu18m'),
            ('fu12m', 'postop'),
            ('fu12m', 'fu3m'),
            ('fu12m', 'fu12m'),
            ('fu12m', 'fu18m'),
            ('fu18m', 'postop'),
            ('fu18m', 'fu3m'),
            ('fu18m', 'fu12m'),
            ('fu18m', 'fu18m')]
    
    3) save values in results_DF with columns:
        - session_1
        - session_2
        - subject_hemisphere
        - spearman_r (r value from -1 to 1, 0=no correlation; 1=positive correlation, ???)
        - pval
    
    4) Calculate the Mean or Median of all STN correlation r values per session combination

    5) Restructure the column of means or medians to 4x4 matrices with floats

    6) Plot a Heatmap using plotly visualizing the mean or medians of all session combinations


    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")

    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

    session_comparison = ["postop_postop", "postop_fu3m", "postop_fu12m", "postop_fu18m", 
                  "fu3m_postop", "fu3m_fu3m", "fu3m_fu12m", "fu3m_fu18m",
                  "fu12m_postop", "fu12m_fu3m", "fu12m_fu12m", "fu12m_fu18m", 
                  "fu18m_postop", "fu18m_fu3m", "fu18m_fu12m", "fu18m_fu18m"]
    

    loaded_fooof_monopolar = loadResults.load_fooof_monoRef_all_contacts_weight_beta()


    # from the list of all existing sub_hem STNs, get only the STNs with existing sessions 1 + 2 
    session_pair_stn_list = {}
    sample_size_dict = {}

    # for each session comparison select STNs that have recordings for both
    for comparison in session_comparison:

        both_sessions = list(comparison.split("_"))

        # define session 1 and 2
        session_1 = both_sessions[0] # e.g. "postop"
        session_2 = both_sessions[1] # e.g. "fu3m"

        session_1_df = loaded_fooof_monopolar.loc[loaded_fooof_monopolar.session == session_1]
        session_2_df = loaded_fooof_monopolar.loc[loaded_fooof_monopolar.session == session_2]

        #find STNs with both sessions
        session_1_stns = list(session_1_df.subject_hemisphere.unique())
        session_2_stns = list(session_2_df.subject_hemisphere.unique())

        stn_comparison_list = list(set(session_1_stns) & set(session_2_stns))
        stn_comparison_list.sort()

        comparison_df_1 = session_1_df.loc[session_1_df["subject_hemisphere"].isin(stn_comparison_list)]
        comparison_df_2 = session_2_df.loc[session_2_df["subject_hemisphere"].isin(stn_comparison_list)]

        comparsion_df = pd.concat([comparison_df_1, comparison_df_2], axis=0)

        if only_segmental == "yes":
            comparsion_df = comparsion_df.loc[comparsion_df.contact.isin(segmental_contacts)] # only rows with segmental contacts are included
            print("only segmental contacts included")
        
        else:
            print("all contacts included")

        # correlate each electrode seperately
        for sub_hem in stn_comparison_list:

            # only run, if sub_hem STN exists in both session Dataframes
            if sub_hem not in comparsion_df.subject_hemisphere.values:
                continue

            # only take one electrode at both sessions and get spearman correlation
            stn_comparison = comparsion_df.loc[comparsion_df["subject_hemisphere"] == sub_hem]

            stn_session1 = stn_comparison.loc[stn_comparison.session == session_1]
            stn_session2 = stn_comparison.loc[stn_comparison.session == session_2]

            # choose which values to correlate
            if values_to_correlate == "not_normalized":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(stn_session1.estimated_monopolar_beta_psd.values, stn_session2.estimated_monopolar_beta_psd.values)

            elif values_to_correlate == "rel_to_rank_1":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(stn_session1.beta_psd_rel_to_rank1.values, stn_session2.beta_psd_rel_to_rank1.values)
            
            elif values_to_correlate == "rel_range_0_to_1":
                # correlate the beta psd of both sessions to each other
                spearman_psd_stn = stats.spearmanr(stn_session1.beta_psd_rel_range_0_to_1.values, stn_session2.beta_psd_rel_range_0_to_1.values)


            # store values in a dictionary
            session_pair_stn_list[f"{comparison}_{sub_hem}"] = [session_1, session_2, comparison, sub_hem, spearman_psd_stn.statistic, spearman_psd_stn.pvalue]

            

    # save the dictionary as a Dataframe
    results_DF = pd.DataFrame(session_pair_stn_list)
    results_DF.rename(index={0: "session_1", 1: "session_2", 2: "session_comparison", 3: "subject_hemisphere", 4: f"spearman_r", 5: f"pval"}, inplace=True)
    results_DF = results_DF.transpose()

    # get sample size
    for s_comp in session_comparison:

        s_comp_df = results_DF.loc[results_DF.session_comparison == s_comp]
        s_comp_count = s_comp_df["session_comparison"].count()

        sample_size_dict[f"{s_comp}"] = [s_comp, s_comp_count]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(index={0: "session_comparison", 1: "sample_size"}, inplace=True)
    sample_size_df = sample_size_df.transpose()



    ################## CALCULATE THE MEAN OR MEDIAN OF ALL SPEARMAN R CORRELATION VALUES OF EACH SESSION COMBINATION ##################
    spearman_m = {}

    # calculate the MEAN or median of each session pair
    for comp in session_comparison:

        # define session 1 and session 2 to correlate
        both_sessions = list(comp.split("_"))

        # define session 1 and 2
        session_1 = both_sessions[0] # e.g. "postop"
        session_2 = both_sessions[1] # e.g. "fu3m"

        pairs_df = results_DF.loc[(results_DF.session_1 == session_1)]
        pairs_df = pairs_df.loc[(pairs_df.session_2 == session_2)]

        if mean_or_median == "mean":
            m_spearmanr = pairs_df.spearman_r.mean() 
            m_pval = pairs_df.pval.mean()
        
        elif mean_or_median == "median":
            m_spearmanr = pairs_df.spearman_r.median() 
            m_pval = pairs_df.pval.median()

        spearman_m[f"{comp}_spearman_m"] = [session_1, session_2, m_spearmanr, m_pval]

    # write a Dataframe with the mean or median spearman values per session combination
    spearman_m_df = pd.DataFrame(spearman_m)
    spearman_m_df.rename(index={0: "session_1", 1: "session_2", 2: f"{mean_or_median}_spearmanr", 3: f"{mean_or_median}_pval"}, inplace=True)
    spearman_m_df = spearman_m_df.transpose()

    ################## PLOT A HEAT MAP OF SPEARMAN CORRELATION MEAN OR MEDIAN VALUES PER SESSION COMBINATION ##################

    # transform spearman mean or median values to floats and 4x4 matrices
    if mean_or_median == "mean":
        spearmanr_to_plot = spearman_m_df.mean_spearmanr.values.astype(float)
        # reshape the mean of spearman r values into 4x4 matrix
        spearmanr_to_plot = spearmanr_to_plot.reshape(4,4)

    elif mean_or_median == "median":
        spearmanr_to_plot = spearman_m_df.median_spearmanr.values.astype(float)
        # reshape the medians of spearman r values into 4x4 matrix
        spearmanr_to_plot = spearmanr_to_plot.reshape(4,4)


    # plot a heatmap
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(spearmanr_to_plot, cmap=plt.cm.YlOrRd)
    # other color options: GnBu, YlOrRd, YlGn, Greys, Blues, PuBuGn, YlGnBu

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(spearmanr_to_plot.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(spearmanr_to_plot.shape[0])+0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)
    ax.set_yticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)


    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label(f"spearman correlation {mean_or_median}")

    # Add the cell values to the heatmap
    for i in range(spearmanr_to_plot.shape[0]):
        for j in range(spearmanr_to_plot.shape[1]):
            plt.text(j + 0.5, i + 0.5, str("{: .2f}".format(spearmanr_to_plot[i, j])), ha='center', va='center') # only show 2 numbers after the comma of a float

    # Add a title
    plt.title(f"{mean_or_median} of spearman correlation of beta psd")

    if only_segmental == "yes":
        file_add = "only_segmental"
    
    else:
        file_add = "all_contacts"

    fig.tight_layout()
    fig.savefig(figures_path + f"\\fooof_monopol_beta_correlations_{mean_or_median}_{file_add}_heatmap.png", bbox_inches="tight")
    fig.savefig(figures_path + f"\\fooof_monopol_beta_correlations_{mean_or_median}_{file_add}_heatmap.svg", bbox_inches="tight", format="svg")

    # save DF as pickle file
    spearman_m_df_filepath = os.path.join(results_path, f"fooof_monopol_beta_correlations_{mean_or_median}_{file_add}_heatmap.pickle")
    with open(spearman_m_df_filepath, "wb") as file:
        pickle.dump(spearman_m_df, file)

    print("file: ", 
          f"fooof_monopol_beta_correlations_{mean_or_median}_{file_add}_heatmap.pickle",
          "\nwritten in: ", results_path
          )


    return {
        "results_DF": results_DF,
        "sample_size_df": sample_size_df,
        "spearman_m_df": spearman_m_df
    }





def mono_rank_differences(
        freq_band:str,
        normalization:str,
        filter_signal:str,
        level_or_direction:str

):

    """
    load the group dataframe of monopolar estimated psd values:
        "GroupMonopolar_weightedPsdCoordinateDistance_relToRank1_beta_rawPsd_band-pass.pickle" 

    Input: 
        - freq_band: str, e.g. "beta", "lowBeta"
        - normalization: str, e.g. "rawPsd"
        - filter_signal: str, e.g. "band-pass"
        - level_or_direction: str, e.g. "level" or "direction"
    
    1) filter and edit the dataframe
        - only take relevant contacts (8)
        - rank contacts by their power in the freq_band
        - calculate the relative PSD to rank1
    
    2) for each session comparison:
        - check which stn have recordings at both sessions
        - for each rank from 1-8 compare the level of session 1 to session 2 
        - store the differences of levels (absolute values) into a dataframe
            


    """


    # load the monolopar beta psd for for each electrode at different timepoints
    data_weightedByCoordinates = loadResults.load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
        freqBand=freq_band,
        normalization=normalization,
        filterSignal=filter_signal
    )

    if level_or_direction == "level":
        contacts = ["0", "1A", "1B", "1C", "2A", "2B", "2C", "3"]
        ranks_range = [1, 2, 3, 4, 5, 6, 7, 8]
    
    elif level_or_direction == "direction":
        contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]
        ranks_range = [1, 2, 3, 4, 5, 6]

    sessions = ["postop", "fu3m", "fu12m", "fu18m"]
    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]

    weightedByCoordinate_Dataframe = pd.DataFrame() # concat all Dataframes from all sub, hem, sessions


    # first check, which STNs and sessions exist in data 
    sub_hem_keys = list(data_weightedByCoordinates.subject_hemisphere.unique())

    ################## CHOOSE ONLY 8 CONTACTS AND RANK AGAIN ##################
    for STN in sub_hem_keys:

        # select only one STN
        STN_data = data_weightedByCoordinates[data_weightedByCoordinates.subject_hemisphere == STN]


        for ses in sessions:

            # first check, if session exists in STN data
            if ses not in STN_data.session.values:
                continue

            
            # get the dataframe per session
            STN_session_data = STN_data[STN_data.session == ses]

            # choose only directional contacts and Ring contacts 0, 3 and rank again only the chosen contacts
            STN_session_data = STN_session_data[STN_session_data["contact"].isin(contacts)]
            STN_session_data["rank_contacts"] = STN_session_data["averaged_monopolar_PSD_beta"].rank(ascending=False) # ranks 1-8
            STN_session_data_copy = STN_session_data.copy()
            STN_session_data_copy.drop(["rank"], axis=1, inplace=True)

            # calculate the relative PSD to the highest PSD of the 8 remaining contacts
            beta_rank_1 = STN_session_data_copy[STN_session_data_copy["rank_contacts"] == 1.0] # taking the row containing 1.0 in rank
            beta_rank_1 = beta_rank_1[f"averaged_monopolar_PSD_{freq_band}"].values[0] # just taking psdAverage of rank 1.0

            STN_session_data_copy[f"relativePSD_to_{freq_band}_rank_contacts"] = STN_session_data_copy.apply(lambda row: row[f"averaged_monopolar_PSD_{freq_band}"] / beta_rank_1, axis=1) # in each row add to new value psd/beta_rank1
            STN_session_data_copy.drop([f"relativePSD_to_{freq_band}_Rank1"], axis=1, inplace=True)
            # session_weightedByCoordinates_copy["subject_hemisphere_monoChannel"] = session_weightedByCoordinates_copy[["subject_hemisphere", "monopolarChannels"]].agg("_".join, axis=1)
            

            weightedByCoordinate_Dataframe = pd.concat([weightedByCoordinate_Dataframe, STN_session_data_copy], ignore_index=True)


    ################## LEVEL DIFFERENCE OF EACH RANK ##################

    # replace all session names by integers
    weightedByCoordinate_Dataframe = weightedByCoordinate_Dataframe.replace(to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18])

    # Type of ranks should be integers 
    weightedByCoordinate_Dataframe["rank_contacts"] = weightedByCoordinate_Dataframe["rank_contacts"].astype(int)

    # new column with level of contact
    weightedByCoordinate_Dataframe_copy = weightedByCoordinate_Dataframe.copy()

    if level_or_direction == "level":
        weightedByCoordinate_Dataframe_copy = weightedByCoordinate_Dataframe_copy.assign(contact_level=weightedByCoordinate_Dataframe_copy["contact"]).rename(columns={"contact_level": "contact_level"})
        weightedByCoordinate_Dataframe_copy["contact_level"] = weightedByCoordinate_Dataframe_copy["contact_level"].replace(to_replace=["0", "3"], value=[0, 3]) # level 0 or 3
        weightedByCoordinate_Dataframe_copy["contact_level"] = weightedByCoordinate_Dataframe_copy["contact_level"].replace(to_replace=["1A", "1B", "1C"], value=[1, 1, 1]) # level 1
        weightedByCoordinate_Dataframe_copy["contact_level"] = weightedByCoordinate_Dataframe_copy["contact_level"].replace(to_replace=["2A", "2B", "2C"], value=[2, 2, 2]) # level 2
    
    elif level_or_direction == "direction":
        weightedByCoordinate_Dataframe_copy = weightedByCoordinate_Dataframe_copy.assign(contact_direction=weightedByCoordinate_Dataframe_copy["contact"]).rename(columns={"contact_direction": "contact_direction"})
        weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1A", "2A"], value=["A", "A"]) # direction A
        weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1B", "2B"], value=["B", "B"]) # direction B
        weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1C", "2C"], value=["C", "C"]) # ldirection C

    difference_dict = {}

    for comp in comparisons:

        comp_split = comp.split("_")
        session_1 = int(comp_split[0]) # first session as integer
        session_2 = int(comp_split[1])

        for stn in sub_hem_keys:

            # check for each STN, which ones have both sessions
            stn_dataframe = weightedByCoordinate_Dataframe_copy.loc[weightedByCoordinate_Dataframe_copy.subject_hemisphere == stn]

            if session_1 not in stn_dataframe.session.values:
                continue

            elif session_2 not in stn_dataframe.session.values:
                continue

            stn_session_1 = stn_dataframe.loc[stn_dataframe.session == session_1]
            stn_session_2 = stn_dataframe.loc[stn_dataframe.session == session_2]

            # go through each rank and calculate the difference of level between two sessions
            for rank in ranks_range:

                rank_session_1 = stn_session_1.loc[stn_session_1.rank_contacts == rank] # row of one rank of session 1
                rank_session_2 = stn_session_2.loc[stn_session_2.rank_contacts == rank] # row of one rank of session 2

                if level_or_direction == "level":

                    rank_contact_level_session_1 = rank_session_1.contact_level.values[0] # level of rank as integer
                    rank_contact_level_session_2 = rank_session_2.contact_level.values[0] # level of rank as integer

                    level_difference_rank = abs(rank_contact_level_session_1 - rank_contact_level_session_2) # difference of level as absolute number

                    # store in dictionary
                    difference_dict[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, rank, rank_contact_level_session_1, rank_contact_level_session_2, level_difference_rank]
                

                elif level_or_direction == "direction":

                    rank_contact_direction_session_1 = rank_session_1.contact_direction.values[0] # direction of rank as str
                    rank_contact_direction_session_2 = rank_session_2.contact_direction.values[0] # direction of rank as str

                    if rank_contact_direction_session_1 == rank_contact_direction_session_2:
                        direction_difference_rank = 0
                    
                    else:
                        direction_difference_rank = 1
                    
                    # store in dictionary
                    difference_dict[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, rank, rank_contact_direction_session_1, rank_contact_direction_session_2, direction_difference_rank]
                


    # transform dictionary to dataframe
    difference_df = pd.DataFrame(difference_dict)
    difference_df.rename(index={0: "session_comparison",
                                    1: "session_1",
                                    2: "session_2",
                                    3: "subject_hemisphere",
                                    4: "rank",
                                    5: f"{level_or_direction}_session_1",
                                    6: f"{level_or_direction}_session_2",
                                    7: f"{level_or_direction}_difference"}, 
                                    inplace=True)

    difference_df = difference_df.transpose()

    return difference_df




def mono_rank_difference_heatmap(
        freq_band:str,
        normalization:str,
        filter_signal:str,
        ranks_included:list,
        difference_to_plot:str,
        level_or_direction:str,
        only_segmental:str,
):
    """
    Research question: how many levels do beta ranks change over time across electrodes?

    Input: 
        - freq_band: str, e.g. "beta", "lowBeta"
        - normalization: str, e.g. "rawPsd"
        - filter_signal: str, e.g. "band-pass"
        - ranks_included: list, e.g. [1,2,3,4,5,6,7,8] or [1,2,3]
        - difference_to_plot:str, e.g. "1_or_less", "more_than_1", "more_than_0", "0", "1", "2", "3"
            -> defining what values to plot in the heatmap
            "1_or_less" will plot relative amount how often a difference <= 1 occured for each session comparison
        - level_or_direction: str, e.g. "level" or "direction"
        - only_segmental: str, "yes" -> then monopolar estimation method only includes segmental bipolar channels and calculates psd only for segmental contacts

    1) load the dataframe written by the function mono_rank_level_differences()
        - containing columns: session_comparison, session_1, session_2, subject_hemisphere, rank, level_session_1, level_session_2, level_abs_difference
        - filter by ranks_included: only keep rows of dataframe containing rank isin rank_included

    2) for each session comparison
        - count how often the level difference 0, 1, 2, 3 occurs or the direction difference 0 or 1
        - calculate relative amount of how often a level or direction difference occurs
    
    3) plot heatmap
        - 
    
    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")

    if only_segmental == "yes":
         # load the dataframe with differences of levels for each rank from 1 to 6 across electrodes
        difference_df = mono_rank_differences_only_segmental_rec_used(
            freq_band=freq_band,
            normalization=normalization,
            filter_signal=filter_signal,
        )

    else:
        # load the dataframe with differences of levels for each rank from 1 to 8 across electrodes
        difference_df = mono_rank_differences(
            freq_band=freq_band,
            normalization=normalization,
            filter_signal=filter_signal,
            level_or_direction=level_or_direction
        )

    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]
    
    session_comparison_difference_dict = {}

    # filter the dataframe and only keep ranks of interest
    if level_or_direction == "direction":
        if 7 or 8 in ranks_included:
            print(f"ranks_included: ranks allowed = [1, 2, 3, 4, 5, 6].")

    difference_df_ranks_included = difference_df[difference_df["rank"].isin(ranks_included)]

    group_description = {}

    for comp in comparisons:

        comp_dataframe = difference_df_ranks_included.loc[difference_df_ranks_included.session_comparison == comp]
        session_1 = comp_dataframe.session_1.values[0]
        session_2 = comp_dataframe.session_2.values[0]

        # quantify percentage of how often differences occur
        total_rank_comparisons = comp_dataframe[f"{level_or_direction}_difference"].count()

        # check if numbers exist for each difference value
        if 0 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_0 = 0
        else: 
            count_0 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[0]

        
        if 1 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_1 = 0
        else: 
            count_1 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[1]

        # differences of 2 and 3 only occur in level_difference
        if 2 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_2 = 0
        else: 
            count_2 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[2]

        
        if 3 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_3 = 0
        else: 
            count_3 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[3]

        count_1_or_less = count_0 + count_1
        count_more_than_1 = count_2 + count_3
        coun_more_than_0 = count_1 + count_2 + count_3

        # relative values to total rank comparisons
        rel_0 = count_0 / total_rank_comparisons
        rel_1 = count_1 / total_rank_comparisons
        rel_2 = count_2 / total_rank_comparisons
        rel_3 = count_3 / total_rank_comparisons

        rel_1_or_less = count_1_or_less / total_rank_comparisons
        rel_more_than_1 = count_more_than_1 / total_rank_comparisons
        rel_more_than_0 = coun_more_than_0 / total_rank_comparisons

        # save in a dict
        session_comparison_difference_dict[f"{comp}"] = [comp, session_1, session_2, total_rank_comparisons, 
                                                            rel_0, rel_1, rel_2, rel_3, rel_1_or_less, rel_more_than_1, rel_more_than_0]
        
        # describe group
        stn_count = len(list(comp_dataframe.subject_hemisphere.unique()))
        group_mean = np.mean(comp_dataframe[f"{level_or_direction}_difference"].values)
        group_std = np.std(comp_dataframe[f"{level_or_direction}_difference"].values)
        

        group_description[f"{comp}"] = [comp, total_rank_comparisons, stn_count, group_mean, group_std]
        

    # transform to dataframe
    session_comparison_difference_df = pd.DataFrame(session_comparison_difference_dict)
    session_comparison_difference_df.rename(index={
        0: "session_comparison",
        1: "session_1",
        2: "session_2", 
        3: "total_rank_comparisons",
        4: "rel_amount_difference_0",
        5: "rel_amount_difference_1",
        6: "rel_amount_difference_2",
        7: "rel_amount_difference_3",
        8: "rel_amount_difference_1_or_less",
        9: "rel_amount_difference_more_than_1",
        10: "rel_amount_difference_more_than_0",
    }, inplace=True)
    session_comparison_difference_df = session_comparison_difference_df.transpose()

    description_results = pd.DataFrame(group_description)
    description_results.rename(index={0: "session_comparison", 1: "number_of_observations", 2: "number_of_stn", 3: "mean", 4: "standard_deviation"}, inplace=True)
    description_results = description_results.transpose()
   

    ########################## PLOT HEATMAP OF REL AMOUNT OF DIFFERENCES IN LEVELS FOR RANKS ##########################

    if level_or_direction == "direction":
        if difference_to_plot not in ["0", "1"]:
            print(f"difference_to_plot: {difference_to_plot} must be in ['0', '1'].")

    # transform difference values to floats and 4x4 matrices
    if difference_to_plot == "1_or_less":
        difference = session_comparison_difference_df.rel_amount_difference_1_or_less.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_1_or_less"

    elif difference_to_plot == "more_than_1":
        difference = session_comparison_difference_df.rel_amount_difference_more_than_1.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_more_than_1"
    
    elif difference_to_plot == "more_than_0":
        difference = session_comparison_difference_df.rel_amount_difference_more_than_0.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_more_than_0"
    
    elif difference_to_plot == "0":
        difference = session_comparison_difference_df.rel_amount_difference_0.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_0"
    
    elif difference_to_plot == "1":
        difference = session_comparison_difference_df.rel_amount_difference_1.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_1"
    
    elif difference_to_plot == "2":
        difference = session_comparison_difference_df.rel_amount_difference_2.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_2"
    
    elif difference_to_plot == "3":
        difference = session_comparison_difference_df.rel_amount_difference_3.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_3"


    fig, ax = plt.subplots()

    heatmap = ax.pcolor(difference, cmap=plt.cm.YlOrRd)
    # other color options: GnBu, YlOrRd, YlGn, Greys, Blues, PuBuGn, YlGnBu

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(difference.shape[1])+0.5, minor=False) # if minor=True it will plot x and y labels differently
    ax.set_yticks(np.arange(difference.shape[0])+0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)
    ax.set_yticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)

    # Rotate the x-axis tick labels to be vertical
    # plt.xticks(rotation=90)

    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label(f"relative to total number of comparisons")

    # Add the cell values to the heatmap
    for i in range(difference.shape[0]):
        for j in range(difference.shape[1]):
            plt.text(j + 0.5, i + 0.5, str("{: .2f}".format(difference[i, j])), ha='center', va='center') # only show 2 numbers after the comma of a float

    plt.title(f"{level_or_direction} differences of {difference_to_plot} \nof {freq_band} ranks {ranks_included}")

    fig.tight_layout()

    if only_segmental == "yes":
        fig_name = f"\\monopol_only_segm_heatmap_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}_{freq_band}_{normalization}_{filter_signal}.png"
    
    else: 
        fig_name = f"\\monopol_heatmap_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}_{freq_band}_{normalization}_{filter_signal}.png"

    fig.savefig(figures_path + fig_name, bbox_inches="tight")


    return {
        "session_comparison_difference_df": session_comparison_difference_df,
        "description_results": description_results}



def mono_rank_differences_only_segmental_rec_used(
        freq_band:str,
        normalization:str,
        filter_signal:str,

):

    """
    load the group dataframe of monopolar estimated psd values:
        e.g. group_monoRef_only_segmental_weight_psd_by_distance_beta_rawPsd_band-pass.pickle

    Input: 
        - freq_band: str, e.g. "beta", "lowBeta"
        - normalization: str, e.g. "rawPsd"
        - filter_signal: str, e.g. "band-pass"
    
    1) for each session comparison:
        - check which stn have recordings at both sessions
        - for each rank from 1-6 compare the direction of session 1 to session 2 
        - store the differences of direction 0 or 1 into a dataframe
            


    """


    # load the monolopar beta psd for for each electrode at different timepoints
    data_weightedByCoordinates = loadResults.load_Group_monoRef_only_segmental_weight_psd_by_distance(
        freqBand=freq_band,
        normalization=normalization,
        filterSignal=filter_signal
    )

    
    ranks_range = [1, 2, 3, 4, 5, 6]

    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]


    # first check, which STNs and sessions exist in data 
    sub_hem_keys = list(data_weightedByCoordinates.subject_hemisphere.unique())


    ################## DIRECTION DIFFERENCE OF EACH RANK ##################

    # replace all session names by integers
    data_weightedByCoordinates = data_weightedByCoordinates.replace(to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18])

    # Type of ranks should be integers 
    data_weightedByCoordinates["rank"] = data_weightedByCoordinates["rank"].astype(int)

    # new column with direction of contact
    weightedByCoordinate_Dataframe_copy = data_weightedByCoordinates.copy()

    weightedByCoordinate_Dataframe_copy = weightedByCoordinate_Dataframe_copy.assign(contact_direction=weightedByCoordinate_Dataframe_copy["contact"]).rename(columns={"contact_direction": "contact_direction"})
    weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1A", "2A"], value=["A", "A"]) # direction A
    weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1B", "2B"], value=["B", "B"]) # direction B
    weightedByCoordinate_Dataframe_copy["contact_direction"] = weightedByCoordinate_Dataframe_copy["contact_direction"].replace(to_replace=["1C", "2C"], value=["C", "C"]) # ldirection C




    difference_dict = {}

    for comp in comparisons:

        comp_split = comp.split("_")
        session_1 = int(comp_split[0]) # first session as integer
        session_2 = int(comp_split[1])

        for stn in sub_hem_keys:

            # check for each STN, which ones have both sessions
            stn_dataframe = weightedByCoordinate_Dataframe_copy.loc[weightedByCoordinate_Dataframe_copy.subject_hemisphere == stn]

            if session_1 not in stn_dataframe.session.values:
                continue

            elif session_2 not in stn_dataframe.session.values:
                continue

            stn_session_1 = stn_dataframe.loc[stn_dataframe.session == session_1]
            stn_session_2 = stn_dataframe.loc[stn_dataframe.session == session_2]

            # go through each rank and calculate the difference of direction between two sessions
            for rank in ranks_range:

                rank_session_1 = stn_session_1.loc[stn_session_1["rank"] == rank] # row of one rank of session 1
                rank_session_2 = stn_session_2.loc[stn_session_2["rank"] == rank] # row of one rank of session 2

                
                rank_contact_direction_session_1 = rank_session_1.contact_direction.values[0] # direction of rank as str
                rank_contact_direction_session_2 = rank_session_2.contact_direction.values[0] # direction of rank as str

                if rank_contact_direction_session_1 == rank_contact_direction_session_2:
                    direction_difference_rank = 0
                
                else:
                    direction_difference_rank = 1
                
                # store in dictionary
                difference_dict[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, rank, rank_contact_direction_session_1, rank_contact_direction_session_2, direction_difference_rank]
            


    # transform dictionary to dataframe
    difference_df = pd.DataFrame(difference_dict)
    difference_df.rename(index={0: "session_comparison",
                                    1: "session_1",
                                    2: "session_2",
                                    3: "subject_hemisphere",
                                    4: "rank",
                                    5: "direction_session_1",
                                    6: "direction_session_2",
                                    7: "direction_difference"}, 
                                    inplace=True)

    difference_df = difference_df.transpose()

    return difference_df



def fooof_mono_rank_differences(
        fooof_spectrum:str,
        level_or_direction:str

):

    """
    load the group dataframe of monopolar estimated psd values:
        e.g. group_monoRef_only_segmental_weight_psd_by_distance_beta_rawPsd_band-pass.pickle

    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - only_segmental: str e.g. "yes" or "no"
    
    1) for each session comparison:
        - check which stn have recordings at both sessions
        - for each rank from 1-6 compare the direction of session 1 to session 2 
        - store the differences of direction 0 or 1 into a dataframe
            


    """

    segmental_contacts = ["1A", "1B", "1C", "2A", "2B", "2C"]

    if level_or_direction == "level":
        ranks_range = [1, 2, 3, 4, 5, 6, 7, 8]
        only_segmental = "no"

    
    elif level_or_direction == "direction":
        ranks_range = [1, 2, 3, 4, 5, 6]
        only_segmental = "yes"

    # load data
    if level_or_direction == "direction":

        loaded_fooof_monopolar = loadResults.load_fooof_monopolar_weighted_psd(
            fooof_spectrum=fooof_spectrum,
            segmental=only_segmental
            )
        
        fooof_monopolar_df = pd.concat([loaded_fooof_monopolar["postop_monopolar_Dataframe"],
                                    loaded_fooof_monopolar["fu3m_monopolar_Dataframe"],
                                    loaded_fooof_monopolar["fu12m_monopolar_Dataframe"],
                                    loaded_fooof_monopolar["fu18m_monopolar_Dataframe"]])
        
        fooof_monopolar_df_copy = fooof_monopolar_df.copy()
        fooof_monopolar_df_copy["rank_beta"] = fooof_monopolar_df["rank"].astype(int)
    
        

    elif level_or_direction == "level":
        fooof_monopolar_df = loadResults.load_fooof_monoRef_all_contacts_weight_beta()

        fooof_monopolar_df_copy = fooof_monopolar_df.copy()
        fooof_monopolar_df_copy["rank_beta"] = fooof_monopolar_df["rank_8"].astype(int)
    
    

    # replace session names by integers
    fooof_monopolar_df_copy = fooof_monopolar_df_copy.replace(to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0, 3, 12, 18])

    # add a column with the direction

    if level_or_direction == "level":
        fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(contact_level=fooof_monopolar_df_copy["contact"]).rename(columns={"contact_level":"contact_level"})
        fooof_monopolar_df_copy["contact_level"] = fooof_monopolar_df_copy["contact_level"].replace(to_replace=["0", "3"], value=[0, 3]) # level 0 or 3
        fooof_monopolar_df_copy["contact_level"] = fooof_monopolar_df_copy["contact_level"].replace(to_replace=["1A", "1B", "1C"], value=[1, 1, 1]) # level 1
        fooof_monopolar_df_copy["contact_level"] = fooof_monopolar_df_copy["contact_level"].replace(to_replace=["2A", "2B", "2C"], value=[2, 2, 2]) # level 2
    
    elif level_or_direction == "direction":
        fooof_monopolar_df_copy = fooof_monopolar_df_copy.assign(contact_direction=fooof_monopolar_df_copy["contact"]).rename(columns={"contact_direction":"contact_direction"})
        fooof_monopolar_df_copy["contact_direction"] = fooof_monopolar_df_copy["contact_direction"].replace(to_replace=["1A", "2A"], value=["A", "A"]) # direction A
        fooof_monopolar_df_copy["contact_direction"] = fooof_monopolar_df_copy["contact_direction"].replace(to_replace=["1B", "2B"], value=["B", "B"]) # direction B
        fooof_monopolar_df_copy["contact_direction"] = fooof_monopolar_df_copy["contact_direction"].replace(to_replace=["1C", "2C"], value=["C", "C"]) # ldirection C


    #################   VARIABLES   #################
    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]

    difference_dict = {}


    # first check, which STNs and sessions exist in data 
    sub_hem_keys = list(fooof_monopolar_df_copy.subject_hemisphere.unique())

    #################   CALCULATE THE DIFFERENCE OF DIRECTION FOR EACH RANK PER SESSION COMPARISON  #################
    for comp in comparisons:

        comp_split = comp.split("_")
        session_1 = int(comp_split[0]) # first session as integer
        session_2 = int(comp_split[1])

        for stn in sub_hem_keys:

            # check for each STN, which ones have both sessions
            stn_dataframe = fooof_monopolar_df_copy.loc[fooof_monopolar_df_copy.subject_hemisphere == stn]

            if session_1 not in stn_dataframe.session.values:
                continue

            elif session_2 not in stn_dataframe.session.values:
                continue

            stn_session_1 = stn_dataframe.loc[stn_dataframe.session == session_1]
            stn_session_2 = stn_dataframe.loc[stn_dataframe.session == session_2]

            # go through each rank and calculate the difference of direction between two sessions
            for rank in ranks_range:

                rank_session_1 = stn_session_1.loc[stn_session_1["rank_beta"] == rank] # row of one rank of session 1
                rank_session_2 = stn_session_2.loc[stn_session_2["rank_beta"] == rank] # row of one rank of session 2

                if level_or_direction == "level":

                    rank_contact_level_session_1 = rank_session_1.contact_level.values[0] # level of rank as integer
                    rank_contact_level_session_2 = rank_session_2.contact_level.values[0] # level of rank as integer

                    level_difference_rank = abs(rank_contact_level_session_1 - rank_contact_level_session_2) # difference of level as absolute number

                    # store in dictionary
                    difference_dict[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, rank, rank_contact_level_session_1, rank_contact_level_session_2, level_difference_rank]
                

                elif level_or_direction == "direction":

                    rank_contact_direction_session_1 = rank_session_1.contact_direction.values[0] # direction of rank as str
                    rank_contact_direction_session_2 = rank_session_2.contact_direction.values[0] # direction of rank as str

                    if rank_contact_direction_session_1 == rank_contact_direction_session_2:
                        direction_difference_rank = 0
                    
                    else:
                        direction_difference_rank = 1
                    
                    # store in dictionary
                    difference_dict[f"{comp}_{stn}_{rank}"] = [comp, session_1, session_2, stn, rank, rank_contact_direction_session_1, rank_contact_direction_session_2, direction_difference_rank]


    # transform dictionary to dataframe
    difference_df = pd.DataFrame(difference_dict)
    difference_df.rename(index={0: "session_comparison",
                                    1: "session_1",
                                    2: "session_2",
                                    3: "subject_hemisphere",
                                    4: "rank",
                                    5: f"{level_or_direction}_session_1",
                                    6: f"{level_or_direction}_session_2",
                                    7: f"{level_or_direction}_difference"}, 
                                    inplace=True)

    difference_df = difference_df.transpose()

    return difference_df



def fooof_mono_rank_difference_heatmap(
        fooof_spectrum:str,
        ranks_included:list,
        difference_to_plot:str,
        level_or_direction:str,
):
    """
    Research question: how many levels do beta ranks change over time across electrodes?

    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit
        
        - only_segmental: str e.g. "yes" or "no"
        
        - ranks_included: list, e.g. [1,2,3,4,5,6,7,8] or [1,2,3]
        - difference_to_plot:str, e.g. "1_or_less", "more_than_1", "more_than_0", "0", "1", "2", "3"
            -> defining what values to plot in the heatmap
            "1_or_less" will plot relative amount how often a difference <= 1 occured for each session comparison
        - level_or_direction: str, e.g. "level" or "direction"
            direction -> uses only segmental recordings!
            level -> uses segments and rings!
        
    1) load the dataframe written by the function mono_rank_level_differences()
        - containing columns: session_comparison, session_1, session_2, subject_hemisphere, rank, level_session_1, level_session_2, level_abs_difference
        - filter by ranks_included: only keep rows of dataframe containing rank isin rank_included

    2) for each session comparison
        - count how often the level difference 0, 1, 2, 3 occurs or the direction difference 0 or 1
        - calculate relative amount of how often a level or direction difference occurs
    
    3) plot heatmap
        - 
    
    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")

    # load the dataframe with differences of levels for each rank from 1 to 6 across electrodes
    difference_df = fooof_mono_rank_differences(
        fooof_spectrum=fooof_spectrum,
        level_or_direction=level_or_direction
    )
    

    comparisons = ["0_0", "0_3", "0_12", "0_18", 
                    "3_0", "3_3", "3_12", "3_18", 
                    "12_0", "12_3", "12_12", "12_18",
                    "18_0", "18_3", "18_12", "18_18"]
    
    session_comparison_difference_dict = {}

    # filter the dataframe and only keep ranks of interest
    if level_or_direction == "direction":
        if 7 or 8 in ranks_included:
            print(f"ranks_included: ranks allowed = [1, 2, 3, 4, 5, 6].")

    difference_df_ranks_included = difference_df[difference_df["rank"].isin(ranks_included)]

    group_description = {}

    for comp in comparisons:

        comp_dataframe = difference_df_ranks_included.loc[difference_df_ranks_included.session_comparison == comp]
        session_1 = comp_dataframe.session_1.values[0]
        session_2 = comp_dataframe.session_2.values[0]

        # quantify percentage of how often differences occur
        total_rank_comparisons = comp_dataframe[f"{level_or_direction}_difference"].count()

        # check if numbers exist for each difference value
        if 0 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_0 = 0
        else: 
            count_0 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[0]

        
        if 1 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_1 = 0
        else: 
            count_1 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[1]

        # differences of 2 and 3 only occur in level_difference
        if 2 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_2 = 0
        else: 
            count_2 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[2]

        
        if 3 not in comp_dataframe[f"{level_or_direction}_difference"].values:
            count_3 = 0
        else: 
            count_3 = comp_dataframe[f"{level_or_direction}_difference"].value_counts()[3]

        count_1_or_less = count_0 + count_1
        count_more_than_1 = count_2 + count_3
        coun_more_than_0 = count_1 + count_2 + count_3

        # relative values to total rank comparisons
        rel_0 = count_0 / total_rank_comparisons
        rel_1 = count_1 / total_rank_comparisons
        rel_2 = count_2 / total_rank_comparisons
        rel_3 = count_3 / total_rank_comparisons

        rel_1_or_less = count_1_or_less / total_rank_comparisons
        rel_more_than_1 = count_more_than_1 / total_rank_comparisons
        rel_more_than_0 = coun_more_than_0 / total_rank_comparisons

        # save in a dict
        session_comparison_difference_dict[f"{comp}"] = [comp, session_1, session_2, total_rank_comparisons, 
                                                            rel_0, rel_1, rel_2, rel_3, rel_1_or_less, rel_more_than_1, rel_more_than_0]
        
        # describe group
        stn_count = len(list(comp_dataframe.subject_hemisphere.unique()))
        group_mean = np.mean(comp_dataframe[f"{level_or_direction}_difference"].values)
        group_std = np.std(comp_dataframe[f"{level_or_direction}_difference"].values)
        

        group_description[f"{comp}"] = [comp, total_rank_comparisons, stn_count, group_mean, group_std]
        

    # transform to dataframe
    session_comparison_difference_df = pd.DataFrame(session_comparison_difference_dict)
    session_comparison_difference_df.rename(index={
        0: "session_comparison",
        1: "session_1",
        2: "session_2", 
        3: "total_rank_comparisons",
        4: "rel_amount_difference_0",
        5: "rel_amount_difference_1",
        6: "rel_amount_difference_2",
        7: "rel_amount_difference_3",
        8: "rel_amount_difference_1_or_less",
        9: "rel_amount_difference_more_than_1",
        10: "rel_amount_difference_more_than_0",
    }, inplace=True)
    session_comparison_difference_df = session_comparison_difference_df.transpose()

    description_results = pd.DataFrame(group_description)
    description_results.rename(index={0: "session_comparison", 1: "number_of_observations", 2: "number_of_stn", 3: "mean", 4: "standard_deviation"}, inplace=True)
    description_results = description_results.transpose()
   

    ########################## PLOT HEATMAP OF REL AMOUNT OF DIFFERENCES IN LEVELS FOR RANKS ##########################

    if level_or_direction == "direction":
        if difference_to_plot not in ["0", "1"]:
            print(f"difference_to_plot: {difference_to_plot} must be in ['0', '1'].")

    # transform difference values to floats and 4x4 matrices
    if difference_to_plot == "1_or_less":
        difference = session_comparison_difference_df.rel_amount_difference_1_or_less.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_1_or_less"

    elif difference_to_plot == "more_than_1":
        difference = session_comparison_difference_df.rel_amount_difference_more_than_1.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_more_than_1"
    
    elif difference_to_plot == "more_than_0":
        difference = session_comparison_difference_df.rel_amount_difference_more_than_0.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_more_than_0"
    
    elif difference_to_plot == "0":
        difference = session_comparison_difference_df.rel_amount_difference_0.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_0"
    
    elif difference_to_plot == "1":
        difference = session_comparison_difference_df.rel_amount_difference_1.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_1"
    
    elif difference_to_plot == "2":
        difference = session_comparison_difference_df.rel_amount_difference_2.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_2"
    
    elif difference_to_plot == "3":
        difference = session_comparison_difference_df.rel_amount_difference_3.values.astype(float)
        # reshape the difference values into 4x4 matrix
        difference = difference.reshape(4,4)
        difference_parameter = "rel_amount_difference_3"


    fig, ax = plt.subplots()

    heatmap = ax.pcolor(difference, cmap=plt.cm.YlOrRd)
    # other color options: GnBu, YlOrRd, YlGn, Greys, Blues, PuBuGn, YlGnBu

    # Set the x and y ticks to show the indices of the matrix
    ax.set_xticks(np.arange(difference.shape[1])+0.5, minor=False) # if minor=True it will plot x and y labels differently
    ax.set_yticks(np.arange(difference.shape[0])+0.5, minor=False)

    # Set the tick labels to show the values of the matrix
    ax.set_xticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)
    ax.set_yticklabels(["postop", "3MFU", "12MFU", "18MFU"], minor=False)

    # Rotate the x-axis tick labels to be vertical
    # plt.xticks(rotation=90)

    # Add a colorbar to the right of the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label(f"relative to total number of comparisons")

    # Add the cell values to the heatmap
    for i in range(difference.shape[0]):
        for j in range(difference.shape[1]):
            plt.text(j + 0.5, i + 0.5, str("{: .2f}".format(difference[i, j])), ha='center', va='center') # only show 2 numbers after the comma of a float

    plt.title(f"Difference in {level_or_direction} of {difference_to_plot} \nof FOOOF beta ranks {ranks_included}")

    fig.tight_layout()

    if level_or_direction == "direction":
        fig_name_png = f"\\fooof_monopol_only_segm_heatmap_beta_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}.png"
        fig_name_svg = f"\\fooof_monopol_only_segm_heatmap_beta_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}.svg"
    
    elif level_or_direction == "level": 
        fig_name_png = f"\\fooof_monopol_heatmap_beta_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}.png"
        fig_name_svg = f"\\fooof_monopol_heatmap_beta_ranks_{ranks_included}_{level_or_direction}_difference_{difference_to_plot}.svg"

    fig.savefig(figures_path + fig_name_png, bbox_inches="tight")
    fig.savefig(figures_path + fig_name_svg, bbox_inches="tight", format="svg")


    return {
        "session_comparison_difference_df": session_comparison_difference_df,
        "description_results": description_results}
