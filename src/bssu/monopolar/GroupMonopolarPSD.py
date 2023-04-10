""" Group all PSD monopolar averages and ranks """



import os
import pandas as pd
import itertools
from scipy import stats
import plotly.express as px

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

    # load the monolopar beta psd for for each electrode at different timepoints
    data_weightedByCoordinates = loadResults.load_GroupMonoRef_weightedPsdCoordinateDistance_pickle(
        freqBand=freqBand,
        normalization="rawPsd",
        filterSignal="band-pass")
    
    # first check, which STNs and sessions exist in data 
    sub_hem_keys = list(data_weightedByCoordinates.subject_hemisphere.unique())

    weightedByCoordinate_Dataframe = pd.DataFrame() # concat all Dataframes from all sub, hem, sessions

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
            session_pair_STNs[f"{session_1}_{session_2}_{STN}"] = [session_1, session_2, STN, spearman_correlation.statistic, spearman_correlation.pvalue]

    # save the dictionary as a Dataframe
    results_DF = pd.DataFrame(session_pair_STNs)
    results_DF.rename(index={0: "session_1", 1: "session_2", 2: "subject_hemisphere", 3: f"spearman_r", 4: f"pval"}, inplace=True)
    results_DF = results_DF.transpose()


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
    
    # plot a heatmap
    fig = px.imshow(spearmanr_to_plot,
                labels=dict(x="session 1", y="session 2", color=f"spearman correlation {mean_or_median}"),
                x=['postop', 'fu3m', 'fu12m', 'fu18m'],
                y=['postop', 'fu3m', 'fu12m', 'fu18m'],
                title=f"Correlation per electrode of {freqBand} {ranks_or_relPsd}",
                text_auto=True
               )
    
    fig.update_xaxes(side="top")
    fig.update_layout(title={
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image(os.path.join(figures_path, f"Monopol_session_correlations_heatmap_{freqBand}_{ranks_or_relPsd}_{mean_or_median}.png"))
    


    return {
        "results_DF":results_DF,
        "spearman_m_df":spearman_m_df
    }






