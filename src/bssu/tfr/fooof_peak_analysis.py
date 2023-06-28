""" Analysis of FOOOF beta peak parameters """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
import scipy
from scipy import stats
from scipy.integrate import simps
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import fooof
from fooof.plts.spectra import plot_spectrum

# Local Imports
from ..classes import mainAnalysis_class
from ..utils import find_folders as findfolders
from ..utils import loadResults as loadResults  



def highest_beta_channels_fooof(
        fooof_spectrum:str,
        highest_beta_session:str
):
    """
    Load the file "fooof_model_group_data.json"
    from the group result folder

    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

    1) calculate beta average for each channel and rank within 1 stn, 1 session and 1 channel group
    
    2) rank beta averages and only select the channels with rank 1.0 

    Output highest_beta_df
        - containing all stns, all sessions, all channels with rank 1.0 within their channel group
    
    """

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

    beta_rank_df = pd.DataFrame()

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

                # save to ranked_beta_df
                beta_rank_df = pd.concat([beta_rank_df, group_comp_df_copy])
    
    # depending on input: keep only rank 1.0 or keep postop rank 1 or 3MFU rank 1 channel
    if highest_beta_session == "highest_each_session":
        # only keep the row with beta rank 1.0
        highest_beta_df = beta_rank_df.loc[beta_rank_df.beta_rank == 1.0]
    
    elif highest_beta_session == "highest_postop":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:

            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]
            
            for ses in sessions:
                # check if postop exists
                if "postop" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue
                
                else: 
                    postop_rank1_channels = stn_data.loc[stn_data.session=="postop"]
                    postop_rank1_channels = postop_rank1_channels.loc[postop_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]
                
                for group in channel_group:

                    if group == "ring":
                        channels = ['01', '12', '23']
                    
                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]
                    
                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in postop in this channel group
                    postop_1_row = postop_rank1_channels.loc[postop_rank1_channels["bipolar_channel"].isin(channels)]
                    postop_1_channelname = postop_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == postop_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, postop_1_row, selected_rows])
        
        # drop index columns 
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"])


    elif highest_beta_session == "highest_fu3m":
        highest_beta_df = pd.DataFrame()
        # for each stn get channel name of beta rank 1 in postop and select the channels for the other timepoints
        for stn in stn_unique:

            stn_data = beta_rank_df.loc[beta_rank_df.subject_hemisphere == stn]
            
            for ses in sessions:

                # # if session is postop, continue, because we´re only interested in follow ups here
                # if ses == "postop":
                #     continue

                # check if fu3m exists
                if "fu3m" not in stn_data.session.values:
                    continue

                elif ses not in stn_data.session.values:
                    continue
                
                else: 
                    fu3m_rank1_channels = stn_data.loc[stn_data.session=="fu3m"]
                    fu3m_rank1_channels = fu3m_rank1_channels.loc[fu3m_rank1_channels.beta_rank == 1.0]

                    stn_ses_data = stn_data.loc[stn_data.session == ses]
                
                for group in channel_group:

                    if group == "ring":
                        channels = ['01', '12', '23']
                    
                    elif group == "segm_inter":
                        channels = ["1A2A", "1B2B", "1C2C"]
                    
                    elif group == "segm_intra":
                        channels = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

                    group_data = stn_ses_data.loc[stn_ses_data["bipolar_channel"].isin(channels)].reset_index()

                    # get channel name of rank 1 channel in fu3m in this channel group
                    fu3m_1_row = fu3m_rank1_channels.loc[fu3m_rank1_channels["bipolar_channel"].isin(channels)]
                    fu3m_1_channelname = fu3m_1_row.bipolar_channel.values[0]

                    # select only this channel in all the other sessions
                    selected_rows = group_data.loc[group_data.bipolar_channel == fu3m_1_channelname]
                    highest_beta_df = pd.concat([highest_beta_df, fu3m_1_row, selected_rows])
        
        # drop index columns 
        # drop duplicated postop rows
        highest_beta_df = highest_beta_df.drop(columns=["level_0", "index"])
        highest_beta_df = highest_beta_df.drop_duplicates(keep="first", subset=["subject_hemisphere", "session", "bipolar_channel"])


    le = LabelEncoder()

    # define split array function
    split_array = lambda x: pd.Series(x)

    channel_group = ["ring", "segm_inter", "segm_intra"]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    group_dict = {} # data with predictions

    ############################## create a single dataframe for each channel group with only one highest beta channels per STN ##############################
    for group in channel_group:

        if group == "ring":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(ring)]

        elif group == "segm_inter":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(segm_inter)]

        elif group == "segm_intra":
            group_df = highest_beta_df.loc[highest_beta_df.bipolar_channel.isin(segm_intra)]
        
        # session values have to be integers, add column group with integers for each STN electrode 
        group_df_copy = group_df.copy()
        group_df_copy["group"] = le.fit_transform(group_df_copy["subject_hemisphere"]) # adds a column "group" with integer values for each subject_hemisphere
        group_df_copy["session"] = group_df_copy.session.replace(to_replace=["postop", "fu3m", "fu12m", "fu18m"], value=[0,3,12,18])

        # split beta peak column into three columns
        group_df_copy[["beta_center_frequency", "beta_peak_power", "beta_band_width"]] = group_df_copy["beta_peak_CF_power_bandWidth"].apply(split_array)
        group_df_copy = group_df_copy.drop(columns=["alpha_peak_CF_power_bandWidth", "gamma_peak_CF_power_bandWidth"])
        
        group_df_copy = group_df_copy.dropna()

        group_dict[group] = group_df_copy
    

    return group_dict






def fooof_mixedlm_highest_beta_channels(
        fooof_spectrum:str,
        highest_beta_session:str,
        data_to_fit:str,
        incl_sessions:list,
        shape_of_model:str
):
    """
    
    Input: 
        - fooof_spectrum: 
            "periodic_spectrum"         -> 10**(model._peak_fit + model._ap_fit) - (10**model._ap_fit)
            "periodic_plus_aperiodic"   -> model._peak_fit + model._ap_fit (log(Power))
            "periodic_flat"             -> model._peak_fit

        - highest_beta_session: "highest_postop", "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc_fu3m_cf"

        - incl_sessions: [0,3] or [3,12,18] o [0,3,12,18]

        - shape_of_model: e.g. "straight", "curved", "asymptotic"


    Load the dataframe with highest beta channels in a given baseline session

    ASYMPTOTIC LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_asymptotic", data=data_analysis, groups=data_analysis["group"], re_formula="1") 
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session_asymptotic
        - groups = electrode ID 
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group
        
    The model will fit a curved line, the sum of a linear function (ax + b) and an asymptotic function a * (x / (1 + x)) + b 
        y = a1 * x + a2 * (x / (1 + x))  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the asymptotic model (output as Coef. of session_asymptotic)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)


    CURVED LINE
    lme model: smf.mixedlm(f"{data_to_fit} ~ session + session_sq", data=data_analysis, groups=data_analysis["group"], re_formula="1") 
        - this is a mixed linear effects model
        - dependent variable = data_to_fit e.g. beta average
        - independent variable = predictor = session + session squared
        - groups = electrode ID 
        - re_formula = "1" specifying a random intercept model, assuming the same effect of the predictor across groups but different intercepts for each group
        
    The model will fit a curved line, the sum of a linear function (ax + b) and an exponential function (ax^2 + b) 
        y = a1 * x + a2 * x**2  + b

        - a1 = coef of the linear model (output as Coef. of session)
        - a2 = coef of the squared model (output as Coef. of session_sq)
        - x = continuous range from 0 to 18 (min and max session)
        - b = model intercept (output as Coef. of Intercept)

    
    Two figures



    

    """

    results_path = findfolders.get_local_path(folder="GroupResults")
    figures_path = findfolders.get_local_path(folder="GroupFigures")
    fontdict = {"size": 25}

    # Load the dataframe with only highest beta channels
    highest_beta_channels = highest_beta_channels_fooof(
        fooof_spectrum=fooof_spectrum,
        highest_beta_session=highest_beta_session
        )
    # output is a dictionary with keys "ring", "segm_inter", "segm_intra"
    

    channel_group = ["ring", "segm_inter", "segm_intra"]
    sessions = [0, 3, 12, 18]

    ring = ['01', '12', '23']
    segm_inter = ["1A2A", "1B2B", "1C2C"]
    segm_intra = ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']

    sample_size_dict = {} # 
    model_output = {} # md.fit()
    group_dict = {}

    ############################## select the center frequency of 3MFU and get the area under the curve of power in a freq range +- 3 Hz around that center frequency ##############################
    for group in channel_group:

        group_df = highest_beta_channels[group]

        stn_unique = list(group_df.subject_hemisphere.unique())

        group_df_with_power_in_frange = pd.DataFrame()

        # select the beta center frequency at 3MFU for every stn
        for stn in stn_unique:

            stn_data = group_df.loc[group_df.subject_hemisphere == stn]
            fu3m_data = stn_data.loc[stn_data.session == 3] 

            fu3m_peak_center_frequency = round(fu3m_data.beta_center_frequency.values[0])
            # now get +- 3 Hz frequency range around peak center frequency
            fu3m_cf_range = np.arange(fu3m_peak_center_frequency - 3, fu3m_peak_center_frequency + 4, 1)

            for ses in sessions: # for each session collect the area under the curve for the selected frequency range of one stn

                if ses not in stn_data.session.values:
                    continue

                else:
                    ses_data = stn_data.loc[stn_data.session == ses]
                    power = ses_data.fooof_power_spectrum.values[0]
                    power_in_freq_range = power[fu3m_cf_range[0] : (fu3m_cf_range[6]+1)] # select the power values by indexing from frequency range first until last value
                    power_area_under_curve = simps(power_in_freq_range, fu3m_cf_range)

                    ses_data_copy = ses_data.copy()
                    ses_data_copy["round_cf_at_fu3m"] = fu3m_peak_center_frequency
                    ses_data_copy["beta_power_auc_fu3m_cf"] = power_area_under_curve

                    group_df_with_power_in_frange = pd.concat([group_df_with_power_in_frange, ses_data_copy])
                    
            group_dict[group] = group_df_with_power_in_frange
            

    ############################## perform linear mixed effects model ##############################
    for g, group in enumerate(channel_group):

        data_analysis = group_dict[group]
        
        #predictor_x = data_analysis.session.values
        data_analysis = data_analysis.copy()
        data_analysis["session_sq"] = data_analysis.session**2 
        data_analysis["session_asymptotic"] = data_analysis.session / (1 + data_analysis.session)
        
        if shape_of_model == "asymptotic":
            md = smf.mixedlm(f"{data_to_fit} ~ session + session_asymptotic", data=data_analysis, groups=data_analysis["group"], 
                                re_formula="1") 

        elif shape_of_model == "curved":
            md = smf.mixedlm(f"{data_to_fit} ~ session + session_sq", data=data_analysis, groups=data_analysis["group"], 
                                re_formula="1") 
        
        elif shape_of_model == "straight":
            md = smf.mixedlm(f"{data_to_fit} ~ session", data=data_analysis, groups=data_analysis["group"], 
                                re_formula="1")
        
        # re_formula defining the random effect 
        # re_formula = 1 specifying random intercept model, assuming same effect of predictor for all groups
        # re_formula = f"1 + session" specifying random intercept and slope model
        mdf = md.fit()

        # save linear model result              
        print(mdf.summary())
        model_output[group] = mdf


        # add predictions column to dataframe
        yp = mdf.fittedvalues
        group_dict[group]["predictions"] = yp

        for ses in incl_sessions:
            ses_data = data_analysis.loc[data_analysis.session==ses]
            count = ses_data.subject_hemisphere.count()

            # save sample size
            sample_size_dict[f"{group}_{ses}mfu"] = [group, ses, count]

    sample_size_df = pd.DataFrame(sample_size_dict)
    sample_size_df.rename(index={
        0: "channel_group",
        1: "session",
        2: "count",
    }, inplace=True)
    sample_size_df = sample_size_df.transpose()

    ############################## plot the observed values and the model ##############################
    fig_1, axes_1 = plt.subplots(3,1,figsize=(10,15)) 
    fig_2, axes_2 = plt.subplots(3,1,figsize=(10,15)) 

    for g, group in enumerate(channel_group):

        data_analysis = group_dict[group] # this is the dataframe with data
        mdf_group = model_output[group] # this is md.fit()

        # get the results
        result_part_2 = mdf_group.summary().tables[1] # part containing model intercept, slope, std.error
        model_intercept = float(result_part_2["Coef."].values[0])
        model_slope = float(result_part_2["Coef."].values[1])

        if shape_of_model == "asymptotic" or "curved":
            model_slope_2 = float(result_part_2["Coef."].values[2])
            #group_variance = float(result_part_2["Coef."].values[3])

        std_error_intercept = float(result_part_2["Std.Err."].values[0])
        # p_val_intercept = float(result_part_2["P>|z|"].values[0])
        # p_val_session = float(result_part_2["P>|z|"].values[1])
        # p_val_session2 = float(result_part_2["P>|z|"].values[2])
        conf_int = mdf_group.conf_int(alpha=0.05) # table with confidence intervals for intercept, session, session2 and group var


        # one subplot per channel group
        axes_1[g].set_title(f"{group} channel group", fontdict=fontdict)
        axes_2[g].set_title(f"{group} channel group", fontdict=fontdict)

        ################## plot the result for each electrode ##################
        for id, group_id in enumerate(data_analysis.group.unique()):

            sub_data = data_analysis[data_analysis.group==group_id]

            # axes[g].scatter(sub_data[f"{data_to_fit}"], sub_data["session"] ,color=plt.cm.twilight_shifted(group_id*10)) # color=plt.cm.tab20(group_id)
            # axes[g].plot(sub_data[f"{data_to_fit}"], sub_data["predictions"], color=plt.cm.twilight_shifted(group_id*10))

            axes_1[g].scatter(sub_data["session"], sub_data[f"{data_to_fit}"] ,color=plt.cm.twilight_shifted((id+1)*10), alpha=0.3) # color=plt.cm.tab20(group_id)
            # plot the predictions
            # axes[g].plot(sub_data["session"], sub_data["predictions"], color=plt.cm.twilight_shifted((id+1)*10), linewidth=1, alpha=0.5)
            axes_1[g].plot(sub_data["session"], sub_data[f"{data_to_fit}"], color=plt.cm.twilight_shifted((id+1)*10), linewidth=1, alpha=0.3)

        # plot the model regression line
        if 0 in incl_sessions:

            if 18 in incl_sessions:
                x=np.arange(0,19)
            
            else:
                x=np.arange(0,4)

        elif 0 not in incl_sessions:
            x=np.arange(3,19)

        ################## plot the modeled curved line ##################
        if shape_of_model == "curved":
            y=x*model_slope + x**2 * model_slope_2 + model_intercept # curved model
        
        elif shape_of_model == "asymptotic":
            y = x*model_slope + (x / (1+x))*model_slope_2 + model_intercept # asymptotic model
        
        elif shape_of_model == "straight":
            y = x*model_slope + model_intercept # straight line

        axes_1[g].plot(x, y, color="k", linewidth=5)
        # linear model: coef*x (=linear) + coef*x^2 (=exponential) + intercept
        # coef defines the slope

        # pred = mdf_group.predict(exog=dict(x=x))
        # calculate the confidence interval
        # cov_params = mdf_group.cov_params()
        # mse = np.mean(mdf_group.resid.values**2)
        # t_value = stats.t.ppf(0.975, df=mdf_group.df_resid)
        # standard_errors = np.sqrt(np.diag(cov_params))
        # lower_bound = y - t_value * standard_errors * np.sqrt(mse)
        # upper_bound = y + t_value * standard_errors * np.sqrt(mse)

        # lower_bound = y + 1.96 *

        # axes[g].plot(prediction_data["session"], prediction_data["mean_yp"], color="k", linewidth=5)
        # axes[g].fill_between(prediction_data["session"], prediction_data["mean_yp"]-prediction_data["sem_yp"], prediction_data["mean_yp"]+prediction_data["sem_yp"], color='lightgray', alpha=0.5)        
        # axes_1[g].fill_between(x, lower_bound, upper_bound, color="k", linewidth=5, alpha=0.3)

        ################### plot the residuals against predictions ##################
        # number of residuals equals the number of observations (number of channels I input)
        resid = mdf_group.resid
        predicted_values = mdf_group.fittedvalues
        axes_2[g].scatter(predicted_values, resid, color="k", alpha=0.2)
        axes_2[g].axhline(y=0, color="red", linestyle="--")

    for ax in axes_1:

        ax.set_ylabel(f"{data_to_fit}", fontsize=25)
        ax.set_xlabel("months post-surgery", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)

    fig_1.suptitle(f"Linear mixed effects model: {highest_beta_session} beta channels", fontsize=30)
    fig_1.subplots_adjust(wspace=0, hspace=0)

    fig_1.tight_layout()
    fig_1.savefig(os.path.join(figures_path, f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png"), bbox_inches="tight")
    fig_1.savefig(os.path.join(figures_path, f"lme_{shape_of_model}_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg"), bbox_inches="tight", format="svg")

    print("figure: ", 
        f"lme_{data_to_fit}_{highest_beta_session}_beta_channels.png",
        "\nwritten in: ", figures_path
        )
    

    for ax in axes_2:
        ax.set_xlabel("Predicted Values", fontsize=25)
        ax.set_ylabel("Residuals", fontsize=25)

        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.grid(False)
    
    fig_2.suptitle(f"Linear mixed effects model residuals: {highest_beta_session} beta channels", fontsize=30)
    fig_2.subplots_adjust(wspace=0, hspace=0)
    fig_2.tight_layout()
    fig_2.savefig(os.path.join(figures_path, f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.png"), bbox_inches="tight")
    fig_2.savefig(os.path.join(figures_path, f"lme_{shape_of_model}_residuals_{data_to_fit}_{highest_beta_session}_beta_channels_sessions{incl_sessions}.svg"), bbox_inches="tight", format="svg")

    

    # save results
    mdf_result_filepath = os.path.join(results_path, f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle")
    with open(mdf_result_filepath, "wb") as file:
        pickle.dump(model_output, file)
    
    print("file: ", 
          f"fooof_lme_{shape_of_model}_model_output_{data_to_fit}_{highest_beta_session}_sessions{incl_sessions}.pickle",
          "\nwritten in: ", results_path
          )



    return {
        "group_dict": group_dict,
        "sample_size_df": sample_size_df,
        "conf_int":conf_int,
        "model_output":model_output,
        "md": md
        }





        