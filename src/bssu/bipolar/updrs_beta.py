""" Correlation of beta power and UPDRS """

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
from ..tfr import fooof_peak_analysis as fooof_peaks

results_path = findfolders.get_local_path(folder="GroupResults")
figures_path = findfolders.get_local_path(folder="GroupFigures")
data_path = findfolders.get_local_path(folder="data")

channel_group = ["ring", "segm_inter", "segm_intra"]
sessions = [0, 3, 12, 18]




def merge_data_updrs_beta(
        highest_beta_session:str,
        data_to_fit:str,
):
    
    """
    Load UPDRS Excel and FOOOF beta power dataframe

    Input:

        - highest_beta_session: "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc"
                            "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"
    

    """

    if highest_beta_session == "highest_fu3m":
        around_cf = "around_cf_at_fixed_session"
    
    elif highest_beta_session == "highest_each_session":
        around_cf = "around_cf_at_each_session"


    ################## calculate Beta power average left and right per STN per session ################## 
    fooof_data_stn_average = {}

    fooof_beta_power = fooof_peaks.calculate_auc_beta_power_fu18or24(
        fooof_spectrum="periodic_spectrum",
        highest_beta_session=highest_beta_session,
        around_cf=around_cf
    )

    for group in channel_group:

        group_beta_data = fooof_beta_power[f"{group}"]
        group_beta_data[["subject", "hemisphere"]] = group_beta_data["subject_hemisphere"].str.split("_", expand=True)

        for ses in sessions:

            session_beta_data = group_beta_data.loc[group_beta_data.session == ses]
            subject_unique = list(session_beta_data["subject"].unique())

            for sub in subject_unique:

                sub_data = session_beta_data.loc[session_beta_data.subject == sub] # left and right

                # calculate the average of the data to fit of left and right hemisphere
                sub_data_average = np.average(sub_data[f"{data_to_fit}"].values)

                left_data = sub_data.loc[sub_data.hemisphere == "Left"]
                left_data = left_data[f"{data_to_fit}"].values[0]

                right_data = sub_data.loc[sub_data.hemisphere == "Right"]
                right_data = right_data[f"{data_to_fit}"].values[0]

                fooof_data_stn_average[f"{group}_{ses}_{sub}"] = [group, ses, sub, sub_data_average, left_data, right_data]


    # save FOOOF data stn average to dataframe
    fooof_dataframe_stn_average = pd.DataFrame(fooof_data_stn_average)
    fooof_dataframe_stn_average.rename(index={
        0: "channel_group",
        1: "session",
        2: "subject",
        3: f"{data_to_fit}_l_r_average",
        4: f"{data_to_fit}_left",
        5: f"{data_to_fit}_right"
    }, inplace=True)
    fooof_dataframe_stn_average = fooof_dataframe_stn_average.transpose()


    ################## add the UPDRS Score to each subject per channel group and per session ##################
    updrs_III_data = pd.read_excel(os.path.join(data_path, "updrs_III_m0s0.xlsx"), sheet_name="UPDRS-III")

    # column subject consists of integers e.g. 17 instead of "017"
    # make a string and add "0" infront
    updrs_III_data["subject"] = updrs_III_data["subject"].apply(lambda x: "0" + str(x))


    merged_updrs_fooof_data = pd.DataFrame()
    # merge updrs scores and fooof values together for each channel group seperately
    
    # search for single updrs score of each subject per session
    for ses in sessions: 

        ses_data_fooof = fooof_dataframe_stn_average.loc[fooof_dataframe_stn_average.session == ses]
        subj_per_session = list(ses_data_fooof.subject.unique())

        ses_data_updrs = updrs_III_data.loc[updrs_III_data.session == ses]

        for sub in subj_per_session:

            sub_data_fooof = ses_data_fooof.loc[ses_data_fooof.subject == sub]
            sub_data_fooof_copy = sub_data_fooof.copy()

            updrs_score = ses_data_updrs.loc[ses_data_updrs.subject == sub]
            updrs_score = updrs_score.updrs_m0s0.values[0]

            # add new column with updrs score to fooof dataframe and add updrs score
            sub_data_fooof_copy["updrs_III_m0s0"] = updrs_score

            # add new dataframe rows to merged dataframe
            merged_updrs_fooof_data = pd.concat([merged_updrs_fooof_data, sub_data_fooof_copy])


    return {
        "fooof_dataframe_stn_average": fooof_dataframe_stn_average,
        "updrs_III_data": updrs_III_data,
        "merged_updrs_fooof_data": merged_updrs_fooof_data
    }


def correlate_scatterplot_updrs_beta_per_session(
        highest_beta_session:str,
        data_to_fit:str,

):
    """
    Input:

        - highest_beta_session: "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc"
                            "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"
    
    
    """

    # load the merged updrs and fooof data
    merged_data = merge_data_updrs_beta(
        highest_beta_session=highest_beta_session,
        data_to_fit=data_to_fit
    )

    merged_data = merged_data["merged_updrs_fooof_data"]

    correlation_results = {}

    for group in channel_group:
        
        fig, axes = plt.subplots(4,1,figsize=(10,15)) # plot for each group a figure with 4 rows, 4 sessions
        
        group_data = merged_data.loc[merged_data.channel_group == group]
        group_data = group_data.dropna()

        for s, ses in enumerate(sessions):

            ses_data = group_data.loc[group_data.session == ses]

            sample_size = ses_data.subject.count()

            spearman_corr = stats.spearmanr(ses_data[f"{data_to_fit}_l_r_average"], ses_data["updrs_III_m0s0"])
            spearman_corr_coeff = spearman_corr.statistic
            spearman_corr_pval = spearman_corr.pvalue

            correlation_results[f"{group}_{ses}"] = [group, ses, sample_size, spearman_corr_coeff, spearman_corr_pval]

            x_values = ses_data[f"{data_to_fit}_l_r_average"].astype(float)
            y_values = ses_data["updrs_III_m0s0"].astype(float)

            axes[s].scatter(x_values, y_values)
            axes[s].set_title(f"session: {ses}", fontdict={"size": 25})
            axes[s].set_xlabel(f"{data_to_fit} left and right average", fontsize=25)
            axes[s].set_ylabel("UPDRS-III", fontsize=25)
            axes[s].tick_params(axis="x", labelsize=25)
            axes[s].tick_params(axis="y", labelsize=25)
            axes[s].grid(False)

            # add a correlation line
            fit = np.polyfit(x_values, y_values, 1)
            line = np.poly1d(fit)
            axes[s].plot(x_values, line(x_values), color='red', label=f'Correlation (r={spearman_corr_coeff:.2f})')

                    
        fig.suptitle(f"{group}-group: UPDRS III and {data_to_fit}", fontsize=30)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()

        filename = f"spearman_corr_updrs_{data_to_fit}_l_r_average_{group}"
        fig.savefig(os.path.join(figures_path, f"{filename}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_path, f"{filename}.svg"), bbox_inches="tight", format="svg")

        print("figure: ", 
            f"{filename}.png",
            "\nwritten in: ", figures_path
            )

    
    correlation_results_dataframe = pd.DataFrame(correlation_results)
    correlation_results_dataframe.rename(index={
        0: "channel_group",
        1: "session",
        2: "sample_size",
        3: "spearman_corr_coeff",
        4: "spearman_corr_pval"
    }, inplace=True)
    correlation_results_dataframe = correlation_results_dataframe.transpose()

    return correlation_results_dataframe


def correlate_scatterplot_updrs_beta_all_sessions(
        highest_beta_session:str,
        data_to_fit:str,

):
    """
    Input:

        - highest_beta_session: "highest_fu3m", "highest_each_session"

        - data_to_fit: str e.g. "beta_average", "beta_peak_power", "beta_center_frequency", "beta_power_auc"
                            "low_beta_center_frequency", "low_beta_power_auc", "high_beta_center_frequency", "high_beta_power_auc"
    
    
    """

    # load the merged updrs and fooof data
    merged_data = merge_data_updrs_beta(
        highest_beta_session=highest_beta_session,
        data_to_fit=data_to_fit
    )

    merged_data = merged_data["merged_updrs_fooof_data"]

    correlation_results = {}

    fig, axes = plt.subplots(3,1,figsize=(10,15)) # plot for each group a subplot 
       
    for g, group in enumerate(channel_group):
        
        group_data = merged_data.loc[merged_data.channel_group == group]
        group_data = group_data.dropna()

        sample_size = group_data.subject.count()

        spearman_corr = stats.spearmanr(group_data[f"{data_to_fit}_l_r_average"], group_data["updrs_III_m0s0"])
        spearman_corr_coeff = spearman_corr.statistic
        spearman_corr_pval = spearman_corr.pvalue

        correlation_results[f"{group}"] = [group, sample_size, spearman_corr_coeff, spearman_corr_pval]

        x_values = group_data[f"{data_to_fit}_l_r_average"].astype(float)
        y_values = group_data["updrs_III_m0s0"].astype(float)

        axes[g].scatter(x_values, y_values)
        axes[g].set_title(f"group: {group}", fontdict={"size": 25})
        axes[g].set_xlabel(f"{data_to_fit} left and right average", fontsize=25)
        axes[g].set_ylabel("UPDRS-III", fontsize=25)
        axes[g].tick_params(axis="x", labelsize=25)
        axes[g].tick_params(axis="y", labelsize=25)
        axes[g].grid(False)

        # add a correlation line
        fit = np.polyfit(x_values, y_values, 1)
        line = np.poly1d(fit)
        axes[g].plot(x_values, line(x_values), color='red', label=f'Correlation (r={spearman_corr_coeff:.2f})')

    fig.suptitle(f"UPDRS III and {data_to_fit}", fontsize=30)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    filename = f"spearman_corr_updrs_{data_to_fit}_l_r_average_all_sessions"
    fig.savefig(os.path.join(figures_path, f"{filename}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_path, f"{filename}.svg"), bbox_inches="tight", format="svg")

    print("figure: ", 
        f"{filename}.png",
        "\nwritten in: ", figures_path
        )

    
    correlation_results_dataframe = pd.DataFrame(correlation_results)
    correlation_results_dataframe.rename(index={
        0: "channel_group",
        1: "sample_size",
        2: "spearman_corr_coeff",
        3: "spearman_corr_pval"
    }, inplace=True)
    correlation_results_dataframe = correlation_results_dataframe.transpose()

    return correlation_results_dataframe












            





