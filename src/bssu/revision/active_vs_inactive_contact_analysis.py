""" Pseudo-monopolar beta power at optimal stimulation settings"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.stats import ttest_ind
import statistics
from scipy.stats import shapiro, friedmanchisquare, wilcoxon
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


import os
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

import plotly.express as px

import itertools
import seaborn as sns


######### PRIVATE PACKAGES #########
from ..utils import find_folders as find_folders
from ..utils import loadResults as loadResults
from ..utils import percept_helpers as helpers
from ..utils import sub_session_dict as sub_session_dict
from ..stimulation import activeStimulationContacts as active_stim_contacts
from ..stimulation import active_contacts_beta_over_time as active_contacts_beta_over_time

RESULTS_PATH = find_folders.get_local_path(folder="GroupResults")
FIGURES_PATH = find_folders.get_local_path(folder="GroupFigures")

LAST_SESSION = {"group_1": "fu12m", "group_2": "fu18or24m", "group_3": "fu18or24m"}


#### get optimal stimulation parameters from the last session per cohort (0-3-12 or 3-12-18)


def get_best_stimulation_settings_possible(cohort: str):
    """
    From the table with stimulation settings, only get the last session per patient.
    For cohort "group_1": last session is "fu12m"
    For cohort "group_2": last session is "fu18or24m"
    """

    # load Excel file with best clinical stimulation parameters
    best_clinical_stimulation = loadResults.load_BestClinicalStimulation_excel()
    best_clinical_contacts = best_clinical_stimulation["BestContacts_one_longterm"]

    # filter by cohort
    included_sub_sessions = sub_session_dict.get_subs_sessions(cohort)  # other options: "group_1", "group_2", "group_3"
    incl_subjects = list(included_sub_sessions["incl_subjects"])  # list of subjects

    best_clinical_contacts["subject"] = best_clinical_contacts['subject_hemisphere'].str.split('_').str[0]
    cohort_clinical_contacts = best_clinical_contacts[best_clinical_contacts["subject"].isin(incl_subjects)]

    # Further filter by the last session of interest
    last_session = LAST_SESSION[cohort]
    cohort_clinical_contacts_last_session = cohort_clinical_contacts[
        cohort_clinical_contacts["session"] == last_session
    ]

    return cohort_clinical_contacts_last_session


### get pseudo-monopolar beta power from each session of the cohort and label as active or inactive


def get_monopolar_beta_power_and_clinical_activity(cohort: str):
    """
    Get the pseudo-monopolar beta power at active and inactive contacts for each session of the cohort
    """

    # load all pseudo-monopolar data
    loaded_fooof_mono_beta = loadResults.select_fooof_data(dataset="monopolar_beta", cohort=cohort)
    loaded_fooof_mono_beta = loaded_fooof_mono_beta["fooof_data"]

    # load the clinical activity data
    clinical_contacts_last_session = get_best_stimulation_settings_possible(cohort)

    # Merge the dataframes based on subject_hemisphere and session
    merged_df = loaded_fooof_mono_beta.merge(clinical_contacts_last_session, on=["subject_hemisphere"], how="left")

    # Ensure the columns CathodalContact and InactiveContacts are strings
    merged_df['CathodalContact'] = merged_df['CathodalContact'].astype(str)
    merged_df['InactiveContacts'] = merged_df['InactiveContacts'].astype(str)

    # Function to determine contact activity status
    def check_contact_status(row):
        contact = row['contact']
        cathodal_contacts = row['CathodalContact'].split("_") if pd.notna(row['CathodalContact']) else []
        inactive_contacts = row['InactiveContacts'].split("_") if pd.notna(row['InactiveContacts']) else []

        if contact in cathodal_contacts:
            return "active"
        elif contact in inactive_contacts:
            return "inactive"
        else:
            return None  # No status if contact is not in either list

    # Apply the function to create a new column
    merged_df['clinical_activity'] = merged_df.apply(check_contact_status, axis=1)

    # Drop unwanted columns and rename columns as requested
    merged_df = merged_df.drop(columns=["subject_y", "session_y"])  # Drop unnecessary columns
    merged_df = merged_df.rename(columns={"subject_x": "subject", "session_x": "session"})  # Rename columns

    # Add a new column combining 'session' and 'clinical_activity' as a string
    merged_df['session_clinical_activity'] = merged_df['session'] + "_" + merged_df['clinical_activity']

    # Result
    return merged_df


def plot_violinplot_active_vs_inactive(cohort: str):
    """
    Plot violin plots with scatter and statistical markers for each session,
    and return a DataFrame with mean, median, Q1, Q3, and sample size for each half (active/inactive).

    Parameters:
        data (pd.DataFrame): Input data with columns:
            'session_clinical_activity' (e.g., "fu3m_active"),
            'beta_psd_rel_to_rank1', 'session', 'clinical_activity'.

    Returns:
        stats_df (pd.DataFrame): A DataFrame with mean, median, Q1, Q3, and sample size for each half.
    """
    # Extract session and activity from 'session_clinical_activity'
    loaded_data = get_monopolar_beta_power_and_clinical_activity(cohort)

    # Determine the y-axis limits based on the range of beta_psd_rel_to_rank1
    y_min = loaded_data['beta_psd_rel_to_rank1'].min() - 0.2
    y_max = loaded_data['beta_psd_rel_to_rank1'].max() + 0.2

    # Initialize a list to store statistics
    stats_list = []

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 8))  # Define ax here

    # Create the violin plot
    sns.violinplot(
        data=loaded_data,
        x='session',
        y='beta_psd_rel_to_rank1',
        hue='clinical_activity',
        hue_order=['active', 'inactive'],
        split=True,
        palette={'active': 'gold', 'inactive': 'grey'},
        scale='width',
        inner=None,
        ax=ax,  # Pass ax to the plot
    )

    # Overlay scatterplot for individual points
    sns.stripplot(
        data=loaded_data,
        x='session',
        y='beta_psd_rel_to_rank1',
        hue='clinical_activity',
        hue_order=['active', 'inactive'],
        dodge=True,
        alpha=0.7,
        palette={'active': 'gold', 'inactive': 'grey'},
        jitter=True,
        size=9,
        linewidth=0.5,
        edgecolor='k',
        ax=ax,  # Pass ax to the plot
    )

    # Calculate and plot statistics for each half
    sessions = loaded_data['session'].unique()
    for session_idx, session in enumerate(sessions):
        for activity, color, offset in zip(['active', 'inactive'], ['gold', 'grey'], [-0.15, 0.15]):
            # Subset data for the specific session and activity
            subset = loaded_data[(loaded_data['session'] == session) & (loaded_data['clinical_activity'] == activity)]

            # Calculate statistics
            mean_value = subset['beta_psd_rel_to_rank1'].mean()
            median_value = subset['beta_psd_rel_to_rank1'].median()
            q1 = subset['beta_psd_rel_to_rank1'].quantile(0.25)
            q3 = subset['beta_psd_rel_to_rank1'].quantile(0.75)
            sample_size = len(subset)

            # Perform Shapiro-Wilk normality test
            if sample_size > 2:  # Shapiro-Wilk requires at least 3 samples
                _, normality_p_value = shapiro(subset['beta_psd_rel_to_rank1'])
            else:
                normality_p_value = None  # Insufficient data

            # Save the statistics in the list
            stats_list.append(
                {
                    "Session": session,
                    "Activity": activity,
                    "Mean": mean_value,
                    "Median": median_value,
                    "Q1": q1,
                    "Q3": q3,
                    "Sample Size": sample_size,
                    "Normality P-Value": normality_p_value,
                    "Normal Distribution (p > 0.05)": normality_p_value is not None and normality_p_value > 0.05,
                }
            )

            # Calculate x position for the markers
            x_position = session_idx + offset

            # Plot mean as +
            ax.scatter(
                x=[x_position],
                y=[mean_value],
                color='black',
                s=150,
                marker='+',
                zorder=10,  # Ensure the mean cross is on top
                label="Mean" if session_idx == 0 and activity == "active" else None,
            )

            # Plot median as o
            ax.scatter(
                x=[x_position],
                y=[median_value],
                color='white',
                s=100,
                marker='o',
                zorder=9,  # Place median below the mean cross
                label="Median" if session_idx == 0 and activity == "active" else None,
            )

            # Plot quartiles as vertical line
            ax.vlines(
                x=x_position,
                ymin=q1,
                ymax=q3,
                color='white',
                linewidth=2,
                zorder=8,  # Place quartiles below the median and mean
                label="IQR" if session_idx == 0 and activity == "active" else None,
            )

    # Customize the plot
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels) // 2  # Split handles and labels into two groups (violin and scatter)
    ax.legend(
        handles[:n] + [handles[-3], handles[-2], handles[-1]],
        labels[:n] + ["Mean", "Median", "IQR"],
        title="Legend",
        loc='upper right',
    )

    ax.set_title('Beta PSD Relative to Rank 1 for Each Session', fontsize=16)
    ax.set_xlabel('Session', fontsize=14)
    ax.set_ylabel('Beta PSD Relative to Rank 1', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # save figure
    helpers.save_fig_png_and_svg(
        path=FIGURES_PATH,
        filename=f"revision_active_vs_inactive_monopol_beta_rel_to_rank1_{cohort}_optimal_stimulation_settings",
        figure=fig,
    )

    # Convert the statistics list into a DataFrame
    stats_df = pd.DataFrame(stats_list)

    return stats_df


def perform_friedman_test_with_posthoc_all_groups(cohort):
    """
    Perform Friedman test and post-hoc Wilcoxon signed-rank tests with multiple comparison correction.

    Parameters:
        data (pd.DataFrame): Input data with columns:
            - 'session': Session label (e.g., "postop", "fu3m", "fu12m", "fu18m")
            - 'clinical_activity': Activity group ("active" or "inactive")
            - 'beta_psd_rel_to_rank1': The metric to compare.
            - 'subject_hemisphere_contact': Unique identifier for paired data.

    Returns:
        results (pd.DataFrame): Summary of the Friedman test, post-hoc tests, and corrected p-values.
    """
    results = []
    all_posthoc_pvalues = []

    # Extract session and activity from 'session_clinical_activity'
    loaded_data = get_monopolar_beta_power_and_clinical_activity(cohort)
    loaded_data['subject_hemisphere_contact'] = loaded_data['subject_hemisphere'] + "_" + loaded_data['contact']

    for activity in ['active', 'inactive']:
        activity_data = loaded_data[loaded_data['clinical_activity'] == activity]

        # Check for duplicates and aggregate if necessary
        if activity_data.duplicated(subset=["subject_hemisphere_contact", "session"]).any():
            print("Duplicate entries found. Aggregating by mean.")
            activity_data = activity_data.groupby(["subject_hemisphere_contact", "session"], as_index=False).mean()

        # Pivot data for Friedman test
        pivoted_data = activity_data.pivot(
            index='subject_hemisphere_contact', columns='session', values='beta_psd_rel_to_rank1'
        )

        # Handle cases for 3 or 4 sessions
        if pivoted_data.shape[1] == 3:
            # Perform Friedman test for 3 sessions
            test_stat, p_value = friedmanchisquare(
                pivoted_data.iloc[:, 0], pivoted_data.iloc[:, 1], pivoted_data.iloc[:, 2]
            )

            results.append(
                {
                    "Activity Group": activity,
                    "Test": "Friedman",
                    "Comparison": "0 vs 1 vs 2",
                    "Statistic": test_stat,
                    "Raw P-Value": p_value,
                }
            )

            # Post-hoc tests for 3 sessions
            if p_value < 0.05:
                comparisons = [(0, 1), (0, 2), (1, 2)]
                for session1, session2 in comparisons:
                    stat, posthoc_p = wilcoxon(pivoted_data.iloc[:, session1], pivoted_data.iloc[:, session2])
                    results.append(
                        {
                            "Activity Group": activity,
                            "Test": "Wilcoxon",
                            "Comparison": f"{session1} vs {session2}",
                            "Statistic": stat,
                            "Raw P-Value": posthoc_p,
                        }
                    )
                    all_posthoc_pvalues.append(posthoc_p)

        elif pivoted_data.shape[1] == 4:
            # Perform Friedman test for 4 sessions
            test_stat, p_value = friedmanchisquare(
                pivoted_data.iloc[:, 0], pivoted_data.iloc[:, 1], pivoted_data.iloc[:, 2], pivoted_data.iloc[:, 3]
            )

            results.append(
                {
                    "Activity Group": activity,
                    "Test": "Friedman",
                    "Comparison": "0 vs 1 vs 2 vs 3",
                    "Statistic": test_stat,
                    "Raw P-Value": p_value,
                }
            )

            # Post-hoc tests for 4 sessions
            if p_value < 0.05:
                comparisons = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                for session1, session2 in comparisons:
                    stat, posthoc_p = wilcoxon(pivoted_data.iloc[:, session1], pivoted_data.iloc[:, session2])
                    results.append(
                        {
                            "Activity Group": activity,
                            "Test": "Wilcoxon",
                            "Comparison": f"{session1} vs {session2}",
                            "Statistic": stat,
                            "Raw P-Value": posthoc_p,
                        }
                    )
                    all_posthoc_pvalues.append(posthoc_p)

        else:
            raise ValueError(f"Expected exactly 3 or 4 sessions, found {pivoted_data.shape[1]} sessions.")

    # Perform multiple comparison correction on all post-hoc p-values, if any
    if all_posthoc_pvalues:
        corrected_results = {}
        correction_methods = ["bonferroni", "holm", "fdr_bh"]
        for method in correction_methods:
            _, corrected_pvalues, _, _ = multipletests(all_posthoc_pvalues, method=method)
            corrected_results[method] = corrected_pvalues

        # Update results with corrected p-values
        posthoc_index = 0
        for result in results:
            if result["Test"] == "Wilcoxon":
                for method, corrected_pvalues in corrected_results.items():
                    result[f"Corrected P-Value ({method})"] = corrected_pvalues[posthoc_index]
                posthoc_index += 1
    else:
        print("No post-hoc tests to correct.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df
