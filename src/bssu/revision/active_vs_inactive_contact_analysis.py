""" Pseudo-monopolar beta power at optimal stimulation settings"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.stats import ttest_ind
import statistics
from scipy.stats import shapiro, friedmanchisquare, wilcoxon, mannwhitneyu
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

    ax.set_title(f'Beta PSD Relative to Rank 1: {cohort}, optimal stimulation from last session', fontsize=16)
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


def perform_mann_whitney_u_test(cohort: str):
    """
    Perform Mann-Whitney U tests comparing active vs. inactive groups for each session.

    Parameters:
        data (pd.DataFrame): DataFrame with columns:
            - 'session_clinical_activity': Combined session and clinical activity column (e.g., "postop_active").
            - 'beta_psd_rel_to_rank1': The metric to compare.

    Returns:
        results_df (pd.DataFrame): Summary of the MWU test results with corrected p-values.
    """
    results = []
    descriptive_stats = []

    # Extract session and activity from 'session_clinical_activity'
    loaded_data = get_monopolar_beta_power_and_clinical_activity(cohort)
    loaded_data['subject_hemisphere_contact'] = loaded_data['subject_hemisphere'] + "_" + loaded_data['contact']

    sessions = loaded_data.session.unique()  # Extract unique session names
    all_pvalues = []

    for session in sessions:
        # Filter active and inactive groups for this session
        active_group = loaded_data.loc[
            loaded_data['session_clinical_activity'] == f"{session}_active", 'beta_psd_rel_to_rank1'
        ]
        inactive_group = loaded_data.loc[
            loaded_data['session_clinical_activity'] == f"{session}_inactive", 'beta_psd_rel_to_rank1'
        ]

        # Perform MWU test
        if len(active_group) > 0 and len(inactive_group) > 0:  # Ensure groups are non-empty
            stat, p_value = mannwhitneyu(active_group, inactive_group, alternative='two-sided')
            results.append(
                {
                    "Session": session,
                    "Test": "Mann-Whitney U",
                    "Statistic": stat,
                    "Raw P-Value": p_value,
                    "Active Sample Size": len(active_group),
                    "Inactive Sample Size": len(inactive_group),
                }
            )
            all_pvalues.append(p_value)

            # Collect descriptive statistics for active and inactive groups
            descriptive_stats.append(
                {
                    "Session": session,
                    "Group": "Active",
                    "Mean": active_group.mean(),
                    "Median": active_group.median(),
                    "1st Quartile": active_group.quantile(0.25),
                    "3rd Quartile": active_group.quantile(0.75),
                    "Standard Deviation": active_group.std(),
                    "Sample Size": len(active_group),
                }
            )

            descriptive_stats.append(
                {
                    "Session": session,
                    "Group": "Inactive",
                    "Mean": inactive_group.mean(),
                    "Median": inactive_group.median(),
                    "1st Quartile": inactive_group.quantile(0.25),
                    "3rd Quartile": inactive_group.quantile(0.75),
                    "Standard Deviation": inactive_group.std(),
                    "Sample Size": len(inactive_group),
                }
            )

        else:
            print(f"Skipped session {session} due to empty groups.")

    # Perform multiple comparison correction
    if all_pvalues:
        correction_methods = ["bonferroni", "holm", "fdr_bh"]
        corrections = {}
        for method in correction_methods:
            _, corrected_pvalues, _, _ = multipletests(all_pvalues, method=method)
            corrections[method] = corrected_pvalues

        # Add corrected p-values to results
        for i, result in enumerate(results):
            for method, corrected_pvalues in corrections.items():
                result[f"Corrected P-Value ({method})"] = corrected_pvalues[i]
    else:
        print("No tests were performed; no corrections applied.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    descriptive_stats_df = pd.DataFrame(descriptive_stats)

    return results_df, descriptive_stats_df


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

    # rename sessions to integers
    if cohort == "group_1":
        loaded_data["session"] = loaded_data["session"].map({"postop": 0, "fu3m": 1, "fu12m": 2})

    elif cohort == "group_2":
        loaded_data["session"] = loaded_data["session"].map({"fu3m": 0, "fu12m": 1, "fu18or24m": 2})

    for activity in ['active', 'inactive']:
        activity_data = loaded_data[loaded_data['clinical_activity'] == activity]

        # Check for duplicates and aggregate if necessary
        # if activity_data.duplicated(subset=["subject_hemisphere_contact", "session"]).any():
        #     print("Duplicate entries found. Aggregating by mean.")
        #     activity_data = activity_data.groupby(["subject_hemisphere_contact", "session"], as_index=False).mean()

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


def plot_active_or_inactive_plot_with_lines(cohort: str, active_or_inactive: str):
    """
    Plot boxplots with scatter and connecting lines for the "active" clinical_activity group
    across sessions.

    Parameters:
        cohort (str): Cohort identifier to filter the data.
        active_or_inactive (str): Activity group to plot ("active" or "inactive").

    Returns:
        stats_df (pd.DataFrame): A DataFrame with mean, median, Q1, Q3, and sample size for each session.
    """
    # Extract session and activity from 'session_clinical_activity'
    loaded_data = get_monopolar_beta_power_and_clinical_activity(cohort)

    # add new column for subject_hemisphere_contact
    loaded_data['subject_hemisphere_contact'] = loaded_data['subject_hemisphere'] + "_" + loaded_data['contact']

    # Filter the data for the "active" clinical activity group
    active_data = loaded_data[loaded_data['clinical_activity'] == active_or_inactive]

    if cohort == "group_1":
        session_order = ['postop', 'fu3m', 'fu12m']
    elif cohort == "group_2":
        session_order = ['fu3m', 'fu12m', 'fu18or24m']

    session_map = {session: i for i, session in enumerate(session_order)}
    active_data['x_vals'] = active_data['session'].map(session_map)
    active_data['session'] = pd.Categorical(active_data['session'], categories=session_order, ordered=True)

    # Sort data by `subject_hemisphere` and `session`
    active_data = active_data.sort_values(by=['subject_hemisphere', 'session'])

    # Determine the y-axis limits based on the range of beta_psd_rel_to_rank1
    y_min = active_data['beta_psd_rel_to_rank1'].min() - 0.3
    y_max = active_data['beta_psd_rel_to_rank1'].max() + 0.3

    # Initialize a list to store statistics
    stats_list = []
    x_val_list = []
    y_val_list = []
    subject_data_list = []

    # Plot settings
    fig, ax = plt.subplots(figsize=(10, 6))  # 12,8

    # Create the boxplot
    if active_or_inactive == "active":
        color = "gold"
    elif active_or_inactive == "inactive":
        color = "lightgray"

    # sns.boxplot(
    #     data=active_data,
    #     x='session',
    #     y='beta_psd_rel_to_rank1',
    #     color=color,
    #     width=0.6,
    #     showcaps=True,
    #     showfliers=False,
    #     boxprops={'facecolor': color, 'edgecolor': 'black', 'linewidth': 1.5},
    #     medianprops={'color': 'black', 'linewidth': 2},
    #     whiskerprops={'color': 'black', 'linewidth': 1.5},
    #     ax=ax,
    # )

    # Create the violin plot
    sns.violinplot(
        data=active_data,
        x='session',
        y='beta_psd_rel_to_rank1',
        color=color,
        inner=None,  # Disable inner lines as we'll add markers for statistics
        scale='width',
        linewidth=1.5,
        width=0.3,
        ax=ax,
    )

    # Adjust x-axis limits to make all violins visible
    session_labels = session_order
    # session_labels = active_data['session'].unique()
    ax.set_xlim(-0.5, len(session_labels) - 0.5)  # Add padding to both ends

    # Overlay scatterplot for individual points
    sns.stripplot(
        data=active_data,
        x='session',
        y='beta_psd_rel_to_rank1',
        jitter=True,
        size=9,
        color='black',
        alpha=0.5,
        edgecolor='k',
        linewidth=0.5,
        ax=ax,
    )

    # Add lines connecting the dots for the same subject across sessions
    for subject in active_data['subject_hemisphere_contact'].unique():
        subject_data = active_data[active_data['subject_hemisphere_contact'] == subject]

        # Sort data by session to ensure correct plotting
        subject_data = subject_data.sort_values(by='x_vals')

        x_vals = subject_data['x_vals'].values
        y_vals = subject_data['beta_psd_rel_to_rank1'].values

        x_val_list.append(x_vals)
        y_val_list.append(y_vals)
        subject_data_list.append(subject_data)

        # Plot connecting lines
        for i in range(len(x_vals) - 1):
            color = 'green' if y_vals[i + 1] > y_vals[i] else 'gray'
            plt.plot(
                [x_vals[i], x_vals[i + 1]],
                [y_vals[i], y_vals[i + 1]],
                color=color,
                alpha=0.3,
                linestyle='-',
                linewidth=1.5,
            )

    # Calculate and plot statistics for each session
    sessions = active_data['session'].unique()
    for session_idx, session in enumerate(sessions):
        # Subset data for the specific session
        subset = active_data[active_data['session'] == session]

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
                "Mean": mean_value,
                "Median": median_value,
                "Q1": q1,
                "Q3": q3,
                "Sample Size": sample_size,
                "Normality P-Value": normality_p_value,
                "Normal Distribution (p > 0.05)": normality_p_value is not None and normality_p_value > 0.05,
            }
        )

        # Plot mean as +
        ax.scatter(
            x=[session_idx],
            y=[mean_value],
            color='black',
            s=200,
            marker='+',
            zorder=10,
            label="Mean" if session_idx == 0 else None,
        )

        # Plot median as o
        ax.scatter(
            x=[session_idx],
            y=[median_value],
            color='white',
            s=100,
            marker='o',
            zorder=9,
            label="Median" if session_idx == 0 else None,
        )

        # Plot quartiles as vertical line
        ax.vlines(
            x=session_idx,
            ymin=q1,
            ymax=q3,
            color='white',
            linewidth=2,
            zorder=8,
            label="IQR" if session_idx == 0 else None,
        )

    # Customize the plot
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Beta PSD Relative to Rank 1 (Active): {cohort}', fontsize=16)
    ax.set_xlabel('Session', fontsize=14)
    ax.set_ylabel('Beta PSD Relative to Rank 1', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    helpers.save_fig_png_and_svg(
        path=FIGURES_PATH,
        filename=f"revision_{active_or_inactive}_monopol_beta_rel_to_rank1_{cohort}_optimal_stimulation_settings",
        figure=fig,
    )

    return stats_list, x_val_list, y_val_list, subject_data_list
