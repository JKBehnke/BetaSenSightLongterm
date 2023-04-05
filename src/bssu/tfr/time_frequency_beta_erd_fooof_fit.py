"""Perform and save time frequency analysis of given files."""
from pathlib import Path

import fooof
import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pte

import plotting_settings
import project_constants


def fit_model_iterative(
    freqs: np.ndarray, power_spectrum: np.ndarray, fig, ax
) -> tuple[fooof.FOOOF, matplotlib.figure.Figure]:
    """Fit FOOOF model iteratively."""
    fit_knee: bool = True
    while True:
        if fit_knee:
            print("Fitting model WITH knee component.")
        else:
            print("Fitting model WITHOUT knee component.")
        model = fit_model(freqs, power_spectrum, fit_knee, ax)
        fig.tight_layout()
        fig.canvas.draw()
        redo_fit = get_input_y_n("Try new fit w or w/o knee")
        ax.clear()
        if redo_fit.lower() == "n":
            break
        fit_knee = not fit_knee
    return model, fig


def fit_model(
    freqs: np.ndarray, power_spectrum: np.ndarray, fit_knee: bool, ax
) -> fooof.FOOOF:
    """Fit fooof model."""
    aperiodic_mode = "knee" if fit_knee else "fixed"
    model = fooof.FOOOF(
        peak_width_limits=(2, 20.0),
        max_n_peaks=4,
        min_peak_height=0.0,
        peak_threshold=1.0,
        aperiodic_mode=aperiodic_mode,
        verbose=True,
    )
    model.fit(freqs=freqs, power_spectrum=power_spectrum)
    model.print_results()
    model.plot(ax=ax)
    return model


def get_input_y_n(message: str) -> str:
    """Get ´y` or `n` user input."""
    while True:
        user_input = input(f"{message} (y/n)? ")
        if user_input.lower() in ["y", "n"]:
            break
        print(
            f"Input must be `y` or `n`. Got: {user_input}."
            " Please provide a valid input."
        )
    return user_input


def main() -> None:
    """Main function of this script."""
    # Load project settings
    DERIVATIVES: Path = project_constants.DERIVATIVES
    RESULTS: Path = project_constants.RESULTS

    PLOT_DIR: Path = project_constants.PLOTS / "fooof_spectra"
    PLOT_DIR.mkdir(exist_ok=True)

    PIPELINE: Path = Path(
        "time_frequency", "tfr_morlet_2022_09_20", "emg_onset"
    )

    IN_ROOT: Path = DERIVATIVES / PIPELINE

    RESULTS_ROOT: Path = RESULTS / PIPELINE
    RESULTS_ROOT.mkdir(exist_ok=True, parents=True)

    FOOOF_ROOT: Path = RESULTS_ROOT / "fooof_models"
    FOOOF_ROOT.mkdir(exist_ok=True)

    CHANNEL = "ECOG"
    TIMES_ROOT = (
        RESULTS
        / "pte_decode"
        / "2022_11_27"
        / "20_predict"
        / f"lda_opt_no_-0.1_trial_onset_100ms_chs_{CHANNEL.lower()}_hem_contralat"
    )

    med_paired = [sub.strip("sub-") for sub in project_constants.MED_PAIRED]
    times_raw = pd.read_csv(TIMES_ROOT / f"timepoints.csv").rename(
        columns={"Earliest Timepoint": "Time"}
    )
    times: dict[str, int | float] = {}
    for sub in med_paired:
        df_sub = times_raw.query(f"Subject == '{sub}'")
        time_min = df_sub.Time.min()
        times[sub] = time_min

    FMIN = 3
    FMAX = 45
    # TMIN = -1.5
    TMAX = 0.0
    MEDICATION = None
    STIMULATION = None
    KEYWORDS = project_constants.MED_PAIRED
    EXCLUDE = "sub-002"
    PICK = "sub-016"

    CHANNELS = ["dbs"]

    file_finder = pte.filetools.get_filefinder(datatype="any")
    file_finder.find_files(
        directory=IN_ROOT,
        extensions=["tfr.h5"],
        keywords=KEYWORDS,
        medication=MEDICATION,
        stimulation=STIMULATION,
        exclude=EXCLUDE,
    )
    file_finder.filter_files(PICK)
    print(file_finder)

    results = []
    peak_fits = []
    fig, ax = plt.subplots(1, 1)
    fig.show()
    for file in file_finder.files[:]:
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(file)
        basename = Path(file).stem.removesuffix("_tfr")

        decoding_time: int | float = times[sub]

        power = pte.time_frequency.load_power(files=[file])[0]
        power = power.crop(
            fmin=FMIN, fmax=FMAX, tmin=decoding_time, tmax=TMAX
        ).average()

        for channel in CHANNELS:
            power_ch = power.copy().pick(channel)
            freqs = power_ch.freqs
            print(power_ch.info["ch_names"])

            for ch in power_ch.info["ch_names"][:]:

                power_spectrum: np.ndarray = power_ch.copy().pick(ch).data
                power_spectrum = power_spectrum.mean(axis=-1).squeeze()
                fig.suptitle(ch)
                model, fig = fit_model_iterative(
                    freqs, power_spectrum, fig=fig, ax=ax
                )
                model.save(
                    file_name=basename + "_model.json",
                    file_path=str(FOOOF_ROOT),
                    save_results=True,
                    save_settings=True,
                    save_data=False,
                )
                model.save_report(
                    file_name=basename + "_model.pdf",
                    file_path=str(FOOOF_ROOT),
                )
                fig.savefig(str(PLOT_DIR / (basename + ".png")))
                beta_peaks = fooof.analysis.get_band_peak_fm(
                    model,
                    band=(13.0, 35.0),
                    select_highest=False,
                    attribute="peak_params",
                )
                if beta_peaks.ndim == 1:
                    beta_peaks = np.expand_dims(beta_peaks, axis=0)
                results.extend(
                    (
                        [
                            sub,
                            med,
                            stim,
                            ch,
                            peak[0],
                            peak[1],
                            peak[2],
                        ]
                        for peak in beta_peaks
                    )
                )
                peak_fits.append([sub, med, stim, ch, *model._peak_fit])
    final = pd.DataFrame(
        results,
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "Channel",
            "CenterFrequency",
            "Power",
            "BandWidth",
        ],
    ).replace({"DBS": "LFP"})
    final.to_csv(
        RESULTS_ROOT / "beta_erd_fooof_peaks_sub-016.csv", index=False
    )
    powers = pd.DataFrame(
        peak_fits,
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "Channel",
            *(freq for freq in freqs),
        ],
    ).replace({"DBS": "LFP"})
    powers.to_csv(
        RESULTS_ROOT / f"beta_erd_fooof_power_sub-016.csv", index=False
    )


if __name__ == "__main__":
    main()
