"""Plotting script for event detection."""

from cli_plotting import _get_parser
from os.path import join as opj

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab

import ev

fontsize = 28
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
pylab.rcParams.update(params)
# Global variables
NRAND = 100
TR = 0.83
FIGSIZE = (45, 30)
HISTORY = "Deconvolution based on event-detection."
# Font size for plots
font = {"weight": "normal", "size": 28}
matplotlib.rc("font", **font)


def plot_comparison(
    rss_orig_sur,
    rssr_orig_sur,
    idxpeak_orig_sur,
    rss_fitt,
    rssr_fitt,
    idxpeak_fitt,
    rss_beta,
    rssr_beta,
    idxpeak_beta,
    rss_auc,
    rssr_auc,
    idxpeak_auc,
    ats,
    outdir,
):
    """
    Plot comparison of different RSS with vertical subplots.
    """
    greymap = cm.get_cmap("Greys")
    colors = greymap(np.linspace(0, 0.65, rssr_orig_sur.shape[1]))

    min_range = np.min(np.minimum(rss_orig_sur, rss_fitt)) * 0.9
    max_range = np.max(np.maximum(rss_orig_sur, rss_fitt)) * 1.1

    _, axs = plt.subplots(5, 1, figsize=FIGSIZE)
    for i in range(rssr_orig_sur.shape[1]):
        axs[0].plot(rssr_orig_sur[:, i], color=colors[i], linewidth=0.5)
    axs[0].plot(
        idxpeak_orig_sur,
        rss_orig_sur[idxpeak_orig_sur],
        "r*",
        label="orig_sur-peaks",
        markersize=20,
    )
    axs[0].plot(
        rss_orig_sur,
        color="k",
        linewidth=3,
        label="orig_sur",
    )
    axs[0].set_ylim([min_range, max_range])
    axs[0].set_title("Original signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[1].plot(rssr_fitt[:, i], color=colors[i], linewidth=0.5)
    axs[1].plot(
        idxpeak_fitt, rss_fitt[idxpeak_fitt], "r*", label="fitt-peaks", markersize=20
    )
    axs[1].plot(rss_fitt, color="k", linewidth=3, label="fitt")
    axs[1].set_ylim([min_range, max_range])
    axs[1].set_title("Fitted signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[2].plot(rssr_beta[:, i], color=colors[i], linewidth=0.5)
    axs[2].plot(
        idxpeak_beta, rss_beta[idxpeak_beta], "r*", label="beta-peaks", markersize=20
    )
    axs[2].plot(rss_beta, color="k", linewidth=3, label="beta")
    axs[2].set_title("Betas")

    for i in range(rssr_orig_sur.shape[1]):
        axs[3].plot(rssr_auc[:, i], color=colors[i], linewidth=0.5)
    axs[3].plot(
        idxpeak_auc, rss_auc[idxpeak_auc], "r*", label="AUC-peaks", markersize=20
    )
    axs[3].plot(rss_auc, color="k", linewidth=3, label="AUC")
    axs[3].set_title("AUCs")

    axs[4].plot(ats, label="ATS", color="black")
    axs[4].set_title("Activation time-series")

    plt.legend()
    plt.savefig(opj(outdir, "event_detection.png"), dpi=300)


def plot_all(
    rss_orig_sur,
    idxpeak_orig_sur,
    rss_beta,
    idxpeak_beta,
    rss_fitt,
    idxpeak_fitt,
    outdir,
):
    """
    Plot all RSS lines on same figure.
    """
    plt.figure(figsize=FIGSIZE)

    # Original signal
    rss_orig_norm = (rss_orig_sur - rss_orig_sur.min()) / (
        rss_orig_sur.max() - rss_orig_sur.min()
    )
    plt.plot(
        idxpeak_orig_sur,
        rss_orig_norm[idxpeak_orig_sur],
        "r*",
        "linewidth",
        3,
        label="orig_sur-peaks",
    )
    plt.plot(
        range(rss_orig_norm.shape[0]),
        rss_orig_norm,
        "k",
        "linewidth",
        3,
        label="orig_sur",
    )

    # Betas
    rss_beta_norm = (rss_beta - rss_beta.min()) / (rss_beta.max() - rss_beta.min())
    plt.plot(
        idxpeak_beta,
        rss_beta_norm[idxpeak_beta],
        "g*",
        "linewidth",
        3,
        label="deconvolved_peaks",
    )
    plt.plot(
        range(rss_beta_norm.shape[0]),
        rss_beta_norm,
        "b",
        "linewidth",
        3,
        label="deconvolved",
    )

    # Fitted signal
    rss_fitt_norm = (rss_fitt - rss_fitt.min()) / (rss_fitt.max() - rss_fitt.min())
    plt.plot(
        idxpeak_fitt,
        rss_fitt_norm[idxpeak_fitt],
        "m*",
        "linewidth",
        3,
        label="fitted_peaks",
    )
    plt.plot(
        range(rss_fitt_norm.shape[0]),
        rss_fitt_norm,
        "y",
        "linewidth",
        3,
        label="fitted",
    )
    plt.legend()
    plt.savefig(opj(outdir, "event_detection_all.png"), dpi=300)


def plot_ets_matrix(ets, outdir, sufix="", dvars=None, enorm=None, peaks=None, vmin=-0.5, vmax=0.5):
    """
    Plots edge-time matrix
    """
    if dvars is not None and enorm is not None:
        # widths = [1]
        # heights = [2, 1, 1]
        # gs = dict(width_ratios=widths, height_ratios=heights)
        # fig, axs = plt.subplots(3, 1, figsize=FIGSIZE,gridspec_kw=gs)
        # im = axs[0].imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        # axs[0].set_title("Edge-time series")
        # axs[0].set_ylabel("Edge-edge connections")
        # fig.colorbar(im, orientation="vertical", ax=axs[0]) # ax=axs.ravel().tolist()
        # axs[1].plot(dvars)
        # axs[1].set_title("DVARS")
        # axs[2].plot(enorm)
        # axs[2].set_title("ENORM")
        # axs[2].set_xlabel("Time (TR)")
        fig = plt.subplots(figsize=FIGSIZE)
        ax0 = plt.subplot(111)
        divider = make_axes_locatable(ax0)
        ax1 = divider.append_axes("bottom", size="25%", pad=1)
        ax2 = divider.append_axes("bottom", size="25%", pad=1)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        im = ax0.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        ax0.set_ylabel("Edge-edge connections")
        cb = plt.colorbar(im, orientation="vertical", ax=ax0, cax=cax)  # ax=axs.ravel().tolist()
        dvars[1] = np.mean(dvars)
        ax1.plot(dvars)
        ax1.set_title("DVARS")
        ax1.margins(0, 0)
        for i in peaks:
            ax1.axvspan(i, i+1, facecolor='b', alpha=0.5)
            ax2.axvspan(i, i+1, facecolor='b', alpha=0.5)
        ax2.plot(enorm)
        ax2.set_title("ENORM")
        ax2.set_xlabel("Time (TR)")
        ax2.margins(0, 0)
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)
    else:
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        plt.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        plt.title("Edge-time series")
        plt.xlabel("Time (TR)")
        plt.ylabel("Edge-edge connections")
        plt.colorbar()
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)


def main(argv=None):
    """
    Main function to perform event detection and plot results.
    """
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)

    # Global variables
    SUBJECT = kwargs["subject"][0]
    NROIS = kwargs["nROI"][0]
    # Paths to files
    MAINDIR = kwargs["dir"][0]
    TEMPDIR = opj(MAINDIR, f"temp_{SUBJECT}_{NROIS}")
    ORIGDIR = "/bcbl/home/public/PARK_VFERRER/PREPROC/" + SUBJECT + "/func/task-restNorm_acq-MB3_run-01"
    ats_name = "pb06." + SUBJECT + ".denoised_no_censor_ATS_abs_95.1D"
    ATS = np.loadtxt(opj(MAINDIR, ats_name))
    ATLAS = opj(TEMPDIR, "atlas.nii.gz")
    DATAFILE = opj(MAINDIR, f"pb06.{SUBJECT}.denoised_no_censor.nii.gz")
    BETAFILE = opj(MAINDIR, f"pb06.{SUBJECT}.denoised_no_censor_beta_95.nii.gz")
    FITTFILE = opj(MAINDIR, f"pb06.{SUBJECT}.denoised_no_censor_fitt_95.nii.gz")
    AUCFILE = opj(MAINDIR, f"{SUBJECT}_AUC_{NROIS}.nii.gz")
    # Perform event detection on BETAS
    print("Performing event-detection on betas...")
    (
        ets_beta,
        rss_beta,
        rssr_beta,
        idxpeak_beta,
        etspeaks_beta,
        mu_beta,
        _,
        _,
        _,
    ) = ev.event_detection(BETAFILE, ATLAS, opj(TEMPDIR, "surrogate_"), "_beta_95")

    # Perform event detection on ORIGINAL data
    print("Performing event-detection on original data...")
    (
        ets_orig_sur,
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        etspeaks_orig_sur,
        mu_orig_sur,
        _,
        _,
        _,
    ) = ev.event_detection(DATAFILE, ATLAS, opj(TEMPDIR, "surrogate_"))

    # Perform event detection on FITTED signal
    print("Performing event-detection on fitted signal...")
    (
        ets_fitt,
        rss_fitt,
        rssr_fitt,
        idxpeak_fitt,
        etspeaks_fitt,
        mu_fitt,
        _,
        _,
        _,
    ) = ev.event_detection(FITTFILE, ATLAS, opj(TEMPDIR, "surrogate_"), "_fitt_95")

    # Perform event detection on AUC
    print("Performing event-detection on AUC...")
    (
        ets_auc,
        rss_auc,
        rssr_auc,
        idxpeak_auc,
        etspeaks_AUC,
        mu_AUC,
        ets_auc_denoised,
        idx_u,
        idx_v,
    ) = ev.event_detection(AUCFILE, ATLAS, opj(TEMPDIR, "surrogate_AUC_"))

    print("Making plots...")
    # Plot comparison of rss time series, null, and significant peaks for
    # original, betas, fitted, AUC and ATS
    plot_comparison(
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        rss_fitt,
        rssr_fitt,
        idxpeak_fitt,
        rss_beta,
        rssr_beta,
        idxpeak_beta,
        rss_auc,
        rssr_auc,
        idxpeak_auc,
        ATS,
        MAINDIR,
    )

    # Plot all rss time series, null, and significant peaks in one plot
    plot_all(
        rss_orig_sur, idxpeak_orig_sur, rss_beta, idxpeak_beta, rss_fitt,
        idxpeak_fitt, MAINDIR
    )

    print("Plotting original, AUC, and AUC-denoised ETS matrices...")
    # Plot ETS matrix of original signal
    DVARS = np.loadtxt(opj(ORIGDIR, SUBJECT + "_dvars.1D"))
    ENORM = np.loadtxt(opj(ORIGDIR, SUBJECT + "_Motion_enorm.1D"))
    plot_ets_matrix(ets_orig_sur, MAINDIR, "_original", DVARS, ENORM, idxpeak_auc)

    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(ets_auc, MAINDIR, "_AUC_original", DVARS, ENORM, idxpeak_auc)
    plot_ets_matrix(ets_auc_denoised, MAINDIR, "_AUC_denoised", DVARS, ENORM, idxpeak_auc)

    # Save RSS time-series as text file for easier visualization on AFNI
    rss_out = np.zeros(rss_auc.shape)
    rss_out[idxpeak_auc] = rss_auc[idxpeak_auc]
    np.savetxt(opj(MAINDIR, f"{DATAFILE[:-7]}_rss.1D"), rss_out)

    # Perform debiasing based on thresholded edge-time matrix
    beta, _ = ev.debiasing(
        DATAFILE, ATLAS, ets_auc_denoised, idx_u, idx_v, TR, MAINDIR, HISTORY
    )

    print("Plotting edge-time matrix of ETS-based deconvolution.")
    denoised_beta_ets, _, _ = ev.calculate_ets(beta, beta.shape[1])
    plot_ets_matrix(denoised_beta_ets, MAINDIR, "_beta_denoised",DVARS, ENORM, idxpeak_auc)

    print("THE END")


if __name__ == "__main__":
    main()
