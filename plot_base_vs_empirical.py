#!/usr/bin/env python
# coding: utf-8

# Compares the Base GENIE and Empirical GENIE (O16) MC tunes, reading back
# the event branches exported by 05_FULL_DATA_signal_band.ipynb instead of
# re-scanning the raw MC/data ntuples. Empirical is rendered exactly as in
# that notebook (alpha-filled stack, hatched MC-stat band, on-beam data
# points, residual row); Base GENIE's total predicted stack is overlaid as
# a dashed red line for model comparison, with no stats band of its own and
# no contribution to the residual row.

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from lib.imports import *
from lib.constants import (
    bunch_sigma, bunch_time_start, bunch_time_start_data,
    ALPHA_SCALING, MC_POT_Empirical, MC_POT_Base,
)

EMPIRICAL_ROOT = "Empirical_GENIE.root"
BASE_ROOT      = "Base_GENIE.root"
PLOTS_DIR      = "plots/comparison/"
os.makedirs(PLOTS_DIR, exist_ok=True)

POT_MC_EMPIRICAL = MC_POT_Empirical / 1e20
POT_MC_BASE      = MC_POT_Base / 1e20

EXPECTED_SKYSHINE = 14.1   # from the 04_Skyshine_Sideband.ipynb sideband analysis; not GENIE-model dependent
EXPECTED_SKYSHINE_MID = EXPECTED_SKYSHINE                 # 1-2 sigma has the same exposure
EXPECTED_SKYSHINE_FAR = EXPECTED_SKYSHINE / (182 / 123.4)  # >2 sigma has the same exposure as the skyshine sideband

# from 05_FULL_DATA_signal_band.ipynb's on-beam/off-beam data cells (printed output)
POT_DATA  = 2.572970591694   # total_POT_TOR860, e20 POT
TRIG_DATA = 60624559         # total_triggers, on-beam
TRIG_EXT  = 24737351         # total_ext_triggers, off-beam

CATEGORY_NAMES = {0: "nuNCQE", 1: "NCother", 2: "CC", 3: "nubarNCQE", 4: "outOfFV", 5: "External"}
CAT_ORDER = [5, 4, 2, 1, 3, 0]   # plotting order used throughout 05_FULL_DATA_signal_band.ipynb


# ------------------------------------------------------------------ #
# 1. Load both exported ROOT files
# ------------------------------------------------------------------ #

def load_sample(root_path):
    with uproot.open(root_path) as f:
        mc         = {i: f[f"mc_{name}"].arrays(library="np") for i, name in CATEGORY_NAMES.items()}
        mc_outside = {i: f[f"mc_outside_{name}"].arrays(library="np") for i, name in CATEGORY_NAMES.items()}
        mc_mid     = {i: f[f"mc_mid_{name}"].arrays(library="np") for i, name in CATEGORY_NAMES.items()}
        mc_far     = {i: f[f"mc_far_{name}"].arrays(library="np") for i, name in CATEGORY_NAMES.items()}
        sample = dict(
            mc=mc, mc_outside=mc_outside, mc_mid=mc_mid, mc_far=mc_far,
            onbeam=f["onbeam"].arrays(library="np"),
            offbeam=f["offbeam"].arrays(library="np"),
            skyshine=f["skyshine"].arrays(library="np"),
            onbeam_outside=f["onbeam_outside"].arrays(library="np"),
            onbeam_mid=f["onbeam_mid"].arrays(library="np"),
            onbeam_far=f["onbeam_far"].arrays(library="np"),
            offbeam_outside=f["offbeam_outside"].arrays(library="np"),
            offbeam_mid=f["offbeam_mid"].arrays(library="np"),
            offbeam_far=f["offbeam_far"].arrays(library="np"),
        )
    return sample


print(f"Loading {EMPIRICAL_ROOT} and {BASE_ROOT}...")
empirical = load_sample(EMPIRICAL_ROOT)
base      = load_sample(BASE_ROOT)


def cat_stack(sample, tree, branch):
    """Category stack in the 5,4,2,1,3,0 plotting order used throughout 05_...ipynb."""
    return [sample[tree][i][branch] for i in CAT_ORDER]


def cat_weight_stack(sample, tree):
    return [sample[tree][i]["genie_weight"] for i in CAT_ORDER]


mc_weights_stack_empirical = cat_weight_stack(empirical, "mc")
mc_weights_stack_base      = cat_weight_stack(base, "mc")

common_args = dict(
    pot_mc=POT_MC_EMPIRICAL, pot_data=POT_DATA, trig_ext=TRIG_EXT, trig_data=TRIG_DATA,
    legend_columns=3, mc_weights_by_cat=mc_weights_stack_empirical,
)


# ------------------------------------------------------------------ #
# 2. Plotting function for single-panel PMT observables + Base overlay
# ------------------------------------------------------------------ #

def plot_with_skyshine_and_base(
        bins, mc_data, on_beam_data, off_beam_data, skyshine_data,
        base_mc_data, base_mc_weights_by_cat, base_pot_mc,
        pot_mc, pot_data, trig_ext, trig_data, expected_skyshine_yield,
        axis_label, y_axis_label, x_limit, y_limit, xaxis_minor,
        plot_name, legend_columns=3, mc_weights_by_cat=None):

    # 1. Scaling factors (Empirical, primary)
    f_pot = pot_data / pot_mc
    f_ext = trig_data / trig_ext
    f_sky = expected_skyshine_yield / len(skyshine_data) if len(skyshine_data) > 0 else 0.0

    prediction_data = [mc_data[0], skyshine_data] + mc_data[1:] + [off_beam_data]

    mc_weights = []
    for i, arr in enumerate(mc_data):
        genie_w = np.array(mc_weights_by_cat[i]) if mc_weights_by_cat is not None else np.ones(len(arr))
        mc_weights.append(genie_w * f_pot)

    weights = ([mc_weights[0], np.ones(len(skyshine_data)) * f_sky]
               + mc_weights[1:] + [np.ones(len(off_beam_data)) * f_ext])

    labels_base = ['External (MC)', 'Skyshine', 'Out-of-FV', 'CC', 'NCother',
                   r'$\bar{\nu}$NCQE', r'$\nu$NCQE', 'Off-Beam Data']
    colors = ['papayawhip', 'darkmagenta', 'lime', 'darkred', 'cyan', 'darkorange', '#0066ff', 'gray']

    weights[0] *= ALPHA_SCALING

    yields = [np.sum(w) for w in weights]
    total_pred_yield = sum(yields) if sum(yields) > 0 else 1
    labels = [f"{l} ({100*y/total_pred_yield:.1f}%)" for l, y in zip(labels_base, yields)]

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = np.diff(bins)

    data_counts, _ = np.histogram(on_beam_data, bins=bins)
    data_err = np.sqrt(data_counts)

    pred_hist_comps = [np.histogram(prediction_data[i], bins=bins, weights=weights[i])[0]
                        for i in range(len(prediction_data))]
    pred_counts = np.sum(pred_hist_comps, axis=0)
    pred_w2_comps = [np.histogram(prediction_data[i], bins=bins, weights=weights[i]**2)[0]
                      for i in range(len(prediction_data))]
    pred_err = np.sqrt(np.sum(pred_w2_comps, axis=0))

    # ---- Base GENIE total-prediction curve ----
    # reuses Empirical's own skyshine/off-beam arrays -- those backgrounds
    # aren't GENIE-tune dependent, so there's no separate "Base" version.
    f_pot_base = pot_data / base_pot_mc
    base_mc_weights = [np.array(base_mc_weights_by_cat[i]) * f_pot_base for i in range(len(base_mc_data))]
    base_prediction_data = [base_mc_data[0], skyshine_data] + base_mc_data[1:] + [off_beam_data]
    base_weights = ([base_mc_weights[0], np.ones(len(skyshine_data)) * f_sky]
                     + base_mc_weights[1:] + [np.ones(len(off_beam_data)) * f_ext])
    base_weights[0] *= ALPHA_SCALING
    base_pred_hist_comps = [np.histogram(base_prediction_data[i], bins=bins, weights=base_weights[i])[0]
                             for i in range(len(base_prediction_data))]
    base_pred_counts = np.sum(base_pred_hist_comps, axis=0)

    # --- Plotting ---
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.hist(prediction_data, bins=bins, stacked=True, weights=weights,
            label=labels, color=colors, linewidth=0, alpha=0.65)

    x_step = np.repeat(bins, 2)[1:-1]
    y_band_hi = np.repeat(pred_counts + pred_err, 2)
    y_band_lo = np.repeat(np.maximum(pred_counts - pred_err, 0), 2)
    ax.fill_between(x_step, y_band_lo, y_band_hi,
                     facecolor='none', edgecolor='dimgray',
                     linewidth=0, hatch='//', zorder=3, label='MC stats')

    ax.stairs(pred_counts, bins, color='black', linewidth=1.8, zorder=4)
    ax.stairs(base_pred_counts, bins, color='red', linestyle='--',
              linewidth=1.8, zorder=5, label='Base GENIE')

    ax.errorbar(bin_centers, data_counts, yerr=data_err, xerr=bin_widths/2,
                fmt='ko', label='On-Beam Data (stats only)', markersize=4, capsize=0)

    # --- Residual plot: Empirical vs data only, Base is not part of this ---
    mask = pred_counts > 0
    res = np.full_like(pred_counts, np.nan, dtype=float)
    res_err = np.full_like(pred_counts, np.nan, dtype=float)
    res[mask] = (data_counts[mask] - pred_counts[mask]) / pred_counts[mask]
    ratio = data_counts[mask] / pred_counts[mask]
    d_term = np.where(data_counts[mask] > 0, (data_err[mask]/data_counts[mask])**2, 0)
    res_err[mask] = ratio * np.sqrt(d_term + (pred_err[mask]/pred_counts[mask])**2)

    ax2.errorbar(bin_centers[mask], res[mask], yerr=res_err[mask], xerr=bin_widths[mask]/2, fmt='ko', markersize=4)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax2.axhline(-0.5, color='gray', linestyle=':', linewidth=0.8)

    ax.set_ylabel(y_axis_label)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    ax.tick_params(axis='x', labelbottom=False)
    ax.xaxis.set_minor_locator(MultipleLocator(xaxis_minor))

    ax2.set_ylabel(r'$\frac{\mathrm{Data}}{\mathrm{Pred}} - 1$', fontsize=14)
    ax2.set_xlabel(axis_label)
    ax2.set_xlim(x_limit)
    ax2.set_ylim([-1.2, 1.2])
    ax2.xaxis.set_minor_locator(MultipleLocator(xaxis_minor))

    ax.legend(ncol=legend_columns, fontsize=7.5, loc='upper right', frameon=False)

    ax.text(0.02, 1.02, 'ANNIE', fontweight='bold', color='blue', transform=ax.transAxes)
    ax.text(0.70, 1.02, f"{pot_data:.2f} " + r"$\times 10^{20}$ POT", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------ #
# 3. Plotting function for the 3-panel miniboone-style delta_t plot + Base overlay
# ------------------------------------------------------------------ #

def plot_with_skyshine_3panel_and_base(
        bins, mc_data, on_beam_data, off_beam_data, skyshine_data,
        mc_data_mid, mc_weights_mid, on_beam_mid, off_beam_mid, skyshine_mid_data,
        mc_data_far, mc_weights_far, on_beam_far, off_beam_far, skyshine_far_data,
        base_mc_data, base_mc_weights_by_cat, base_pot_mc,
        base_mc_data_mid, base_mc_weights_mid,
        base_mc_data_far, base_mc_weights_far,
        pot_mc, pot_data, trig_ext, trig_data,
        expected_skyshine_yield, expected_skyshine_mid_yield, expected_skyshine_far_yield,
        axis_label, y_axis_label, x_limit, y_limit, y_limit_outside,
        xaxis_minor, yaxis_minor, yaxis_minor_outside,
        plot_name, legend_columns=3, mc_weights_by_cat=None, combined_errors=False):

    # ---- shared scaling factors (Empirical, primary) ----
    f_pot = pot_data / pot_mc
    f_ext = trig_data / trig_ext
    f_sky     = expected_skyshine_yield     / len(skyshine_data)     if len(skyshine_data)     > 0 else 0.0
    f_sky_mid = expected_skyshine_mid_yield / len(skyshine_mid_data) if len(skyshine_mid_data) > 0 else 0.0
    f_sky_far = expected_skyshine_far_yield / len(skyshine_far_data) if len(skyshine_far_data) > 0 else 0.0

    # ---- INSIDE (main panel) ----
    mc_weights = []
    for i, arr in enumerate(mc_data):
        genie_w = np.array(mc_weights_by_cat[i]) if mc_weights_by_cat is not None else np.ones(len(arr))
        mc_weights.append(genie_w * f_pot)

    prediction_data = [mc_data[0], skyshine_data] + mc_data[1:] + [off_beam_data]
    weights = ([mc_weights[0], np.ones(len(skyshine_data)) * f_sky]
               + mc_weights[1:] + [np.ones(len(off_beam_data)) * f_ext])

    # ---- MID (1-2 sigma) ----
    mid_mc_w  = [np.array(mc_weights_mid[i]) * f_pot for i in range(len(mc_data_mid))]
    mid_pred_data    = [mc_data_mid[0], skyshine_mid_data] + mc_data_mid[1:] + [off_beam_mid]
    mid_pred_weights = ([mid_mc_w[0], np.ones(len(skyshine_mid_data)) * f_sky_mid]
                         + mid_mc_w[1:] + [np.ones(len(off_beam_mid)) * f_ext])

    # ---- FAR (>2 sigma) ----
    far_mc_w  = [np.array(mc_weights_far[i]) * f_pot for i in range(len(mc_data_far))]
    far_pred_data    = [mc_data_far[0], skyshine_far_data] + mc_data_far[1:] + [off_beam_far]
    far_pred_weights = ([far_mc_w[0], np.ones(len(skyshine_far_data)) * f_sky_far]
                         + far_mc_w[1:] + [np.ones(len(off_beam_far)) * f_ext])

    # ---- aesthetics ----
    labels_base = ['External (MC)', 'Skyshine', 'Out-of-FV', 'CC',
                    'NCother', r'$\bar{\nu}$NCQE', r'$\nu$NCQE', 'Off-Beam Data']
    colors      = ['papayawhip', 'darkmagenta', 'lime', 'darkred',
                    'cyan', 'darkorange', '#0066ff', 'gray']

    weights[0]          *= ALPHA_SCALING
    mid_pred_weights[0] *= ALPHA_SCALING
    far_pred_weights[0] *= ALPHA_SCALING

    # ---- main panel histogram math ----
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths  = np.diff(bins)

    data_counts, _ = np.histogram(on_beam_data, bins=bins)
    data_err = np.sqrt(data_counts)

    pred_hist_comps = [np.histogram(prediction_data[i], bins=bins, weights=weights[i])[0]
                        for i in range(len(prediction_data))]
    pred_counts = np.sum(pred_hist_comps, axis=0)
    pred_w2_comps = [np.histogram(prediction_data[i], bins=bins, weights=weights[i]**2)[0]
                        for i in range(len(prediction_data))]
    pred_err = np.sqrt(np.sum(pred_w2_comps, axis=0))
    plot_err = np.sqrt(data_err**2 + pred_err**2) if combined_errors else data_err

    def single_bin_stats(on_beam, pred_weights):
        data_count = len(on_beam)
        data_err_  = np.sqrt(data_count)
        yields     = [np.sum(w) for w in pred_weights]
        pred_total = sum(yields)
        pred_err_  = np.sqrt(sum(np.sum(w**2) for w in pred_weights))
        plot_err_  = np.sqrt(data_err_**2 + pred_err_**2) if combined_errors else data_err_
        return data_count, yields, pred_total, pred_err_, plot_err_

    mid_data_count, mid_yields, mid_pred_total, mid_pred_err, mid_plot_err = \
        single_bin_stats(on_beam_mid, mid_pred_weights)
    far_data_count, far_yields, far_pred_total, far_pred_err, far_plot_err = \
        single_bin_stats(on_beam_far, far_pred_weights)

    yields = [np.sum(w) for w in weights]
    total_pred_yield = sum(yields) if sum(yields) > 0 else 1
    labels = [f"{l} ({100*y/total_pred_yield:.1f}%)" for l, y in zip(labels_base, yields)]

    # ---- Base GENIE total-prediction curves (main + mid + far) ----
    # reuses Empirical's own skyshine/off-beam arrays throughout, same reasoning as above.
    f_pot_base = pot_data / base_pot_mc

    base_mc_w = [np.array(base_mc_weights_by_cat[i]) * f_pot_base for i in range(len(base_mc_data))]
    base_prediction_data = [base_mc_data[0], skyshine_data] + base_mc_data[1:] + [off_beam_data]
    base_weights = ([base_mc_w[0], np.ones(len(skyshine_data)) * f_sky]
                     + base_mc_w[1:] + [np.ones(len(off_beam_data)) * f_ext])
    base_weights[0] *= ALPHA_SCALING
    base_pred_hist_comps = [np.histogram(base_prediction_data[i], bins=bins, weights=base_weights[i])[0]
                             for i in range(len(base_prediction_data))]
    base_pred_counts = np.sum(base_pred_hist_comps, axis=0)

    def base_single_bin_total(base_mc_data_, base_mc_weights_, skyshine_data_, off_beam_data_, f_sky_):
        base_mc_w_ = [np.array(base_mc_weights_[i]) * f_pot_base for i in range(len(base_mc_data_))]
        base_mc_w_[0] = base_mc_w_[0] * ALPHA_SCALING
        base_pred_weights_ = ([base_mc_w_[0], np.ones(len(skyshine_data_)) * f_sky_]
                               + base_mc_w_[1:] + [np.ones(len(off_beam_data_)) * f_ext])
        return sum(np.sum(w) for w in base_pred_weights_)

    base_mid_pred_total = base_single_bin_total(base_mc_data_mid, base_mc_weights_mid, skyshine_mid_data, off_beam_mid, f_sky_mid)
    base_far_pred_total = base_single_bin_total(base_mc_data_far, base_mc_weights_far, skyshine_far_data, off_beam_far, f_sky_far)

    # ------------------------------------------------------------------ #
    # Layout: wide main | mid (1-2 sigma) | far (>2 sigma), residual row each
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 6))
    gs  = fig.add_gridspec(2, 5, height_ratios=[4, 1],
                            width_ratios=[10, 0.4, 1.7, 0.12, 1.7],
                            hspace=0.1, wspace=0.08)

    ax      = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[1, 0])
    ax_mid  = fig.add_subplot(gs[0, 2])
    ax_mid2 = fig.add_subplot(gs[1, 2])
    ax_far  = fig.add_subplot(gs[0, 4])
    ax_far2 = fig.add_subplot(gs[1, 4])

    # ---- main panel: alpha fill + hatch MC-stat band + bold total outline ----
    ax.hist(prediction_data, bins=bins, stacked=True, weights=weights,
            label=labels, color=colors, linewidth=0, alpha=0.65)

    x_step    = np.repeat(bins, 2)[1:-1]
    y_band_hi = np.repeat(pred_counts + pred_err, 2)
    y_band_lo = np.repeat(np.maximum(pred_counts - pred_err, 0), 2)
    ax.fill_between(x_step, y_band_lo, y_band_hi,
                     facecolor='none', edgecolor='dimgray',
                     linewidth=0, hatch='//', zorder=3, label='MC stats')
    ax.stairs(pred_counts, bins, color='black', linewidth=1.8, zorder=4)
    ax.stairs(base_pred_counts, bins, color='red', linestyle='--',
              linewidth=1.8, zorder=5, label='Base GENIE')

    ax.errorbar(bin_centers, data_counts, yerr=plot_err, xerr=bin_widths/2,
                fmt='ko', label='On-Beam Data (stats only)', markersize=4, capsize=0)

    # ---- main residual (Empirical vs data only) ----
    mask = pred_counts > 0
    res = np.full_like(pred_counts, np.nan, dtype=float)
    res_err = np.full_like(pred_counts, np.nan, dtype=float)
    res[mask] = (data_counts[mask] - pred_counts[mask]) / pred_counts[mask]
    ratio  = data_counts[mask] / pred_counts[mask]
    d_term = np.where(data_counts[mask] > 0, (data_err[mask]/data_counts[mask])**2, 0)
    res_err[mask] = ratio * np.sqrt(d_term + (pred_err[mask]/pred_counts[mask])**2)

    ax2.errorbar(bin_centers[mask], res[mask], yerr=res_err[mask],
                 xerr=bin_widths[mask]/2, fmt='ko', markersize=4)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax2.axhline(-0.5, color='gray', linestyle=':', linewidth=0.8)

    # ---- helper: draw one "outside" stacked-bar panel + its residual ----
    def draw_outside_panel(ax_top, ax_bot, yields_, pred_total_, pred_err_,
                            data_count_, plot_err_, tick_label, base_pred_total_):
        bar_x, bar_w = 0.5, 0.7
        bottom = 0.0
        for yv, col in zip(yields_, colors):
            ax_top.bar(bar_x, yv, bar_w, bottom=bottom,
                       color=col, edgecolor='none', linewidth=0, alpha=0.65)
            bottom += yv

        out_total = pred_total_ if pred_total_ > 0 else 1
        bottom = 0.0
        for yv, col in zip(yields_, colors):
            if yv / out_total > 0.05:
                ax_top.text(bar_x, bottom + yv/2, f'{100*yv/out_total:.1f}%',
                            ha='center', va='center', fontsize=8, color='black')
            bottom += yv

        lo, hi = max(pred_total_ - pred_err_, 0), pred_total_ + pred_err_
        ax_top.fill_between([bar_x - bar_w/2, bar_x + bar_w/2], lo, hi,
                             facecolor='none', edgecolor='dimgray',
                             linewidth=0, hatch='//', zorder=3)
        ax_top.plot([bar_x - bar_w/2, bar_x + bar_w/2], [pred_total_, pred_total_],
                    color='black', linewidth=1.8, zorder=4)
        ax_top.plot([bar_x - bar_w/2, bar_x + bar_w/2], [base_pred_total_, base_pred_total_],
                    color='red', linestyle='--', linewidth=1.8, zorder=5)

        ax_top.errorbar(bar_x, data_count_, yerr=plot_err_, fmt='ko', markersize=4, capsize=0)

        if pred_total_ > 0:
            res_     = (data_count_ - pred_total_) / pred_total_
            ratio_   = data_count_ / pred_total_
            d_term_  = (np.sqrt(data_count_)/data_count_)**2 if data_count_ > 0 else 0
            res_err_ = ratio_ * np.sqrt(d_term_ + (pred_err_/pred_total_)**2)
            ax_bot.errorbar(bar_x, res_, yerr=res_err_, fmt='ko', markersize=4)

        ax_bot.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_bot.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax_bot.axhline(-0.5, color='gray', linestyle=':', linewidth=0.8)

        ax_top.set_xlim([0, 1]); ax_top.tick_params(axis='x', labelbottom=False); ax_top.set_xticks([])
        ax_bot.set_xlim([0, 1]); ax_bot.set_ylim([-1.2, 1.2])
        ax_bot.set_xticks([0.5]); ax_bot.set_xticklabels([tick_label], fontsize=14)

    draw_outside_panel(ax_mid, ax_mid2, mid_yields, mid_pred_total, mid_pred_err,
                        mid_data_count, mid_plot_err, r'$1$-$2\sigma_{\nu\mathrm{NCQE}}$', base_mid_pred_total)
    draw_outside_panel(ax_far, ax_far2, far_yields, far_pred_total, far_pred_err,
                        far_data_count, far_plot_err, r'$>2\sigma_{\nu\mathrm{NCQE}}$', base_far_pred_total)

    ax_mid.set_ylim(y_limit_outside)
    ax_mid.yaxis.set_minor_locator(MultipleLocator(yaxis_minor_outside))
    ax_mid.tick_params(axis='y', labelleft=False)
    ax_mid2.tick_params(axis='y', labelleft=False)

    ax_far.set_ylim(y_limit_outside)
    ax_far.yaxis.set_minor_locator(MultipleLocator(yaxis_minor_outside))
    ax_far.yaxis.tick_right()
    ax_far.tick_params(axis='y', labelright=True, labelleft=False)

    ax_far2.yaxis.tick_right()
    ax_far2.tick_params(axis='y', labelright=True, labelleft=False)

    # ---- formatting ----
    ax.set_ylabel(y_axis_label, fontsize=16)
    ax.set_xlim(x_limit); ax.set_ylim(y_limit)
    ax.tick_params(axis='x', labelbottom=False)
    ax.xaxis.set_minor_locator(MultipleLocator(xaxis_minor))
    ax.yaxis.set_minor_locator(MultipleLocator(yaxis_minor))

    ax2.set_ylabel(r'$\frac{\mathrm{Data}}{\mathrm{Pred}} - 1$', fontsize=16)
    ax2.set_xlabel(axis_label, fontsize=16)
    ax2.set_xlim(x_limit); ax2.set_ylim([-1.2, 1.2])
    ax2.xaxis.set_minor_locator(MultipleLocator(xaxis_minor))

    ax.legend(ncol=legend_columns, fontsize=7.5, loc='upper right', frameon=False)
    ax.text(0.02, 1.02, 'ANNIE', fontweight='bold', color='blue', transform=ax.transAxes)
    ax.text(0.60, 1.02, f"{pot_data:.2f}" + r" $\times 10^{20}$ POT", transform=ax.transAxes)

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------ #
# 4. Single-panel PMT observables
# ------------------------------------------------------------------ #

plot_with_skyshine_and_base(
    bins=np.arange(5, 13, 1),
    mc_data=cat_stack(empirical, "mc", "reco_E"),
    on_beam_data=empirical["onbeam"]["reco_E"],
    off_beam_data=empirical["offbeam"]["reco_E"],
    skyshine_data=empirical["skyshine"]["reco_E"],
    base_mc_data=cat_stack(base, "mc", "reco_E"),
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    axis_label='Reconstructed energy [MeV]',
    y_axis_label='Events / 1 MeV bin',
    x_limit=[5, 12],
    y_limit=[0, 200],
    xaxis_minor=0.5,
    plot_name=PLOTS_DIR + 'reco_energy.png',
    **common_args,
)

plot_with_skyshine_and_base(
    bins=np.arange(10, 50, 4),
    mc_data=cat_stack(empirical, "mc", "hits"),
    on_beam_data=empirical["onbeam"]["hits"],
    off_beam_data=empirical["offbeam"]["hits"],
    skyshine_data=empirical["skyshine"]["hits"],
    base_mc_data=cat_stack(base, "mc", "hits"),
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    axis_label='PMT hits',
    y_axis_label='Events / 4 hit bin',
    x_limit=[10, 50],
    y_limit=[0, 250],
    xaxis_minor=1,
    plot_name=PLOTS_DIR + 'PMT_hits.png',
    **common_args,
)

plot_with_skyshine_and_base(
    bins=np.arange(0, 0.8, 0.1),
    mc_data=cat_stack(empirical, "mc", "radius"),
    on_beam_data=empirical["onbeam"]["radius"],
    off_beam_data=empirical["offbeam"]["radius"],
    skyshine_data=empirical["skyshine"]["radius"],
    base_mc_data=cat_stack(base, "mc", "radius"),
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    axis_label='Reconstructed distance to center [m]',
    y_axis_label='Events / 10 cm bin',
    x_limit=[0, 0.7],
    y_limit=[0, 300],
    xaxis_minor=0.1,
    plot_name=PLOTS_DIR + 'radius.png',
    **common_args,
)

plot_with_skyshine_and_base(
    bins=np.arange(-1, 1.25, 0.25),
    mc_data=cat_stack(empirical, "mc", "vertex_y"),
    on_beam_data=empirical["onbeam"]["vertex_y"],
    off_beam_data=empirical["offbeam"]["vertex_y"],
    skyshine_data=empirical["skyshine"]["vertex_y"],
    base_mc_data=cat_stack(base, "mc", "vertex_y"),
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    axis_label='Reconstructed vertex (Y) [m]',
    y_axis_label='Events / 50 cm bin',
    x_limit=[-1, 1],
    y_limit=[0, 175],
    xaxis_minor=0.2,
    plot_name=PLOTS_DIR + 'Y.png',
    **common_args,
)

for name, branch, cv_bins, cv_xlim, cv_ylim in [
        ('x', 'CV_x', np.arange(-1, 1.25, 0.25), [-1, 1],   [0, 200]),
        ('y', 'CV_y', np.arange(-1, 1.25, 0.25), [-1, 1],   [0, 225]),
        ('z', 'CV_z', np.arange(-1, 0.6, 0.2),   [-1, 0.4], [0, 200]),
]:
    plot_with_skyshine_and_base(
        bins=cv_bins,
        mc_data=cat_stack(empirical, "mc", branch),
        on_beam_data=empirical["onbeam"][branch],
        off_beam_data=empirical["offbeam"][branch],
        skyshine_data=empirical["skyshine"][branch],
        base_mc_data=cat_stack(base, "mc", branch),
        base_mc_weights_by_cat=mc_weights_stack_base,
        base_pot_mc=POT_MC_BASE,
        expected_skyshine_yield=EXPECTED_SKYSHINE,
        axis_label=r'$C_V$ ' + f'({name})',
        y_axis_label='Events / bin',
        x_limit=cv_xlim,
        y_limit=cv_ylim,
        xaxis_minor=0.1,
        plot_name=f'{PLOTS_DIR}Cv_{name}.png',
        **common_args,
    )


# ------------------------------------------------------------------ #
# 5. Time-since-spill-start (bunches_6)
# ------------------------------------------------------------------ #
# bunches_80_* in 05_...ipynb is just bunch_time shifted by a constant
# (bunch_time_start for MC, bunch_time_start_data for data/ext, no
# per-run branching despite the notebook's get_normalized_times() name),
# so it's re-derivable directly from the already-exported bunch_time
# branches -- no export changes needed for this observable.

def normalized_times(bunch_time, mc):
    shift = bunch_time_start if mc else bunch_time_start_data
    return np.asarray(bunch_time) - shift


bunches_80_mc_empirical = [normalized_times(empirical["mc"][i]["bunch_time"], mc=True) for i in CAT_ORDER]
bunches_80_mc_base      = [normalized_times(base["mc"][i]["bunch_time"], mc=True) for i in CAT_ORDER]
bunches_80_data = normalized_times(empirical["onbeam"]["bunch_time"], mc=False)
bunches_80_ext  = normalized_times(empirical["offbeam"]["bunch_time"], mc=False)

# skyshine_times in 05_...ipynb is a synthetic inverse-transform-sampled draw,
# not a real simulated sample -- regenerated the same way here. Seeded (unlike
# the notebook) so this script's own output is reproducible run-to-run.
t0 = 353.6
t_max = 18.936 * 25 + bunch_sigma
N_skyshine_mc = len(empirical["skyshine"]["reco_E"])
np.random.seed(0)
u = np.random.uniform(0, 1, N_skyshine_mc)
skyshine_times = t0 + (t_max - t0) * np.sqrt(u)

bunch_width = 18.936 * 6
bunches_bins = np.arange(18.936, 18.936 * 25 + bunch_width, bunch_width)

plot_with_skyshine_and_base(
    bins=bunches_bins,
    mc_data=bunches_80_mc_empirical,
    on_beam_data=bunches_80_data,
    off_beam_data=bunches_80_ext,
    skyshine_data=skyshine_times,
    base_mc_data=bunches_80_mc_base,
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    axis_label='Time since first bunch [ns]',
    y_axis_label='Events / 6 BNB bunches',
    x_limit=[18.936, 18.936 * 25],
    y_limit=[0, 300],
    xaxis_minor=50,
    plot_name=PLOTS_DIR + 'bunches_6.png',
    **common_args,
)


# ------------------------------------------------------------------ #
# 6. Miniboone-style 3-panel delta_t plot
# ------------------------------------------------------------------ #
# mc_mid_*/mc_far_*/onbeam_mid/onbeam_far/offbeam_mid/offbeam_far are
# already split by sigma region in the ROOT export itself, so (unlike the
# live notebook session) no re-masking against delta_t is needed here --
# straight readback.

mc_stack_mid_empirical     = cat_stack(empirical, "mc_mid", "delta_t")
mc_weights_mid_empirical   = cat_weight_stack(empirical, "mc_mid")
mc_stack_far_empirical     = cat_stack(empirical, "mc_far", "delta_t")
mc_weights_far_empirical   = cat_weight_stack(empirical, "mc_far")

mc_stack_mid_base   = cat_stack(base, "mc_mid", "delta_t")
mc_weights_mid_base = cat_weight_stack(base, "mc_mid")
mc_stack_far_base   = cat_stack(base, "mc_far", "delta_t")
mc_weights_far_base = cat_weight_stack(base, "mc_far")

data_mid = empirical["onbeam_mid"]["delta_t"]
data_far = empirical["onbeam_far"]["delta_t"]
ext_mid  = empirical["offbeam_mid"]["delta_t"]
ext_far  = empirical["offbeam_far"]["delta_t"]

plot_with_skyshine_3panel_and_base(
    bins=np.array([-bunch_sigma, -2.5, -1, 1, 2.5, bunch_sigma]),
    mc_data=cat_stack(empirical, "mc", "delta_t"),
    on_beam_data=empirical["onbeam"]["delta_t"],
    off_beam_data=empirical["offbeam"]["delta_t"],
    skyshine_data=empirical["skyshine"]["delta_t"],

    mc_data_mid=mc_stack_mid_empirical, mc_weights_mid=mc_weights_mid_empirical,
    on_beam_mid=data_mid, off_beam_mid=ext_mid, skyshine_mid_data=empirical["skyshine"]["delta_t"],

    mc_data_far=mc_stack_far_empirical, mc_weights_far=mc_weights_far_empirical,
    on_beam_far=data_far, off_beam_far=ext_far, skyshine_far_data=empirical["skyshine"]["delta_t"],

    base_mc_data=cat_stack(base, "mc", "delta_t"),
    base_mc_weights_by_cat=mc_weights_stack_base,
    base_pot_mc=POT_MC_BASE,
    base_mc_data_mid=mc_stack_mid_base, base_mc_weights_mid=mc_weights_mid_base,
    base_mc_data_far=mc_stack_far_base, base_mc_weights_far=mc_weights_far_base,

    pot_mc=POT_MC_EMPIRICAL, pot_data=POT_DATA, trig_ext=TRIG_EXT, trig_data=TRIG_DATA,
    expected_skyshine_yield=EXPECTED_SKYSHINE,
    expected_skyshine_mid_yield=EXPECTED_SKYSHINE_MID,
    expected_skyshine_far_yield=EXPECTED_SKYSHINE_FAR,
    axis_label=r'$\Delta t_{\mathrm{nearest\ bunch}}$ [ns]',
    y_axis_label='Events / bin',
    x_limit=[-bunch_sigma, bunch_sigma],
    y_limit=[0, 350],
    y_limit_outside=[0, 650],
    xaxis_minor=0.5,
    yaxis_minor=10,
    yaxis_minor_outside=50,
    plot_name=PLOTS_DIR + 'delta_t_3panel.png',
    legend_columns=3,
    mc_weights_by_cat=mc_weights_stack_empirical,
)

print('done')
