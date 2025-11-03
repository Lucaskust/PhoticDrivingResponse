"""
Kladkladklad 
Deze module gebruik ik puur om functies te testen.
"""
import glob, os

from pathlib import Path
import argparse

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

import mne
from mne.io.base import BaseRaw

from patients import parse_args, patient_files, eeg, filter_files
from power import Power
from phase import Phase
from analytics import stats_base_power, stats_power, stats_plv

# parser = argparse.ArgumentParser()
# trial_map = {
#     "t0": "T0_T1_T2",
#     "t1": "T0_T1",
#     "t2": "T0_T2"}
# parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t2",
# help = "Choose trial map: t0, t1 or t2")

# time_map = {
#     "t0": ["t0", "t1", "t2"],
#     "t1": ["t0", "t1"],
#     "t2": ["t0", "t2"]
# }
# parser.add_argument("-t", "--time", choices = time_map.keys(), default="t0",
#                     help = "Choose point in time: t0, t1 or t2")

# while True:
#     args = parser.parse_args()
#     if args.time not in time_map[args.trial]:
#         print(f"Error: timepoint {args.time} is not in trial {args.trial}")
#         print("Please choose again.")
#     break

# data_map = {
#     "t0": "1",
#     "t1": "2",
#     "t2": "3"
# }

# pt = "VEP38_" + data_map[args.time] + ".cnt"
# src = Path('/Volumes/Docs/Bruikbare Data') / trial_map[args.trial] / args.time / pt
# passband = [0.5, 100]

# # Comparing EEG without max filter
# eeg = eeg(src, passband, notch = 50, plot=True)
# raw = mne.io.read_raw_ant(src, preload=True, verbose='ERROR')
# raw.filter(l_freq=0.05, h_freq=1999, picks="eeg", verbose='ERROR')
# line_freq = 50 if (freq := raw.info["line_freq"]) is None else freq
# lowpass = np.arange(line_freq, raw.info["lowpass"]+1, line_freq)
# raw.notch_filter(freqs=(lowpass), notch_widths=(lowpass)/line_freq, picks=["eeg"], verbose='ERROR')
# raw.plot(scalings = "auto", title="Non-Filtered EEG data", show=True, block=False)

# Testing power functions
# eeg = eeg(src, passband, notch = 50, occi=False, plot=False)
# df = Power._stimulation_power(eeg, save=False)
# epochs, df_epochs = Power._epoch_power(df, eeg, save=False, plot=False)
# fft_powers, fft_freqs = Power._fft_power(epochs, df_epochs, trim=0.0, padding= "zeros", upper_lim = 40, plot=True)
# powers = Power._snr(epochs, fft_powers, fft_freqs, save=False, plot=True, harms=4, upper_lim=40, montage="standard_1020")

# Testing phase functions
# raw = eeg(src, passband, notch = 50, occi = True, plot=False)
# df = Phase._stimulation_phase(raw, save=False, base=False)
# epochs =  Phase._epoch_phase(df, raw)
# phases = Phase._fft_phase(epochs, plot = True, save=False)

# Baseline phase functions
# raw = eeg(src, passband, notch = 50, occi=True, plot=False)
# df, df_base = Phase._stimulation_phase(raw, save=True, base=True)
# epochs_baseline =  Phase._epoch_phase(df_base, raw)
# phases_baseline = Phase._fft_phase(epochs_baseline, plot=True, save=True)

responder_ids = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}
FOLDER_POWER = "results_POWER"
FOLDER_PLV = "results_PLV"
stats_power(responder_ids, FOLDER_POWER, paired=True, save=False, plot=True)
#stats_plv(responder_ids, FOLDER_PLV, paired=True, save=False, plot=True)

## Filtering files function

# # === MOCK / LOAD TRIAL MAP AND ARGS ===
# # Adjust to match your actual trial_map and args structure
# time_map = {
#     "t0": ["t0", "t1", "t2"],
#     "t1": ["t0", "t1"],
#     "t2": ["t0", "t1"]
# }
# class Args:
#     """Beepboop"""
#     time = "all"
#     trial = "t0"
# args = Args()


# # # === FILTER FILES ===
# # complete = filter_files(list(Path("./results_POWER").glob("*_power.pkl")), time_map, args)
# # print(f"Complete sets of files for: {complete}")

# # === Analytics ===
# power_path = Path("./results_POWER")
# stats_base(power_path, paired=True, save=True)



# # --- Aslabels en layout ---
# g.set(ylim=(df["Average_ABS"][df["Average_ABS"] > 0].min()*0.8, None))
# g.set_axis_labels("Frequency–Harmonic pair (Hz)", "Absolute Power (dB)")
# g.set_titles("{col_name}")
# for ax in g.axes.flatten():
#     ax.grid(axis='y', linestyle='--', alpha=0.4)
#     ax.tick_params(axis='x', rotation=45)


# # # === PHASE RESULTS AAhH visualisatie ===
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re

# # ----- SETTINGS -----
# order = [
#     (6, 6), (6, 12), (6, 18), (6, 24), (6, 30), (6, 36),
#     (10, 10), (10, 20), (10, 30), (10, 40),
#     (20, 20), (20, 40)
# ]
# freq_labels = [f"{f1}-{f2}" for f1, f2 in order]

# # ----- LOAD -----
# plv_stim = pd.read_csv("PLV_stim.csv")
# plv_base = pd.read_csv("PLV_base.csv")

# # ---- Ensure structure ----
# if plv_stim.shape[0] != len(order):
#     raise ValueError(f"Expected 12 rows (frequency pairs), got {plv_stim.shape[0]}.")

# # ---- Extract condition (t0, t1, t2) from column names ----
# def extract_condition(name):
#     match = re.search(r"_(\d)$", name)
#     if match:
#         idx = match.group(1)
#         return {"1": "t0 (no meds)", "2": "t1 (1 unit)", "3": "t2 (max dose)"}.get(idx, "unknown")
#     return "unknown"

# # Melt stimulation data (long format)
# stim_long = plv_stim.melt(var_name="Patient_Time", value_name="PLV")
# stim_long["FreqPair"] = freq_labels * (plv_stim.shape[1])
# stim_long["Condition"] = stim_long["Patient_Time"].apply(extract_condition)
# stim_long["Patient"] = stim_long["Patient_Time"].apply(lambda x: re.sub(r"_\d$", "", x))

# # Melt baseline data (no time condition)
# base_long = plv_base.melt(var_name="Patient_Time", value_name="PLV")
# base_long["FreqPair"] = freq_labels * (plv_base.shape[1])
# base_long["Condition"] = "Baseline"
# base_long["Patient"] = base_long["Patient_Time"]

# # Combine both datasets
# df_all = pd.concat([stim_long, base_long], ignore_index=True)

# # ---- Define responders ----
# responders = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}

# # ---- Assign responder group ----
# def classify_responder(patient):
#     # Extract only digits from patient ID
#     match = re.search(r"(\d+)", patient)
#     if match and match.group(1) in responders:
#         return "Responder"
#     else:
#         return "Non-responder"

# df_all["ResponseGroup"] = df_all["Patient"].apply(classify_responder)

# # ---- Order and palette ----
# condition_order = ["Baseline", "t0 (no meds)", "t1 (1 unit)", "t2 (max dose)"]
# group_order = [f"{c} - Responder" for c in condition_order] + [f"{c} - Non-responder" for c in condition_order]

# # ---- Filter valid conditions ----
# condition_order = ["Baseline", "t0 (no meds)", "t1 (1 unit)", "t2 (max dose)"]
# palette = {
#     "Baseline": "royalblue",
#     "t0 (no meds)": "firebrick",
#     "t1 (1 unit)": "darkorange",
#     "t2 (max dose)": "mediumorchid"
# }

# # ----- PLOTTING -----
# sns.set(style="whitegrid")

# # Create the catplot, but don't draw a legend automatically
# g = sns.catplot(
#     data=df_all,
#     x="FreqPair", y="PLV",
#     hue="Condition",
#     hue_order=condition_order,
#     col="ResponseGroup",
#     kind="box",
#     palette=palette,
#     linewidth=0.8,
#     width=0.55,
#     fliersize=3,
#     sharey=True,
#     col_order=["Responder", "Non-responder"],
#     height=6, aspect=1.3,
#     legend=False
# )

# # Axis and layout formatting
# g.set_axis_labels("Frequency pair (Hz)", "Phase Locking Value (PLV)")
# g.set_titles("{col_name}")
# g.set(ylim=(0, 1))

# for ax in g.axes.flatten():
#     ax.grid(axis='y', linestyle='--', alpha=0.4)
#     ax.tick_params(axis='x', rotation=45)

# # ---- Build legend handles from all data (once) ----
# handles, labels = [], []
# for cond, color in palette.items():
#     handles.append(plt.Line2D([], [], color=color, lw=8))
#     labels.append(cond)

# # ---- Add legends manually to each subplot ----
# for ax in g.axes.flatten():
#     leg = ax.legend(
#         handles, labels,
#         title="",
#         loc="upper right",
#         frameon=True,
#         facecolor="white",
#         edgecolor="gray",
#         fontsize=9
#     )
#     leg.get_frame().set_alpha(0.9)

# # ---- Title ----
# g.fig.suptitle(
#     "PLV distributions by dose condition and response group",
#     fontsize=15,
#     y=1
# )

# # Adjust spacing to make everything visible
# g.fig.subplots_adjust(top=0.90, right=0.97, wspace=0.15)
# plt.show()
