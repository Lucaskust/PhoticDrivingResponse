"""
Module phase evaluates the phase-locking of the stimulation frequencies.
"""
import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

import mne
from mne.io.base import BaseRaw
from numpy.fft import rfft, rfftfreq

import matplotlib.pyplot as plt


def _resolve_outdir(out_dir: Optional[str | Path]) -> Optional[Path]:
    if out_dir is None:
        return None
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv(df: pd.DataFrame, filename: str, out_dir: Optional[Path]) -> None:
    """Save df to out_dir/filename if out_dir is set; otherwise save to CWD (legacy)."""
    if out_dir is None:
        df.to_csv(filename, index=False)
    else:
        df.to_csv(out_dir / filename, index=False)


@dataclass
class Phase:
    """Implements the full phase calculation for EEG data."""
    passband: list
    eeg: BaseRaw
    out_dir: Optional[Path] = None

    def run(
        self,
        save_intermediates: bool = False,
        plot: bool = False,
        upper_lim: int = 40,
        base: bool = True,
    ):
        """
        Run the full pipeline on EEG data and returns (plv_stim, plv_base) DataFrames.
        """
        self.out_dir = _resolve_outdir(self.out_dir)

        if base:
            df_stim, df_base = self._stimulation_phase(self.eeg, save=save_intermediates, base=True, out_dir=self.out_dir)
            epochs_stim = self._epoch_phase(df_stim, self.eeg, upper_lim=upper_lim)
            epochs_base = self._epoch_phase(df_base, self.eeg, upper_lim=upper_lim)

            plv_stim = self._fft_phase(epochs_stim, plot=plot, save=save_intermediates, out_dir=self.out_dir, filename="phases_stim.csv")
            plv_base = self._fft_phase(epochs_base, plot=False, save=save_intermediates, out_dir=self.out_dir, filename="phases_base.csv")
            return plv_stim, plv_base

        # if you ever want stim-only
        df_stim = self._stimulation_phase(self.eeg, save=save_intermediates, base=False, out_dir=self.out_dir)
        epochs_stim = self._epoch_phase(df_stim, self.eeg, upper_lim=upper_lim)
        plv_stim = self._fft_phase(epochs_stim, plot=plot, save=save_intermediates, out_dir=self.out_dir, filename="phases.csv")
        return plv_stim

    # -------------------------
    # Subfunctions
    # -------------------------

    @staticmethod
    def _stimulation_phase(raw: BaseRaw, save: bool = False, base: bool = False, out_dir: Optional[Path] = None):
        """
        Extracts the stimulation blocks from trigger-annotations of EEG Data.
        Optionally creates baseline events for each stim freq per repetition.
        """
        sfreq = raw.info["sfreq"]
        block_threshold = 1.0 * sfreq

        sample, _ = mne.events_from_annotations(raw)
        df = pd.DataFrame(sample, columns=["sample", "previous", "event_id"])
        df["block"] = (np.diff(np.r_[0, df["sample"].to_numpy()]) > block_threshold).cumsum()

        def compute_freq(samples):
            return int(round(1 / np.mean(np.diff(samples) / sfreq), 2)) if len(samples) > 1 else np.nan

        df["freq"] = df.groupby("block")["sample"].transform(compute_freq)

        rep, rep_freqs = 1, set()
        for block_id, freq in df.groupby("block")["freq"].first().items():
            if pd.isna(freq):
                continue
            if freq in rep_freqs:
                rep += 1
                rep_freqs = {freq}
            else:
                rep_freqs.add(freq)
            df.loc[df["block"] == block_id, "rep"] = int(rep)

        df.drop(["event_id"], axis=1, inplace=True)

        df_base = None
        if base:
            baseline_blocks = []
            df_epochs = df.loc[df.groupby("block")["sample"].idxmin()].reset_index(drop=True)
            df_epochs["ends"] = df.groupby("block")["sample"].max().values
            tmax = int(math.ceil(((df_epochs["ends"] - df_epochs["sample"]) / sfreq).min()))

            for rep, rep_epochs in df.groupby("rep"):
                # per rep: maak baseline events voor elke freq die in die rep voorkomt
                freqs_rep = rep_epochs["freq"].dropna().astype(int).unique()
                for f in freqs_rep:
                    if f <= 0:
                        continue
                    start = rep_epochs["sample"].min() - 0.1 * sfreq - tmax * sfreq
                    if start < 0:
                        continue

                    period_samples = sfreq / f  # float
                    baseline_samples = np.arange(start, start + (tmax * sfreq), period_samples)

                    used_samples = set()
                    for s in baseline_samples:
                        event_sample = int(s + rep)  # ensure non-identical
                        while event_sample in used_samples:
                            event_sample += 1
                        used_samples.add(event_sample)

                        baseline_blocks.append(
                            {
                                "sample": int(event_sample),
                                "previous": int(0),
                                "freq": int(f),
                                "rep": int(rep),
                            }
                        )
            df_base = pd.DataFrame(baseline_blocks)

        if save:
            _save_csv(df, "phase_stimulation_info.csv", out_dir)
            if base and df_base is not None:
                _save_csv(df_base, "baseline_info.csv", out_dir)
                print("Saved: phase_stimulation_info.csv + baseline_info.csv")
            else:
                print("Saved: phase_stimulation_info.csv")

        return (df, df_base) if base else df

    @staticmethod
    def _epoch_phase(df: pd.DataFrame, raw: BaseRaw, upper_lim: int = 40) -> pd.Series:
        """
        Find epochs of the raw EEG data for different stimulation frequencies and harmonics.
        Returns a pandas Series with MultiIndex (Frequency, Harmonic) -> mne.Epochs
        """
        # Filter out NaNs / invalid frequencies
        freqs_stim = sorted([int(f) for f in df["freq"].dropna().unique() if int(f) > 0])

        filtered_epochs = {}
        cycles = 1.5

        np_epochs = df[["sample", "previous", "freq"]].to_numpy(dtype=int)

        for f in freqs_stim:
            max_harm = math.floor(upper_lim / f)
            for i in range(1, max_harm + 1):
                h = f * i

                bandwidth = 1
                l_freq = max(h - bandwidth, 0.1)
                h_freq = h + bandwidth

                raw_filt = raw.copy().filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    picks="eeg",
                    phase="zero",
                    verbose="ERROR",
                )

                tmax = cycles / h
                epochs = mne.Epochs(
                    raw_filt,
                    np_epochs,
                    event_id={str(f): f},
                    tmin=0,
                    tmax=tmax,
                    baseline=None,
                    preload=True,
                    verbose="ERROR",
                )

                filtered_epochs[(f, h)] = epochs

        ser = pd.Series(filtered_epochs)
        ser.index = pd.MultiIndex.from_tuples(ser.index, names=["Frequency", "Harmonic"])
        return ser

    @staticmethod
    def _fft_phase(
        filtered_epochs: pd.Series,
        plot: bool = False,
        save: bool = False,
        out_dir: Optional[Path] = None,
        filename: str = "phases.csv",
    ) -> pd.DataFrame:
        """
        Compute PLV for each (stim_freq, harmonic_freq) and per channel.
        """
        angles = {}
        phases = {}

        for (f, h) in filtered_epochs.index:
            epoch = filtered_epochs[(f, h)]
            ch_names = epoch.info["ch_names"]
            sfreq = epoch.info["sfreq"]
            data = epoch.get_data()
            _, _, n_times = data.shape

            window = np.hanning(n_times)
            data_windowed = data * window[np.newaxis, np.newaxis, :]
            fft_data = rfft(data_windowed, axis=2)

            freqs = rfftfreq(n_times, 1 / sfreq)
            center_idx = int(np.argmin(np.abs(freqs - h)))
            center_idx = np.clip(center_idx, 0, fft_data.shape[2] - 1)

            angles[(f, h)] = np.angle(fft_data[:, :, center_idx])

            plv = np.abs(np.mean(np.exp(1j * angles[(f, h)]), axis=0))
            mean_plv = float(np.mean(plv))
            mean_phase = float(np.angle(np.mean(np.exp(1j * np.angle(np.mean(np.exp(1j * angles[(f, h)]), axis=0))))))

            phases[(f, h)] = {
                "angles": angles[(f, h)],
                "plv": dict(zip(ch_names, plv)),
                "ch_names": ch_names,
                "mean_plv": mean_plv,
                "mean_phase": mean_phase,
            }

        if plot and len(phases) > 0:
            colors = plt.get_cmap("tab10").colors
            n_pairs = len(angles)
            n_cols = max(1, math.ceil(math.log(n_pairs, 2)))
            n_rows = math.ceil(n_pairs / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(5 * n_cols, 5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            for idx, ((f, h), content) in enumerate(phases.items()):
                p = content["angles"]
                ax = axes[idx]
                n_epochs, n_channels = p.shape

                ax.set_title(fr"$p(\theta \mid f={f}\,Hz, h={h})$", fontsize=10)

                for ch_idx in range(n_channels):
                    ch_name = content["ch_names"][ch_idx]
                    color = colors[ch_idx % len(colors)]
                    jitter = 1 + np.random.normal(-0.05, 0.05, size=n_epochs)
                    ax.scatter(p[:, ch_idx], jitter, s=5, c=[color], alpha=0.7, label=ch_name if idx == 0 else None)

                mean_vector = np.mean(np.exp(1j * p.flatten()))
                mean_angle = np.angle(mean_vector)
                plv_mean = np.abs(mean_vector)
                ax.plot([0, mean_angle], [0, plv_mean], linewidth=3, label="Mean vector" if idx == 0 else None)
                ax.set_ylim(0, 1.2)

            for ax in axes[n_pairs:]:
                ax.axis("off")

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.05, 0.5))
            fig.suptitle("Phase distributions across frequencies and harmonics", fontsize=14)
            plt.tight_layout()
            plt.show()

        # Build output dataframe
        rows = []
        for (f, h), ch_dict in phases.items():
            row = {"Frequency": f, "Harmonic": h}

            for ch_idx, ch_name in enumerate(ch_dict["ch_names"]):
                row[f"{ch_name}_plv"] = ch_dict["plv"][ch_name]
                row[f"{ch_name}_angles"] = np.degrees(np.angle(np.mean(np.exp(1j * ch_dict["angles"][:, ch_idx]))))

            row["mean_plv"] = ch_dict["mean_plv"]
            row["mean_phase"] = np.degrees(ch_dict["mean_phase"])
            rows.append(row)

        df = pd.DataFrame(rows).sort_values(["Frequency", "Harmonic"]).reset_index(drop=True)

        if save:
            _save_csv(df, filename, out_dir)
            print(f"Saved: {filename}")

        return df
