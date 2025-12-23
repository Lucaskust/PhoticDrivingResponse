"""
Module power calculates the power spectral densities of the different flash-stimuli blocks and 
calculates the SNR (flash-stimulation-harmonics vs baseline).
"""
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

import mne
from mne.io.base import BaseRaw
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
matplotlib.use("TkAgg")

@dataclass
class Power:
    """ Implements the full power calculation pipeline for EEG data. """
    passband: list
    raw: BaseRaw
    eeg: Optional[BaseRaw] = field(default=None, init=False)

    def run(self):
        """
        Run the full pipeline on EEg data and return the SNR DataFrame.
        
        :EEG: filtered EEG data, should contain trigger timing in annotations.
        """
        self.eeg = self.raw.copy()
        df_stim = self._stimulation_power(self.eeg, save=False)
        epochs, df_epochs = self._epoch_power(df_stim, self.eeg, save=False, plot=False)
        fft_powers_db, fft_powers_lin, fft_freq = self._fft_power(epochs, df_epochs, trim=0.0, padding="zeros", plot=False)
        powers = self._snr(epochs, fft_powers_db, fft_freq, harms=5, upper_lim=40, save=False, plot=False)


        return powers

    # -------------------------
    # Subfunctions
    # -------------------------

    @staticmethod
    def _stimulation_power(raw:BaseRaw, save: bool = False) -> pd.DataFrame:
        """
        Extracts the stimulation blocks from trigger-annotations of EEG Data.

        Parameters
        ----------        
        :raw: mne.BaseRaw
            EEG data, should contain trigger timing in annotations.
        :save: bool, optional
            Option to save the made dataframe.

        Returns
        -------
        :df: Pandas DataFrame
            Dataframe containing EEG information.
        """
        sfreq = raw.info["sfreq"]
        block_threshold = 1.0 * sfreq

        sample, _ = mne.events_from_annotations(raw)
        df = pd.DataFrame(sample, columns=["sample", "previous", "event_id"])
        df["block"] = (np.diff(np.r_[0, df["sample"].to_numpy()]) > block_threshold).cumsum()

        def compute_freq(samples):
            return int(round(1/np.mean(np.diff(samples)/sfreq),2)) if len(samples) > 1 else np.nan

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
            df.loc[df["block"] == block_id, "block base"] = int(rep + block_id)
            df.loc[df["block"] == block_id, "rep"] = int(rep)
        df.drop(["block", "event_id"], axis=1, inplace=True)

        # Option to save the dataframe
        if save:
            df.to_csv("power_stimulation_info.csv", index=False)
            print("Stimulation dataframe saved as power_stimulation_info.csv")

        return df

    @staticmethod
    def _epoch_power(df: pd.DataFrame, raw: BaseRaw, save: bool = False, plot: bool = False
               )-> tuple[mne.Epochs, pd.DataFrame]:
        """
        Adds baseline blocks with no stimulation and epochs the raw EEG data around all blocks. 

        Parameters
        ----------
        :df: Pandas DataFrame
            Dataframe containing information about the stimulation blocks.
        :raw: mne.BaseRaw
            EEG data, should contain trigger timing in annotations.
        :save: bool, optional
            Option to save the made dataframe.
        :plot: bool, optional
            Option to plot the made epochs.

        Returns
        -------
        :epochs: mne.Epochs
            EEG epoched around the different frequencies.
        :df_epochs:
            Dataframe containing information about the epochs.
        """
        sfreq = raw.info["sfreq"]
        df_epochs = df.loc[df.groupby("block base")["sample"].idxmin()].reset_index(drop=True)
        df_epochs["ends"] = df.groupby("block base")["sample"].max().values
        tmax = int(math.ceil(((df_epochs["ends"] - df_epochs["sample"]) / sfreq).min()))

        blocks_baseline = []
        for rep, rep_epochs in df_epochs.groupby("rep"):
            start = rep_epochs["sample"].min() - 0.1*sfreq - tmax*sfreq

            if start < 0:
                print("-- skipping first baseline due to too little prestimulation data.")
                continue
            block_baseline = {
                "sample": int(start),
                "previous": 0,
                "block base": int((rep-1)*(len(rep_epochs)+1)+1),
                "freq": 0,
                "rep": int(rep),
                "ends": int(start + tmax * sfreq)
            }
            blocks_baseline.append(block_baseline)

        df_epochs = pd.concat([df_epochs, pd.DataFrame(blocks_baseline)], ignore_index=True)
        df_epochs = df_epochs.sort_values(by="sample").reset_index(drop=True)

        # Option to save DataFrame as .csv file
        if save:
            df_epochs.to_csv("power_epochs.csv", float_format="%.3f", index=False)
            print("Epoch dataframe saved as power_epochs.csv")

        np_epochs = df_epochs[["sample", "previous", "freq"]].to_numpy(dtype=int)
        epochs = mne.Epochs(raw, np_epochs, event_id=None, tmin=0, tmax=tmax,
                            baseline=None, preload=True, verbose='ERROR') # pylint: disable=not-callable

        # Option to plot Epochs
        if plot:
            epochs.plot(block=True)

        return epochs, df_epochs

    @staticmethod
    def _fft_power(epochs: mne.Epochs, df_epochs: pd.DataFrame, trim: float = 0.0, padding: str = "copy",
                    upper_lim: int = 40, plot: bool = False)-> tuple[dict, dict]:
        """
        Automatically compute FFT for all blocks.

        Parameters
        ----------
        :epochs: mne.Epochs
            EEG data epoched around the different frequencies.
        :df_epochs:
            Dataframe containing information about the epochs.
        :trim: float
            Time in seconds we trim at epoch start in order to avoid spectral leakage.
        :padding: string, optional
            Option to use padding for epoching, choices: copy, zeros, none. 
        :upper_lim: int
            Maximum analyzed frequency, default 40 Hz.
        :plot: bool, optional
            Option to plot the PSDs pet stimulation frequency. 

        Returns
        -------
        :fft_powers: dictionary
            Containing (reps x freq x chs) amount of lists with power in dB µV²/Hzper bin. 
            - [rep] = repetition number, zero contains the average over repetitions
            - [freq] = stimulation frequency 
            - [ch_name] = channel name
        :fft_freqs: list
            Contains the frequencies in Hz per bin. Should be 1 Hz per bin, truncated by the upper passband frequency. 
        """

        # Option to trim epochs to avoid transient onset of the brain.
        if trim > 0:
            t0 = epochs.tmin + trim
            if t0 >= epochs.tmax:
                raise ValueError("trim too large for epoch duration")
            epochs = epochs.crop(tmin=t0, tmax=epochs.tmax)

        # Data retrieval
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        ch_names = epochs.info["ch_names"]

        reps =int((df_epochs["rep"]).max())
        freqs = sorted(df_epochs["freq"].dropna().astype(int).unique())

        # FFT preparation (Hanning window & Welch)
        fft_powers = {rep: {f: {ch: [] for ch in ch_names} for f in freqs} for rep in range(0, reps + 1)}
        fft_powers_lin = {rep: {f: {ch: [] for ch in ch_names} for f in freqs} for rep in range(0, reps + 1)}

        overlap = 0.5
        window_length = int(sfreq)
        step = int(window_length * overlap)
        fft_freq= rfftfreq(window_length, 1/sfreq).squeeze()
        mask = fft_freq <= upper_lim
        fft_freq = fft_freq[mask]

        window = np.hanning(window_length)
        window_scale = np.sum(window**2) / window_length

        for index, epoch_info in df_epochs.iterrows():
            freq = epoch_info["freq"]
            rep = epoch_info["rep"]
            epoch_all = data[index]

            for ch_idx, ch_name in enumerate(ch_names):
                epoch = epoch_all[ch_idx]

                # Option: padding
                if padding == "copy":
                    epoch_padded = np.concatenate([epoch[:window_length], epoch, epoch[-window_length:]])
                elif padding == "zeros":
                    zeros = np.zeros(window_length)
                    epoch_padded = np.concatenate([zeros, epoch, zeros])
                elif padding == "none":
                    epoch_padded = epoch
                else:
                    raise ValueError ("Chosen padding not found. Please choose: copy, zeros or none.")

                segments_values = []
                segments_powers = []

                # FFT, windowing & Welch's method
                for start in range(0, len(epoch_padded) - window_length + 1, step):
                    segment = epoch_padded[start:start+window_length]
                    segment_windowed = segment * window
                    fft_value =  rfft(segment_windowed)
                    segments_values.append(fft_value[mask])
                    power = (np.abs(fft_value)**2 / window_length)/window_scale
                    segments_powers.append(power[mask])

                # Convert to dB
                # Mean linear power (µV^2 / "bin")  - FOOOF/specparam uses linear
                mean_lin = np.mean(segments_powers, axis=0) * 1e12
                mean_lin = np.maximum(mean_lin, 1e-30)  # avoid log(0)

                fft_powers_lin[rep][freq][ch_name] = mean_lin
                fft_powers[rep][freq][ch_name] = 10 * np.log10(mean_lin)


        # Average across reps (both dB and linear)
        for freq in freqs:
            for ch_name in ch_names:
                vals_db = []
                vals_lin = []
                for r in fft_powers:
                    if r == 0:
                        continue
                    if freq in fft_powers[r] and isinstance(fft_powers[r][freq][ch_name], np.ndarray):
                        vals_db.append(fft_powers[r][freq][ch_name])
                    if freq in fft_powers_lin[r] and isinstance(fft_powers_lin[r][freq][ch_name], np.ndarray):
                        vals_lin.append(fft_powers_lin[r][freq][ch_name])

                fft_powers[0][freq][ch_name] = np.mean(vals_db, axis=0) if len(vals_db) else np.full_like(fft_freq, np.nan)
                fft_powers_lin[0][freq][ch_name] = np.mean(vals_lin, axis=0) if len(vals_lin) else np.full_like(fft_freq, np.nan)


        # Option to plot different PSD's averaged over all channels
        if plot:
            _, axes = plt.subplots(1, len(freqs), figsize=(16, 4))

            for ax, f in zip(axes, freqs):
                # Channel means
                channel_means = np.array([fft_powers[0][f][ch] for ch in ch_names])
                grand_mean = np.nanmean(channel_means, axis=0)

                for ch_idx, ch_name in enumerate(ch_names):
                    ax.plot(fft_freq, channel_means[ch_idx], color="green", alpha=0.1, linewidth=1)
                ax.plot([], [], color="green", alpha=0.5, label="solo channels")
                ax.plot(fft_freq, grand_mean, color="black", linestyle="-", linewidth=1.1, label="overall mean")
                ax.set_title(f"{f:.1f} Hz")
                ax.set_xlim(0, upper_lim)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (dB µV^2/Hz)")
                ax.legend()
                ax.grid(True, linestyle='dotted', alpha=0.6)

                if f > 0:
                    for h in range(1, int(math.floor(upper_lim/f))):
                        harmonic = f * h
                        ax.axvline(harmonic, color='gray', linestyle='dotted', linewidth=1)
            plt.tight_layout()
            plt.show()
        return fft_powers, fft_powers_lin, fft_freq

    @staticmethod
    def _snr(epochs: mne.Epochs, fft_powers: dict, fft_freq: dict, save: bool = False, plot: bool = False,
             harms: int = 4, upper_lim: int = 40, montage="standard_1020")-> pd.DataFrame:
        """
        Computes the SNR's for different frequencies. 

        Parameters
        ----------
        :epochs: mne.Epochs
            EEG data epoched around the different frequencies.
        :fft_powers: dictionary
            Containing (reps x freq x chs) amount of lists with power in dB µV²/Hzper bin. 
            - [rep] = repetition number, zero contains the average over repetitions
            - [freq] = stimulation frequency 
            - [ch_name] = channel name
        :fft_freqs: list
            Contains the frequencies in Hz per bin. Should be 1 Hz per bin, truncated by the upper passband frequency. 
        :save: bool, optional
            Option to save the made SNR and baseline dataframe.
        :plot: bool, optional
            Option to plot a topomop of the different SNRs.
        :harms: int
            The number of harmonics of the stimulation frequencies to plot in topomap.
        :upper_lim: int
            Maximum analyzed frequency, default 40 Hz.
        :montage: string
            Type of montage used during data acquisition. Standard: 10-20 system. Needed for topomap.

        Returns
        -------
        :df_all: Pandas DataFrame
            Dataframe containing the power values for all frequency-pairs.
        """

        freqs = [f for f in fft_powers[1].keys() if f != 0]
        ch_names = epochs.info["ch_names"]
        rows_all = []

        for freq in freqs:
            harmonics = [freq * i for i in range(1, math.floor(upper_lim/freq)+1)]

            for h in harmonics:
                bin_idx = int(np.argmin(np.abs(np.array(fft_freq) - h)))

                powers_absolute = {ch: fft_powers[0][freq][ch][bin_idx] for ch in ch_names}
                powers_baseline = {ch: fft_powers[0][0][ch][bin_idx] for ch in ch_names}
                powers_snr = {
                    ch: fft_powers[0][freq][ch][bin_idx] - fft_powers[0][0][ch][bin_idx]
                    for ch in ch_names}

                row = {"Frequency": freq,
                        "Harmonic": h,
                        # compute averages for each metric across channels
                        "Average_SNR": np.mean(list(powers_snr.values())),
                        "Average_BASE": np.mean(list(powers_baseline.values())),
                        "Average_PWR": np.mean(list(powers_absolute.values())),
                        **{f"{ch}_SNR": val for ch, val in powers_snr.items()},
                        **{f"{ch}_BASE": val for ch, val in powers_baseline.items()},
                        **{f"{ch}_PWR": val for ch, val in powers_absolute.items()}}
                rows_all.append(row)
        df_all = pd.DataFrame(rows_all)

        if save:
            df_all.to_csv("powers.csv", float_format="%.3f", index=False)

        if plot:
            df_snr = df_all[["Frequency", "Harmonic"] + [c for c in df_all.columns if "_SNR" in c]]
            ch_names = [f"{ch}_SNR" for ch in ch_names]
            fig, axes = plt.subplots(len(freqs), harms, figsize=((harms*6), len(freqs)*5))
            axes = np.atleast_2d(axes)
            epochs.set_montage(montage)
            pos = mne.channels.layout._find_topomap_coords(epochs.info, picks='eeg') # pylint: disable=protected-access
            dx, dy = [0, -0.02]
            pos_shifted = pos+np.array([dx, dy])  #Om ook in de outline te passen
            vmax = df_snr[ch_names].max().max()
            vmin = -vmax
            sm = matplotlib.cm.ScalarMappable(cmap='RdBu_r', norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])

            for i, freq in enumerate(freqs):
                for h in range(harms):
                    powers_snr = df_snr[(df_snr["Frequency"] == freq)
                                        & (df_snr["Harmonic"] == freq * (h+1))][ch_names].values.flatten()
                    if powers_snr.shape[0] == 0:
                        axes[i, h].text(0.5, 0.5, 'No data', ha='center', va='center')
                        axes[i, h].set_xticks([])
                        axes[i, h].set_yticks([])
                        continue
                    axes[i, h].set_title(f"{freq} Hz, h={freq * (h+1)}")
                    mne.viz.plot_topomap(powers_snr, pos_shifted, axes=axes[i, h],
                                         show=False, outlines="head", sensors=True,
                                         vlim=(vmin, vmax), names=[f"{v:.1f}" for v in powers_snr])
                    for text in axes[i, h].texts:
                        text.set_fontsize(5)

            fig.suptitle("SNR Topomaps", fontsize=16)
            fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.90, wspace=0.3, hspace=0.4)
            cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax, label='SNR (dB µV^2 / Hz)')
            plt.show()

        return df_all
