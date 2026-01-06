#!/usr/bin/env python3
"""
concat_psd_test.py

Doel
-----
Aantonen dat:
(1) "gemiddelde van PSD's over 3 herhalingen"  ~=  (2) "PSD van geconcateneerde herhalingen"
als je EXACT dezelfde Welch settings gebruikt (1s Hann windows, 50% overlap, zelfde scaling).

Script doet:
- Load .cnt via patients.eeg() preprocessing (jullie bestaande functie).
- Extract stimulation blocks uit annotations (zelfde logica als power._stimulation_power).
- Epoch elk block met gemeenschappelijke duur tmax (zelfde logica als power._epoch_power).
- Maakt 2 PSD schattingen per stimulatiefrequentie:
    A) AVG: Welch PSD per herhaling-epoch, daarna average over reps
    B) CONCAT: concat herhaling-epochs tot één lange serie (optioneel cross-fade), daarna één Welch PSD
- Slaat op:
    - <stem>_concat_vs_avg_psd.png
    - <stem>_concat_vs_avg_metrics.csv

Run (single file):
    python concat_psd_test.py --cnt "D:/cenobamate_eeg_1/VEP02_1.cnt" --outdir out_test

Run (alle .cnt in directory, niet-recursief):
    python concat_psd_test.py --dir "D:/cenobamate_eeg_1" --outdir out_test

Knoppen:
    --upper-lim 40        # match je huidige pipeline
    --trim 0.0            # seconds trim aan start van elk epoch
    --padding copy        # {copy, zeros, none}
    --fade 0.25           # cross-fade in seconden tussen blocks (0 = hard concat)
    --all-channels        # default is alleen O1/O2/Oz (occi=True) zoals je pipeline
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne
from mne import Epochs
from numpy.fft import rfft, rfftfreq

# Gebruik jouw bestaande preprocessing
from PhoticDrivingResponse.pdr.core.patients import eeg as load_eeg


def extract_stimulation_df(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Repliceert power._stimulation_power() logica."""
    sfreq = raw.info["sfreq"]
    block_threshold = 1.0 * sfreq

    sample, _event_id = mne.events_from_annotations(raw)
    df = pd.DataFrame(sample, columns=["sample", "previous", "event_id"])
    df["block"] = (np.diff(np.r_[0, df["sample"].to_numpy()]) > block_threshold).cumsum()

    def compute_freq(samples: pd.Series) -> float:
        arr = samples.to_numpy()
        return int(round(1 / np.mean(np.diff(arr) / sfreq), 2)) if len(arr) > 1 else np.nan

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
    return df


def epoch_blocks(df: pd.DataFrame, raw: mne.io.BaseRaw) -> Tuple[mne.Epochs, pd.DataFrame, int]:
    """Repliceert power._epoch_power() logica."""
    sfreq = raw.info["sfreq"]

    df_epochs = df.loc[df.groupby("block base")["sample"].idxmin()].reset_index(drop=True)
    df_epochs["ends"] = df.groupby("block base")["sample"].max().values
    tmax = int(math.ceil(((df_epochs["ends"] - df_epochs["sample"]) / sfreq).min()))

    blocks_baseline = []
    for rep, rep_epochs in df_epochs.groupby("rep"):
        start = rep_epochs["sample"].min() - 0.1 * sfreq - tmax * sfreq
        if start < 0:
            print("-- skipping first baseline due to too little prestimulation data.")
            continue
        block_baseline = {
            "sample": int(start),
            "previous": 0,
            "block base": int((rep - 1) * (len(rep_epochs) + 1) + 1),
            "freq": 0,
            "rep": int(rep),
            "ends": int(start + tmax * sfreq),
        }
        blocks_baseline.append(block_baseline)

    df_epochs = pd.concat([df_epochs, pd.DataFrame(blocks_baseline)], ignore_index=True)
    df_epochs = df_epochs.sort_values(by="sample").reset_index(drop=True)

    np_epochs = df_epochs[["sample", "previous", "freq"]].to_numpy(dtype=int)
    epochs = Epochs(
        raw,
        events=np_epochs,
        event_id=None,
        tmin=0.0,
        tmax=tmax - 1 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )


    return epochs, df_epochs, tmax


def crossfade_concat(segments: List[np.ndarray], fade_samples: int) -> np.ndarray:
    """
    Concatenate 2D segments (n_channels, n_samples) met overlap-add cross-fade.
    fade_samples = 0 => hard concat
    """
    if not segments:
        raise ValueError("No segments provided for concatenation.")
    if fade_samples <= 0:
        return np.concatenate(segments, axis=1)

    out = segments[0].copy()
    for seg in segments[1:]:
        fade = int(min(fade_samples, out.shape[1], seg.shape[1]))
        if fade <= 0:
            out = np.concatenate([out, seg], axis=1)
            continue

        w = np.linspace(0, np.pi, fade, endpoint=True)
        fade_out = 0.5 * (1 + np.cos(w))  # 1 -> 0
        fade_in = 0.5 * (1 - np.cos(w))   # 0 -> 1
        fade_out = fade_out.reshape(1, -1)
        fade_in = fade_in.reshape(1, -1)

        out_tail = out[:, -fade:] * fade_out
        seg_head = seg[:, :fade] * fade_in
        out[:, -fade:] = out_tail + seg_head
        out = np.concatenate([out, seg[:, fade:]], axis=1)

    return out


def welch_psd_1d(
    x: np.ndarray,
    sfreq: float,
    upper_lim: float,
    padding: str = "copy",
    out: str = "log10",   # <-- nieuw: "log10" (zoals SpecParam) of "db"
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD: 1s Hann, 50% overlap.
    - out="log10": geeft log10(V^2/Hz) (matcht SpecParam-plot)
    - out="db": geeft 10*log10(µV^2/Hz) (ook ok, maar andere schaal)
    """
    window_length = int(sfreq)          # 1 second
    step = int(window_length * 0.5)     # 50% overlap

    freqs_full = rfftfreq(window_length, 1 / sfreq).squeeze()
    mask = freqs_full <= upper_lim
    freqs = freqs_full[mask]

    window = np.hanning(window_length)
    win_pow = np.sum(window**2)  # nodig voor density normalisatie

    if padding == "copy":
        x_pad = np.concatenate([x[:window_length], x, x[-window_length:]])
    elif padding == "zeros":
        zeros = np.zeros(window_length)
        x_pad = np.concatenate([zeros, x, zeros])
    elif padding == "none":
        x_pad = x
    else:
        raise ValueError("padding must be one of: copy, zeros, none")

    seg_psd = []
    for start in range(0, len(x_pad) - window_length + 1, step):
        segment = x_pad[start : start + window_length]
        segw = segment * window

        # Density PSD (V^2/Hz) – standaard vorm voor Welch/FOOOF
        X = rfft(segw)
        psd = (np.abs(X) ** 2) / (sfreq * win_pow)

        seg_psd.append(psd[mask])

    mean_psd = np.mean(seg_psd, axis=0)          # V^2/Hz
    mean_psd = np.maximum(mean_psd, 1e-30)

    if out == "log10":
        y = np.log10(mean_psd)                   # log10(V^2/Hz)
    elif out == "db":
        y = 10 * np.log10(mean_psd * 1e12)       # dB(µV^2/Hz)
    else:
        raise ValueError('out must be "log10" or "db"')

    return y, freqs



def compute_avg_psd_db(
    epochs: mne.Epochs,
    df_epochs: pd.DataFrame,
    upper_lim: float,
    trim: float,
    padding: str,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray]:
    """AVG: PSD per rep-epoch, daarna average over reps (freq>0)."""
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.info["ch_names"]
    data = epochs.get_data()

    trim_samp = int(round(trim * sfreq))
    freqs_unique = sorted(df_epochs["freq"].dropna().astype(int).unique())
    freqs_unique = [f for f in freqs_unique if f > 0]

    per_freq_ch: Dict[int, Dict[str, List[np.ndarray]]] = {
        f: {ch: [] for ch in ch_names} for f in freqs_unique
    }
    freqs_axis = None

    for idx, row in df_epochs.iterrows():
        f = int(row["freq"])
        if f <= 0:
            continue
        epoch_all = data[idx]

        for ch_i, ch in enumerate(ch_names):
            x = epoch_all[ch_i]
            if trim_samp > 0:
                x = x[trim_samp:]
            psd_log10, freqs = welch_psd_1d(x, sfreq, upper_lim, padding=padding, out="log10")
            per_freq_ch[f][ch].append(psd_log10)
            if freqs_axis is None:
                freqs_axis = freqs

    if freqs_axis is None:
        raise RuntimeError("No stimulation epochs found (freq > 0).")

    avg_db: Dict[int, Dict[str, np.ndarray]] = {f: {} for f in freqs_unique}
    for f in freqs_unique:
        for ch in ch_names:
            vals = per_freq_ch[f][ch]
            avg_db[f][ch] = np.mean(vals, axis=0) if len(vals) else np.full_like(freqs_axis, np.nan)

    return avg_db, freqs_axis


def compute_concat_psd_db(
    epochs: mne.Epochs,
    df_epochs: pd.DataFrame,
    upper_lim: float,
    trim: float,
    padding: str,
    fade: float,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray]:
    """CONCAT: concat reps per freq (freq>0), daarna één PSD."""
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.info["ch_names"]
    data = epochs.get_data()

    trim_samp = int(round(trim * sfreq))
    fade_samp = int(round(fade * sfreq))

    freqs_unique = sorted(df_epochs["freq"].dropna().astype(int).unique())
    freqs_unique = [f for f in freqs_unique if f > 0]

    freqs_axis = None
    concat_db: Dict[int, Dict[str, np.ndarray]] = {f: {} for f in freqs_unique}

    for f in freqs_unique:
        sel = df_epochs[df_epochs["freq"].astype(int) == f].copy()
        sel = sel.sort_values(by=["rep", "sample"])
        idxs = sel.index.to_list()
        if not idxs:
            continue

        segments = []
        for idx in idxs:
            seg = data[idx]  # (n_ch, n_times)
            if trim_samp > 0:
                seg = seg[:, trim_samp:]
            segments.append(seg)

        xcat = crossfade_concat(segments, fade_samp)

        for ch_i, ch in enumerate(ch_names):
            psd_log10, freqs = welch_psd_1d(xcat[ch_i], sfreq, upper_lim, padding=padding, out="log10")
            concat_db[f][ch] = psd_log10
            if freqs_axis is None:
                freqs_axis = freqs

    if freqs_axis is None:
        raise RuntimeError("No stimulation epochs found (freq > 0).")

    return concat_db, freqs_axis

def crossfade_concat_with_boundaries(segments: List[np.ndarray], fade_samples: int):
    """
    Concatenate 2D segments (n_channels, n_samples) met overlap-add cross-fade.
    Geeft ook boundaries terug: start-index van elk segment in de output.
    boundaries[0] = 0
    boundaries[1] = start van segment 2 in output, etc.
    """
    if not segments:
        raise ValueError("No segments provided.")
    if fade_samples <= 0:
        out = np.concatenate(segments, axis=1)
        boundaries = [0]
        pos = 0
        for seg in segments[:-1]:
            pos += seg.shape[1]
            boundaries.append(pos)
        return out, boundaries

    out = segments[0].copy()
    boundaries = [0]
    pos = segments[0].shape[1]

    for seg in segments[1:]:
        fade = int(min(fade_samples, out.shape[1], seg.shape[1]))
        # Start van nieuw segment (na overlap) ligt op (pos - fade)
        boundaries.append(pos - fade)

        w = np.linspace(0, np.pi, fade, endpoint=True)
        fade_out = 0.5 * (1 + np.cos(w))  # 1 -> 0
        fade_in = 0.5 * (1 - np.cos(w))   # 0 -> 1
        fade_out = fade_out.reshape(1, -1)
        fade_in = fade_in.reshape(1, -1)

        out_tail = out[:, -fade:] * fade_out
        seg_head = seg[:, :fade] * fade_in
        out[:, -fade:] = out_tail + seg_head
        out = np.concatenate([out, seg[:, fade:]], axis=1)

        pos = out.shape[1]

    return out, boundaries


def inspect_concat_timeseries(
    outdir: Path,
    file_stem: str,
    freq_hz: int,
    ch_names: List[str],
    sfreq: float,
    segments: List[np.ndarray],          # lijst van (n_ch, n_samples) vóór concat
    fade_s: float,
    inspect_ch: str = "Oz",
    zoom_s: float = 1.0,
    save_fif: bool = False,
):
    """
    Maakt debug plots + saves voor de geconcateneerde tijdserie.
    - Slaat concat array op als .npy
    - Plot: volledige concat trace (1 kanaal) met join markers
    - Plot: zoom rond joins + ook (end seg_i) vs (start seg_{i+1}) vóór crossfade
    """
    import matplotlib.pyplot as plt
    import mne

    fade_samples = int(round(fade_s * sfreq))
    xcat, boundaries = crossfade_concat_with_boundaries(segments, fade_samples)

    # Kanaal kiezen
    if inspect_ch not in ch_names:
        inspect_ch = ch_names[0]
    ci = ch_names.index(inspect_ch)

    # Save arrays
    npy_path = outdir / f"{file_stem}_{freq_hz:02d}Hz_concat_{inspect_ch}.npy"
    np.save(npy_path, xcat[ci])

    # Optioneel: als MNE Raw opslaan (handig om interactief te inspecteren)
    if save_fif:
        info = mne.create_info([inspect_ch], sfreq=sfreq, ch_types=["eeg"])
        raw_concat = mne.io.RawArray(xcat[ci][None, :], info, verbose="ERROR")
        fif_path = outdir / f"{file_stem}_{freq_hz:02d}Hz_concat_{inspect_ch}.fif"
        raw_concat.save(fif_path, overwrite=True)

    # --- Plot 1: volledige concat + join markers
    t = np.arange(xcat.shape[1]) / sfreq
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(t, xcat[ci], linewidth=1)
    ax.set_title(f"{file_stem} | {freq_hz} Hz | concat ({inspect_ch}) | fade={fade_s}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")  # MNE houdt EEG meestal in Volt intern

    for b in boundaries[1:]:
        ax.axvline(b / sfreq, linestyle="--", linewidth=1, alpha=0.8)
        if fade_samples > 0:
            ax.axvline((b + fade_samples) / sfreq, linestyle=":", linewidth=1, alpha=0.8)

    ax.grid(True, linestyle="dotted", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / f"{file_stem}_{freq_hz:02d}Hz_concat_{inspect_ch}_full.png", dpi=200)
    plt.close(fig)

    # --- Plot 2: zoom rond joins + pre-join end/start (voor/na)
    joins = boundaries[1:]  # startindex nieuwe segmenten
    if len(joins) > 0:
        z = int(round(zoom_s * sfreq))
        nrows = len(joins)
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.5 * nrows), squeeze=False)

        for i, b in enumerate(joins):
            # (A) Zoom op concat rond join
            s0 = max(0, b - z)
            s1 = min(xcat.shape[1], b + z)
            tt = np.arange(s0, s1) / sfreq
            axes[i, 0].plot(tt, xcat[ci, s0:s1], linewidth=1)
            axes[i, 0].axvline(b / sfreq, linestyle="--", linewidth=1)
            if fade_samples > 0:
                axes[i, 0].axvline((b + fade_samples) / sfreq, linestyle=":", linewidth=1)
            axes[i, 0].set_title(f"Concat zoom join {i+1} (start seg {i+2})")
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].set_ylabel("V")
            axes[i, 0].grid(True, linestyle="dotted", alpha=0.6)

            # (B) Voor de concat: einde seg_i en begin seg_{i+1} naast elkaar (zelfde tijdas)
            prev_seg = segments[i][ci]
            next_seg = segments[i+1][ci]
            n = min(len(prev_seg), len(next_seg), z)
            tloc = np.arange(n) / sfreq
            axes[i, 1].plot(tloc, prev_seg[-n:], label="end seg_i", linewidth=1)
            axes[i, 1].plot(tloc, next_seg[:n], label="start seg_{i+1}", linewidth=1)
            axes[i, 1].set_title("Pre-join raw chunks (end vs start)")
            axes[i, 1].set_xlabel("Local time (s)")
            axes[i, 1].set_ylabel("V")
            axes[i, 1].grid(True, linestyle="dotted", alpha=0.6)
            axes[i, 1].legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(outdir / f"{file_stem}_{freq_hz:02d}Hz_concat_{inspect_ch}_joins.png", dpi=200)
        plt.close(fig)

    print(f"[INSPECT] saved: {npy_path.name} (+plots)")


def similarity_metrics(avg_db: np.ndarray, cat_db: np.ndarray) -> Tuple[float, float, float]:
    """(mean_abs_diff_db, max_abs_diff_db, corr_db)."""
    a = np.asarray(avg_db)
    b = np.asarray(cat_db)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    a = a[mask]
    b = b[mask]
    mad = float(np.mean(np.abs(a - b)))
    mx = float(np.max(np.abs(a - b)))
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return mad, mx, corr


def plot_overlay(
    out_png: Path,
    freqs_axis: np.ndarray,
    avg_db: Dict[int, Dict[str, np.ndarray]],
    cat_db: Dict[int, Dict[str, np.ndarray]],
    ch_names: List[str],
    upper_lim: float,
):
    import matplotlib.pyplot as plt

    freqs_list = sorted(avg_db.keys())
    n = len(freqs_list)
    fig, axes = plt.subplots(1, n, figsize=(max(12, 4 * n), 4), squeeze=False)
    axes = axes[0]

    for ax, f in zip(axes, freqs_list):
        avg_ch = np.array([avg_db[f][ch] for ch in ch_names])
        cat_ch = np.array([cat_db[f][ch] for ch in ch_names])

        avg_mean = 10*np.log10(np.nanmean(10**(avg_ch/10), axis=0))
        cat_mean = 10*np.log10(np.nanmean(10**(cat_ch/10), axis=0))


        ax.plot(freqs_axis, avg_mean, linewidth=2, label="AVG mean")
        ax.plot(freqs_axis, cat_mean, linewidth=2, linestyle="--", label="CONCAT mean")

        ax.set_title(f"{f} Hz")
        ax.set_xlim(0, upper_lim)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log10(Power) (V²/Hz)")
        ax.grid(True, linestyle="dotted", alpha=0.6)

        for h in range(1, int(math.floor(upper_lim / f)) + 1):
            ax.axvline(f * h, color="gray", linestyle="dotted", linewidth=1, alpha=0.7)

        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_one_file(
    cnt_path: Path,
    outdir: Path,
    passband: Tuple[float, float],
    notch: float,
    occi: bool,
    upper_lim: float,
    trim: float,
    padding: str,
    fade: float,
    args,
):
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_eeg(cnt_path, list(passband), notch=notch, occi=occi, plot=False)
    df = extract_stimulation_df(raw)
    epochs, df_epochs, tmax = epoch_blocks(df, raw)
    # -----------------------------
    # Inspect concatenated raw data
    # -----------------------------
    if args.inspect:
        freqs_unique = sorted(df_epochs["freq"].dropna().astype(int).unique())
        freqs_unique = [f for f in freqs_unique if f > 0]
        if not freqs_unique:
            raise RuntimeError("No stimulation freqs found to inspect.")

        f_inspect = args.inspect_freq if args.inspect_freq is not None else freqs_unique[0]

        sel = df_epochs[df_epochs["freq"].astype(int) == int(f_inspect)].copy()
        sel = sel.sort_values(by=["rep", "sample"])
        idxs = sel.index.to_list()

        data = epochs.get_data()  # (n_epochs, n_ch, n_times)
        sfreq = epochs.info["sfreq"]
        trim_samp = int(round(args.trim * sfreq))

        segments = []
        for idx in idxs:
            seg = data[idx]
            if trim_samp > 0:
                seg = seg[:, trim_samp:]
            segments.append(seg)

        inspect_concat_timeseries(
            outdir=outdir,
            file_stem=cnt_path.stem,
            freq_hz=int(f_inspect),
            ch_names=epochs.info["ch_names"],
            sfreq=sfreq,
            segments=segments,
            fade_s=args.fade,
            inspect_ch=args.inspect_ch,
            zoom_s=args.inspect_zoom,
            save_fif=args.inspect_save_fif,
        )

    avg_db, faxis = compute_avg_psd_db(epochs, df_epochs, upper_lim, trim, padding)
    cat_db, _ = compute_concat_psd_db(epochs, df_epochs, upper_lim, trim, padding, fade)

    ch_names = epochs.info["ch_names"]

    rows = []
    for f in sorted(avg_db.keys()):
        avg_mean = np.nanmean(np.array([avg_db[f][ch] for ch in ch_names]), axis=0)
        cat_mean = np.nanmean(np.array([cat_db[f][ch] for ch in ch_names]), axis=0)
        mad, mx, corr = similarity_metrics(avg_mean, cat_mean)
        rows.append(
            dict(
                file=cnt_path.name,
                freq_hz=f,
                tmax_s=tmax,
                trim_s=trim,
                padding=padding,
                fade_s=fade,
                mean_abs_diff_db=mad,
                max_abs_diff_db=mx,
                corr_db=corr,
            )
        )

    dfm = pd.DataFrame(rows).sort_values("freq_hz")
    dfm.to_csv(outdir / f"{cnt_path.stem}_concat_vs_avg_metrics.csv", index=False)

    plot_overlay(
        out_png=outdir / f"{cnt_path.stem}_concat_vs_avg_psd.png",
        freqs_axis=faxis,
        avg_db=avg_db,
        cat_db=cat_db,
        ch_names=ch_names,
        upper_lim=upper_lim,
    )

    print(f"[OK] {cnt_path.name} -> {outdir.resolve()}")
    print(dfm[["freq_hz", "mean_abs_diff_db", "max_abs_diff_db", "corr_db"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Compare AVG-of-block-PSDs vs CONCAT-block-PSD (same Welch settings).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cnt", type=str, help="Path to a single .cnt file.")
    src.add_argument("--dir", type=str, help="Directory containing .cnt files (non-recursive).")

    ap.add_argument("--outdir", type=str, default="concat_psd_test_outputs", help="Output directory.")
    ap.add_argument("--passband", type=float, nargs=2, default=(0.5, 100.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--notch", type=float, default=50.0, help="Notch filter frequency (e.g., 50).")
    ap.add_argument("--all-channels", action="store_true", help="Use all EEG channels instead of only O1/O2/Oz.")
    ap.add_argument("--upper-lim", type=float, default=40.0, help="Upper frequency limit for PSD axis (Hz).")
    ap.add_argument("--trim", type=float, default=0.0, help="Trim (seconds) from start of each epoch.")
    ap.add_argument("--padding", type=str, default="copy", choices=["copy", "zeros", "none"])
    ap.add_argument("--fade", type=float, default=0.25, help="Cross-fade seconds between concatenated blocks (0=hard concat).")
    ap.add_argument("--inspect", action="store_true", help="Save/debug plot the concatenated raw timeseries.")
    ap.add_argument("--inspect-freq", type=int, default=None, help="Which stimulation frequency to inspect (e.g. 6). Default=first available.")
    ap.add_argument("--inspect-ch", type=str, default="Oz", help="Channel to inspect (default Oz).")
    ap.add_argument("--inspect-zoom", type=float, default=1.0, help="Seconds to zoom around joins.")
    ap.add_argument("--inspect-save-fif", action="store_true", help="Also save a .fif Raw for MNE viewing.")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    occi = not args.all_channels

    if args.cnt:
        run_one_file(
            cnt_path=Path(args.cnt),
            outdir=outdir,
            passband=tuple(args.passband),
            notch=args.notch,
            occi=occi,
            upper_lim=args.upper_lim,
            trim=args.trim,
            padding=args.padding,
            fade=args.fade,
            args=args,
        )
        return

    d = Path(args.dir)
    files = sorted([p for p in d.iterdir() if p.suffix.lower() == ".cnt"])
    if not files:
        raise FileNotFoundError(f"No .cnt files found in {d.resolve()}")

    for p in files:
        run_one_file(
            cnt_path=p,
            outdir=outdir,
            passband=tuple(args.passband),
            notch=args.notch,
            occi=occi,
            upper_lim=args.upper_lim,
            trim=args.trim,
            padding=args.padding,
            fade=args.fade,
            args=args,
        )


if __name__ == "__main__":
    main()
