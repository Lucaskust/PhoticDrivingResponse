import argparse
import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from specparam import SpectralModel
from PhoticDrivingResponse.unused.LK_config import resolve_specparam_config, DEFAULT_PRESET, SPEC_PRESETS


# -------------------------
# Channel selection helpers
# -------------------------
def pick_occipital(raw: mne.io.BaseRaw):
    """Robuust occipitaal (werkt ook bij suffix/prefix zoals 'O1-REF')."""
    keys = ["O1", "O2", "OZ", "PO3", "PO4", "POZ", "IZ"]
    occ = [ch for ch in raw.ch_names if any(k in ch.upper() for k in keys)]
    return occ


# -------------------------
# Block detection from annotations
# -------------------------
def extract_stimulation_blocks(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """
    Maakt een dataframe met per block:
    - start_sample
    - end_sample
    - freq_hz (geschat uit inter-event interval)
    - rep index

    Zelfde principe als je power/concat scripts:
    - annotations/events representeren de flitsen
    - blocks zijn gescheiden door gaps > 1s
    """
    sfreq = raw.info["sfreq"]
    block_threshold = int(round(1.0 * sfreq))  # 1 second gap => new block

    events, _ = mne.events_from_annotations(raw)
    if events is None or len(events) < 2:
        raise RuntimeError("Not enough annotation events found to build stimulation blocks.")

    df = pd.DataFrame(events, columns=["sample", "previous", "event_id"])
    df["gap"] = df["sample"].diff().fillna(0)
    df["block"] = (df["gap"] > block_threshold).cumsum()

    def compute_freq(samples: pd.Series) -> float:
        arr = samples.to_numpy()
        if len(arr) < 2:
            return np.nan
        isi = np.diff(arr) / sfreq
        f = 1.0 / np.mean(isi)
        return float(np.round(f, 2))

    df["freq"] = df.groupby("block")["sample"].transform(compute_freq)

    # Rep counter: increases when a frequency repeats again
    rep = 1
    rep_freqs = set()
    df["rep"] = 0
    for block_id, freq in df.groupby("block")["freq"].first().items():
        if np.isnan(freq):
            continue
        if freq in rep_freqs:
            rep += 1
            rep_freqs = {freq}
        else:
            rep_freqs.add(freq)
        df.loc[df["block"] == block_id, "rep"] = rep

    # Summarize to per-block table
    blocks = []
    for block_id, g in df.groupby("block"):
        freq = float(g["freq"].iloc[0])
        if np.isnan(freq) or freq <= 0:
            continue
        start = int(g["sample"].min())
        end = int(g["sample"].max())
        blocks.append(
            dict(
                block=int(block_id),
                rep=int(g["rep"].iloc[0]),
                freq_hz=int(round(freq)),
                start_sample=start,
                end_sample=end,
                n_samples=int(end - start),
            )
        )

    if not blocks:
        raise RuntimeError("No stimulation blocks (freq>0) detected from annotations.")

    df_blocks = pd.DataFrame(blocks).sort_values(["rep", "freq_hz", "start_sample"]).reset_index(drop=True)
    return df_blocks


# -------------------------
# PSD + SpecParam
# -------------------------
def welch_psd_block(
    data: np.ndarray,          # shape (n_channels, n_times)
    sfreq: float,
    fmin: float,
    fmax: float,
    seg_s: float,
    overlap: float,
):
    """
    Returns:
      freqs (n_freqs,), psd_mean_lin (n_freqs,)
    Uses MNE psd_array_welch (linear PSD density: V^2/Hz).
    """
    n_per_seg = int(round(seg_s * sfreq))
    n_per_seg = max(8, n_per_seg)
    n_overlap = int(round(overlap * n_per_seg))
    n_fft = n_per_seg

    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        n_fft=n_fft,
        average="mean",
        verbose="error",
    )
    # psds shape: (n_channels, n_freqs)
    psd_mean = np.mean(psds, axis=0)
    psd_mean = np.maximum(psd_mean, 1e-30)
    return freqs, psd_mean


def fit_specparam(freqs, psd_lin, settings, freq_range):
    fm = SpectralModel(**settings)
    fm.fit(freqs, psd_lin, freq_range=freq_range)
    return fm


def extract_fit_outputs(fm: SpectralModel):
    # robust across versions
    r2 = getattr(fm, "r_squared_", np.nan)
    err = getattr(fm, "error_", np.nan)

    res = getattr(fm, "results_", None)
    if res is None:
        res = getattr(fm, "results", None)
    if res is not None:
        r2 = getattr(res, "r_squared_", r2)
        r2 = getattr(res, "r_squared", r2)
        err = getattr(res, "error_", err)
        err = getattr(res, "error", err)

    # Aperiodic params
    ap = None
    for key in ["aperiodic", "aperiodic_params"]:
        try:
            ap = fm.get_params(key)
            break
        except Exception:
            pass

    offset = np.nan
    knee = np.nan
    exponent = np.nan
    if ap is not None and len(ap) >= 2:
        offset = float(ap[0])
        exponent = float(ap[-1])
        if len(ap) == 3:
            knee = float(ap[1])

    # Peaks
    peaks = None
    for key in ["peak", "peaks", "peak_params"]:
        try:
            peaks = fm.get_params(key)
            break
        except Exception:
            pass

    if peaks is None:
        peaks = np.empty((0, 3))
    peaks = np.asarray(peaks)
    if peaks.ndim == 1 and peaks.size == 0:
        peaks = np.empty((0, 3))
    if peaks.ndim == 1 and peaks.size == 3:
        peaks = peaks.reshape(1, 3)

    df_peaks = pd.DataFrame(peaks, columns=["center_freq", "peak_power", "bandwidth"])
    df_peaks = df_peaks.sort_values("peak_power", ascending=False).reset_index(drop=True)

    return r2, err, offset, knee, exponent, df_peaks


def run_specparam_blocks(
    cnt_path: Path,
    out_dir: Path,
    preset: str,
    overrides: list[str] | None,
    fmin: float | None,
    fmax: float | None,
    seg_s: float,
    overlap: float,
    trim_s: float,
    use_occipital: bool,
    by: str,                 # "freq" or "block"
    stim_hz: int | None,     # optional filter
):
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_ant(str(cnt_path), preload=True, verbose="error")
    raw.pick(picks="eeg")

    sfreq = raw.info["sfreq"]
    base = cnt_path.stem
    dur = raw.n_times / sfreq
    print(f"File: {base} | Duration: {dur:.1f}s | sfreq: {sfreq:.1f}Hz")

    # settings + freq_range from preset
    settings, freq_range = resolve_specparam_config(
        preset=preset,
        fmin=fmin,
        fmax=fmax,
        overrides=overrides,
    )
    fmin_eff, fmax_eff = float(freq_range[0]), float(freq_range[1])
    print(f"Preset: {preset} | freq_range={freq_range} | settings={settings}")

    # channel picks
    occ = pick_occipital(raw) if use_occipital else []
    if use_occipital and len(occ) > 0:
        picks = occ
        channels_used = ",".join(occ)
        print(f"Using occipital channels ({len(occ)}): {occ}")
    else:
        picks = None
        channels_used = "ALL_EEG"
        if use_occipital:
            print("WARNING: No occipital channels found -> using all EEG channels")

    df_blocks = extract_stimulation_blocks(raw)

    if stim_hz is not None:
        df_blocks = df_blocks[df_blocks["freq_hz"] == int(stim_hz)].copy()
        if df_blocks.empty:
            raise RuntimeError(f"No blocks found for stim_hz={stim_hz}")

    # choose fixed length (like your power pipeline): use minimum block length across selected blocks
    trim_samp = int(round(trim_s * sfreq))
    min_len = int(df_blocks["n_samples"].min())
    fixed_len = min_len - trim_samp
    fixed_len = max(1, fixed_len)
    fixed_len_s = fixed_len / sfreq
    print(f"Using fixed block length: {fixed_len_s:.3f}s (min_len={min_len/sfreq:.3f}s, trim={trim_s}s)")

    # collect PSDs per block
    rows_aperiodic = []
    all_peaks_rows = []

    # For by="freq": accumulate psds per freq before fitting
    psd_bank = {}  # freq_hz -> list of psd vectors
    meta_bank = {} # freq_hz -> dict for saving info

    for i, row in df_blocks.iterrows():
        freq_hz = int(row["freq_hz"])
        rep = int(row["rep"])
        block_id = int(row["block"])
        start = int(row["start_sample"]) + trim_samp
        stop = start + fixed_len

        # safety: ensure within block end
        block_end = int(row["end_sample"])
        if stop > block_end:
            # skip blocks that are too short after trimming
            continue

        data = raw.get_data(picks=picks, start=start, stop=stop)  # (n_ch, n_times)
        if data.ndim != 2 or data.shape[1] < 16:
            continue

        freqs, psd_lin = welch_psd_block(
            data=data,
            sfreq=sfreq,
            fmin=fmin_eff,
            fmax=fmax_eff,
            seg_s=seg_s,
            overlap=overlap,
        )

        if by == "block":
            # fit per block
            fm = fit_specparam(freqs, psd_lin, settings=settings, freq_range=(fmin_eff, fmax_eff))
            r2, err, offset, knee, exponent, df_peaks = extract_fit_outputs(fm)

            tag = f"{base}_{freq_hz:02d}Hz_rep{rep}_block{block_id}_{preset}"
            # plot
            fig, ax = plt.subplots(figsize=(8, 4))
            fm.plot(ax=ax, plot_peaks="shade")
            fig.savefig(out_dir / f"{tag}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            # save arrays
            np.save(out_dir / f"{tag}_freqs.npy", freqs)
            np.save(out_dir / f"{tag}_psd_lin.npy", psd_lin)

            rows_aperiodic.append(
                dict(
                    file=base,
                    preset=preset,
                    freq_hz=freq_hz,
                    rep=rep,
                    block=block_id,
                    r2=r2,
                    error=err,
                    offset=offset,
                    knee=knee,
                    exponent=exponent,
                    n_channels_used=(len(picks) if isinstance(picks, list) else len(raw.ch_names)),
                    channels_used=channels_used,
                    welch_seg_s=seg_s,
                    welch_overlap=overlap,
                    trim_s=trim_s,
                    fixed_len_s=fixed_len_s,
                )
            )

            if df_peaks is not None and len(df_peaks) > 0:
                df_peaks = df_peaks.copy()
                df_peaks.insert(0, "file", base)
                df_peaks.insert(1, "preset", preset)
                df_peaks.insert(2, "freq_hz", freq_hz)
                df_peaks.insert(3, "rep", rep)
                df_peaks.insert(4, "block", block_id)
                all_peaks_rows.append(df_peaks)

        else:
            # by == "freq": accumulate for later mean
            psd_bank.setdefault(freq_hz, []).append(psd_lin)
            meta_bank.setdefault(freq_hz, dict(reps=set(), blocks=set()))
            meta_bank[freq_hz]["reps"].add(rep)
            meta_bank[freq_hz]["blocks"].add(block_id)

    if by == "freq":
        for freq_hz, psds in sorted(psd_bank.items()):
            psds = np.asarray(psds)
            psd_mean = np.mean(psds, axis=0)

            fm = fit_specparam(freqs, psd_mean, settings=settings, freq_range=(fmin_eff, fmax_eff))
            r2, err, offset, knee, exponent, df_peaks = extract_fit_outputs(fm)

            tag = f"{base}_{freq_hz:02d}Hz_{preset}"
            fig, ax = plt.subplots(figsize=(8, 4))
            fm.plot(ax=ax, plot_peaks="shade")
            fig.savefig(out_dir / f"{tag}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            np.save(out_dir / f"{tag}_freqs.npy", freqs)
            np.save(out_dir / f"{tag}_psd_lin.npy", psd_mean)

            rows_aperiodic.append(
                dict(
                    file=base,
                    preset=preset,
                    freq_hz=freq_hz,
                    rep="ALL",
                    block="ALL",
                    r2=r2,
                    error=err,
                    offset=offset,
                    knee=knee,
                    exponent=exponent,
                    n_channels_used=(len(picks) if isinstance(picks, list) else len(raw.ch_names)),
                    channels_used=channels_used,
                    welch_seg_s=seg_s,
                    welch_overlap=overlap,
                    trim_s=trim_s,
                    fixed_len_s=fixed_len_s,
                    n_blocks=len(meta_bank[freq_hz]["blocks"]),
                    n_reps=len(meta_bank[freq_hz]["reps"]),
                )
            )

            if df_peaks is not None and len(df_peaks) > 0:
                df_peaks = df_peaks.copy()
                df_peaks.insert(0, "file", base)
                df_peaks.insert(1, "preset", preset)
                df_peaks.insert(2, "freq_hz", freq_hz)
                df_peaks.insert(3, "rep", "ALL")
                df_peaks.insert(4, "block", "ALL")
                all_peaks_rows.append(df_peaks)

    # write summary outputs
    df_ap = pd.DataFrame(rows_aperiodic)
    df_ap.to_csv(out_dir / f"{base}_aperiodic_{preset}_{by}.csv", index=False)

    if all_peaks_rows:
        df_peaks_all = pd.concat(all_peaks_rows, ignore_index=True)
    else:
        df_peaks_all = pd.DataFrame(columns=["file", "preset", "freq_hz", "rep", "block", "center_freq", "peak_power", "bandwidth"])
    df_peaks_all.to_csv(out_dir / f"{base}_peaks_{preset}_{by}.csv", index=False)

    print(f"[OK] Saved: {df_ap.shape[0]} aperiodic rows, {df_peaks_all.shape[0]} peaks rows -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True, help="Path to .cnt file")
    ap.add_argument("--out", default="specparam_out", help="Output directory")

    ap.add_argument("--preset", type=str, default=DEFAULT_PRESET, choices=list(SPEC_PRESETS.keys()))
    ap.add_argument("--override", action="append", default=None,
                    help="Override setting: key=value (repeatable). Example: --override peak_threshold=2.5")

    ap.add_argument("--fmin", type=float, default=None, help="Override preset fmin")
    ap.add_argument("--fmax", type=float, default=None, help="Override preset fmax")

    ap.add_argument("--all-channels", action="store_true",
                    help="Use all EEG channels (default: occipital selection).")

    ap.add_argument("--seg-s", type=float, default=1.0, help="Welch segment length (seconds)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Welch overlap fraction (0-1)")
    ap.add_argument("--trim", type=float, default=0.0, help="Trim seconds from start of each block")
    ap.add_argument("--by", type=str, default="freq", choices=["freq", "block"],
                    help="Fit level: per freq (average reps) or per block (each rep separately)")
    ap.add_argument("--stim-hz", type=int, default=None, help="Only fit this stim frequency (e.g., 6)")

    args = ap.parse_args()

    run_specparam_blocks(
        cnt_path=Path(args.cnt),
        out_dir=Path(args.out),
        preset=args.preset,
        overrides=args.override,
        fmin=args.fmin,
        fmax=args.fmax,
        seg_s=args.seg_s,
        overlap=args.overlap,
        trim_s=args.trim,
        use_occipital=(not args.all_channels),
        by=args.by,
        stim_hz=args.stim_hz,
    )


if __name__ == "__main__":
    main()
