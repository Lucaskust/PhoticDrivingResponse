import os
import argparse
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from specparam import SpectralModel

TARGETS = np.array([6.0, 10.0, 20.0])
TOL_HZ = 1.0
MIN_PULSES_PER_BLOCK = 20

def pick_occipital(raw):
    keys = ["O1", "O2", "OZ", "PO3", "PO4", "POZ", "IZ"]
    occ = [ch for ch in raw.ch_names if any(k in ch.upper() for k in keys)]
    return occ

def infer_blocks_from_triggers(raw):
    onsets = np.sort(np.array(raw.annotations.onset, dtype=float))
    dt = np.diff(onsets)
    freq = 1.0 / dt

    nearest_idx = np.argmin(np.abs(freq[:, None] - TARGETS[None, :]), axis=1)
    nearest_hz = TARGETS[nearest_idx]
    ok = np.abs(freq - nearest_hz) <= TOL_HZ

    blocks = []
    i = 0
    while i < len(dt):
        if not ok[i]:
            i += 1
            continue
        hz = float(nearest_hz[i])
        start_i = i
        while i < len(dt) and ok[i] and nearest_hz[i] == hz:
            i += 1
        end_i = i

        start_pulse = start_i
        end_pulse = end_i
        n_pulses = (end_pulse - start_pulse) + 1

        if n_pulses >= MIN_PULSES_PER_BLOCK:
            t_start = float(onsets[start_pulse])
            t_end = float(onsets[end_pulse])
            blocks.append((hz, t_start, t_end, n_pulses))
    return blocks

def compute_psd_for_segment(raw, t_start, t_end, fmin, fmax, occ_picks):
    seg = raw.copy().crop(tmin=t_start, tmax=t_end)
    sfreq = seg.info["sfreq"]

    # Welch settings: 1s windows -> stabiel voor ~5s blocks
    psd = seg.compute_psd(
        method="welch",
        fmin=fmin, fmax=fmax,
        n_per_seg=int(sfreq * 1),
        n_overlap=int(sfreq * 0.5),
        n_fft=int(sfreq * 1),
        verbose="error"
    )
    freqs = psd.freqs
    p = psd.get_data(picks=occ_picks) if occ_picks is not None else psd.get_data()
    mean_lin = np.mean(p, axis=0)
    mean_lin = np.maximum(mean_lin, 1e-30)
    return freqs, mean_lin

def bin_power_at(freqs, psd_lin, target_hz):
    """Return (closest_freq, log10(power)) at nearest frequency bin."""
    freqs = np.asarray(freqs, dtype=float)
    psd_lin = np.asarray(psd_lin, dtype=float)
    idx = int(np.argmin(np.abs(freqs - float(target_hz))))
    cf = float(freqs[idx])
    lp = float(np.log10(np.maximum(psd_lin[idx], 1e-30)))
    return cf, lp

def fit_specparam(freqs, mean_lin, fmin, fmax, aperiodic_mode):
    fm = SpectralModel(
        aperiodic_mode=aperiodic_mode,
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0
    )
    fm.fit(freqs, mean_lin, freq_range=(fmin, fmax))
    return fm

def main(cnt_path, out_dir="specparam_out_blocks", fmin=2, fmax=45, aperiodic_mode="fixed"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(cnt_path).replace(".cnt", "")

    raw = mne.io.read_raw_ant(cnt_path, preload=True, verbose="error")
    raw.pick(picks="eeg")

    occ = pick_occipital(raw)
    occ_picks = occ if len(occ) else None
    print(f"Occipital picks: {occ if occ_picks else 'ALL_EEG'}")

    blocks = infer_blocks_from_triggers(raw)
    print(f"Found {len(blocks)} blocks")

    # PSD per block opslaan per frequentie
    by_hz = {6.0: [], 10.0: [], 20.0: []}
    freqs_ref = None

    for (hz, t0, t1, n) in blocks:
        freqs, psd_lin = compute_psd_for_segment(raw, t0, t1, fmin, fmax, occ_picks)
        if freqs_ref is None:
            freqs_ref = freqs
        by_hz[hz].append(psd_lin)

    rows = []
    for hz in [6.0, 10.0, 20.0]:
        if len(by_hz[hz]) == 0:
            continue

        # Average PSD across repeats
        psd_avg = np.mean(np.vstack(by_hz[hz]), axis=0)
        stim_bin_cf, stim_bin_logp = bin_power_at(freqs_ref, psd_avg, hz)
        harm2_bin_cf, harm2_bin_logp = bin_power_at(freqs_ref, psd_avg, 2*hz)
        harm3_bin_cf, harm3_bin_logp = bin_power_at(freqs_ref, psd_avg, 3*hz)


        fm = fit_specparam(freqs_ref, psd_avg, fmin, fmax, aperiodic_mode)

        # Plot per freq
        fig, ax = plt.subplots(figsize=(8, 4))
        fm.plot(ax=ax, plot_peaks="shade")
        ax.set_title(f"{base} | {hz:.0f} Hz blocks (avg PSD)")
        fig.savefig(os.path.join(out_dir, f"{base}_{int(hz)}Hz_specparam.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Peaks + aperiodic
        r2 = getattr(fm, "r_squared_", np.nan)
        err = getattr(fm, "error_", np.nan)

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
        if peaks.ndim == 1 and peaks.size == 3:
            peaks = peaks.reshape(1, 3)
        df_peaks = pd.DataFrame(peaks, columns=["center_freq", "peak_power", "bandwidth"])
        df_peaks = df_peaks.sort_values("peak_power", ascending=False).reset_index(drop=True)
        def nearest_peak(df, target, tol=1.0):
            if df is None or len(df) == 0:
                return np.nan, np.nan, np.nan
            d = np.abs(df["center_freq"].values - float(target))
            i = int(np.argmin(d))
            if d[i] <= tol:
                return float(df.loc[i, "center_freq"]), float(df.loc[i, "peak_power"]), float(df.loc[i, "bandwidth"])
            return np.nan, np.nan, np.nan

        stim_cf, stim_pw, stim_bw = nearest_peak(df_peaks, hz, tol=1.0)
        harm2_cf, harm2_pw, harm2_bw = nearest_peak(df_peaks, 2*hz, tol=1.0)
        harm3_cf, harm3_pw, harm3_bw = nearest_peak(df_peaks, 3*hz, tol=1.0)

        row = {
            "file": base,
            "stim_hz": hz,
            "n_blocks": len(by_hz[hz]),
            "aperiodic_mode": aperiodic_mode,
            "r2": r2,
            "error": err,
            "offset": offset,
            "knee": knee,
            "exponent": exponent,
            "stim_cf": stim_cf, "stim_pw": stim_pw, "stim_bw": stim_bw,
            "harm2_cf": harm2_cf, "harm2_pw": harm2_pw, "harm2_bw": harm2_bw,
            "harm3_cf": harm3_cf, "harm3_pw": harm3_pw, "harm3_bw": harm3_bw,
            "stim_bin_cf": stim_bin_cf, "stim_bin_logp": stim_bin_logp,
            "harm2_bin_cf": harm2_bin_cf, "harm2_bin_logp": harm2_bin_logp,
            "harm3_bin_cf": harm3_bin_cf, "harm3_bin_logp": harm3_bin_logp,
        }

        # Top-3 peaks as columns
        for i in range(3):
            if i < len(df_peaks):
                row[f"peak{i+1}_cf"] = float(df_peaks.loc[i, "center_freq"])
                row[f"peak{i+1}_pw"] = float(df_peaks.loc[i, "peak_power"])
                row[f"peak{i+1}_bw"] = float(df_peaks.loc[i, "bandwidth"])
            else:
                row[f"peak{i+1}_cf"] = np.nan
                row[f"peak{i+1}_pw"] = np.nan
                row[f"peak{i+1}_bw"] = np.nan

        # Save peaks csv per freq (optioneel)
        df_peaks.insert(0, "file", base)
        df_peaks.insert(1, "stim_hz", hz)
        df_peaks.to_csv(os.path.join(out_dir, f"{base}_{int(hz)}Hz_peaks.csv"), index=False)

        rows.append(row)

        # Save averaged PSD arrays (handig voor debug)
        np.save(os.path.join(out_dir, f"{base}_{int(hz)}Hz_freqs.npy"), freqs_ref)
        np.save(os.path.join(out_dir, f"{base}_{int(hz)}Hz_psd_avg.npy"), psd_avg)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{base}_specparam_blocks_summary.csv"), index=False)
    print(f"Saved summary: {base}_specparam_blocks_summary.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True)
    ap.add_argument("--out", default="specparam_out_blocks")
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--mode", type=str, default="fixed", choices=["fixed", "knee"])
    args = ap.parse_args()
    main(args.cnt, args.out, args.fmin, args.fmax, args.mode)
