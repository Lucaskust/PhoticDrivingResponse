import argparse
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from specparam import SpectralModel

def pick_occipital(raw):
    # werkt ook als kanalen suffix/prefix hebben (bijv. "O1-REF")
    keys = ["O1", "O2", "OZ", "PO3", "PO4", "POZ", "IZ"]
    occ = [ch for ch in raw.ch_names if any(k in ch.upper() for k in keys)]
    return occ


def fit_one(cnt_path, out_dir="specparam_out", fmin=2, fmax=45, aperiodic_mode="fixed"):
    os.makedirs(out_dir, exist_ok=True)

    raw = mne.io.read_raw_ant(cnt_path, preload=True, verbose="error")
    raw.pick(picks="eeg")
    sfreq = raw.info["sfreq"]
    dur = raw.n_times / sfreq
    print(f"Duration: {dur:.2f} s | sfreq: {sfreq:.1f} Hz")


    # --- 1) Kies occipitaal-kanalen ---
    occ_chs = pick_occipital(raw)
    if len(occ_chs) == 0:
        print("WARNING: No occipital channels found -> using all EEG channels")
        occ_picks = None
    else:
        print(f"Using occipital channels ({len(occ_chs)}): {occ_chs}")
        occ_picks = occ_chs

    # --- 2) PSD (lineair) - Welch ---
    sfreq = raw.info["sfreq"]
    psd = raw.compute_psd(
        method="welch",
        fmin=fmin, fmax=fmax,
        n_per_seg=int(sfreq * 1),   # 2s segment
        n_overlap=int(sfreq * 0.5),   # 50% overlap
        n_fft=int(sfreq * 1),
        verbose="error"
    )
    freqs = psd.freqs

    # HIER pas occipitaal selectie toe
    p = psd.get_data(picks=occ_picks) if occ_picks is not None else psd.get_data()

    # Gemiddelde spectrum over gekozen kanalen
    # TEMP: pak alleen 1 kanaal (eerste occipitaal kanaal) om te debuggen
    mean_lin = np.mean(p, axis=0)
    mean_lin = np.maximum(mean_lin, 1e-30)  # avoid zeros

    # --- 3) Specparam/FOOOF fit ---
    fm = SpectralModel(
        aperiodic_mode=aperiodic_mode,
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0
    )
    fm.fit(freqs, mean_lin, freq_range=(fmin, fmax))

    # --- 4) Save plot ---
    base = os.path.basename(cnt_path).replace(".cnt", "")
    fig, ax = plt.subplots(figsize=(8, 4))
    fm.plot(ax=ax, plot_peaks="shade")
    fig.savefig(os.path.join(out_dir, f"{base}_specparam.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 5) Save parameters (robust across specparam versions) ---
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

    row = {
        "file": base,
        "aperiodic_mode": aperiodic_mode,
        "r2": r2,
        "error": err,
        "offset": offset,
        "knee": knee,
        "exponent": exponent,
        "n_channels_used": (len(occ_chs) if occ_picks is not None else len(raw.ch_names)),
        "channels_used": ",".join(occ_chs) if occ_picks is not None else "ALL_EEG",
    }

    # Peak params
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

    df_peaks.insert(0, "file", base)

    # Write outputs
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, f"{base}_aperiodic.csv"), index=False)
    df_peaks.to_csv(os.path.join(out_dir, f"{base}_peaks.csv"), index=False)

    # Save PSD arrays
    np.save(os.path.join(out_dir, f"{base}_freqs.npy"), freqs)
    np.save(os.path.join(out_dir, f"{base}_mean_lin_psd.npy"), mean_lin)

    print(f"[OK] {base}  r2={r2}  error={err}  peaks={len(df_peaks)}")
    print(f"Saved to: {out_dir}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True)
    ap.add_argument("--out", default="specparam_out")
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--mode", type=str, default="fixed", choices=["fixed", "knee"])
    args = ap.parse_args()
    fit_one(args.cnt, args.out, args.fmin, args.fmax, args.mode)

if __name__ == "__main__":
    main()
