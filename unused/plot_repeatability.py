import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGETS = [6, 10, 20]


def ensure_patient_timepoint(df: pd.DataFrame) -> pd.DataFrame:
    """Add patient + timepoint if missing, based on file name like VEP02_1."""
    if "patient" in df.columns and "timepoint" in df.columns:
        return df

    patients, tps = [], []
    for f in df["file"].astype(str).values:
        m = re.match(r"^(VEP\d+)[_-](\d+)$", f)
        if m:
            patients.append(m.group(1))
            tps.append(int(m.group(2)))
        else:
            patients.append(f)
            tps.append(np.nan)

    df = df.copy()
    df.insert(0, "patient", patients)
    df.insert(1, "timepoint", tps)
    return df


def load_repeats(psd_dir: str, base: str, stim_hz: int):
    f_rep = os.path.join(psd_dir, f"{base}_{stim_hz}Hz_psd_repeats.npy")
    f_frq = os.path.join(psd_dir, f"{base}_{stim_hz}Hz_freqs.npy")

    if os.path.exists(f_rep) and os.path.exists(f_frq):
        freqs = np.load(f_frq)
        reps = np.load(f_rep)  # shape (n_repeats, n_freqs)
        return freqs, reps

    # fallback glob
    cand_rep = glob.glob(os.path.join(psd_dir, f"{base}_{stim_hz}Hz_*psd_repeats.npy"))
    cand_frq = glob.glob(os.path.join(psd_dir, f"{base}_{stim_hz}Hz_*freqs.npy"))
    if cand_rep and cand_frq:
        freqs = np.load(cand_frq[0])
        reps = np.load(cand_rep[0])
        return freqs, reps

    return None, None


def bin_logp_at(freqs, psd_lin_1d, target_hz):
    """Return log10 power at nearest bin to target_hz. If target outside range -> NaN."""
    freqs = np.asarray(freqs, float)
    psd_lin_1d = np.asarray(psd_lin_1d, float)

    if target_hz < np.min(freqs) or target_hz > np.max(freqs):
        return np.nan

    idx = int(np.argmin(np.abs(freqs - float(target_hz))))
    return float(np.log10(np.maximum(psd_lin_1d[idx], 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to all_specparam_blocks_summary.csv")
    ap.add_argument("--psd_dir", required=True, help="Folder containing *_psd_repeats.npy and *_freqs.npy")
    ap.add_argument("--out", default="t0_repeatability", help="Output folder")
    ap.add_argument("--timepoint", type=int, default=1, help="1=T0, 2=T1, 3=T2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = ensure_patient_timepoint(df)
    df_tp = df[df["timepoint"] == args.timepoint].copy()

    if len(df_tp) == 0:
        raise RuntimeError(f"No rows found for timepoint={args.timepoint} in summary.")

    bases = sorted(df_tp["file"].unique().tolist())

    rows = []
    for base in bases:
        for stim in TARGETS:
            freqs, reps = load_repeats(args.psd_dir, base, stim)
            if freqs is None:
                continue

            # compute per-repeat bin features for stim + harmonics
            stim_vals = []
            h2_vals = []
            h3_vals = []

            for r in reps:
                stim_vals.append(bin_logp_at(freqs, r, stim))
                h2_vals.append(bin_logp_at(freqs, r, 2*stim))
                h3_vals.append(bin_logp_at(freqs, r, 3*stim))

            stim_vals = np.array(stim_vals, float)
            h2_vals = np.array(h2_vals, float)
            h3_vals = np.array(h3_vals, float)

            rows.append({
                "file": base,
                "patient": re.match(r"^(VEP\d+)", base).group(1) if re.match(r"^(VEP\d+)", base) else base,
                "timepoint": args.timepoint,
                "stim_hz": stim,
                "n_repeats": int(len(reps)),
                "stim_mean": float(np.nanmean(stim_vals)),
                "stim_sd": float(np.nanstd(stim_vals)),
                "harm2_mean": float(np.nanmean(h2_vals)),
                "harm2_sd": float(np.nanstd(h2_vals)),
                "harm3_mean": float(np.nanmean(h3_vals)),
                "harm3_sd": float(np.nanstd(h3_vals)),
                "stim_repeat_vals": stim_vals,
                "harm2_repeat_vals": h2_vals,
                "harm3_repeat_vals": h3_vals,
            })

    if len(rows) == 0:
        raise RuntimeError("No repeatability data found. Are *_psd_repeats.npy files present?")

    # Save a tidy CSV (repeat values expanded)
    tidy = []
    for r in rows:
        for i in range(r["n_repeats"]):
            tidy.append({
                "file": r["file"],
                "patient": r["patient"],
                "timepoint": r["timepoint"],
                "stim_hz": r["stim_hz"],
                "repeat": i + 1,
                "stim_bin_logp": float(r["stim_repeat_vals"][i]),
                "harm2_bin_logp": float(r["harm2_repeat_vals"][i]),
                "harm3_bin_logp": float(r["harm3_repeat_vals"][i]),
            })
    df_tidy = pd.DataFrame(tidy)
    df_tidy.to_csv(os.path.join(args.out, f"repeatability_T{args.timepoint}_tidy.csv"), index=False)

    # ---- Plot: per-stim scatter of repeats per patient + mean±SD
    # We keep it readable by sorting patients by mean at 10 Hz (if available)
    patients = sorted(df_tidy["patient"].unique().tolist())

    # helper to order patients by 10Hz mean
    tmp = df_tidy[df_tidy["stim_hz"] == 10].groupby("patient")["stim_bin_logp"].mean()
    if len(tmp) > 0:
        patients = tmp.sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(3, 1, figsize=(12, max(8, 0.25*len(patients) + 6)), sharex=True)
    for ax, stim in zip(axes, TARGETS):
        sub = df_tidy[df_tidy["stim_hz"] == stim].copy()
        ax.set_title(f"T{args.timepoint} repeatability — stim_bin_logp at {stim} Hz (3 repeats)")
        ax.set_ylabel("log10 power (V^2/Hz)")

        # plot per patient
        for xi, p in enumerate(patients):
            vals = sub[sub["patient"] == p]["stim_bin_logp"].values
            if len(vals) == 0:
                continue
            # jittered points
            x = xi + (np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0]))
            ax.plot(x, vals, "o", markersize=4, alpha=0.8)
            # mean±sd
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            ax.plot([xi-0.18, xi+0.18], [m, m], "-", linewidth=2)
            ax.plot([xi, xi], [m-s, m+s], "-", linewidth=1.5, alpha=0.9)

        ax.grid(True, alpha=0.2)

    axes[-1].set_xticks(range(len(patients)))
    axes[-1].set_xticklabels(patients, rotation=90)
    axes[-1].set_xlabel("Patient (sorted by 10 Hz mean if available)")

    plt.tight_layout()
    out_fig = os.path.join(args.out, f"repeatability_T{args.timepoint}_stim_bin_logp.png")
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    # ---- Plot: SD distribution per stim (quick QC)
    fig, ax = plt.subplots(figsize=(8, 4))
    sds = []
    labels = []
    for stim in TARGETS:
        sd_vals = []
        for r in rows:
            if r["stim_hz"] == stim:
                sd_vals.append(r["stim_sd"])
        sds.append(sd_vals)
        labels.append(str(stim))
    ax.boxplot(sds, labels=labels)
    ax.set_title(f"T{args.timepoint} within-measure SD (stim_bin_logp) across patients")
    ax.set_xlabel("Stimulus frequency (Hz)")
    ax.set_ylabel("SD of 3 repeats (log10 power)")
    ax.grid(True, alpha=0.2)

    out_fig2 = os.path.join(args.out, f"repeatability_T{args.timepoint}_SD_boxplot.png")
    plt.tight_layout()
    fig.savefig(out_fig2, dpi=200)
    plt.close(fig)

    print(f"[OK] Wrote: {out_fig}")
    print(f"[OK] Wrote: {out_fig2}")
    print(f"[OK] Wrote: {os.path.join(args.out, f'repeatability_T{args.timepoint}_tidy.csv')}")


if __name__ == "__main__":
    main()
