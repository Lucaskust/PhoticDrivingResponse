import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_patient_timepoint(df: pd.DataFrame) -> pd.DataFrame:
    """Add patient + timepoint if missing, based on file name like VEP02_1."""
    if "patient" in df.columns and "timepoint" in df.columns:
        return df

    patients = []
    tps = []
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


def save_heatmap(df_t0: pd.DataFrame, value_col: str, out_path: str, title: str):
    # pivot: rows=patient, cols=stim_hz
    piv = df_t0.pivot_table(index="patient", columns="stim_hz", values=value_col, aggfunc="mean")
    # order columns nicely
    cols = [c for c in [6.0, 10.0, 20.0] if c in piv.columns]
    piv = piv[cols]

    # sort patients by 10Hz (if exists) else by mean
    if 10.0 in piv.columns:
        piv = piv.sort_values(by=10.0, ascending=False)
    else:
        piv = piv.sort_values(by=cols[0], ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.22 * len(piv) + 2)))
    im = ax.imshow(piv.values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Stimulus frequency (Hz)")
    ax.set_ylabel("Patient")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(int(c)) for c in piv.columns])

    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_psd_avg(psd_dir: str, base: str, stim_hz: int):
    """Load freqs + avg PSD for a given file base (e.g., VEP02_1) and stim (6/10/20)."""
    f_psd = os.path.join(psd_dir, f"{base}_{stim_hz}Hz_psd_avg.npy")
    f_frq = os.path.join(psd_dir, f"{base}_{stim_hz}Hz_freqs.npy")

    if os.path.exists(f_psd) and os.path.exists(f_frq):
        return np.load(f_frq), np.load(f_psd)

    # fallback via glob (if paths differ)
    cand_psd = glob.glob(os.path.join(psd_dir, f"{base}_{stim_hz}Hz_*psd_avg.npy"))
    cand_frq = glob.glob(os.path.join(psd_dir, f"{base}_{stim_hz}Hz_*freqs.npy"))
    if cand_psd and cand_frq:
        return np.load(cand_frq[0]), np.load(cand_psd[0])

    return None, None


def make_psd_overlay(df_t0: pd.DataFrame, psd_dir: str, out_path: str, stim_hz: int):
    files = sorted(df_t0["file"].unique().tolist())

    all_log = []
    freqs_ref = None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"T0 PSD overlay (avg over 3 repeats) — {stim_hz} Hz condition")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10(PSD) [V^2/Hz]")

    for base in files:
        freqs, psd = load_psd_avg(psd_dir, base, stim_hz)
        if freqs is None:
            continue

        freqs = np.asarray(freqs, float)
        psd = np.asarray(psd, float)
        logp = np.log10(np.maximum(psd, 1e-30))

        if freqs_ref is None:
            freqs_ref = freqs
        else:
            # only keep if same freq grid
            if len(freqs) != len(freqs_ref) or np.max(np.abs(freqs - freqs_ref)) > 1e-6:
                continue

        all_log.append(logp)
        ax.plot(freqs, logp, linewidth=0.8, alpha=0.25)

    if len(all_log) == 0:
        ax.text(0.05, 0.6, "No PSD avg files found for this stim.", transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    arr = np.vstack(all_log)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)

    ax.plot(freqs_ref, med, linewidth=2.2, label="Median")
    ax.fill_between(freqs_ref, q1, q3, alpha=0.25, label="IQR (25–75%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to all_specparam_blocks_summary.csv")
    ap.add_argument("--psd_dir", required=True, help="Folder containing *_psd_avg.npy and *_freqs.npy")
    ap.add_argument("--out", default="t0_overview_plots", help="Output folder for plots")
    ap.add_argument("--timepoint", type=int, default=1, help="Timepoint number: 1=T0, 2=T1, 3=T2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = ensure_patient_timepoint(df)

    # filter T0
    df_t0 = df[df["timepoint"] == args.timepoint].copy()
    if len(df_t0) == 0:
        raise RuntimeError(f"No rows found for timepoint={args.timepoint} in summary.")

    # ---- HEATMAPS (bin-based; works even if peak columns are empty)
    if "stim_bin_logp" in df_t0.columns:
        save_heatmap(
            df_t0, "stim_bin_logp",
            os.path.join(args.out, f"heatmap_T{args.timepoint}_stim_bin_logp.png"),
            title=f"T{args.timepoint} heatmap: log10 power at stimulus frequency (stim_bin_logp)"
        )

    if "harm2_bin_logp" in df_t0.columns:
        save_heatmap(
            df_t0, "harm2_bin_logp",
            os.path.join(args.out, f"heatmap_T{args.timepoint}_harm2_bin_logp.png"),
            title=f"T{args.timepoint} heatmap: log10 power at 2× stimulus (harm2_bin_logp)"
        )

    # ---- PSD overlays
    for stim in [6, 10, 20]:
        make_psd_overlay(
            df_t0, args.psd_dir,
            os.path.join(args.out, f"psd_overlay_T{args.timepoint}_{stim}Hz.png"),
            stim_hz=stim
        )

    print(f"[OK] Wrote plots to: {args.out}")


if __name__ == "__main__":
    main()
