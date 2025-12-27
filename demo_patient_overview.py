import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne

from specparam import SpectralModel

# --- Helpers -------------------------------------------------

TARGETS = np.array([6.0, 10.0, 20.0])

def pick_occipital(raw):
    # voorkeur: jouw occipital set
    wanted = ["POz", "O1", "O2", "PO3", "PO4", "Oz"]
    present = [ch for ch in wanted if ch in raw.ch_names]
    return present

def infer_blocks_from_triggers(raw, tol=0.8, min_pulses=20):
    """
    Uses annotation onsets as trigger times.
    Converts inter-trigger intervals to instantaneous frequency (1/dt),
    classifies to nearest of {6,10,20} Hz and groups consecutive into blocks.
    Returns list of tuples: (hz, t_start, t_end, n_pulses)
    """
    onsets = np.sort(np.array(raw.annotations.onset, dtype=float))
    if len(onsets) < 3:
        return [], onsets, np.array([]), np.array([]), np.array([])

    dt = np.diff(onsets)
    # avoid division issues
    dt = np.where(dt <= 0, np.nan, dt)
    inst_f = 1.0 / dt
    t_inst = onsets[1:]  # inst_f corresponds to interval after each onset

    # classify each interval
    labels = np.full(inst_f.shape, np.nan)
    for i, f in enumerate(inst_f):
        if not np.isfinite(f):
            continue
        j = int(np.argmin(np.abs(TARGETS - f)))
        if abs(TARGETS[j] - f) <= tol:
            labels[i] = TARGETS[j]

    # group into blocks: consecutive same label
    blocks = []
    i = 0
    while i < len(labels):
        if not np.isfinite(labels[i]):
            i += 1
            continue

        hz = labels[i]
        start_idx = i
        while i < len(labels) and labels[i] == hz:
            i += 1
        end_idx = i - 1

        # pulses in block = number of triggers involved = (end-start+2)
        n_pulses = (end_idx - start_idx + 2)
        if n_pulses >= min_pulses:
            t0 = onsets[start_idx]
            t1 = onsets[end_idx + 1]  # last trigger time in that run
            blocks.append((float(hz), float(t0), float(t1), int(n_pulses)))

    return blocks, onsets, t_inst, inst_f, labels

def compute_psd_for_segment(raw, t0, t1, fmin=2, fmax=45, picks=None):
    """
    Compute Welch PSD on cropped segment, then average across picks.
    Returns freqs, mean_psd_lin (V^2/Hz)
    """
    seg = raw.copy().crop(t0, t1)
    if picks is not None:
        seg.pick(picks=picks)
    else:
        seg.pick(picks="eeg")

    sfreq = seg.info["sfreq"]

    psd = seg.compute_psd(
        method="welch",
        fmin=fmin, fmax=fmax,
        n_per_seg=int(sfreq * 1),     # 1s window -> 1 Hz resolution
        n_overlap=int(sfreq * 0.5),   # 50%
        n_fft=int(sfreq * 1),
        verbose="error"
    )
    freqs = psd.freqs
    p = psd.get_data()  # (n_ch, n_freq)
    mean_lin = np.mean(p, axis=0)
    mean_lin = np.maximum(mean_lin, 1e-30)
    return freqs, mean_lin

def fit_specparam(freqs, psd_lin, fmin=2, fmax=45, aperiodic_mode="fixed"):
    fm = SpectralModel(
        aperiodic_mode=aperiodic_mode,
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0
    )
    fm.fit(freqs, psd_lin, freq_range=(fmin, fmax))
    return fm

def get_peak_params(fm):
    """Robuust peak ophalen voor verschillende specparam versies."""
    peaks = None

    # 1) probeer via get_params (meest betrouwbaar)
    if hasattr(fm, "get_params"):
        for key in ["peak_params", "peaks", "peak"]:
            try:
                peaks = fm.get_params(key)
                break
            except Exception:
                pass

    # 2) fallback op attribute (als die bestaat)
    if peaks is None:
        peaks = getattr(fm, "peak_params_", None)

    # 3) normaliseer naar Nx3
    if peaks is None:
        peaks = np.empty((0, 3))
    peaks = np.asarray(peaks)

    if peaks.ndim == 1 and peaks.size == 0:
        peaks = np.empty((0, 3))
    elif peaks.ndim == 1 and peaks.size == 3:
        peaks = peaks.reshape(1, 3)

    return peaks


def nearest_peak(df, target, tol=1.0):
    if df is None or len(df) == 0:
        return np.nan, np.nan, np.nan
    d = np.abs(df["center_freq"].values - float(target))
    i = int(np.argmin(d))
    if d[i] <= tol:
        row = df.iloc[i]
        return float(row["center_freq"]), float(row["peak_power"]), float(row["bandwidth"])
    return np.nan, np.nan, np.nan

def bin_power_at(freqs, psd_lin, target_hz):
    freqs = np.asarray(freqs, dtype=float)
    psd_lin = np.asarray(psd_lin, dtype=float)
    idx = int(np.argmin(np.abs(freqs - float(target_hz))))
    cf = float(freqs[idx])
    lp = float(np.log10(np.maximum(psd_lin[idx], 1e-30)))
    return cf, lp

# --- Report --------------------------------------------------

def make_report(cnt_path, out_dir="report_out", fmin=2, fmax=45, aperiodic_mode="fixed"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(cnt_path).replace(".cnt", "")

    # Read without preloading full dataset (memory-safe)
    raw = mne.io.read_raw_ant(cnt_path, preload=False, verbose="error")
    raw.pick(picks="eeg")

    occ = pick_occipital(raw)
    picks = occ if len(occ) else None

    # ---- Step: triggers -> blocks
    blocks, onsets, t_inst, inst_f, labels = infer_blocks_from_triggers(raw, tol=0.8, min_pulses=20)

    # tabular overview of blocks
    block_rows = []
    for k, (hz, t0, t1, n) in enumerate(blocks, start=1):
        block_rows.append({
            "block": k,
            "hz": hz,
            "t_start_s": t0,
            "t_end_s": t1,
            "dur_s": (t1 - t0),
            "pulses": n
        })
    df_blocks = pd.DataFrame(block_rows)

    # group blocks by frequency
    by_hz = {6.0: [], 10.0: [], 20.0: []}
    for (hz, t0, t1, n) in blocks:
        if hz in by_hz:
            by_hz[hz].append((t0, t1))

    # ---- Step: PSD per block + average per frequency
    psd_rep = {6.0: [], 10.0: [], 20.0: []}
    freqs_ref = None
    for hz in [6.0, 10.0, 20.0]:
        for (t0, t1) in by_hz[hz]:
            freqs, psd_lin = compute_psd_for_segment(raw, t0, t1, fmin=fmin, fmax=fmax, picks=picks)
            if freqs_ref is None:
                freqs_ref = freqs
            psd_rep[hz].append(psd_lin)

    psd_avg = {}
    for hz in [6.0, 10.0, 20.0]:
        if len(psd_rep[hz]) > 0:
            psd_avg[hz] = np.mean(np.vstack(psd_rep[hz]), axis=0)
        else:
            psd_avg[hz] = None

    # ---- Step: specparam fit per frequency + params table
    sum_rows = []
    fit_models = {}
    peak_dfs = {}

    for hz in [6.0, 10.0, 20.0]:
        if psd_avg[hz] is None:
            continue

        fm = fit_specparam(freqs_ref, psd_avg[hz], fmin=fmin, fmax=fmax, aperiodic_mode=aperiodic_mode)
        fit_models[hz] = fm

        # peaks dataframe
        # peaks dataframe (robust)
        peaks = get_peak_params(fm)
        dfp = pd.DataFrame(peaks, columns=["center_freq", "peak_power", "bandwidth"])
        dfp = dfp.sort_values("peak_power", ascending=False).reset_index(drop=True)
        peak_dfs[hz] = dfp


        stim_cf, stim_pw, stim_bw = nearest_peak(dfp, hz, tol=1.0)
        harm2_cf, harm2_pw, harm2_bw = nearest_peak(dfp, 2*hz, tol=1.0)
        harm3_cf, harm3_pw, harm3_bw = nearest_peak(dfp, 3*hz, tol=1.0)

        stim_bin_cf, stim_bin_logp = bin_power_at(freqs_ref, psd_avg[hz], hz)
        harm2_bin_cf, harm2_bin_logp = bin_power_at(freqs_ref, psd_avg[hz], 2*hz)
        harm3_bin_cf, harm3_bin_logp = bin_power_at(freqs_ref, psd_avg[hz], 3*hz)

        sum_rows.append({
            "file": base,
            "stim_hz": hz,
            "n_blocks": len(psd_rep[hz]),
            "offset": float(fm.get_params("aperiodic")[0]) if hasattr(fm, "get_params") else np.nan,
            "exponent": float(fm.get_params("aperiodic")[-1]) if hasattr(fm, "get_params") else np.nan,
            "stim_cf": stim_cf, "stim_pw": stim_pw, "stim_bw": stim_bw,
            "harm2_cf": harm2_cf, "harm2_pw": harm2_pw, "harm2_bw": harm2_bw,
            "harm3_cf": harm3_cf, "harm3_pw": harm3_pw, "harm3_bw": harm3_bw,
            "stim_bin_cf": stim_bin_cf, "stim_bin_logp": stim_bin_logp,
            "harm2_bin_cf": harm2_bin_cf, "harm2_bin_logp": harm2_bin_logp,
            "harm3_bin_cf": harm3_bin_cf, "harm3_bin_logp": harm3_bin_logp,
        })

    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_dir, f"{base}_report_summary.csv"), index=False)
    if len(df_blocks) > 0:
        df_blocks.to_csv(os.path.join(out_dir, f"{base}_blocks.csv"), index=False)

    # ---- Create PDF report (stacked steps)
    pdf_path = os.path.join(out_dir, f"{base}_pipeline_report.pdf")
    with PdfPages(pdf_path) as pdf:

        # Page 1: overview + block table
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle(f"Pipeline report: {base}\nOccipital picks: {picks if picks is not None else 'ALL EEG'}", fontsize=14)

        ax1 = plt.subplot(2, 1, 1)
        ax1.axis("off")
        txt = (
            "STEPS\n"
            "1) RAW CNT EEG (not fully loaded)\n"
            "2) Trigger extraction from annotations (onsets)\n"
            "3) Instantaneous frequency = 1/dt between triggers\n"
            "4) Classification to 6 / 10 / 20 Hz and grouping into blocks\n"
            "5) PSD per block (Welch) on occipital channels\n"
            "6) Average PSD over the 3 repeats per frequency\n"
            "7) Specparam fit per frequency (aperiodic + peaks)\n"
            "8) Export features (peak-based + bin-based)\n"
        )
        ax1.text(0.01, 0.98, txt, va="top", fontsize=11)

        ax2 = plt.subplot(2, 1, 2)
        ax2.axis("off")
        if len(df_blocks) == 0:
            ax2.text(0.01, 0.9, "No blocks detected.", fontsize=12)
        else:
            ax2.text(0.01, 0.95, "Detected stimulus blocks:", fontsize=12, weight="bold")
            table = ax2.table(
                cellText=df_blocks.round(3).values,
                colLabels=df_blocks.columns,
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

        # =========================
        # Page 2: TRIGGERS -> BLOCKS (clean)
        # =========================
        from matplotlib.patches import Rectangle

        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Trigger step: markers → dt → instantaneous freq → blocks", fontsize=14)

        # bepaal tijd-as grenzen (zodat het niet 0..1 wordt)
        if len(blocks) > 0:
            tmin = min(b[1] for b in blocks) - 5
            tmax = max(b[2] for b in blocks) + 5
        else:
            # fallback: neem trigger-onsets als er geen blocks zijn
            tmin = float(np.min(onsets)) - 5 if len(onsets) else 0
            tmax = float(np.max(onsets)) + 5 if len(onsets) else 1

        color_map = {6.0: "#4C72B0", 10.0: "#55A868", 20.0: "#C44E52"}

        # (A) timeline met blokken + triggers als rug
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("Detected stimulus blocks (timeline)")
        ax1.set_xlim(tmin, tmax)
        ax1.set_ylim(4, 22)
        ax1.set_yticks([6, 10, 20])
        ax1.set_ylabel("Stim (Hz)")
        ax1.grid(True, axis="x", alpha=0.25)

        # triggers als kleine streepjes onderaan (niet als mega-barcode)
        if len(onsets) > 0:
            ax1.plot(onsets, np.full_like(onsets, 4.5), "|", color="k", markersize=6, alpha=0.25, label="trigger")

        # blokken als horizontale stroken
        if len(blocks) == 0:
            ax1.text(0.02, 0.5, "No blocks detected.", transform=ax1.transAxes, fontsize=12)
        else:
            for (hz, t0, t1, n_pulses) in blocks:
                ax1.add_patch(
                    Rectangle((t0, hz - 0.8), (t1 - t0), 1.6,
                            facecolor=color_map.get(hz, "gray"), alpha=0.35, edgecolor="none")
                )
                ax1.text((t0 + t1) / 2, hz, f"{int(hz)} Hz",
                        ha="center", va="center", fontsize=9)

        ax1.legend(loc="upper right")

        # (B) instant freq punten (geen diagonalen) + referentielijnen
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("Instantaneous frequency (1/dt), classified to 6/10/20 Hz")
        ax2.set_xlim(tmin, tmax)
        ax2.set_ylim(0, 25)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("1/dt (Hz)")
        ax2.grid(True, alpha=0.25)

        mask = np.isfinite(labels) & np.isfinite(inst_f) & np.isfinite(t_inst)
        for hz in [6.0, 10.0, 20.0]:
            m = mask & (labels == hz)
            if np.any(m):
                ax2.scatter(t_inst[m], inst_f[m], s=14, alpha=0.9,
                            color=color_map.get(hz, None), label=f"{int(hz)} Hz")
            ax2.axhline(hz, linestyle="--", linewidth=1.0, alpha=0.35, color=color_map.get(hz, None))

        ax2.legend(loc="upper right")

        fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)



        # =========================
        # Page 3: DASHBOARD (no table)
        # =========================
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Patient overview: photic-driving features (per stimulus frequency)", fontsize=14)

        # layout: top text, bottom-left lines, bottom-right exponent
        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 3], width_ratios=[3, 2])

        ax_text = fig.add_subplot(gs[0, :])
        ax_l = fig.add_subplot(gs[1, 0])
        ax_r = fig.add_subplot(gs[1, 1])

        ax_text.axis("off")

        if len(df_sum) == 0:
            ax_text.text(0.01, 0.6, "No results available (no blocks found).", fontsize=12, weight="bold")
        else:
            # --- Extract the core values ---
            stim_hz = df_sum["stim_hz"].values

            stim_bin = df_sum["stim_bin_logp"].values
            h2_bin = df_sum["harm2_bin_logp"].values
            h3_bin = df_sum["harm3_bin_logp"].values

            exponent = df_sum["exponent"].values
            offset = df_sum["offset"].values

            # simple takeaways (presentation-friendly)
            # 1) strongest driving = highest stim_bin_logp (less negative = more power)
            idx_best = int(np.nanargmax(stim_bin))
            best_hz = float(stim_hz[idx_best])
            best_val = float(stim_bin[idx_best])

            # 2) exponent trend
            exp_min = float(np.nanmin(exponent))
            exp_max = float(np.nanmax(exponent))

            ax_text.text(0.01, 0.82, "Key takeaways", fontsize=12, weight="bold")
            ax_text.text(
                0.01, 0.52,
                f"• Strongest stimulus-locked power is at {best_hz:.0f} Hz (stim_bin_logp = {best_val:.3f}).\n"
                f"• Aperiodic exponent ranges from {exp_min:.3f} to {exp_max:.3f} across conditions.\n"
                f"• These features are computed from occipital channels and averaged over the 3 repeats per stimulus frequency.",
                fontsize=10
            )
            ax_text.text(
                0.01, 0.18,
                "Interpretation:\n"
                "stim_bin_logp / harm2_bin_logp / harm3_bin_logp = log10 power exactly at stim and its harmonics.\n"
                "offset/exponent = background (aperiodic) fit describing overall spectral slope.",
                fontsize=9, alpha=0.95
            )

            # --- Bottom-left: driving features ---
            ax_l.set_title("Stimulus-locked power (stim & harmonics)")
            ax_l.plot(stim_hz, stim_bin, marker="o", label="stim (1×)")
            ax_l.plot(stim_hz, h2_bin, marker="o", label="harmonic (2×)")
            ax_l.plot(stim_hz, h3_bin, marker="o", label="harmonic (3×)")
            ax_l.set_xlabel("Stimulus frequency (Hz)")
            ax_l.set_ylabel("log10 power (V^2/Hz)")
            ax_l.grid(True, alpha=0.25)
            ax_l.legend(loc="best")

            # --- Bottom-right: aperiodic exponent ---
            ax_r.set_title("Aperiodic exponent")
            ax_r.plot(stim_hz, exponent, marker="o")
            ax_r.set_xlabel("Stimulus frequency (Hz)")
            ax_r.set_ylabel("Exponent")
            ax_r.grid(True, alpha=0.25)

            # Optional: also show offset as a faint 2nd line (comment out if you don't want it)
            ax_r2 = ax_r.twinx()
            ax_r2.plot(stim_hz, offset, marker="s", linestyle="--", alpha=0.5)
            ax_r2.set_ylabel("Offset", alpha=0.7)

        fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.93])
        pdf.savefig(fig)
        plt.close(fig)


        # Page 4: PSD repeats + average (what you already made, but in the report)
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("PSD per block (3 repeats) + average per stimulus frequency", fontsize=14)

        panel = 1
        for hz in [6.0, 10.0, 20.0]:
            ax = plt.subplot(3, 1, panel); panel += 1
            if len(psd_rep[hz]) == 0:
                ax.text(0.1, 0.5, f"No PSD blocks for {hz} Hz", fontsize=12)
                ax.axis("off")
                continue

            arr = np.vstack(psd_rep[hz])
            for i in range(arr.shape[0]):
                ax.plot(freqs_ref, np.log10(arr[i]), linewidth=1.0)
            ax.plot(freqs_ref, np.log10(psd_avg[hz]), linewidth=2.5)
            ax.set_title(f"{hz:.0f} Hz blocks: repeats + average")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("log10(V^2/Hz)")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # Page 5: Specparam fits (one per frequency)
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Specparam fits (aperiodic + peaks)", fontsize=14)

        panel = 1
        for hz in [6.0, 10.0, 20.0]:
            ax = plt.subplot(3, 1, panel); panel += 1
            if hz not in fit_models:
                ax.text(0.1, 0.5, f"No fit for {hz} Hz", fontsize=12)
                ax.axis("off")
                continue
            fit_models[hz].plot(ax=ax, plot_peaks="shade")
            ax.set_title(f"Specparam fit on AVG PSD | {hz:.0f} Hz")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

    # Also save key pages as PNG for quick VS Code viewing
    # (PDF is the main artifact)
    print(f"[OK] Wrote: {pdf_path}")
    print(f"[OK] Wrote CSVs: {os.path.join(out_dir, base+'_report_summary.csv')} and blocks.csv (if blocks found)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True, help="Path to .cnt")
    ap.add_argument("--out", default="report_out", help="Output folder")
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--mode", type=str, default="fixed", choices=["fixed", "knee"])
    args = ap.parse_args()
    make_report(args.cnt, out_dir=args.out, fmin=args.fmin, fmax=args.fmax, aperiodic_mode=args.mode)

if __name__ == "__main__":
    main()
