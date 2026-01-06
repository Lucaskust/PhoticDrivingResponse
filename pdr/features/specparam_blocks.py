from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from specparam import SpectralModel

from pdr.features.power import Power


# Start-presets (kunnen we later in config zetten)
SPEC_PRESETS = {
    "baseline": dict(
        aperiodic_mode="fixed",
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0,
    ),
    "conservative": dict(
        aperiodic_mode="fixed",
        peak_width_limits=(2, 10),
        max_n_peaks=4,
        min_peak_height=0.10,
        peak_threshold=2.5,
    ),
    "harmonics": dict(
        aperiodic_mode="fixed",
        peak_width_limits=(2, 8),
        max_n_peaks=8,
        min_peak_height=0.05,
        peak_threshold=1.5,
    ),
}


def _ensure_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_get_r2_err(fm) -> tuple[float, float]:
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

    return float(r2) if r2 is not None else np.nan, float(err) if err is not None else np.nan


def _safe_get_aperiodic(fm) -> tuple[float, float, float]:
    # returns offset, knee, exponent (knee can be NaN)
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
    return offset, knee, exponent


def _safe_get_peaks(fm) -> np.ndarray:
    peaks = None
    for key in ["peak", "peaks", "peak_params"]:
        try:
            peaks = fm.get_params(key)
            break
        except Exception:
            pass

    if peaks is None:
        return np.empty((0, 3))

    peaks = np.asarray(peaks)
    if peaks.ndim == 1 and peaks.size == 0:
        return np.empty((0, 3))
    if peaks.ndim == 1 and peaks.size == 3:
        peaks = peaks.reshape(1, 3)
    return peaks


@dataclass
class SpecParamBlocks:
    out_dir: Path
    fmin: float = 2.0
    fmax: float = 45.0
    upper_lim_psd: int = 40  # must be >= fmax ideally
    trim: float = 0.0
    padding: str = "zeros"
    presets: Optional[list[str]] = None  # default: all keys in SPEC_PRESETS

    def run_from_raw(self, raw, base_name: str) -> pd.DataFrame:
        """
        Computes PSD-per-block (avg reps) using Power internals, then fits SpecParam for each freq.
        Returns summary dataframe (one row per freq per preset).
        """
        self.out_dir = _ensure_dir(self.out_dir)

        # 1) Reuse Power's block extraction + epoching + PSD logic
        df_stim = Power._stimulation_power(raw, save=False, out_dir=None)
        epochs, df_epochs = Power._epoch_power(df_stim, raw, save=False, plot=False, out_dir=None)
        _, fft_lin, fft_freq = Power._fft_power(
            epochs,
            df_epochs,
            trim=self.trim,
            padding=self.padding,
            upper_lim=self.upper_lim_psd,
            plot=False,
        )

        ch_names = epochs.info["ch_names"]
        freqs_present = sorted(df_epochs["freq"].dropna().astype(int).unique())

        # Fit range must be inside fft_freq range
        fmin = max(self.fmin, float(np.min(fft_freq)))
        fmax = min(self.fmax, float(np.max(fft_freq)))
        if fmax <= fmin:
            raise ValueError(f"SpecParam freq_range invalid after clipping: ({fmin}, {fmax})")

        preset_names = self.presets or list(SPEC_PRESETS.keys())

        summary_rows = []

        for preset in preset_names:
            if preset not in SPEC_PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(SPEC_PRESETS.keys())}")

            preset_dir = _ensure_dir(self.out_dir / preset)

            for f in freqs_present:
                # Average across channels (rep=0 is mean over reps)
                # fft_lin is dict: [rep][freq][ch] -> np.array
                try:
                    stack = np.stack([fft_lin[0][f][ch] for ch in ch_names], axis=0)
                    mean_lin = np.nanmean(stack, axis=0)
                except Exception:
                    continue

                mean_lin = np.asarray(mean_lin, dtype=float)
                mean_lin = np.maximum(mean_lin, 1e-30)

                fm = SpectralModel(**SPEC_PRESETS[preset])
                fm.fit(fft_freq, mean_lin, freq_range=(fmin, fmax))

                r2, err = _safe_get_r2_err(fm)
                offset, knee, exponent = _safe_get_aperiodic(fm)

                # Save plot
                fig, ax = plt.subplots(figsize=(8, 4))
                fm.plot(ax=ax, plot_peaks="shade")
                ax.set_title(f"{base_name} | f={f} | preset={preset}")
                fig.savefig(preset_dir / f"{base_name}_f{f}_specparam.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

                # Save peak table
                peaks = _safe_get_peaks(fm)
                df_peaks = pd.DataFrame(peaks, columns=["center_freq", "peak_power", "bandwidth"])
                df_peaks.insert(0, "file", base_name)
                df_peaks.insert(1, "freq_hz", f)
                df_peaks.insert(2, "preset", preset)
                df_peaks.to_csv(preset_dir / f"{base_name}_f{f}_peaks.csv", index=False)

                # Save aperiodic row
                row = {
                    "file": base_name,
                    "freq_hz": f,
                    "preset": preset,
                    "fmin_fit": fmin,
                    "fmax_fit": fmax,
                    "r2": r2,
                    "error": err,
                    "offset": offset,
                    "knee": knee,
                    "exponent": exponent,
                    "n_channels": len(ch_names),
                    "channels": ",".join(ch_names),
                }
                pd.DataFrame([row]).to_csv(preset_dir / f"{base_name}_f{f}_aperiodic.csv", index=False)

                summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.out_dir / f"{base_name}_specparam_summary.csv", index=False)
        return df_summary
