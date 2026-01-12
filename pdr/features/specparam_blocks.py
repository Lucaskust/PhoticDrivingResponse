from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from specparam import SpectralModel

from pdr.core.config import resolve_specparam_config
from pdr.features.power import Power
from typing import Optional
from pathlib import Path


# Python-side defaults (kan later via TOML overschreven worden met preset_overrides)
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
    """Specparam v2 stores metrics in fm.results.metrics.results."""
    r2 = np.nan
    err = np.nan

    # Specparam v2: metrics live here
    try:
        md = fm.results.metrics.results  # dict
        err = md.get("error_mae", md.get("error_rmse", md.get("error", np.nan)))
        r2 = md.get("gof_rsquared", md.get("rsquared", md.get("gof_r2", np.nan)))
        return float(r2), float(err)
    except Exception:
        pass

    # Specparam v2 also exposes FitResults via results.get_results()
    try:
        fres = fm.results.get_results()
        md = getattr(fres, "metrics", {}) or {}
        err = md.get("error_mae", md.get("error_rmse", np.nan))
        r2 = md.get("gof_rsquared", np.nan)
        return float(r2), float(err)
    except Exception:
        pass

    # Backwards-compatible (old fooof-style attrs)
    r2 = getattr(fm, "r_squared_", getattr(fm, "r_squared", np.nan))
    err = getattr(fm, "error_", getattr(fm, "error", np.nan))
    return float(r2) if r2 is not None else np.nan, float(err) if err is not None else np.nan


def _safe_get_aperiodic(fm) -> tuple[float, float, float]:
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


class SpecParamBlocks:
    def __init__(
        self,
        out_dir: Path,
        cfg: dict | None = None,
        fig_dir: Path | None = None,
        fmin: float = 2.0,
        fmax: float = 45.0,
        upper_lim_psd: int = 70,
        trim: float = 0.0,
        padding: str = "zeros",
        presets: list[str] | None = None,
    ):
        self.out_dir = _ensure_dir(Path(out_dir))
        self.fig_dir = _ensure_dir(Path(fig_dir)) if fig_dir is not None else None
        self.cfg = cfg or {}

        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.upper_lim_psd = int(upper_lim_psd)
        self.trim = float(trim)
        self.padding = str(padding)
        self.presets = presets  # als None: resolve_specparam_config bepaalt wat er gebeurt

    def run_from_raw(self, raw, base_name: str) -> pd.DataFrame:
        """
        PSD per stimulus-freq (rep-mean), SpecParam fit per preset, + figuren opslaan.
        """
        # 1) PSD via Power internals
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

        preset_names = self.presets
        if preset_names is None:
            # als cfg presets bevat: pak die keys, anders alleen 'baseline'
            preset_names = list((self.cfg.get("presets", {}) or {}).keys()) or [self.cfg.get("preset", "baseline")]

        summary_rows: list[dict] = []

        for preset in preset_names:
            fallback_presets = (self.cfg.get("presets") or {})
            fmin_fit, fmax_fit, model_params = resolve_specparam_config(self.cfg, preset=preset, fallback_presets=fallback_presets)
            fallback_presets = self.cfg.get("presets") or {
                "baseline": dict(aperiodic_mode="fixed", peak_width_limits=(2, 12), max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0),
                "conservative": dict(aperiodic_mode="fixed", peak_width_limits=(2, 10), max_n_peaks=4, min_peak_height=0.10, peak_threshold=2.5),
                "harmonics": dict(aperiodic_mode="fixed", peak_width_limits=(2, 8),  max_n_peaks=10, min_peak_height=0.05, peak_threshold=1.5),
            }
            fmin_fit, fmax_fit, model_params = resolve_specparam_config(self.cfg, preset, fallback_presets)



            for f in freqs_present:
                # rep=0 = mean over reps (legacy)
                try:
                    stack = np.stack([fft_lin[0][f][ch] for ch in ch_names], axis=0)
                    mean_lin = np.nanmean(stack, axis=0)
                except Exception:
                    continue

                mean_lin = np.asarray(mean_lin, dtype=float)
                mean_lin = np.maximum(mean_lin, 1e-30)

                fm = SpectralModel(**model_params)
                fm.fit(fft_freq, mean_lin, freq_range=(fmin_fit, fmax_fit))

                r2, err = _safe_get_r2_err(fm)
                offset, knee, exponent = _safe_get_aperiodic(fm)

                # ---- FIG SAVE (hard) ----
                save_root = self.fig_dir if self.fig_dir is not None else self.out_dir
                save_dir = save_root / preset
                save_dir.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                fm.plot(ax=ax, plot_peaks="shade")
                ax.set_title(f"{base_name} | f={f} | preset={preset}")
                fig_path = save_dir / f"{base_name}_f{f}_specparam.png"
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                # debug print (laat staan tot je het ziet)
                print(f"[OK] SpecParam figure -> {fig_path}")

                # ---- tables ----
                preset_out = (self.out_dir / preset)
                preset_out.mkdir(parents=True, exist_ok=True)

                peaks = _safe_get_peaks(fm)
                df_peaks = pd.DataFrame(peaks, columns=["center_freq", "peak_power", "bandwidth"])
                df_peaks.insert(0, "file", base_name)
                df_peaks.insert(1, "freq_hz", f)
                df_peaks.insert(2, "preset", preset)
                df_peaks.to_csv(preset_out / f"{base_name}_f{f}_peaks.csv", index=False)

                row = {
                    "file": base_name,
                    "freq_hz": f,
                    "preset": preset,
                    "fmin_fit": fmin_fit,
                    "fmax_fit": fmax_fit,
                    "r2": r2,
                    "error": err,
                    "offset": offset,
                    "knee": knee,
                    "exponent": exponent,
                    "n_channels": len(ch_names),
                    "channels": ",".join(ch_names),
                }
                summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.out_dir / f"{base_name}_specparam_summary.csv", index=False)
        return df_summary