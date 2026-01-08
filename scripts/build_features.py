from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _base_from_name(name: str) -> str:
    # VEP02_1_power.csv -> VEP02_1
    name = name.replace(".csv", "")
    for suf in ["_power", "_plv_stim", "_plv_base", "_specparam_summary", "_specparam_summary_ALL"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _patient_id_from_file(file_stem: str) -> str:
    # VEP02_1 -> VEP02
    return str(file_stem).split("_")[0]


def _timepoint_from_file(file_stem: str) -> str:
    # VEP02_1 -> t0, VEP02_2 -> t1, VEP02_3 -> t2 (als het niet matcht -> "")
    m = re.search(r"_([123])$", str(file_stem))
    if not m:
        return ""
    return {"1": "t0", "2": "t1", "3": "t2"}[m.group(1)]


def _read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _power_features_for_freq(df_power: pd.DataFrame, stim_f: int, harms: int = 5) -> dict:
    out: dict = {}

    cF = _col(df_power, ["Frequency", "freq", "stim_freq"])
    cH = _col(df_power, ["Harmonic", "harmonic", "harm_freq"])
    cS = _col(df_power, ["Average_SNR", "snr", "SNR"])
    cP = _col(df_power, ["Average_PWR", "pwr", "PWR"])
    cB = _col(df_power, ["Average_BASE", "base", "BASE"])

    if not all([cF, cH, cS, cP, cB]):
        # niks bruikbaars
        for k in range(1, harms + 1):
            out[f"power_snr_h{k}"] = np.nan
            out[f"power_pwr_h{k}"] = np.nan
            out[f"power_base_h{k}"] = np.nan
        out["power_snr_mean_allharm"] = np.nan
        out["power_snr_max_allharm"] = np.nan
        out["power_snr_sum_allharm"] = np.nan
        return out

    sub = df_power[df_power[cF].astype(float).round().astype(int) == int(stim_f)].copy()

    snrs = []
    for k in range(1, harms + 1):
        harm_f = int(stim_f * k)
        hit = sub[sub[cH].astype(float).round().astype(int) == harm_f]
        if len(hit) >= 1:
            out[f"power_snr_h{k}"] = float(hit[cS].iloc[0])
            out[f"power_pwr_h{k}"] = float(hit[cP].iloc[0])
            out[f"power_base_h{k}"] = float(hit[cB].iloc[0])
        else:
            out[f"power_snr_h{k}"] = np.nan
            out[f"power_pwr_h{k}"] = np.nan
            out[f"power_base_h{k}"] = np.nan
        snrs.append(out[f"power_snr_h{k}"])

    snrs_arr = np.asarray(snrs, dtype=float)
    out["power_snr_mean_allharm"] = float(np.nanmean(snrs_arr)) if np.isfinite(np.nanmean(snrs_arr)) else np.nan
    out["power_snr_max_allharm"] = float(np.nanmax(snrs_arr)) if np.isfinite(np.nanmax(snrs_arr)) else np.nan
    out["power_snr_sum_allharm"] = float(np.nansum(snrs_arr)) if np.isfinite(np.nansum(snrs_arr)) else np.nan
    return out


def _plv_features_for_freq(df_plv_stim: pd.DataFrame, df_plv_base: pd.DataFrame, stim_f: int, harms: int = 5) -> dict:
    out: dict = {}

    def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None, str | None]:
        cF = _col(df, ["Frequency", "freq", "stim_freq"])
        cH = _col(df, ["Harmonic", "harmonic", "harm_freq"])
        cV = _col(df, ["PLV", "plv", "value"])
        return df, cF, cH, cV

    dfS, cFS, cHS, cVS = _prep(df_plv_stim)
    dfB, cFB, cHB, cVB = _prep(df_plv_base)

    # init NaNs
    for k in range(1, harms + 1):
        out[f"plv_stim_h{k}"] = np.nan
        out[f"plv_base_h{k}"] = np.nan
        out[f"plv_delta_h{k}"] = np.nan

    out["plv_stim_mean_allharm"] = np.nan
    out["plv_stim_max_allharm"] = np.nan
    out["plv_base_mean_allharm"] = np.nan
    out["plv_base_max_allharm"] = np.nan

    if not all([cFS, cHS, cVS]) or not all([cFB, cHB, cVB]):
        return out

    subS = dfS[dfS[cFS].astype(float).round().astype(int) == int(stim_f)].copy()
    subB = dfB[dfB[cFB].astype(float).round().astype(int) == int(stim_f)].copy()

    stim_vals = []
    base_vals = []
    for k in range(1, harms + 1):
        harm_f = int(stim_f * k)

        hs = subS[subS[cHS].astype(float).round().astype(int) == harm_f]
        hb = subB[subB[cHB].astype(float).round().astype(int) == harm_f]

        s = float(hs[cVS].iloc[0]) if len(hs) >= 1 else np.nan
        b = float(hb[cVB].iloc[0]) if len(hb) >= 1 else np.nan

        out[f"plv_stim_h{k}"] = s
        out[f"plv_base_h{k}"] = b
        out[f"plv_delta_h{k}"] = (s - b) if (np.isfinite(s) and np.isfinite(b)) else np.nan

        stim_vals.append(s)
        base_vals.append(b)

    stim_arr = np.asarray(stim_vals, dtype=float)
    base_arr = np.asarray(base_vals, dtype=float)

    out["plv_stim_mean_allharm"] = float(np.nanmean(stim_arr)) if np.isfinite(np.nanmean(stim_arr)) else np.nan
    out["plv_stim_max_allharm"] = float(np.nanmax(stim_arr)) if np.isfinite(np.nanmax(stim_arr)) else np.nan
    out["plv_base_mean_allharm"] = float(np.nanmean(base_arr)) if np.isfinite(np.nanmean(base_arr)) else np.nan
    out["plv_base_max_allharm"] = float(np.nanmax(base_arr)) if np.isfinite(np.nanmax(base_arr)) else np.nan
    return out


def _spec_features_for_freq(df_spec: pd.DataFrame, stim_f: int, preset: str) -> dict:
    out: dict = {}
    if df_spec.empty:
        out[f"spec_{preset}_r2"] = np.nan
        out[f"spec_{preset}_error"] = np.nan
        out[f"spec_{preset}_offset"] = np.nan
        out[f"spec_{preset}_exponent"] = np.nan
        return out

    cF = _col(df_spec, ["freq_hz", "freq", "Frequency"])
    cP = _col(df_spec, ["preset"])
    if not cF or not cP:
        out[f"spec_{preset}_r2"] = np.nan
        out[f"spec_{preset}_error"] = np.nan
        out[f"spec_{preset}_offset"] = np.nan
        out[f"spec_{preset}_exponent"] = np.nan
        return out

    sub = df_spec[(df_spec[cP].astype(str) == str(preset)) & (df_spec[cF].astype(float).round().astype(int) == int(stim_f))]
    if len(sub) < 1:
        out[f"spec_{preset}_r2"] = np.nan
        out[f"spec_{preset}_error"] = np.nan
        out[f"spec_{preset}_offset"] = np.nan
        out[f"spec_{preset}_exponent"] = np.nan
        return out

    row = sub.iloc[0]
    out[f"spec_{preset}_r2"] = float(row["r2"]) if "r2" in sub.columns else np.nan
    out[f"spec_{preset}_error"] = float(row["error"]) if "error" in sub.columns else np.nan
    out[f"spec_{preset}_offset"] = float(row["offset"]) if "offset" in sub.columns else np.nan
    out[f"spec_{preset}_exponent"] = float(row["exponent"]) if "exponent" in sub.columns else np.nan
    return out


def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return df_long

    id_cols = [c for c in ["file", "patient_id", "timepoint"] if c in df_long.columns]
    value_cols = [c for c in df_long.columns if c not in (id_cols + ["freq_hz"])]

    wide = df_long.pivot(index=id_cols, columns="freq_hz", values=value_cols)
    wide.columns = [f"{feat}_f{int(freq)}" for feat, freq in wide.columns]
    wide = wide.reset_index()

    # kolommen sorteren: file/patient_id/timepoint eerst
    first = [c for c in ["file", "patient_id", "timepoint"] if c in wide.columns]
    rest = [c for c in wide.columns if c not in first]
    return wide[first + rest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--harms", type=int, default=5)
    ap.add_argument("--presets", default="baseline,conservative,harmonics")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    feats_dir = run_dir / "features"
    power_dir = feats_dir / "power"
    plv_dir = feats_dir / "plv"
    spec_dir = feats_dir / "specparam"
    ml_dir = run_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    presets = [p.strip() for p in str(args.presets).split(",") if p.strip()]

    power_files = sorted(power_dir.glob("*_power.csv")) if power_dir.exists() else []
    plv_stim_files = sorted(plv_dir.glob("*_plv_stim.csv")) if plv_dir.exists() else []
    plv_base_files = sorted(plv_dir.glob("*_plv_base.csv")) if plv_dir.exists() else []
    spec_files = []
    if spec_dir.exists():
        spec_files = sorted(spec_dir.glob("*_specparam_summary*.csv"))

    bases = set()
    for p in power_files + plv_stim_files + plv_base_files + spec_files:
        bases.add(_base_from_name(p.name))
    bases = sorted(bases)

    rows = []

    for base in bases:
        df_power = _read_csv_safe(power_dir / f"{base}_power.csv")
        df_plv_stim = _read_csv_safe(plv_dir / f"{base}_plv_stim.csv")
        df_plv_base = _read_csv_safe(plv_dir / f"{base}_plv_base.csv")

        # specparam summary: pak de "meest complete" als er meerdere zijn
        df_spec = pd.DataFrame()
        cand = sorted(spec_dir.glob(f"{base}_specparam_summary*.csv")) if spec_dir.exists() else []
        if cand:
            # voorkeur: zonder _ALL of met? maakt niet uit, kies grootste
            cand = sorted(cand, key=lambda p: p.stat().st_size, reverse=True)
            df_spec = _read_csv_safe(cand[0])

        # bepaal stim freqs uit power/plv/spec
        freqs = set()

        cF_pow = _col(df_power, ["Frequency", "freq", "stim_freq"])
        if cF_pow:
            freqs |= set(df_power[cF_pow].dropna().astype(float).round().astype(int).tolist())

        cF_plv = _col(df_plv_stim, ["Frequency", "freq", "stim_freq"])
        if cF_plv:
            freqs |= set(df_plv_stim[cF_plv].dropna().astype(float).round().astype(int).tolist())

        if not df_spec.empty and "freq_hz" in df_spec.columns:
            freqs |= set(df_spec["freq_hz"].dropna().astype(float).round().astype(int).tolist())

        freqs = sorted([f for f in freqs if f > 0])
        if not freqs:
            continue

        pid = _patient_id_from_file(base)
        tp = _timepoint_from_file(base)

        for f in freqs:
            row = {"file": base, "patient_id": pid, "timepoint": tp, "freq_hz": int(f)}
            row.update(_power_features_for_freq(df_power, int(f), harms=args.harms))
            row.update(_plv_features_for_freq(df_plv_stim, df_plv_base, int(f), harms=args.harms))
            for preset in presets:
                row.update(_spec_features_for_freq(df_spec, int(f), preset))
            rows.append(row)

    df_long = pd.DataFrame(rows)
    df_wide = make_wide(df_long)

    df_long.to_csv(ml_dir / "features_long.csv", index=False)
    df_wide.to_csv(ml_dir / "features_wide.csv", index=False)

    print("[OK] Wrote:")
    print(f"  {ml_dir / 'features_long.csv'}")
    print(f"  {ml_dir / 'features_wide.csv'}")
    print(f"[INFO] Long shape: {df_long.shape} | Wide shape: {df_wide.shape}")


if __name__ == "__main__":
    main()
