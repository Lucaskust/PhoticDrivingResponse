import sys
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

# Ensure repo root import works when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stem_base_from_power(path: Path) -> str:
    # VEP02_1_power.pkl -> VEP02_1
    return path.name.replace("_power.pkl", "")


def _stem_base_from_plv(path: Path, kind: str) -> str:
    # VEP02_1_plv_stim.pkl -> VEP02_1
    return path.name.replace(f"_plv_{kind}.pkl", "")


def _safe_read_pickle(p: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(p)
    except Exception as e:
        raise RuntimeError(f"Failed to read pickle: {p} ({e})")


def build_power_long(run_dir: Path, max_harm: int = 5) -> pd.DataFrame:
    power_dir = run_dir / "features" / "power"
    files = sorted(power_dir.glob("*_power.pkl"))
    rows = []

    for f in files:
        base = _stem_base_from_power(f)
        df = _safe_read_pickle(f)

        if "Frequency" not in df.columns or "Harmonic" not in df.columns:
            continue

        # stim freqs only (exclude 0 baseline rows)
        freqs = sorted([int(x) for x in df["Frequency"].dropna().unique() if int(x) != 0])

        for freq in freqs:
            sub = df[df["Frequency"] == freq].copy()

            row = {"file": base, "freq_hz": int(freq)}

            # harmonic features (Average_SNR / PWR / BASE)
            for k in range(1, max_harm + 1):
                h = int(freq * k)
                hit = sub[sub["Harmonic"] == h]
                if len(hit) == 1:
                    row[f"power_snr_h{k}"] = float(hit["Average_SNR"].iloc[0])
                    row[f"power_pwr_h{k}"] = float(hit["Average_PWR"].iloc[0])
                    row[f"power_base_h{k}"] = float(hit["Average_BASE"].iloc[0])
                else:
                    row[f"power_snr_h{k}"] = np.nan
                    row[f"power_pwr_h{k}"] = np.nan
                    row[f"power_base_h{k}"] = np.nan

            # aggregates across ALL harmonics in df for this freq
            if "Average_SNR" in sub.columns:
                row["power_snr_mean_allharm"] = float(np.nanmean(sub["Average_SNR"].to_numpy()))
                row["power_snr_max_allharm"] = float(np.nanmax(sub["Average_SNR"].to_numpy()))
                row["power_snr_sum_allharm"] = float(np.nansum(sub["Average_SNR"].to_numpy()))
            else:
                row["power_snr_mean_allharm"] = np.nan
                row["power_snr_max_allharm"] = np.nan
                row["power_snr_sum_allharm"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_plv_long(run_dir: Path, max_harm: int = 5) -> pd.DataFrame:
    plv_dir = run_dir / "features" / "plv"
    stim_files = sorted(plv_dir.glob("*_plv_stim.pkl"))
    rows = []

    for stim_path in stim_files:
        base = _stem_base_from_plv(stim_path, "stim")
        base_path = plv_dir / f"{base}_plv_base.pkl"
        if not base_path.exists():
            continue

        stim = _safe_read_pickle(stim_path)
        base_df = _safe_read_pickle(base_path)

        if "Frequency" not in stim.columns or "Harmonic" not in stim.columns or "mean_plv" not in stim.columns:
            continue

        freqs = sorted([int(x) for x in stim["Frequency"].dropna().unique() if int(x) != 0])

        for freq in freqs:
            ssub = stim[stim["Frequency"] == freq].copy()
            bsub = base_df[base_df["Frequency"] == freq].copy()

            row = {"file": base, "freq_hz": int(freq)}

            for k in range(1, max_harm + 1):
                h = int(freq * k)

                s_hit = ssub[ssub["Harmonic"] == h]
                b_hit = bsub[bsub["Harmonic"] == h]

                s_val = float(s_hit["mean_plv"].iloc[0]) if len(s_hit) == 1 else np.nan
                b_val = float(b_hit["mean_plv"].iloc[0]) if len(b_hit) == 1 else np.nan

                row[f"plv_stim_h{k}"] = s_val
                row[f"plv_base_h{k}"] = b_val
                row[f"plv_delta_h{k}"] = s_val - b_val if np.isfinite(s_val) and np.isfinite(b_val) else np.nan

            # aggregates across all harmonics available for this freq
            row["plv_stim_mean_allharm"] = float(np.nanmean(ssub["mean_plv"].to_numpy())) if len(ssub) else np.nan
            row["plv_stim_max_allharm"] = float(np.nanmax(ssub["mean_plv"].to_numpy())) if len(ssub) else np.nan
            row["plv_base_mean_allharm"] = float(np.nanmean(bsub["mean_plv"].to_numpy())) if len(bsub) else np.nan
            row["plv_base_max_allharm"] = float(np.nanmax(bsub["mean_plv"].to_numpy())) if len(bsub) else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_specparam_long_and_baseline(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - spec_long: indexed by (file, freq_hz) with columns spec_<preset>_<param>
      - spec0_file: indexed by file only (freq_hz==0), columns spec0_<preset>_<param>
    """
    spec_dir = run_dir / "features" / "specparam"
    summary_files = sorted(spec_dir.glob("*_specparam_summary_ALL.csv"))
    if not summary_files:
        # fallback to *_specparam_summary.csv if you didn't save _ALL
        summary_files = sorted(spec_dir.glob("*_specparam_summary.csv"))

    if not summary_files:
        return pd.DataFrame(), pd.DataFrame()

    dfs = []
    for p in summary_files:
        df = pd.read_csv(p)
        # normalize types
        if "freq_hz" in df.columns:
            df["freq_hz"] = df["freq_hz"].astype(int)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Keep only core cols we want
    keep = [c for c in ["file", "freq_hz", "preset", "r2", "error", "offset", "knee", "exponent"] if c in all_df.columns]
    all_df = all_df[keep].copy()

    # Long (freq != 0)
    spec_long_raw = all_df[all_df["freq_hz"] != 0].copy()
    if len(spec_long_raw):
        spec_long = spec_long_raw.pivot_table(
            index=["file", "freq_hz"],
            columns="preset",
            values=["r2", "error", "offset", "knee", "exponent"],
            aggfunc="first",
        )
        spec_long.columns = [f"spec_{preset}_{param}" for (param, preset) in spec_long.columns]
        spec_long = spec_long.reset_index()
    else:
        spec_long = pd.DataFrame()

    # Baseline per file (freq == 0)
    spec0_raw = all_df[all_df["freq_hz"] == 0].copy()
    if len(spec0_raw):
        spec0 = spec0_raw.pivot_table(
            index=["file"],
            columns="preset",
            values=["r2", "error", "offset", "knee", "exponent"],
            aggfunc="first",
        )
        spec0.columns = [f"spec0_{preset}_{param}" for (param, preset) in spec0.columns]
        spec0 = spec0.reset_index()
    else:
        spec0 = pd.DataFrame()

    return spec_long, spec0


def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long table (file, freq_hz, features...) into wide table: 1 row per file.
    """
    if df_long.empty:
        return df_long

    id_cols = ["file", "freq_hz"]
    feat_cols = [c for c in df_long.columns if c not in id_cols]

    wide = df_long.pivot(index="file", columns="freq_hz", values=feat_cols)
    # Flatten columns: (feature, freq) -> feature_f6
    wide.columns = [f"{feat}_f{int(freq)}" for (feat, freq) in wide.columns]
    wide = wide.reset_index()
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to outputs/<run_id> created by run_pipeline.py")
    ap.add_argument("--max-harm", type=int, default=5, help="How many harmonics (h1..hK) to include as explicit features")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-modality long tables
    df_power = build_power_long(run_dir, max_harm=args.max_harm)
    df_plv = build_plv_long(run_dir, max_harm=args.max_harm)
    df_spec_long, df_spec0 = build_specparam_long_and_baseline(run_dir)

    # Merge long
    # Start from outer merge of power + plv (keeps whichever exists)
    df_long = pd.merge(df_power, df_plv, on=["file", "freq_hz"], how="outer")

    if not df_spec_long.empty:
        df_long = pd.merge(df_long, df_spec_long, on=["file", "freq_hz"], how="left")

    # Add baseline spec0 (file-level) into long too (repeat per freq)
    if not df_spec0.empty:
        df_long = pd.merge(df_long, df_spec0, on=["file"], how="left")

    # Sort
    if not df_long.empty:
        df_long = df_long.sort_values(["file", "freq_hz"]).reset_index(drop=True)

    # Save long
    long_csv = out_dir / "features_long.csv"
    df_long.to_csv(long_csv, index=False)

    # Wide
    df_wide = make_wide(df_long)

    wide_csv = out_dir / "features_wide.csv"
    df_wide.to_csv(wide_csv, index=False)

    # Also save pickles for convenience
    df_long.to_pickle(out_dir / "features_long.pkl")
    df_wide.to_pickle(out_dir / "features_wide.pkl")

    print(f"[OK] Wrote:\n  {long_csv}\n  {wide_csv}")
    print(f"[INFO] Long shape: {df_long.shape} | Wide shape: {df_wide.shape}")


if __name__ == "__main__":
    main()
