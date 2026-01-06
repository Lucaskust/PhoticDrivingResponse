import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


TIME_TO_SUFFIX = {"t0": "_1", "t1": "_2", "t2": "_3"}


def extract_patient_id(file_stem: str) -> str:
    """
    From 'VEP02_3' -> 'VEP02'
    From 'VEP02'   -> 'VEP02'
    """
    s = str(file_stem).strip().upper()
    m = re.match(r"^(VEP\d+)", s)
    return m.group(1) if m else ""


def load_labels(labels_path: Path) -> pd.DataFrame:
    if labels_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(labels_path)
    else:
        df = pd.read_csv(labels_path)

    # normalize column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"file", "t1", "t2", "t2_t1"} 
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels file missing columns: {missing}. Found: {list(df.columns)}")

    # normalize patient id column
    df["patient_id"] = df["file"].astype(str).str.strip().str.upper().apply(extract_patient_id)

    # convert x -> NaN, then numeric
    for c in ["t1", "t2", "t2_t1"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"x": np.nan, "nan": np.nan, "none": np.nan})
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["patient_id", "t1", "t2", "t2_t1"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help=r"outputs\<run_id>")
    ap.add_argument("--labels", required=True, help="xlsx/csv with columns: file,t1,t2,t2_t1 (0/1 or x)")
    ap.add_argument("--timepoint", default="t2", choices=["t1", "t2"], help="Which session’s files to keep (_2 or _3)")
    ap.add_argument("--label-col", default="t2_t1", choices=["t1", "t2", "t2_t1"],
                    help="Which label column to use. Recommended: t2_t1")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ml_dir = run_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    wide_path = ml_dir / "features_wide.csv"
    long_path = ml_dir / "features_long.csv"
    if not wide_path.exists() or not long_path.exists():
        raise FileNotFoundError("Run build_features.py first (features_wide.csv / features_long.csv missing).")

    df_wide = pd.read_csv(wide_path)
    df_long = pd.read_csv(long_path)

    # add patient_id
    df_wide["patient_id"] = df_wide["file"].apply(extract_patient_id)
    df_long["patient_id"] = df_long["file"].apply(extract_patient_id)

    # keep only this session’s files
    suf = TIME_TO_SUFFIX[args.timepoint]
    df_wide = df_wide[df_wide["file"].astype(str).str.endswith(suf)].copy()
    df_long = df_long[df_long["file"].astype(str).str.endswith(suf)].copy()

    # load labels & choose label column
    lab = load_labels(Path(args.labels))
    lab = lab[["patient_id", args.label_col]].rename(columns={args.label_col: "label"})

    # merge
    wide = df_wide.merge(lab, on="patient_id", how="left")
    long = df_long.merge(lab, on="patient_id", how="left")

    # log missing labels
    wide_missing = wide[wide["label"].isna()][["file", "patient_id"]].copy()
    long_missing = long[long["label"].isna()][["file", "patient_id", "freq_hz"]].copy()

    # drop missing labels
    wide = wide.dropna(subset=["label"]).copy()
    long = long.dropna(subset=["label"]).copy()

    wide["label"] = wide["label"].astype(int)
    long["label"] = long["label"].astype(int)

    out_wide = ml_dir / f"dataset_{args.timepoint}_{args.label_col}_wide.csv"
    out_long = ml_dir / f"dataset_{args.timepoint}_{args.label_col}_long.csv"
    wide.to_csv(out_wide, index=False)
    long.to_csv(out_long, index=False)

    if len(wide_missing):
        wide_missing.to_csv(ml_dir / f"missing_labels_{args.timepoint}_{args.label_col}_wide.csv", index=False)
    if len(long_missing):
        long_missing.to_csv(ml_dir / f"missing_labels_{args.timepoint}_{args.label_col}_long.csv", index=False)

    # quick class balance
    counts = wide["label"].value_counts(dropna=False).to_dict()

    print(f"[OK] Wrote:\n  {out_wide}\n  {out_long}")
    print(f"[INFO] wide shape: {wide.shape} | long shape: {long.shape}")
    print(f"[INFO] missing labels: wide={len(wide_missing)} long={len(long_missing)}")
    print(f"[INFO] label counts (wide): {counts}")


if __name__ == "__main__":
    main()
