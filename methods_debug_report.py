# methods_debug_report.py
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import textwrap

def _read_first_existing(run_dir: Path, candidates):
    for rel in candidates:
        p = run_dir / rel
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str, help="e.g. outputs/20260115_105801_dev")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="patient_id")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    # Try common dataset locations/names
    ds_path = _read_first_existing(run_dir, [
        "ml/features_wide.csv",
        "ml/dataset_wide.csv",
        "dataset_wide.csv",
        "features_wide.csv",
    ])

    print(f"[INFO] run_dir: {run_dir}")
    if ds_path is None:
        print("[ERROR] Could not find a dataset CSV (features_wide/dataset_wide).")
        return

    df = pd.read_csv(ds_path)
    print(f"[INFO] dataset: {ds_path}")
    print(f"[INFO] shape: {df.shape}")

    # Basic columns check
    for c in [args.label_col, args.group_col, "file", "timepoint"]:
        if c in df.columns:
            if c == args.label_col:
                vc = df[c].value_counts(dropna=False).to_dict()
                print(f"[INFO] label distribution ({args.label_col}): {vc}")
            elif c == args.group_col:
                print(f"[INFO] unique patients ({args.group_col}): {df[c].nunique(dropna=True)}")
            else:
                print(f"[INFO] column present: {c}")
        else:
            print(f"[WARN] column missing: {c}")

    # Missingness
    miss_frac = df.isna().mean()
    all_nan = miss_frac[miss_frac == 1.0].index.tolist()
    if all_nan:
        print(f"[WARN] all-NaN columns (n={len(all_nan)}):")
        print(textwrap.fill(", ".join(all_nan), width=100))
    top_miss = miss_frac.sort_values(ascending=False).head(20)
    print("\n[INFO] top-20 missingness columns:")
    for k, v in top_miss.items():
        print(f"  {k}: {v:.2f}")

    # Constant columns (excluding NaN-only)
    const_cols = []
    for c in df.columns:
        s = df[c]
        if s.dropna().empty:
            continue
        if s.dropna().nunique() == 1:
            const_cols.append(c)
    if const_cols:
        print(f"\n[WARN] constant columns (n={len(const_cols)}):")
        print(textwrap.fill(", ".join(const_cols[:200]), width=100))
        if len(const_cols) > 200:
            print("  ... (truncated)")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
