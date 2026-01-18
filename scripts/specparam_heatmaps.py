from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_merged_csv(run_dir: Path) -> Path:
    # zoekt overal in run_dir naar specparam_long_merged.csv
    cands = list(run_dir.rglob("specparam_long_merged.csv"))
    if not cands:
        raise FileNotFoundError(f"Could not find specparam_long_merged.csv under: {run_dir}")
    # pak nieuwste
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def make_heatmap(df: pd.DataFrame, preset: str, out_png: Path):
    # filter preset + drop 0 Hz
    d = df.copy()
    if "preset" in d.columns:
        d = d[d["preset"].astype(str) == str(preset)]
    if "freq_hz" in d.columns:
        d = d[d["freq_hz"].fillna(-1).astype(float) > 0]

    # kies alleen numerieke “specparam” features (laat ids weg)
    drop_cols = {"file", "patient_id", "timepoint", "stim_hz", "label", "preset", "channels"}
    cols = [c for c in d.columns if c not in drop_cols]
    X = d[cols].select_dtypes(include=[np.number]).copy()

    # gooi kolommen weg die helemaal NaN zijn
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 2:
        print(f"[SKIP] preset={preset}: not enough numeric columns after cleaning (n={X.shape[1]})")
        return

    corr = X.corr(method="spearman")

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.title(f"SpecParam feature correlation (Spearman) – preset={preset}")
    plt.colorbar()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--presets", default="baseline,conservative,harmonics")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/figures/specparam")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    merged_csv = find_merged_csv(run_dir)
    df = pd.read_csv(merged_csv)

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures" / "specparam")

    for p in presets:
        out_png = out_dir / f"08_spec_corr_heatmap_{p}.png"
        make_heatmap(df, p, out_png)


if __name__ == "__main__":
    main()