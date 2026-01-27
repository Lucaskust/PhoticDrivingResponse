# scripts/longitudinal_t0_t2_summary.py
"""
Longitudinal T0 vs T2 summary report (paired, within-subject deltas).

Goal: produce *few* main-text-ready outputs:
  - SpecParam forest/lollipop plot: mean Δ(T2–T0) ± 95% CI for {r2,error,offset,exponent} across presets & f={6,10,20}
  - Power/PLV summary heatmap: mean Δ for a small set of summary features
  - Top-10 Δ tables (overall + per family)
  - Full delta summary table (optional, for appendix)

Run:
  python scripts/longitudinal_t0_t2_summary.py --run-dir outputs/20260115_105801_dev
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
ID_COLS_DEFAULT = ["file", "patient_id", "timepoint", "label", "trial"]

FAMILY_PREFIX = {
    "Power": "power_",
    "PLV": "plv_",
    "SpecParam": "spec_",
}

SPEC_RE = re.compile(
    r"^spec_(?P<preset>baseline|conservative|harmonics)_(?P<metric>r2|error|offset|exponent)_f(?P<f>\d+)$"
)

# Choose a minimal set of summary features for one heatmap figure
POWER_PLV_SUMMARY_CANDIDATES = [
    # Power SNR summaries
    "power_snr_mean_allharm_f6",
    "power_snr_mean_allharm_f10",
    "power_snr_mean_allharm_f20",
    "power_snr_max_allharm_f6",
    "power_snr_max_allharm_f10",
    "power_snr_max_allharm_f20",
    # Optional: stimulation power summaries (if you have them)
    # "power_pwr_h1_f6", "power_pwr_h1_f10", "power_pwr_h1_f20",

    # PLV summaries (only include if they exist in your table)
    "plv_stim_mean_allharm_f6",
    "plv_stim_mean_allharm_f10",
    "plv_stim_mean_allharm_f20",
    "plv_delta_mean_allharm_f6",
    "plv_delta_mean_allharm_f10",
    "plv_delta_mean_allharm_f20",
]


def _resolve_run_dir(run_dir: str) -> Path:
    p = Path(run_dir)
    if p.exists():
        return p
    # allow passing just run_id
    p2 = Path("outputs") / run_dir
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find run_dir: {run_dir} (also tried outputs/{run_dir})")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _infer_family(col: str) -> str:
    for fam, pref in FAMILY_PREFIX.items():
        if col.startswith(pref):
            return fam
    return "Other"


def _summary_stats(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise per feature:
      n, mean, sd, sem, ci95_low/high, dz, t, p
    """
    out = []
    try:
        from scipy.stats import ttest_1samp  # type: ignore
        have_scipy = True
    except Exception:
        have_scipy = False
        ttest_1samp = None  # type: ignore

    for c in delta_df.columns:
        x = pd.to_numeric(delta_df[c], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(x.size)
        if n < 2:
            continue
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        sem = float(sd / np.sqrt(n)) if sd > 0 else 0.0
        ci = 1.96 * sem
        ci_low = mean - ci
        ci_high = mean + ci
        dz = float(mean / sd) if sd > 0 else np.nan

        t = np.nan
        p = np.nan
        if have_scipy:
            res = ttest_1samp(x, popmean=0.0, nan_policy="omit")
            t = float(res.statistic)
            p = float(res.pvalue)

        out.append(
            {
                "feature": c,
                "family": _infer_family(c),
                "n_pairs": n,
                "mean_delta": mean,
                "sd_delta": sd,
                "sem_delta": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "cohen_dz": dz,
                "t_stat": t,
                "p_value": p,
            }
        )

    return pd.DataFrame(out).sort_values("mean_delta", ascending=False).reset_index(drop=True)


def _build_paired_delta(df_wide: pd.DataFrame, id_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Build paired delta table (T2 - T0) per patient_id.

    Returns:
      delta_df: index=patient_id, columns=features (numeric)
      meta: dict with counts
    """
    if "patient_id" not in df_wide.columns or "timepoint" not in df_wide.columns:
        raise ValueError("features_wide.csv must contain 'patient_id' and 'timepoint' columns")

    # Only T0 and T2 for longitudinal
    df = df_wide[df_wide["timepoint"].isin(["t0", "t2"])].copy()

    # Infer feature columns: everything except identifiers
    feat_cols = [c for c in df.columns if c not in set(id_cols)]
    # Keep numeric-ish columns
    # (Some may still be object; we'll coerce later.)
    feat_cols = [c for c in feat_cols if not c.endswith("_id")]

    # Handle potential duplicates: keep first (and report)
    dup = df.duplicated(subset=["patient_id", "timepoint"], keep=False)
    n_dup = int(dup.sum())
    if n_dup > 0:
        df = df.sort_values(["patient_id", "timepoint"]).drop_duplicates(["patient_id", "timepoint"], keep="first")

    # Pivot to patient_id × timepoint (columns become MultiIndex: (feature, timepoint))
    df_pt = df.set_index(["patient_id", "timepoint"])[feat_cols]
    df_w = df_pt.unstack("timepoint")  # columns: (feature, timepoint)

    # Select paired patients with both t0 and t2 present
    have_t0 = df_w.columns.get_level_values(1).isin(["t0"])
    have_t2 = df_w.columns.get_level_values(1).isin(["t2"])
    # The unstack always has both levels, but entries can be NaN; pairing is by existence of row in original
    # Determine patient availability via original df:
    pt_t0 = set(df[df["timepoint"] == "t0"]["patient_id"].unique().tolist())
    pt_t2 = set(df[df["timepoint"] == "t2"]["patient_id"].unique().tolist())
    paired_pts = sorted(list(pt_t0.intersection(pt_t2)))

    # Slice to paired patients
    df_w = df_w.loc[paired_pts]

    # Extract t0 and t2 feature matrices (align columns)
    t0 = df_w.xs("t0", level=1, axis=1)
    t2 = df_w.xs("t2", level=1, axis=1)
    common = sorted(list(set(t0.columns).intersection(set(t2.columns))))
    t0 = t0[common]
    t2 = t2[common]

    # Vectorized delta
    delta = (t2.astype(float) - t0.astype(float))

    # Drop all-NaN features
    delta = delta.dropna(axis=1, how="all").copy()

    meta = {
        "n_rows_total": int(len(df_wide)),
        "n_t0": int((df_wide["timepoint"] == "t0").sum()) if "timepoint" in df_wide.columns else np.nan,
        "n_t2": int((df_wide["timepoint"] == "t2").sum()) if "timepoint" in df_wide.columns else np.nan,
        "n_patients_total": int(df_wide["patient_id"].nunique()),
        "n_patients_t0": int(len(pt_t0)),
        "n_patients_t2": int(len(pt_t2)),
        "n_pairs": int(len(paired_pts)),
        "n_duplicate_patient_timepoints_removed": n_dup,
        "n_features_delta": int(delta.shape[1]),
    }
    return delta, meta


def _plot_specparam_forest(stats: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    """
    Forest/lollipop plot for SpecParam metrics only (r2/error/offset/exponent), f in {6,10,20}.
    Returns the filtered table used for plotting.
    """
    rows = []
    for _, r in stats.iterrows():
        feat = r["feature"]
        m = SPEC_RE.match(str(feat))
        if not m:
            continue
        f = int(m.group("f"))
        if f not in (6, 10, 20):
            continue
        rows.append(
            {
                **r.to_dict(),
                "preset": m.group("preset"),
                "metric": m.group("metric"),
                "stim_f": f,
                "label": f"{m.group('preset')} | {m.group('metric')} | {f} Hz",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        # still create an empty placeholder message via exception
        raise RuntimeError("No SpecParam features found for forest plot. Expected columns like spec_<preset>_<metric>_f6.")

    # Order: preset -> metric -> frequency
    preset_order = ["baseline", "conservative", "harmonics"]
    metric_order = ["r2", "error", "offset", "exponent"]
    df["preset"] = pd.Categorical(df["preset"], categories=preset_order, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
    df = df.sort_values(["preset", "metric", "stim_f"]).reset_index(drop=True)

    y = np.arange(len(df))
    x = df["mean_delta"].to_numpy(float)
    lo = df["ci95_low"].to_numpy(float)
    hi = df["ci95_high"].to_numpy(float)
    xerr = np.vstack([x - lo, hi - x])

    fig_h = max(6, 0.22 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=9)
    ax.set_xlabel("Mean paired Δ (T2 - T0) with 95% CI")
    ax.set_title("SpecParam longitudinal summary (paired T0 vs T2)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


def _plot_power_plv_heatmap(stats: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    """
    Small heatmap of mean deltas for selected Power/PLV summary features.
    """
    # Keep only candidates that exist
    keep = [c for c in POWER_PLV_SUMMARY_CANDIDATES if c in set(stats["feature"].tolist())]
    if len(keep) == 0:
        raise RuntimeError("None of the predefined Power/PLV summary features were found in the delta table.")

    sub = stats[stats["feature"].isin(keep)].copy()

    # Order rows by family then name
    sub["family"] = sub["feature"].apply(_infer_family)
    sub = sub.sort_values(["family", "feature"]).reset_index(drop=True)

    # Build a matrix with columns f6/f10/f20 (if present in names)
    def _col_from_name(name: str) -> str:
        if name.endswith("_f6"):
            return "6 Hz"
        if name.endswith("_f10"):
            return "10 Hz"
        if name.endswith("_f20"):
            return "20 Hz"
        return "NA"

    sub["cond"] = sub["feature"].apply(_col_from_name)
    sub = sub[sub["cond"].isin(["6 Hz", "10 Hz", "20 Hz"])].copy()

    # Create pivot: rows = base feature name without suffix, cols = condition
    def _base_name(name: str) -> str:
        return re.sub(r"_f(6|10|20)$", "", name)

    sub["base"] = sub["feature"].apply(_base_name)

    piv = sub.pivot_table(index=["family", "base"], columns="cond", values="mean_delta", aggfunc="mean")
    piv = piv.reindex(columns=["6 Hz", "10 Hz", "20 Hz"])

    # Plot
    mat = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * mat.shape[0] + 2)))
    im = ax.imshow(mat, aspect="auto")  # default colormap is fine
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns.tolist())
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([f"{idx[0]} | {idx[1]}" for idx in piv.index], fontsize=9)

    ax.set_title("Longitudinal mean Δ (T2 - T0) for selected Power/PLV summaries")
    ax.set_xlabel("Stimulation condition")
    ax.set_ylabel("Feature (family | name)")
    fig.colorbar(im, ax=ax, label="Mean Δ")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_df = piv.reset_index()
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory (e.g., outputs/20260115_105801_dev) or run_id")
    ap.add_argument("--wide-path", default="", help="Optional override path to features_wide.csv")
    ap.add_argument("--out-tag", default="t0_t2_summary", help="Tag for output folder name")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = run_dir / "longitudinal_summary" / f"{ts}_{args.out_tag}"
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    _safe_mkdir(fig_dir)
    _safe_mkdir(tab_dir)

    wide_path = Path(args.wide_path) if args.wide_path else (run_dir / "ml" / "features_wide.csv")
    if not wide_path.exists():
        raise FileNotFoundError(f"Could not find wide feature table at: {wide_path}")

    df_wide = pd.read_csv(wide_path)

    # Build paired delta
    delta_df, meta = _build_paired_delta(df_wide, id_cols=ID_COLS_DEFAULT)

    # Write meta
    meta_path = out_root / "meta_counts.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    # Summaries
    stats = _summary_stats(delta_df)
    stats.to_csv(tab_dir / "delta_summary_all_features.csv", index=False)

    # Top-10 overall by |mean_delta|
    stats["abs_mean_delta"] = stats["mean_delta"].abs()
    top10_all = stats.sort_values("abs_mean_delta", ascending=False).head(10).drop(columns=["abs_mean_delta"])
    top10_all.to_csv(tab_dir / "top10_delta_overall.csv", index=False)

    # Top-10 per family
    for fam in ["Power", "PLV", "SpecParam", "Other"]:
        sub = stats[stats["family"] == fam].copy()
        if sub.empty:
            continue
        sub["abs_mean_delta"] = sub["mean_delta"].abs()
        top10 = sub.sort_values("abs_mean_delta", ascending=False).head(10).drop(columns=["abs_mean_delta"])
        top10.to_csv(tab_dir / f"top10_delta_{fam.lower()}.csv", index=False)

    # Figures
    # (1) SpecParam forest plot
    spec_png = fig_dir / "summary_delta_specparam_forest.png"
    try:
        spec_used = _plot_specparam_forest(stats, spec_png)
        spec_used.to_csv(tab_dir / "specparam_forest_table.csv", index=False)
    except Exception as e:
        with open(out_root / "WARN_specparam_plot.txt", "w", encoding="utf-8") as f:
            f.write(str(e) + "\n")

    # (2) Power/PLV heatmap
    heat_png = fig_dir / "summary_delta_power_plv_heatmap.png"
    try:
        heat_tbl = _plot_power_plv_heatmap(stats, heat_png)
        heat_tbl.to_csv(tab_dir / "power_plv_heatmap_table.csv", index=False)
    except Exception as e:
        with open(out_root / "WARN_power_plv_plot.txt", "w", encoding="utf-8") as f:
            f.write(str(e) + "\n")

    print("[OK] Longitudinal T0 vs T2 summary written to:")
    print(f"  {out_root}")
    print("[OK] Key outputs:")
    print(f"  {fig_dir / 'summary_delta_specparam_forest.png'}")
    print(f"  {fig_dir / 'summary_delta_power_plv_heatmap.png'}")
    print(f"  {tab_dir / 'top10_delta_overall.csv'}")
    print(f"  {tab_dir / 'delta_summary_all_features.csv'}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
