# scripts/specparam_report.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn is optional (maar handig voor spec-only testjes)
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    SimpleImputer = StandardScaler = Pipeline = LogisticRegression = None


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_toml(path: Path) -> dict:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _infer_spec_summary_files(run_dir: Path) -> list[Path]:
    # supports both *_specparam_summary.csv and *_specparam_summary_ALL.csv etc
    feat_dir = run_dir / "features" / "specparam"
    if not feat_dir.exists():
        return []
    cands = sorted(feat_dir.glob("*_specparam_summary*.csv"))
    # filter out "peaks" tables etc if those exist
    cands = [p for p in cands if "peaks" not in p.name.lower() and "aperiodic" not in p.name.lower()]
    return cands


def _load_long_specparam(run_dir: Path) -> pd.DataFrame:
    files = _infer_spec_summary_files(run_dir)
    if not files:
        raise FileNotFoundError(f"No specparam summary CSVs found in {run_dir/'features/specparam'}")

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            # ensure expected cols
            for c in ["file", "freq_hz", "preset", "r2", "error", "offset", "exponent"]:
                if c not in df.columns:
                    # some older versions may differ; skip
                    pass
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise RuntimeError("Found specparam summary files but none could be read.")

    out = pd.concat(dfs, ignore_index=True)

    # normalize types
    if "freq_hz" in out.columns:
        out["freq_hz"] = pd.to_numeric(out["freq_hz"], errors="coerce")
    for c in ["r2", "error", "offset", "knee", "exponent", "fmin_fit", "fmax_fit"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # drop garbage rows
    out = out.dropna(subset=["file", "preset"], how="any")
    return out


def _load_dataset_meta(dataset_csv: Path, file_col: str, patient_col: str, time_col: str, label_col: str | None):
    df = pd.read_csv(dataset_csv)
    if file_col not in df.columns:
        raise ValueError(f"file_col='{file_col}' not found in dataset. Found: {list(df.columns)[:20]}...")
    for c in [patient_col, time_col]:
        if c not in df.columns:
            raise ValueError(f"'{c}' not found in dataset. Found: {list(df.columns)[:20]}...")

    meta_cols = [file_col, patient_col, time_col]
    if label_col and label_col in df.columns:
        meta_cols.append(label_col)

    meta = df[meta_cols].copy()
    meta = meta.drop_duplicates(subset=[file_col])
    return meta, df


def _merge_long_with_meta(df_long: pd.DataFrame, meta: pd.DataFrame, file_col_meta: str, file_col_long: str = "file"):
    meta2 = meta.rename(columns={file_col_meta: file_col_long})
    out = df_long.merge(meta2, on=file_col_long, how="left")
    return out


def _filter_timepoints(df: pd.DataFrame, time_col: str, tps: list[str]) -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    return df[df[time_col].astype(str).isin(tps)].copy()


def _counts_table(df: pd.DataFrame, time_col: str, label_col: str | None, patient_col: str) -> pd.DataFrame:
    cols = [time_col]
    if label_col and label_col in df.columns:
        cols.append(label_col)

    # counts per row (files)
    g = df.groupby(cols, dropna=False).size().reset_index(name="n_files")
    # unique patients
    gp = df.groupby(cols, dropna=False)[patient_col].nunique().reset_index(name="n_patients")
    out = g.merge(gp, on=cols, how="left")
    return out.sort_values(cols)


def _save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def _plot_counts(counts: pd.DataFrame, out_png: Path, time_col: str, label_col: str | None):
    plt.figure(figsize=(8, 4))
    if label_col and label_col in counts.columns:
        # stacked-ish bars: one bar per timepoint+label as separate categories
        x = [f"{r[time_col]}|{r[label_col]}" for _, r in counts.iterrows()]
        y = counts["n_files"].values
        plt.bar(range(len(x)), y)
        plt.xticks(range(len(x)), x, rotation=45, ha="right")
        plt.ylabel("n files")
        plt.title("Sample counts (files) per timepoint/label")
    else:
        x = counts[time_col].astype(str).values
        y = counts["n_files"].values
        plt.bar(range(len(x)), y)
        plt.xticks(range(len(x)), x)
        plt.ylabel("n files")
        plt.title("Sample counts (files) per timepoint")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _summary_stats_long(df_long: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    keep = [c for c in group_cols if c in df_long.columns]
    for m in metrics:
        if m not in df_long.columns:
            raise ValueError(f"Metric '{m}' not found in long specparam DF.")
    g = df_long.groupby(keep, dropna=False)
    rows = []
    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(keep, keys))
        row["n"] = len(sub)
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _jitter(x, scale=0.08):
    return x + np.random.uniform(-scale, scale, size=len(np.atleast_1d(x)))


def _plot_metric_by_freq(df: pd.DataFrame, out_png: Path, metric: str, preset: str, time_col: str):
    sub = df[df["preset"].astype(str) == str(preset)].copy()
    if sub.empty:
        return

    freqs = sorted(sub["freq_hz"].dropna().unique())
    tps = sorted(sub[time_col].dropna().astype(str).unique()) if time_col in sub.columns else ["all"]

    plt.figure(figsize=(9, 4))
    for j, tp in enumerate(tps):
        ss = sub[sub[time_col].astype(str) == tp] if time_col in sub.columns else sub
        xs_all, ys_all = [], []
        for i, f in enumerate(freqs):
            vals = pd.to_numeric(ss.loc[ss["freq_hz"] == f, metric], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            x0 = i + (j - (len(tps)-1)/2)*0.18
            xs = _jitter(np.full(len(vals), x0), 0.06)
            ys = vals
            plt.scatter(xs, ys, s=18, alpha=0.7, label=tp if i == 0 else None)
            # mean marker
            plt.scatter([x0], [np.mean(ys)], s=60, marker="D")
            xs_all.append(x0); ys_all.append(np.mean(ys))

    plt.xticks(range(len(freqs)), [str(int(f)) for f in freqs])
    plt.xlabel("freq_hz")
    plt.ylabel(metric)
    plt.title(f"{metric} by freq | preset={preset}")
    if len(tps) > 1:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _spaghetti(df: pd.DataFrame, out_png: Path, metric: str, preset: str, time_col: str, file_col: str = "file"):
    sub = df[df["preset"].astype(str) == str(preset)].copy()
    if sub.empty:
        return

    freqs = sorted(sub["freq_hz"].dropna().unique())
    if len(freqs) < 2:
        return

    # ensure sorted per file
    plt.figure(figsize=(9, 4))
    if time_col in sub.columns:
        tps = sorted(sub[time_col].dropna().astype(str).unique())
    else:
        tps = ["all"]

    # per file lines
    for tp in tps:
        ss = sub[sub[time_col].astype(str) == tp] if time_col in sub.columns else sub
        for f_id, g in ss.groupby(file_col):
            g2 = g.sort_values("freq_hz")
            xs = g2["freq_hz"].values
            ys = pd.to_numeric(g2[metric], errors="coerce").values
            if np.all(np.isnan(ys)):
                continue
            plt.plot(xs, ys, alpha=0.25)

        # mean line
        means = []
        for f in freqs:
            vals = pd.to_numeric(ss.loc[ss["freq_hz"] == f, metric], errors="coerce").dropna().values
            means.append(np.mean(vals) if len(vals) else np.nan)
        plt.plot(freqs, means, linewidth=3, label=f"mean {tp}")

    plt.xlabel("freq_hz")
    plt.ylabel(metric)
    plt.title(f"Overlay (spaghetti) | {metric} vs freq | preset={preset}")
    if len(tps) > 1:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _effect_size_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    # Cohen's d (unpaired)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa = np.var(a, ddof=1)
    sb = np.var(b, ddof=1)
    sp = np.sqrt(((len(a)-1)*sa + (len(b)-1)*sb) / (len(a)+len(b)-2))
    if sp == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / sp  # b - a


def _unpaired_effects(df_long: pd.DataFrame, time_col: str, t0: str, t2: str) -> pd.DataFrame:
    # effects per (preset,freq,metric): timepoint t2 - t0
    rows = []
    for (preset, freq), sub in df_long.groupby(["preset", "freq_hz"], dropna=False):
        s0 = sub[sub[time_col].astype(str) == t0] if time_col in sub.columns else pd.DataFrame()
        s2 = sub[sub[time_col].astype(str) == t2] if time_col in sub.columns else pd.DataFrame()
        if s0.empty or s2.empty:
            continue
        for metric in ["r2", "error", "offset", "exponent"]:
            if metric not in sub.columns:
                continue
            a = pd.to_numeric(s0[metric], errors="coerce").values
            b = pd.to_numeric(s2[metric], errors="coerce").values
            d = _effect_size_unpaired(a, b)
            rows.append({
                "preset": preset,
                "freq_hz": freq,
                "metric": metric,
                "mean_t0": float(np.nanmean(a)),
                "mean_t2": float(np.nanmean(b)),
                "delta_t2_minus_t0": float(np.nanmean(b) - np.nanmean(a)),
                "cohen_d": float(d) if d == d else np.nan,
                "n_t0": int(np.sum(~np.isnan(a))),
                "n_t2": int(np.sum(~np.isnan(b))),
            })
    out = pd.DataFrame(rows)
    # rank by |d|
    if not out.empty:
        out["abs_d"] = out["cohen_d"].abs()
        out = out.sort_values("abs_d", ascending=False)
    return out


def _plot_top_effects(effects: pd.DataFrame, out_png: Path, top_k: int = 15):
    if effects.empty:
        return
    sub = effects.head(top_k).copy()
    labels = [f"{r.preset}|f{int(r.freq_hz)}|{r.metric}" for r in sub.itertuples()]
    vals = sub["cohen_d"].values
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(vals)), vals)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right")
    plt.ylabel("Cohen's d (t2 - t0)")
    plt.title(f"Top {top_k} unpaired effects (exploratory)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _wide_spec_features(df_dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    spec_cols = [c for c in df_dataset.columns if c.startswith("spec_")]
    return df_dataset, spec_cols


def _paired_deltas(df_dataset: pd.DataFrame, patient_col: str, time_col: str, file_col: str, t0: str, t2: str) -> pd.DataFrame:
    # paired per patient: delta = t2 - t0 for each spec feature
    _, spec_cols = _wide_spec_features(df_dataset)
    need = [patient_col, time_col, file_col] + spec_cols
    for c in [patient_col, time_col, file_col]:
        if c not in df_dataset.columns:
            raise ValueError(f"'{c}' missing in dataset.")
    if not spec_cols:
        raise ValueError("No spec_* columns found in dataset.")

    df = df_dataset[need].copy()
    # keep only t0 and t2
    df = df[df[time_col].astype(str).isin([t0, t2])].copy()

    # pivot per patient: two rows (t0,t2) -> paired delta
    # If there are duplicates per patient/timepoint, we average them first.
    agg = df.groupby([patient_col, time_col], dropna=False)[spec_cols].mean().reset_index()
    piv0 = agg[agg[time_col].astype(str) == t0].set_index(patient_col)
    piv2 = agg[agg[time_col].astype(str) == t2].set_index(patient_col)

    common = sorted(set(piv0.index) & set(piv2.index))
    if len(common) == 0:
        return pd.DataFrame()

    d0 = piv0.loc[common, spec_cols]
    d2 = piv2.loc[common, spec_cols]
    delta = d2.values - d0.values
    out = pd.DataFrame(delta, columns=spec_cols)
    out.insert(0, patient_col, common)
    return out


def _rank_paired(delta_df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    spec_cols = [c for c in delta_df.columns if c != patient_col]
    rows = []
    for c in spec_cols:
        v = pd.to_numeric(delta_df[c], errors="coerce").values
        v = v[~np.isnan(v)]
        if len(v) < 2:
            continue
        mean = np.mean(v)
        sd = np.std(v, ddof=1)
        score = np.abs(mean) / sd if sd > 0 else np.nan
        rows.append({"feature": c, "mean_delta": float(mean), "std_delta": float(sd), "abs_mean_over_sd": float(score), "n": int(len(v))})
    out = pd.DataFrame(rows).sort_values("abs_mean_over_sd", ascending=False)
    return out


def _plot_top_paired(rank_df: pd.DataFrame, out_png: Path, top_k: int = 15):
    if rank_df.empty:
        return
    sub = rank_df.head(top_k).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sub)), sub["mean_delta"].values)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(len(sub)), sub["feature"].values, rotation=60, ha="right")
    plt.ylabel("mean (t2 - t0)")
    plt.title(f"Top {top_k} paired deltas (patients with both t0 & t2)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _corr_heatmap(df_dataset: pd.DataFrame, out_png: Path):
    _, spec_cols = _wide_spec_features(df_dataset)
    if len(spec_cols) < 3:
        return
    X = df_dataset[spec_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    # simple impute to compute corr
    X = X.fillna(X.median(numeric_only=True))
    C = np.corrcoef(X.values.T)
    plt.figure(figsize=(10, 8))
    plt.imshow(C, aspect="auto")
    plt.colorbar(label="corr")
    plt.title("Correlation heatmap (spec_* features)")
    plt.xticks(range(len(spec_cols)), spec_cols, rotation=90, fontsize=6)
    plt.yticks(range(len(spec_cols)), spec_cols, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _spec_only_classifier_timepoint(df_dataset: pd.DataFrame, out_dir: Path, patient_col: str, time_col: str, t0: str, t2: str):
    if Pipeline is None:
        return None

    _, spec_cols = _wide_spec_features(df_dataset)
    if not spec_cols:
        return None

    df = df_dataset.copy()
    df = df[df[time_col].astype(str).isin([t0, t2])].copy()
    if df.empty:
        return None

    y = (df[time_col].astype(str) == t2).astype(int).values
    groups = df[patient_col].astype(str).values

    # if too small, skip
    if len(np.unique(groups[y == 0])) < 2 or len(np.unique(groups[y == 1])) < 2:
        return None

    X = df[spec_cols].apply(pd.to_numeric, errors="coerce").values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # simple repeated group holdout: random split of patient IDs
    rng = np.random.RandomState(42)
    uniq = np.unique(groups)

    results = []
    rocs = []

    for rep in range(20):
        rng.shuffle(uniq)
        # 80/20 split by patients
        n_test = max(1, int(0.2 * len(uniq)))
        test_pat = set(uniq[:n_test])
        test_idx = np.array([g in test_pat for g in groups], dtype=bool)
        train_idx = ~test_idx

        # guard: both classes in train and test
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue

        pipe.fit(X[train_idx], y[train_idx])
        p = pipe.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], p)
        results.append({"rep": rep, "auc": float(auc), "n_test": int(np.sum(test_idx))})

        fpr, tpr, _ = roc_curve(y[test_idx], p)
        rocs.append((fpr, tpr))

    if not results:
        return None

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "spec_only_timepoint_classifier.csv", index=False)

    # average ROC (rough interpolation)
    grid = np.linspace(0, 1, 200)
    tprs = []
    for fpr, tpr in rocs:
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)

    plt.figure(figsize=(5, 5))
    plt.plot(grid, mean_tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Spec-only timepoint ROC (mean over reps)\nmean AUC={df_res['auc'].mean():.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "spec_only_timepoint_roc.png", dpi=200)
    plt.close()

    # fit once on all data for coefficient table (interpretability)
    pipe.fit(X, y)
    lr = pipe.named_steps["lr"]
    coefs = lr.coef_.ravel()
    df_coef = pd.DataFrame({"feature": spec_cols, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False)
    df_coef.to_csv(out_dir / "spec_only_timepoint_coefficients.csv", index=False)

    # plot top coefficients
    top = df_coef.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top)), top["coef"].values)
    plt.yticks(range(len(top)), top["feature"].values, fontsize=8)
    plt.axvline(0, linewidth=1)
    plt.title("Top 20 LR coefficients (spec-only, timepoint)")
    plt.tight_layout()
    plt.savefig(out_dir / "spec_only_timepoint_top_coefficients.png", dpi=200)
    plt.close()

    return df_res


def _make_contact_sheet(images: list[Path], out_png: Path, ncols: int = 4, title: str = ""):
    if not images:
        return
    n = min(len(images), ncols * ncols)  # 4x4 default max
    images = images[:n]

    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i, p in enumerate(images):
        ax = plt.subplot(nrows, ncols, i + 1)
        try:
            im = plt.imread(p)
            ax.imshow(im)
            ax.set_title(p.name[:30], fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, "failed", ha="center", va="center")
        ax.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/<run_id> folder")
    ap.add_argument("--dataset", default="", help="Optional: dataset CSV with patient_id/timepoint/(label). Recommended.")
    ap.add_argument("--config", default="", help="Optional: configs/default.toml (only used to read preset list etc)")
    ap.add_argument("--out-subdir", default="specparam_report", help="Subfolder in run-dir where report outputs are written")

    ap.add_argument("--file-col", default="file")
    ap.add_argument("--patient-col", default="patient_id")
    ap.add_argument("--time-col", default="timepoint")
    ap.add_argument("--label-col", default="label")

    ap.add_argument("--compare-timepoints", default="t0,t2", help="e.g. t0,t2")
    ap.add_argument("--presets", default="", help="comma list, else infer from data/config")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    tps = [s.strip() for s in args.compare_timepoints.split(",") if s.strip()]
    if len(tps) != 2:
        raise ValueError("--compare-timepoints must be like 't0,t2'")
    t0, t2 = tps[0], tps[1]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _mkdir(run_dir / args.out_subdir / stamp)
    out_fig = _mkdir(out_root / "figures")
    out_tab = _mkdir(out_root / "tables")

    print("[INFO] run_dir:", run_dir)
    print("[INFO] out_root:", out_root)

    # 1) Load specparam long results (from features/specparam/*.csv)
    df_long = _load_long_specparam(run_dir)

    # 2) Load dataset/meta if provided
    df_dataset = None
    meta = None
    if args.dataset:
        ds_path = Path(args.dataset)
        if not ds_path.exists():
            raise FileNotFoundError(ds_path)
        meta, df_dataset = _load_dataset_meta(
            ds_path,
            file_col=args.file_col,
            patient_col=args.patient_col,
            time_col=args.time_col,
            label_col=args.label_col,
        )
        df_long = _merge_long_with_meta(df_long, meta, file_col_meta=args.file_col)
    else:
        # no dataset -> still save raw
        pass

    # Determine presets
    presets = None
    if args.presets.strip():
        presets = [s.strip() for s in args.presets.split(",") if s.strip()]
    elif args.config.strip():
        cfg = _load_toml(Path(args.config))
        sp = cfg.get("specparam", {})
        enabled = sp.get("enabled_presets", None)
        if isinstance(enabled, list) and enabled:
            presets = enabled
        else:
            one = sp.get("preset", "")
            presets = [one] if one else None

    if presets is None:
        presets = sorted(df_long["preset"].astype(str).unique())

    # 3) Filter on timepoints if possible
    if args.time_col in df_long.columns:
        df_long = _filter_timepoints(df_long, args.time_col, [t0, t2])

    # Save merged long table
    _save_df(df_long, out_tab / "specparam_long_merged.csv")

    # 4) Counts
    if meta is not None:
        counts = _counts_table(meta, args.time_col, args.label_col, args.patient_col)
        _save_df(counts, out_tab / "sample_counts.csv")
        _plot_counts(counts, out_fig / "01_sample_counts.png", args.time_col, args.label_col)

    # 5) Fit quality summaries
    if args.time_col in df_long.columns:
        qual = _summary_stats_long(df_long, ["preset", "freq_hz", args.time_col], ["r2", "error"])
    else:
        qual = _summary_stats_long(df_long, ["preset", "freq_hz"], ["r2", "error"])
    _save_df(qual, out_tab / "fit_quality_summary.csv")

    # 6) Aperiodic summaries
    if args.time_col in df_long.columns:
        ap_sum = _summary_stats_long(df_long, ["preset", "freq_hz", args.time_col], ["offset", "exponent"])
    else:
        ap_sum = _summary_stats_long(df_long, ["preset", "freq_hz"], ["offset", "exponent"])
    _save_df(ap_sum, out_tab / "aperiodic_summary.csv")

    # 7) Metric plots by freq
    for preset in presets:
        _plot_metric_by_freq(df_long, out_fig / f"02_r2_by_freq_{preset}.png", "r2", preset, args.time_col)
        _plot_metric_by_freq(df_long, out_fig / f"03_error_by_freq_{preset}.png", "error", preset, args.time_col)
        _spaghetti(df_long, out_fig / f"04_exponent_spaghetti_{preset}.png", "exponent", preset, args.time_col)
        _spaghetti(df_long, out_fig / f"05_offset_spaghetti_{preset}.png", "offset", preset, args.time_col)

    # 8) Unpaired exploratory effects (t2 - t0)
    if args.time_col in df_long.columns:
        eff = _unpaired_effects(df_long, args.time_col, t0=t0, t2=t2)
        _save_df(eff, out_tab / "effects_unpaired_t2_minus_t0.csv")
        _plot_top_effects(eff, out_fig / "06_top_unpaired_effects.png", top_k=15)

    # 9) Paired deltas per patient (requires dataset)
    if df_dataset is not None:
        try:
            ddelta = _paired_deltas(df_dataset, args.patient_col, args.time_col, args.file_col, t0=t0, t2=t2)
            if not ddelta.empty:
                _save_df(ddelta, out_tab / "paired_deltas_per_patient.csv")
                prank = _rank_paired(ddelta, args.patient_col)
                _save_df(prank, out_tab / "paired_feature_rank.csv")
                _plot_top_paired(prank, out_fig / "07_top_paired_deltas.png", top_k=15)
        except Exception as e:
            print("[WARN] paired delta failed:", e)

        # Correlation heatmap of spec_* columns
        _corr_heatmap(df_dataset, out_fig / "08_spec_corr_heatmap.png")

        # Spec-only timepoint classifier (exploratory)
        try:
            _spec_only_classifier_timepoint(df_dataset, out_tab, args.patient_col, args.time_col, t0=t0, t2=t2)
        except Exception as e:
            print("[WARN] spec-only timepoint classifier skipped:", e)

    # 10) Contact sheets from existing SpecParam fit PNGs
    # We search common figure locations:
    fig_dirs = [
        run_dir / "figures" / "specparam",
        run_dir / "features" / "specparam",
    ]
    all_pngs = []
    for d in fig_dirs:
        if d.exists():
            all_pngs.extend(sorted(d.rglob("*.png")))
    if all_pngs:
        # also make per-preset galleries if preset name is in path
        _make_contact_sheet(all_pngs, out_fig / "09_gallery_overview.png", ncols=4, title="SpecParam fit gallery (sample)")

        for preset in presets:
            p_pngs = [p for p in all_pngs if re.search(rf"\b{re.escape(str(preset))}\b", str(p), re.IGNORECASE)]
            if p_pngs:
                _make_contact_sheet(p_pngs, out_fig / f"10_gallery_{preset}.png", ncols=4, title=f"Preset: {preset}")

    # done
    print("\n[OK] SpecParam report written to:")
    print(" ", out_root)
    print("\nTables:")
    print(" ", out_tab)
    print("\nFigures:")
    print(" ", out_fig)


if __name__ == "__main__":
    main()
