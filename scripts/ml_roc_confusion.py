import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

try:
    # newer sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def _auto_dataset(run_dir: Path) -> Path:
    ml_dir = run_dir / "ml"
    cands = sorted(ml_dir.glob("dataset_*_wide.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No dataset_*_wide.csv found in {ml_dir}")
    return cands[0]


def _make_candidates(seed: int):
    # Keep this set stable & robust for tiny datasets
    return {
        "LogReg": LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear"),
        "SVC_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
        "RF": RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced_subsample"),
        "KNN1": KNeighborsClassifier(n_neighbors=1),
        # "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),  # enable if you really want; can be unstable
    }


def _pipeline_for(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler(with_centering=True, with_scaling=True)),
        ("clf", model),
    ])


def _safe_inner_cv_auc(X, y, groups, model, inner_cv: int, seed: int):
    # returns mean AUC across valid folds; np.nan if not computable
    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = GroupKFold(n_splits=inner_cv)
        splits = splitter.split(X, y, groups=groups)

    aucs = []
    pipe = _pipeline_for(model)

    for tr_idx, va_idx in splits:
        y_tr, y_va = y[tr_idx], y[va_idx]
        # fold can still end up single-class on very small data
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        try:
            pipe.fit(X[tr_idx], y_tr)
            proba = pipe.predict_proba(X[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_va, proba))
        except Exception:
            continue

    if len(aucs) == 0:
        return np.nan
    return float(np.mean(aucs))


def main():
    # reduce noise (optional)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.discriminant_analysis")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/<RUN_ID> (folder with ml/)")
    ap.add_argument("--dataset", default="", help="Optional path to dataset_*_wide.csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="patient_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--inner-cv", type=int, default=3)
    ap.add_argument("--timepoint", default="t2", help="Filter rows by timepoint if column exists. Use 'all' to disable.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    dataset_path = Path(args.dataset) if args.dataset else _auto_dataset(run_dir)
    df = pd.read_csv(dataset_path)

    # optional filter
    if args.timepoint != "all" and "timepoint" in df.columns:
        df = df[df["timepoint"].astype(str) == str(args.timepoint)].copy()

    # basic checks
    for c in [args.label_col, args.group_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in dataset. Columns: {list(df.columns)[:30]} ...")

    # keep only rows with label
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    # numeric features only (avoid strings like 'file', 'timepoint')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # remove label from features
    feat_cols = [c for c in num_cols if c != args.label_col]

    # drop all-NaN feature columns (THIS removes your imputer warnings)
    non_empty = df[feat_cols].notna().any(axis=0)
    dropped = [c for c in feat_cols if not bool(non_empty[c])]
    feat_cols = [c for c in feat_cols if bool(non_empty[c])]

    if len(feat_cols) == 0:
        raise RuntimeError("No usable numeric feature columns after dropping all-NaN columns.")

    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Rows: {len(df)} | Patients: {df[args.group_col].nunique()} | Features kept: {len(feat_cols)} | Dropped all-NaN: {len(dropped)}")

    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df[args.label_col].to_numpy(dtype=int)
    g_all = df[args.group_col].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=args.n_repeats, test_size=args.test_size, random_state=args.seed)
    candidates = _make_candidates(args.seed)

    y_true_all = []
    y_score_all = []
    picked_models = []

    repeats_used = 0
    for r, (tr_idx, te_idx) in enumerate(gss.split(X_all, y_all, groups=g_all), start=1):
        X_tr, y_tr, g_tr = X_all[tr_idx], y_all[tr_idx], g_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        # if train has only one class, skip (can happen with group splits + small N)
        if len(np.unique(y_tr)) < 2:
            continue

        # select best model by inner-CV AUC
        best_name, best_auc = None, -np.inf
        for name, model in candidates.items():
            auc = _safe_inner_cv_auc(X_tr, y_tr, g_tr, model, inner_cv=args.inner_cv, seed=args.seed + r)
            if np.isnan(auc):
                continue
            if auc > best_auc:
                best_auc = auc
                best_name = name

        if best_name is None:
            continue

        best_model = candidates[best_name]
        pipe = _pipeline_for(best_model)
        pipe.fit(X_tr, y_tr)

        # test probabilities
        try:
            proba = pipe.predict_proba(X_te)[:, 1]
        except Exception:
            # fallback: decision_function -> min-max to [0,1]
            scores = pipe.decision_function(X_te)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            proba = scores

        y_true_all.append(y_te)
        y_score_all.append(proba)
        picked_models.append(best_name)
        repeats_used += 1

    if repeats_used == 0:
        raise RuntimeError("No repeats produced valid predictions (likely too few labeled patients per split).")

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)

    pooled_auc = roc_auc_score(y_true, y_score)
    print(f"[OK] repeats_used={repeats_used}/{args.n_repeats} | pooled AUC={pooled_auc:.3f}")
    print("[INFO] picked_models counts:", pd.Series(picked_models).value_counts().to_dict())

    out_dir = run_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pooled predictions
    pred_df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    pred_df.to_csv(out_dir / "roc_confusion_pooled_predictions.csv", index=False)

    # ROC curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled ROC (AUC={pooled_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_pooled.png", dpi=200)
    plt.close()

    # Confusion matrix at threshold 0.5
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-responder (0)", "responder (1)"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", cmap=None)
    plt.title("Pooled confusion matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_pooled.png", dpi=200)
    plt.close()

    print("[OK] Wrote:")
    print(" ", out_dir / "roc_curve_pooled.png")
    print(" ", out_dir / "confusion_matrix_pooled.png")
    print(" ", out_dir / "roc_confusion_pooled_predictions.csv")


if __name__ == "__main__":
    main()
