"""
run_classifier.py

Legacy-style ML runner (in the spirit of Livia's pipeline):
- Repeated train/test splits (n_repeats)
- Preprocessing + feature selection:
    * per-feature adaptive scaler (Shapiro + outliers)   [fallback if scipy missing]
    * variance threshold
    * correlation-based feature removal
    * mean imputation
    * RFECV with L1-logistic regression
- Model screening on train set (CV AUC) + optional grid search
- Evaluation on held-out test set
- Group-aware splitting by patient_id (prevents leakage when multiple rows per patient exist)

Inputs:
- A "wide" dataset CSV produced by attach_labels.py, e.g.:
  outputs/<run_id>/ml/dataset_t2_t2_t1_wide.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, UndefinedMetricWarning
from sklearn.model_selection import (
    StratifiedKFold,
    GroupShuffleSplit,
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFECV

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import sys
try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:
    import tomli as tomllib  # py<3.11


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _cli_has(flag: str) -> bool:
    return flag in sys.argv


# --- Optional scipy (Shapiro) ---
try:
    from scipy.stats import shapiro  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
try:
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings(
        "ignore",
        category=LinAlgWarning,
        module="sklearn.discriminant_analysis",
    )
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=".*covariance matrix of class .* is not full rank.*",
    )

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


META_DEFAULT = {"file", "patient_id", "label", "timepoint", "trial", "session"}


def _extract_patient_id(file_stem: str) -> str:
    s = str(file_stem).strip().upper()
    m = re.match(r"^(VEP\d+)", s)
    return m.group(1) if m else ""


def _infer_timepoint_from_file(file_stem: str) -> str:
    s = str(file_stem).strip()
    if s.endswith("_1"):
        return "t0"
    if s.endswith("_2"):
        return "t1"
    if s.endswith("_3"):
        return "t2"
    return ""


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _score_to_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return probability-like scores for AUC."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        s = np.asarray(s, dtype=float)
        # min-max to [0,1]
        if np.allclose(s.max(), s.min()):
            return np.zeros_like(s)
        return (s - s.min()) / (s.max() - s.min())
    # last resort: hard labels
    return clf.predict(X).astype(float)


def _pick_feature_columns(df: pd.DataFrame, meta_cols: set[str]) -> List[str]:
    # keep only numeric feature cols, excluding meta + label
    cols = []
    for c in df.columns:
        if c.lower() in meta_cols:
            continue
        if c == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _apply_feature_set(cols: List[str], feature_set: str) -> List[str]:
    """
    Best-effort heuristic filters, because naming can differ.
    Use --include / --exclude if you want exact control.
    """
    fs = feature_set.lower()
    if fs == "all":
        return cols
    if fs == "power":
        pat = re.compile(r"(power|snr|pwr|base)", re.IGNORECASE)
        return [c for c in cols if pat.search(c)]
    if fs == "plv":
        pat = re.compile(r"(plv|phase)", re.IGNORECASE)
        return [c for c in cols if pat.search(c)]
    if fs == "specparam":
        pat = re.compile(r"(spec_|spec0_|specparam|aperiodic|offset|knee|exponent|peak|bandwidth|center_freq|r2|error)", re.IGNORECASE)
        return [c for c in cols if pat.search(c)]

    raise ValueError(f"Unknown feature_set: {feature_set}")


def _apply_regex_filters(cols: List[str], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    out = cols[:]
    if include:
        inc = [re.compile(p, re.IGNORECASE) for p in include]
        out = [c for c in out if any(p.search(c) for p in inc)]
    if exclude:
        exc = [re.compile(p, re.IGNORECASE) for p in exclude]
        out = [c for c in out if not any(p.search(c) for p in exc)]
    return out


@dataclass
class TransformRecord:
    pre_imputer: SimpleImputer
    pre_keep_cols: List[str]
    scalers: Dict[str, Any]
    variance_selector: VarianceThreshold
    to_drop_corr: List[str]
    imputer: SimpleImputer
    rfecv: Optional[RFECV]
    final_columns_after_var_corr: List[str]
    rfe_feature_names: List[str]



def _choose_scaler_per_feature(x_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Livia-style per-feature scaling:
    - if normal -> StandardScaler
    - if not normal and no outliers -> MinMaxScaler
    - else -> RobustScaler

    If scipy is missing, fallback to StandardScaler for all.
    """
    scalers: Dict[str, Any] = {}

    for col in x_train.columns:
        col_data = x_train[col].astype(float).values
        col_data = col_data[~np.isnan(col_data)]
        if len(col_data) < 5 or not _HAS_SCIPY:
            scalers[col] = StandardScaler()
            continue

        # normality
        try:
            _, pvalue = shapiro(col_data[:5000])  # cap length
            not_normal = float(pvalue) <= 0.05
        except Exception:
            not_normal = True

        # outliers (IQR)
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_count = int(np.sum((col_data < lower) | (col_data > upper)))

        if not_normal is False:
            scalers[col] = StandardScaler()
        elif not_normal is True and outliers_count == 0:
            scalers[col] = MinMaxScaler()
        else:
            scalers[col] = RobustScaler()

    return scalers


def _fit_transform_record(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
    do_rfecv: bool,
    inner_cv: int,
    corr_thresh: float = 0.95,
) -> Tuple[np.ndarray, TransformRecord]:

    x_train = x_train.astype(float)

    # 0) Drop columns that are entirely NaN in this TRAIN split
    x_train = x_train.dropna(axis=1, how="all")

    # 1) Impute early (so variance/scalers won't break)
    pre_imp = SimpleImputer(strategy="mean")
    x_imp = pd.DataFrame(pre_imp.fit_transform(x_train), columns=x_train.columns)

    # Drop columns where mean couldn't be computed (all-NaN columns)
    bad = np.isnan(pre_imp.statistics_)
    keep_cols = list(x_train.columns[~bad])
    x_imp = x_imp[keep_cols]

    # 2) Variance threshold (on imputed data!)
    var_sel = VarianceThreshold(threshold=1e-12)
    x_var = var_sel.fit_transform(x_imp)
    cols_var = list(x_imp.columns[var_sel.get_support()])

    x_var_df = pd.DataFrame(x_var, columns=cols_var)

    # 3) Per-feature scalers (on cleaned data)
    scalers = _choose_scaler_per_feature(x_var_df)
    x_scaled = x_var_df.copy()
    for col, scaler in scalers.items():
        scaler.fit(x_var_df[[col]])
        x_scaled[col] = scaler.transform(x_var_df[[col]]).reshape(-1)

    # 4) Correlation filter
    corr = x_scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    x_corr_df = x_scaled.drop(columns=to_drop)

    # 5) Final imputer (safety; should be redundant now)
    imp = SimpleImputer(strategy="mean")
    x_imp2 = pd.DataFrame(imp.fit_transform(x_corr_df), columns=x_corr_df.columns)

    # 6) RFECV
    rfecv = None
    rfe_feature_names = list(x_imp2.columns)
    x_sel = x_imp2.values

    if do_rfecv:
        estimator = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            penalty="l1",
        )
        cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_obj,
            scoring="roc_auc",
            min_features_to_select=2,
        )
        rfecv.fit(x_imp2, y_train)
        x_sel = rfecv.transform(x_imp2)
        rfe_feature_names = list(x_imp2.columns[rfecv.get_support()])

    rec = TransformRecord(
        pre_imputer=pre_imp,
        pre_keep_cols=keep_cols,
        scalers=scalers,
        variance_selector=var_sel,
        to_drop_corr=to_drop,
        imputer=imp,
        rfecv=rfecv,
        final_columns_after_var_corr=list(x_imp2.columns),
        rfe_feature_names=rfe_feature_names,
    )
    return x_sel, rec



def _apply_transform_record(x: pd.DataFrame, rec: TransformRecord) -> np.ndarray:
    x = x.astype(float)

    # Align columns like train (missing cols -> NaN)
    for c in rec.pre_keep_cols:
        if c not in x.columns:
            x[c] = np.nan
    x = x[rec.pre_keep_cols]

    # Pre-impute
    x_imp = pd.DataFrame(rec.pre_imputer.transform(x), columns=rec.pre_keep_cols)

    # Variance selector
    x_var = rec.variance_selector.transform(x_imp)
    cols_var = list(x_imp.columns[rec.variance_selector.get_support()])
    x_var_df = pd.DataFrame(x_var, columns=cols_var)

    # Scaling
    x_scaled = x_var_df.copy()
    for col, scaler in rec.scalers.items():
        if col not in x_scaled.columns:
            x_scaled[col] = np.nan
        x_scaled[col] = scaler.transform(x_scaled[[col]]).reshape(-1)

    # Correlation drop + align
    x_corr_df = x_scaled.drop(columns=[c for c in rec.to_drop_corr if c in x_scaled.columns])
    x_corr_df = x_corr_df.reindex(columns=rec.final_columns_after_var_corr)

    # Final impute
    x_imp2 = pd.DataFrame(rec.imputer.transform(x_corr_df), columns=rec.final_columns_after_var_corr)

    if rec.rfecv is not None:
        return rec.rfecv.transform(x_imp2)
    return x_imp2.values



def _get_models() -> Tuple[List[str], List[Any]]:
    """
    Mirrors Livia's set (LDA, QDA, best-effort KNN, GNB, LR, SGD, RF, SVC).
    """
    models = [
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        QuadraticDiscriminantAnalysis(reg_param=0.9),
        KNeighborsClassifier(),
        GaussianNB(),
        LogisticRegression(max_iter=500, class_weight="balanced"),
        SGDClassifier(max_iter=1000, tol=1e-3),
        RandomForestClassifier(),
        SVC(probability=True),
    ]
    names = [
        "Linear Discriminant Analysis",
        "Quadratic Discriminant Analysis",
        "K-Neighbors",
        "GaussianNB",
        "Logistic Regression",
        "SGD",
        "Random Forest",
        "SVC",
    ]
    return names, models


def _get_param_grid(name: str) -> Optional[Dict[str, List[Any]]]:
    """
    Compact grids (you can expand later).
    """
    if name == "K-Neighbors":
        return {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]}
    if name == "Logistic Regression":
        return {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]}
    if name == "SGD":
        return {"loss": ["hinge", "log_loss", "modified_huber"], "alpha": [1e-4, 1e-3, 1e-2]}
    if name == "Random Forest":
        return {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
    if name == "SVC":
        return {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    return None


def _stratified_group_split(
    df: pd.DataFrame, label_col: str, group_col: str, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:

    grp = df[[group_col, label_col]].drop_duplicates(group_col)
    groups = grp[group_col].values
    y = grp[label_col].values

    # check of stratify kan
    ok_stratify = True
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2 or np.min(counts) < 2:
        ok_stratify = False

    if ok_stratify:
        try:
            train_g, test_g = train_test_split(groups, test_size=test_size, stratify=y, random_state=seed)
        except ValueError:
            ok_stratify = False

    if not ok_stratify:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(groups)
        n_groups = len(groups)
        n_test = max(1, int(np.ceil(test_size * n_groups)))
        n_test = min(n_test, n_groups - 1)  # zorg dat train niet leeg is
        test_g = perm[:n_test]
        train_g = perm[n_test:]

    train_idx = df.index[df[group_col].isin(train_g)].to_numpy()
    test_idx = df.index[df[group_col].isin(test_g)].to_numpy()
    return train_idx, test_idx



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help=r"outputs\<run_id>")
    ap.add_argument("--dataset", type=str, default=None, help="Path to dataset wide csv. If omitted, auto-picks latest dataset_*_wide.csv in run_dir/ml.")
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--group-col", type=str, default="patient_id")
    ap.add_argument("--file-col", type=str, default="file")
    ap.add_argument("--timepoint", type=str, default="t2", choices=["t0", "t1", "t2", "all"])
    ap.add_argument("--feature-set", type=str, default="all", choices=["all", "power", "plv", "specparam"])
    ap.add_argument("--include", action="append", default=None, help="Regex include filter for columns (repeatable).")
    ap.add_argument("--exclude", action="append", default=None, help="Regex exclude filter for columns (repeatable).")
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--inner-cv", type=int, default=3)
    ap.add_argument("--do-rfecv", action="store_true", help="Enable RFECV (Livia-style).")
    ap.add_argument("--no-rfecv", action="store_true", help="Disable RFECV.")
    ap.add_argument("--gridsearch", action="store_true", help="Enable GridSearchCV for best model on each repeat.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, default=None, help="Path to .toml (loads [ml] defaults). CLI overrides config.")
    args = ap.parse_args()
    
    # ---- Optional config defaults (CLI overrides config) ----
    if args.config:
        cfg = _load_toml(Path(args.config))
        ml = cfg.get("ml", {})

        if not _cli_has("--timepoint"):
            args.timepoint = ml.get("timepoint", args.timepoint)

        if not _cli_has("--feature-set"):
            args.feature_set = ml.get("feature_set", args.feature_set)

        if not _cli_has("--n-repeats"):
            args.n_repeats = int(ml.get("n_repeats", args.n_repeats))

        if not _cli_has("--test-size"):
            args.test_size = float(ml.get("test_size", args.test_size))

        if not _cli_has("--inner-cv"):
            args.inner_cv = int(ml.get("inner_cv", args.inner_cv))

        if not _cli_has("--seed"):
            args.seed = int(ml.get("seed", args.seed))

        # booleans: alleen zetten als gebruiker niet expliciet flags meegaf
        if (not _cli_has("--do-rfecv")) and (not _cli_has("--no-rfecv")):
            args.do_rfecv = bool(ml.get("rfecv", False))
            args.no_rfecv = False

        if not _cli_has("--gridsearch"):
            args.gridsearch = bool(ml.get("gridsearch", args.gridsearch))

        if not _cli_has("--include"):
            args.include = ml.get("include", args.include)

        if not _cli_has("--exclude"):
            args.exclude = ml.get("exclude", args.exclude)
    # ---- End config defaults ----

    run_dir = Path(args.run_dir)
    ml_dir = run_dir / "ml"
    if not ml_dir.exists():
        raise FileNotFoundError(f"Could not find {ml_dir}. Did you run build_features.py + attach_labels.py?")

    # dataset path
    if args.dataset:
        ds_path = Path(args.dataset)
    else:
        cands = sorted(ml_dir.glob("dataset_*_wide.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No dataset_*_wide.csv found in {ml_dir}. Run attach_labels.py first.")
        ds_path = cands[0]

    df = pd.read_csv(ds_path)
    if args.group_col not in df.columns:
        # derive from file column
        if args.file_col not in df.columns:
            raise ValueError(f"Neither {args.group_col} nor {args.file_col} exist in dataset.")
        df[args.group_col] = df[args.file_col].apply(_extract_patient_id)

    if args.timepoint != "all":
        # filter by file suffix if present
        if args.file_col in df.columns:
            suffix = {"t0": "_1", "t1": "_2", "t2": "_3"}[args.timepoint]
            df = df[df[args.file_col].astype(str).str.endswith(suffix)].copy()
        else:
            # fallback: infer
            df["timepoint"] = df[args.file_col].apply(_infer_timepoint_from_file)
            df = df[df["timepoint"] == args.timepoint].copy()

    # label
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in dataset. Columns: {list(df.columns)[:20]}...")
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    # features
    meta_cols = set([c.lower() for c in META_DEFAULT] + [args.file_col.lower(), args.group_col.lower(), args.label_col.lower()])
    feat_cols = _pick_feature_columns(df, meta_cols=meta_cols)

    # heuristic feature-set filtering + regex filters
    feat_cols = _apply_feature_set(feat_cols, args.feature_set)
    feat_cols = _apply_regex_filters(feat_cols, args.include, args.exclude)

    if len(feat_cols) < 2:
        raise ValueError(f"Too few feature columns selected ({len(feat_cols)}). Try --feature-set all or remove filters.")

    # clean invalid numbers
    X_df = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y = df[args.label_col].astype(int)
    groups = df[args.group_col].astype(str)

    # RFECV flag
    do_rfecv = args.do_rfecv and not args.no_rfecv
    if args.no_rfecv:
        do_rfecv = False

    # output folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ml_dir / "results" / f"{stamp}_legacy_like"
    _ensure_dir(out_dir)

    # save run config
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    names, models = _get_models()

    rows_repeat_best = []
    rows_repeat_all = []
    best_overall = None  # (repeat_auc, repeat_idx, model_name, model_obj, transform_record)

    rng = np.random.RandomState(args.seed)

    print(f"[INFO] Dataset: {ds_path}")
    print(f"[INFO] N={len(df)} | positives={int(y.sum())} | negatives={int((1-y).sum())}")
    min_class = int(pd.Series(y).value_counts().min()) if len(y) else 0
    n_groups = df[args.group_col].nunique()

    if min_class < 2 or n_groups < 4:
        raise ValueError(
            f"Te weinig data voor group+stratified split. "
            f"N={len(df)}, groups={n_groups}, min_class={min_class}. "
            f"Run pipeline op meer files (hogere --max-files) of gebruik een run-dir met meer samples."
        )

    print(f"[INFO] Features selected: {len(feat_cols)} | rfecv={do_rfecv} | gridsearch={args.gridsearch}")

    for rep in range(args.n_repeats):
        rep_seed = int(rng.randint(0, 10_000_000))

        # group-stratified split
        train_idx, test_idx = _stratified_group_split(df, args.label_col, args.group_col, args.test_size, rep_seed)

        x_train = X_df.loc[train_idx].copy()
        y_train = y.loc[train_idx].copy()
        g_train = groups.loc[train_idx].copy()

        x_test = X_df.loc[test_idx].copy()
        y_test = y.loc[test_idx].copy()
        g_test = groups.loc[test_idx].copy()

        # fit transformations on train
        x_train_sel, trec = _fit_transform_record(
            x_train=x_train,
            y_train=y_train,
            groups_train=g_train,
            do_rfecv=do_rfecv,
            inner_cv=args.inner_cv,
            corr_thresh=0.95,
        )
        x_test_sel = _apply_transform_record(x_test, trec)

        # screen models on train (CV AUC)
        # (group-awareness inside CV only matters when multiple rows per patient are present)
        cv_obj = StratifiedKFold(n_splits=args.inner_cv, shuffle=True, random_state=rep_seed)

        cv_results = []
        for name, clf in zip(names, models):
            try:
                aucs = cross_val_score(clf, x_train_sel, y_train, cv=cv_obj, scoring="roc_auc")
                mean_auc = float(np.mean(aucs))
                std_auc = float(np.std(aucs))
            except Exception:
                mean_auc = np.nan
                std_auc = np.nan
            cv_results.append((name, clf, mean_auc, std_auc))

            rows_repeat_all.append({
                "repeat": rep,
                "model": name,
                "train_cv_auc_mean": mean_auc,
                "train_cv_auc_std": std_auc,
                "n_features_after_sel": len(trec.rfe_feature_names),
            })

        cv_results = sorted(cv_results, key=lambda t: (-np.nan_to_num(t[2], nan=-1.0), t[0]))
        best_name, best_clf, best_train_auc, best_train_auc_std = cv_results[0]

        # optional grid search on best model
        final_clf = best_clf
        best_params = None
        if args.gridsearch:
            grid = _get_param_grid(best_name)
            if grid:
                gs = GridSearchCV(
                    estimator=best_clf,
                    param_grid=grid,
                    scoring="roc_auc",
                    cv=cv_obj,
                    n_jobs=-1,
                    error_score="raise",
                )
                gs.fit(x_train_sel, y_train)
                final_clf = gs.best_estimator_
                best_params = gs.best_params_

        # fit final model on train, evaluate on test
        final_clf.fit(x_train_sel, y_train)
        y_score = _score_to_proba(final_clf, x_test_sel)
        y_pred = (y_score >= 0.5).astype(int)

        test_auc = _safe_auc(y_test.to_numpy(), y_score)
        test_acc = float(accuracy_score(y_test, y_pred))
        test_bacc = float(balanced_accuracy_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred, zero_division=0))

        rows_repeat_best.append({
            "repeat": rep,
            "best_model": best_name,
            "train_cv_auc_mean": best_train_auc,
            "train_cv_auc_std": best_train_auc_std,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_bal_acc": test_bacc,
            "test_f1": test_f1,
            "n_features_after_sel": len(trec.rfe_feature_names),
            "best_params": json.dumps(best_params) if best_params else "",
        })

        if best_overall is None or (not np.isnan(test_auc) and test_auc > best_overall[0]):
            best_overall = (test_auc, rep, best_name, final_clf, trec)

        print(f"[{rep+1:02d}/{args.n_repeats}] best={best_name:28s} "
              f"trainAUC={best_train_auc:.3f}  testAUC={test_auc:.3f}  nfeat={len(trec.rfe_feature_names)}")

    df_all = pd.DataFrame(rows_repeat_all)
    df_best = pd.DataFrame(rows_repeat_best)

    # summary across repeats
    summary = (
        df_best.groupby("best_model")
        .agg(
            n=("repeat", "count"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_bal_acc_mean=("test_bal_acc", "mean"),
            test_bal_acc_std=("test_bal_acc", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            nfeat_mean=("n_features_after_sel", "mean"),
        )
        .reset_index()
        .sort_values("test_auc_mean", ascending=False)
    )

    df_all.to_csv(out_dir / "train_cv_screening_all_models.csv", index=False)
    df_best.to_csv(out_dir / "repeat_best_model_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_by_best_model.csv", index=False)

    # dump best overall details
    if best_overall is not None:
        best_auc, best_rep, best_name, best_clf, best_trec = best_overall
        (out_dir / "best_overall.txt").write_text(
            "\n".join([
                f"best_test_auc: {best_auc}",
                f"repeat: {best_rep}",
                f"model: {best_name}",
                f"n_features_after_sel: {len(best_trec.rfe_feature_names)}",
                "selected_features:",
                *best_trec.rfe_feature_names,
            ])
        )

        # evaluate best overall on its held-out split again is not possible here (we didn't store indices),
        # but we can at least store selected features list.
        try:
            import joblib  # type: ignore
            joblib.dump({"model": best_clf, "transform": best_trec, "feature_cols": feat_cols}, out_dir / "best_model_bundle.joblib")
        except Exception:
            pass

    print("\n[OK] Results written to:")
    print(f"  {out_dir}")
    print("[INFO] Top of summary_by_best_model.csv:")
    if len(summary):
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
