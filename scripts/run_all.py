from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _run(cmd: list[str]) -> None:
    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_steps(raw_steps: list[str]) -> list[int]:
    """
    Accept:
      --steps 2 3 4
      --steps 2-4
      --steps 2,3,4
      --steps 2,3 4
    """
    out: set[int] = set()
    for token in raw_steps:
        token = token.strip()
        if not token:
            continue
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a_i, b_i = int(a), int(b)
                lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Existing run folder (outputs/<run_id>)")
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--steps", nargs="+", default=["2-4"], help="bijv: --steps 2 3 4 | --steps 2-4 | --steps 2,3,4")

    # Step 3 (labels)
    ap.add_argument("--labels", default=None, help="Path to Excel labels file")
    ap.add_argument("--timepoint", default=None, help="t0/t1/t2, used by attach_labels")
    ap.add_argument("--label-col", dest="excel_label_col", default=None, help="Excel column to use as label source (e.g. t2_t1)")

    # Step 4 (classifier)
    ap.add_argument("--dataset-kind", choices=["wide", "long"], default=None)
    ap.add_argument("--dataset-label-col", default=None, help="Column in dataset used as y (default from config, usually 'label')")
    ap.add_argument("--feature-set", default=None, help="all | power | plv | specparam")
    ap.add_argument("--n-repeats", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--inner-cv", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--file-col", default=None)
    ap.add_argument("--rfecv", action="store_true")
    ap.add_argument("--gridsearch", action="store_true")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    scripts_dir = Path(__file__).resolve().parent

    cfg = _load_toml(Path(args.config))
    labels_cfg = cfg.get("labels", {})
    ml_cfg = cfg.get("ml", {})

    steps = _parse_steps(args.steps)
    py = sys.executable  # always use current venv python

    # -------- Step 2: build_features --------
    if 2 in steps:
        _run([py, str(scripts_dir / "build_features.py"), "--run-dir", str(run_dir)])

    # -------- Step 3: attach_labels --------
    if 3 in steps:
        labels_path = args.labels or labels_cfg.get("path") or labels_cfg.get("file") or ""
        if not labels_path:
            raise ValueError("Missing labels path. Provide --labels or set [labels].path in config.")
        timepoint = args.timepoint or ml_cfg.get("timepoint") or "t2"
        excel_label_col = args.excel_label_col or labels_cfg.get("excel_col") or labels_cfg.get("label_col") or "t2_t1"

        _run([
            py, str(scripts_dir / "attach_labels.py"),
            "--run-dir", str(run_dir),
            "--labels", str(labels_path),
            "--timepoint", str(timepoint),
            "--label-col", str(excel_label_col),
        ])

    # -------- Step 4: run_classifier --------
    if 4 in steps:
        timepoint = args.timepoint or ml_cfg.get("timepoint") or "t2"
        excel_label_col = args.excel_label_col or labels_cfg.get("excel_col") or labels_cfg.get("label_col") or "t2_t1"
        dataset_kind = args.dataset_kind or ml_cfg.get("dataset_kind", "wide")

        # expected dataset path created by attach_labels
        ds_path = run_dir / "ml" / f"dataset_{timepoint}_{excel_label_col}_{dataset_kind}.csv"
        if not ds_path.exists():
            # fallback: latest dataset_*_{kind}.csv
            cands = sorted((run_dir / "ml").glob(f"dataset_*_{dataset_kind}.csv"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                raise FileNotFoundError(f"No dataset found in {run_dir/'ml'}")
            ds_path = cands[0]

        dataset_label_col = args.dataset_label_col or ml_cfg.get("label_col", "label")
        feature_set = args.feature_set or ml_cfg.get("feature_set", "all")
        n_repeats = args.n_repeats or int(ml_cfg.get("n_repeats", 20))
        test_size = args.test_size or float(ml_cfg.get("test_size", 0.2))
        inner_cv = args.inner_cv or int(ml_cfg.get("inner_cv", 3))
        seed = args.seed or int(ml_cfg.get("seed", 42))
        group_col = args.group_col or ml_cfg.get("group_col", "patient_id")
        file_col = args.file_col or ml_cfg.get("file_col", "file")

        cmd = [
            py, str(scripts_dir / "run_classifier.py"),
            "--run-dir", str(run_dir),
            "--dataset", str(ds_path),
            "--label-col", str(dataset_label_col),
            "--feature-set", str(feature_set),
            "--n-repeats", str(n_repeats),
            "--test-size", str(test_size),
            "--inner-cv", str(inner_cv),
            "--seed", str(seed),
            "--group-col", str(group_col),
            "--file-col", str(file_col),
        ]

        # pass config so classifier can use prefixes
        cmd += ["--config", str(args.config)]

        if args.rfecv:
            cmd.append("--do-rfecv")
        else:
            cmd.append("--no-rfecv")
        if args.gridsearch:
            cmd.append("--gridsearch")

        _run(cmd)

    print("\n[OK] Done. Check:", run_dir / "ml")


if __name__ == "__main__":
    main()
