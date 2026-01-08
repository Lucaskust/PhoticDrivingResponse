from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # py>=3.11
except Exception:  # pragma: no cover
    tomllib = None


def load_toml(path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError("tomllib not available (need Python 3.11+).")
    with path.open("rb") as f:
        return tomllib.load(f)


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help=r"outputs\<run_id> (bestaande run, stap 1 al gedaan)")
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--steps", default="2-4", help="bijv: 2-4 of 2,3,4")

    # optionele overrides (handig voor snel testen)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--timepoint", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--feature-set", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_toml(Path(args.config))

    labels_path = Path(args.labels or cfg.get("labels", {}).get("path", ""))
    ml_cfg = cfg.get("ml", {})

    timepoint = args.timepoint or ml_cfg.get("timepoint", "t2")
    label_col = args.label_col or ml_cfg.get("label_col", "label")
    dataset_kind = ml_cfg.get("dataset_kind", "wide")
    feature_set = args.feature_set or ml_cfg.get("feature_set", "all")

    n_repeats = int(ml_cfg.get("n_repeats", 20))
    test_size = float(ml_cfg.get("test_size", 0.2))
    inner_cv = int(ml_cfg.get("inner_cv", 3))
    seed = int(ml_cfg.get("seed", 42))

    rfecv = bool(ml_cfg.get("rfecv", False))
    gridsearch = bool(ml_cfg.get("gridsearch", False))

    group_col = ml_cfg.get("group_col", "patient_id")
    file_col = ml_cfg.get("file_col", "file")

    # steps parsing
    steps = set()
    s = args.steps.replace(" ", "")
    if "-" in s:
        a, b = s.split("-", 1)
        steps = set(range(int(a), int(b) + 1))
    else:
        steps = {int(x) for x in s.split(",") if x}

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent

    # ---- Step 2: build_features ----
    if 2 in steps:
        run([py, str(scripts_dir / "build_features.py"), "--run-dir", str(run_dir)])

    # ---- Step 3: attach_labels ----
    if 3 in steps:
        if not labels_path.exists():
            raise FileNotFoundError(f"labels file not found: {labels_path}")
        run([
            py, str(scripts_dir / "attach_labels.py"),
            "--run-dir", str(run_dir),
            "--labels", str(labels_path),
            "--timepoint", str(timepoint),
            "--label-col", str(label_col),
        ])

    # ---- Step 4: run_classifier ----
    if 4 in steps:
        clf_label_col = "label"
        # verwachtte datasetnaam (zoals jij nu ook krijgt)
        ds_path = run_dir / "ml" / f"dataset_{timepoint}_{label_col}_{dataset_kind}.csv"
        # fallback: pak laatste dataset_*_wide.csv
        if not ds_path.exists():
            cands = sorted((run_dir / "ml").glob("dataset_*_wide.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                raise FileNotFoundError(f"No dataset found in {run_dir/'ml'}")
            ds_path = cands[0]

        cmd = [
            py, str(scripts_dir / "run_classifier.py"),
            "--run-dir", str(run_dir),
            "--dataset", str(ds_path),
            "--timepoint", str(timepoint),
            "--label-col", clf_label_col,
            "--feature-set", str(feature_set),
            "--n-repeats", str(n_repeats),
            "--test-size", str(test_size),
            "--inner-cv", str(inner_cv),
            "--seed", str(seed),
            "--group-col", str(group_col),
            "--file-col", str(file_col),
        ]
        if rfecv:
            cmd.append("--do-rfecv")
        else:
            cmd.append("--no-rfecv")
        if gridsearch:
            cmd.append("--gridsearch")

        run(cmd)

    print("\n[OK] Done. Check:", run_dir / "ml")


if __name__ == "__main__":
    main()
