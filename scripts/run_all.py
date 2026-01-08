from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
import os


def _run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())



def _parse_steps(raw_steps: list[str]) -> list[int]:
    # accepteert: --steps 2 3 4  | --steps 2-4 | --steps 2,3,4
    tokens = []
    for s in raw_steps:
        tokens += [t for t in s.split(",") if t.strip()]
    out = set()
    for t in tokens:
        t = t.strip()
        if "-" in t:
            a, b = t.split("-", 1)
            a, b = int(a), int(b)
            for k in range(min(a, b), max(a, b) + 1):
                out.add(k)
        else:
            out.add(int(t))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="configs/default.toml")

    # welke stappen
    ap.add_argument("--steps", nargs="+", default=["2", "3", "4"])

    # stap 3 (labels)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--timepoint", default="t2")
    ap.add_argument("--label-col", default="t2_t1")  # excel-kolomnaam
    ap.add_argument("--dataset-kind", default="wide")  # wide|long

    # stap 4 (classifier)
    ap.add_argument("--feature-set", default="all")

    args = ap.parse_args()
    steps = _parse_steps(args.steps)

    run_dir = Path(args.run_dir)
    scripts_dir = Path(__file__).resolve().parent
    py = str(Path(subprocess.check_output(["where", "python"], text=True).splitlines()[0]).resolve()) if Path().anchor else "python"
    py = sys.executable
    print("[INFO] Using python:", py)

    # Step 2: build_features
    if 2 in steps:
        _run([py, str(scripts_dir / "build_features.py"), "--run-dir", str(run_dir)])

    # Step 3: attach_labels
    if 3 in steps:
        if not args.labels:
            raise ValueError("Step 3 gekozen maar --labels ontbreekt.")
        _run([
            py, str(scripts_dir / "attach_labels.py"),
            "--run-dir", str(run_dir),
            "--labels", str(args.labels),
            "--timepoint", str(args.timepoint),
            "--label-col", str(args.label_col),
        ])

    # Step 4: run_classifier
    if 4 in steps:
        # datasetnaam zoals attach_labels hem maakt
        ds_path = run_dir / "ml" / f"dataset_{args.timepoint}_{args.label_col}_{args.dataset_kind}.csv"
        if not ds_path.exists():
            # fallback: meest recente dataset_*_{kind}.csv
            cands = sorted((run_dir / "ml").glob(f"dataset_*_{args.dataset_kind}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                raise FileNotFoundError(f"Geen dataset gevonden in {run_dir/'ml'}")
            ds_path = cands[0]

        _run([
            py, str(scripts_dir / "run_classifier.py"),
            "--run-dir", str(run_dir),
            "--dataset", str(ds_path),

            # BELANGRIJK: classifier gebruikt outputkolom 'label' (niet t2_t1)
            "--label-col", "label",

            "--feature-set", str(args.feature_set),
        ])

    print("\n[OK] Done. Check:", run_dir / "ml")


if __name__ == "__main__":
    main()
