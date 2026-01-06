import sys
from pathlib import Path
import argparse
from types import SimpleNamespace

# Make repo root importable (so `import pdr...` works)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdr.utils.config import load_config
from pdr.utils.output import make_run_dir, copy_config

from pdr.core.patients import patient_files, eeg
from pdr.features.power import Power
from pdr.features.phase import Phase
from pdr.features.specparam_blocks import SpecParamBlocks


def _trial_timepoints(trial: str, time: str):
    """Match legacy logic: which timepoints belong to which trial."""
    time_map = {
        "t0": ["t0", "t1", "t2"],
        "t1": ["t0", "t1"],
        "t2": ["t0", "t2"],
    }
    if time == "all":
        return time_map[trial]
    if time not in time_map[trial]:
        raise ValueError(f"time={time} is not valid for trial={trial}")
    return [time]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--trial", choices=["t0", "t1", "t2"], default="t2")
    ap.add_argument("--time", choices=["t0", "t1", "t2", "all"], default="all")
    ap.add_argument("--max-files", type=int, default=1, help="For quick tests. Use 0 to run all.")
    ap.add_argument("--upper-lim", type=int, default=40)
    ap.add_argument("--save-intermediates", action="store_true",
                    help="Save intermediate CSVs per file into outputs/<run_id>/intermediates/<file>/...")
    args_cli = ap.parse_args()

    cfg = load_config(args_cli.config)

    # ---- Run folder ----
    run_dir = make_run_dir(cfg["outputs"]["root"], cfg["outputs"].get("run_name", "run"))
    copy_config(args_cli.config, run_dir)
    print(f"[OK] run_dir = {run_dir}")

    # ---- Build an args-like object for patient_files (keeps legacy signature) ----
    timepoints = _trial_timepoints(args_cli.trial, args_cli.time)

    args_pf = SimpleNamespace(
        trial=args_cli.trial,
        time=timepoints,  # list like ['t0','t2']
        data_root=cfg["data"]["root"],
        t0_folder=cfg["data"].get("t0_folder", "cenobamate_eeg_1"),
        t1_folder=cfg["data"].get("t1_folder", "cenobamate_eeg_2"),
        t2_folder=cfg["data"].get("t2_folder", "cenobamate_eeg_3"),
    )

    # trial_map is unused inside patient_files but kept for compatibility
    trial_map = {"t0": "T0_T1_T2", "t1": "T0_T1", "t2": "T0_T2"}
    files = patient_files(trial_map, args_pf)

    if not files:
        print("[WARN] No .cnt files found. Check configs/default.toml [data].root and folder names.")
        sys.exit(1)

    if args_cli.max_files and args_cli.max_files > 0:
        files = files[: args_cli.max_files]

    # ---- Parameters ----
    passband = [float(cfg["preprocess"]["passband_low"]), float(cfg["preprocess"]["passband_high"])]
    use_occipital = bool(cfg["preprocess"].get("use_occipital", True))

    out_power = run_dir / "features" / "power"
    out_plv = run_dir / "features" / "plv"
    out_power.mkdir(parents=True, exist_ok=True)
    out_plv.mkdir(parents=True, exist_ok=True)
    
    out_spec = run_dir / "features" / "specparam"
    out_spec.mkdir(parents=True, exist_ok=True)

    skipped = []

    for i, cnt_path in enumerate(files, start=1):
        print(f"\n--- [{i}/{len(files)}] Processing: {cnt_path.name}")

        try:
            raw = eeg(cnt_path, passband, occi=use_occipital, plot=False)
        except Exception as e:
            print(f"[SKIP] eeg() failed for {cnt_path.name}: {e}")
            skipped.append((cnt_path.name, "eeg", str(e)))
            continue

        base = cnt_path.stem

        # Optional: per-file intermediates (avoid overwriting)
        per_file_inter = None
        if args_cli.save_intermediates:
            per_file_inter = run_dir / "intermediates" / base
            (per_file_inter / "power").mkdir(parents=True, exist_ok=True)
            (per_file_inter / "plv").mkdir(parents=True, exist_ok=True)

        # ---- POWER ----
        try:
            p = Power(passband, raw, out_dir=(per_file_inter / "power") if per_file_inter else None)
            df_power = p.run(
                save_intermediates=bool(per_file_inter),
                plot=False,
                upper_lim=args_cli.upper_lim,
                padding="zeros",
                trim=0.0,
                harms=5,
            )
            df_power.to_pickle(out_power / f"{base}_power.pkl")
            df_power.to_csv(out_power / f"{base}_power.csv", index=False)
            print(f"[OK] Power saved -> {out_power}")
        except Exception as e:
            print(f"[SKIP] Power failed for {cnt_path.name}: {e}")
            skipped.append((cnt_path.name, "power", str(e)))
            continue

        # ---- PLV ----
        try:
            ph = Phase(passband, raw, out_dir=(per_file_inter / "plv") if per_file_inter else None)
            df_plv_stim, df_plv_base = ph.run(
                save_intermediates=bool(per_file_inter),
                plot=False,
                upper_lim=args_cli.upper_lim,
                base=True,
            )
            df_plv_stim.to_pickle(out_plv / f"{base}_plv_stim.pkl")
            df_plv_base.to_pickle(out_plv / f"{base}_plv_base.pkl")
            df_plv_stim.to_csv(out_plv / f"{base}_plv_stim.csv", index=False)
            df_plv_base.to_csv(out_plv / f"{base}_plv_base.csv", index=False)
            print(f"[OK] PLV saved -> {out_plv}")
        except Exception as e:
            print(f"[SKIP] PLV failed for {cnt_path.name}: {e}")
            skipped.append((cnt_path.name, "plv", str(e)))
            continue
        # ---- SPECPARAM ----
            # ---- SPECPARAM ----
        sp_cfg = cfg.get("specparam", {})
        if bool(sp_cfg.get("enabled", False)):
            try:
                sp = SpecParamBlocks(
                    out_dir=out_spec,
                    fmin=float(sp_cfg.get("fmin", 2.0)),
                    fmax=float(sp_cfg.get("fmax", 45.0)),
                    upper_lim_psd=int(sp_cfg.get("upper_lim_psd", 70)),
                    trim=float(sp_cfg.get("trim", 0.0)),
                    padding=str(sp_cfg.get("padding", "zeros")),
                    presets=list(sp_cfg.get("presets", ["baseline"])),
                )
                df_sp = sp.run_from_raw(raw, base_name=base)
                df_sp.to_csv(out_spec / f"{base}_specparam_summary_ALL.csv", index=False)
                print(f"[OK] SpecParam saved -> {out_spec}")
            except Exception as e:
                print(f"[SKIP] SpecParam failed for {cnt_path.name}: {e}")
                skipped.append((cnt_path.name, "specparam", str(e)))

            
    # ---- Summary ----
    if skipped:
        df_skip = pd.DataFrame(skipped, columns=["file", "stage", "error"])
        df_skip.to_csv(run_dir / "logs" / "skipped.csv", index=False)
        print(f"\n[WARN] Skipped {len(skipped)} files. Details: {run_dir / 'logs' / 'skipped.csv'}")
    else:
        print("\n[OK] All files processed without skips.")

    print(f"\nDone. Outputs in:\n{run_dir}")


if __name__ == "__main__":
    # pandas used in summary only; import here to avoid unused import warnings
    import pandas as pd
    main()
