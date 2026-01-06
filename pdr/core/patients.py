"""
Module patients selects patientfiles, loads their EEG data and filters them.
"""
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
import shutil

import numpy as np
import pandas as pd
import mne
from mne.io.base import BaseRaw

mne.set_log_level("error")


# Default mapping (kan je straks ook in config stoppen)
DEFAULT_TIME_TO_FOLDER = {
    "t0": "cenobamate_eeg_1",
    "t1": "cenobamate_eeg_2",
    "t2": "cenobamate_eeg_3",
}


def parse_args() -> tuple[dict[str, str], dict[str, list[str]], argparse.Namespace]:
    """
    Parse command-line arguments for trial and time selection.

    Returns
    -------
    trial_map, time_map, args
    """
    parser = argparse.ArgumentParser()

    # Trial logic (zoals in Livia's setup)
    trial_map = {"t0": "T0_T1_T2", "t1": "T0_T1", "t2": "T0_T2"}
    parser.add_argument(
        "-tr",
        "--trial",
        choices=trial_map.keys(),
        default="t2",
        help="Choose between t0, t1, t2",
    )

    # Valid timepoints per trial
    time_map = {
        "t0": ["t0", "t1", "t2"],
        "t1": ["t0", "t1"],
        "t2": ["t0", "t2"],
    }

    # ✅ FIX: maak 'all' ook echt een geldig choice
    parser.add_argument(
        "-t",
        "--time",
        choices=["all"] + list(time_map.keys()),
        default="all",
        help="Choose point in time: t0, t1, t2 or all",
    )

    # ✅ NEW: portable datapad (ipv hardcoded D:/)
    # Je kunt ook PDR_DATA_ROOT env var gebruiken als je wil.
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PDR_DATA_ROOT", "D:/"),
        help="Root folder that contains cenobamate_eeg_1/2/3 (e.g., D:/).",
    )

    # ✅ NEW: folders per timepoint (kan je later vanuit config vullen)
    parser.add_argument("--t0-folder", default=DEFAULT_TIME_TO_FOLDER["t0"])
    parser.add_argument("--t1-folder", default=DEFAULT_TIME_TO_FOLDER["t1"])
    parser.add_argument("--t2-folder", default=DEFAULT_TIME_TO_FOLDER["t2"])

    args = parser.parse_args()

    # Normalize args.time
    if args.time == "all":
        args.time = time_map[args.trial]
    elif args.time not in time_map[args.trial]:
        raise ValueError(f"Error: timepoint {args.time} is not valid for trial {args.trial}")

    return trial_map, time_map, args


def _time_to_folder_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Build TIME_TO_FOLDER mapping from CLI args."""
    return {
        "t0": args.t0_folder,
        "t1": args.t1_folder,
        "t2": args.t2_folder,
    }


def patient_files(trial_map: dict[str, str], args: argparse.Namespace) -> list[Path]:
    """
    Returns a list of patient files for a given trial/time combo.

    Expected folder structure (default):
        <data_root>/cenobamate_eeg_1  -> t0
        <data_root>/cenobamate_eeg_2  -> t1
        <data_root>/cenobamate_eeg_3  -> t2
    """
    _ = trial_map  # kept for compatibility with legacy calling signature

    data_root = Path(args.data_root)
    time_to_folder = _time_to_folder_from_args(args)

    timepoints = args.time if isinstance(args.time, list) else [args.time]
    files: list[Path] = []

    for tp in timepoints:
        folder_name = time_to_folder.get(tp)
        if folder_name is None:
            continue

        folder = data_root / folder_name
        print(f"Zoek in map: {folder}")

        if not folder.exists():
            print("  LET OP: map bestaat niet.")
            continue

        new_files = sorted(folder.glob("*.cnt"))
        print(f"  Gevonden .cnt-bestanden: {len(new_files)}")
        files.extend(new_files)

    print(f"Totaal aantal .cnt-bestanden dat verwerkt gaat worden: {len(files)}")
    return files


def eeg(src, passband, notch=50, occi: bool = False, plot: bool = False) -> BaseRaw:
    """
    Loads the EEG data and applies basic filtering.

    Parameters
    ----------
    src: Path
    passband: [low, high]
    notch: line frequency to notch out
    occi: if True only O1/O2/Oz
    plot: show raw plot

    Returns
    -------
    raw: mne BaseRaw
    """
    raw = mne.io.read_raw_ant(src, preload=True, verbose="ERROR")
    raw.filter(l_freq=passband[0], h_freq=passband[1], picks="eeg", verbose="ERROR")

    line_freq = notch if (freq := raw.info["line_freq"]) is None else freq
    lowpass = np.arange(line_freq, raw.info["lowpass"] + 1, line_freq)
    raw.notch_filter(freqs=(lowpass), notch_widths=(lowpass) / line_freq, picks=["eeg"], verbose="ERROR")

    if "EOG" in raw.ch_names:
        raw.drop_channels("EOG")

    if occi:
        raw = raw.copy().pick(["O1", "O2", "Oz"])

    threshold = 1e-6
    channels_dropped = set([ch for ch in raw.ch_names if np.all(np.abs(raw.get_data(picks=ch)) < threshold)])
    if channels_dropped:
        raw.drop_channels([ch for ch in channels_dropped if ch in raw.ch_names])
        print(f"Dropped channels: {channels_dropped}")

    if plot:
        raw.plot(scalings="auto", title="Filtered EEG data", show=True, block=True)

    return raw


def save_pickle_results(data: pd.DataFrame, pt_file: str, folder_name: str, feat: str = "power"):
    """
    Saves dataframes as pickle files in designated pathway.
    (Nog legacy: schrijft naar ./<folder_name>. Later koppelen we dit aan outputs/<run_id>/...)
    """
    path = Path(f"./{folder_name}")
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict) and feat == "plv":
        for ending, item in data.items():
            single_path = path / f"{pt_file.stem}_{feat}_{ending}.pkl"
            item.to_pickle(single_path)
    else:
        single_path = path / f"{pt_file.stem}_{feat}.pkl"
        data.to_pickle(single_path)


def filter_files(folder: str, time_map: dict[str, list[str]], args: argparse.Namespace, feat: str = "power") -> list[str]:
    """
    Filters patients with complete data across required timepoints for either power or PLV results.
    Moves incomplete patient files to ./results_incomplete.
    """
    print(f"Finding complete patient datasets in {folder}...")

    path = Path(f"./{folder}")
    if feat == "power":
        files = list(path.glob("*_power.pkl"))
        file_pattern = r"(VEP\d+)_([123])_power"
        required_suffixes = ["_power.pkl"]
    elif feat == "plv":
        files = list(path.glob("*_plv_stim.pkl"))
        file_pattern = r"(VEP\d+)_([123])_plv_stim"
        required_suffixes = ["_plv_stim.pkl", "_plv_base.pkl"]
    else:
        raise ValueError("Feature type must be 'power' or 'plv'")

    if not files:
        print(f"No power files found in {path.resolve()}")
        return []

    timepoints = args.time if isinstance(args.time, list) else [args.time]

    timepoint_mapping = {"1": "t0", "2": "t1", "3": "t2"}
    patient_times = defaultdict(set)

    for f in files:
        file = f.stem
        match = re.match(file_pattern, file)
        if not match:
            print(f"Unexpected filename: {file}")
            continue
        patient_id, time_num = match.groups()
        timepoint = timepoint_mapping.get(time_num)
        if timepoint:
            if feat == "power":
                patient_times[patient_id].add(timepoint)
            if feat == "plv":
                stim_file = path / f"{patient_id}_{time_num}_plv_stim.pkl"
                base_file = path / f"{patient_id}_{time_num}_plv_base.pkl"
                if stim_file.exists() and base_file.exists():
                    patient_times[patient_id].add(timepoint)

    complete = {pid for pid, tps in patient_times.items() if set(timepoints).issubset(tps)}

    all_files = []
    for suffix in required_suffixes:
        all_files.extend(path.glob(f"*{suffix}"))

    to_remove = [f for f in all_files if f.stem.split("_", 1)[0] not in complete]

    trash_folder = Path("./results_incomplete")
    trash_folder.mkdir(parents=True, exist_ok=True)

    for f in to_remove:
        dest = trash_folder / f.name
        if dest.exists():
            f.unlink()
        else:
            f.rename(dest)

    return sorted(complete)


def sync(folder_power: str, folder_plv: str, folder_incomplete: str) -> None:
    """
    Ensures only patients with both power and PLV (stim only) files for the same timepoint remain.
    Any unmatched files are moved to folder_incomplete.
    """
    power_path = Path(folder_power)
    plv_path = Path(folder_plv)
    trash = Path(folder_incomplete)
    trash.mkdir(parents=True, exist_ok=True)

    def extract_patient_time(fname: str, feat: str):
        if feat == "power":
            match = re.match(r"(VEP\d+)_([123])_power", fname)
        elif feat == "plv":
            match = re.match(r"(VEP\d+)_([123])_plv_stim", fname)
        return match.groups() if match else (None, None)

    power_files = list(power_path.glob("*_power.pkl"))
    power_map = defaultdict(list)
    for f in power_files:
        pid, t = extract_patient_time(f.name, "power")
        if pid and t:
            power_map[(pid, t)].append(f)

    plv_files = list(plv_path.glob("*_plv_stim.pkl"))
    plv_map = defaultdict(list)
    for f in plv_files:
        pid, t = extract_patient_time(f.name, "plv")
        if pid and t:
            plv_map[(pid, t)].append(f)

    valid_keys = set(power_map.keys()) & set(plv_map.keys())

    for (pid, t), files in list(power_map.items()):
        if (pid, t) not in valid_keys:
            for f in files:
                shutil.move(str(f), str(trash / f.name))

    for (pid, t), files in list(plv_map.items()):
        if (pid, t) not in valid_keys:
            for f in files:
                shutil.move(str(f), str(trash / f.name))


def add_patients(args: argparse.Namespace, processed_ids: set[str]) -> list[Path]:
    """
    Recover unprocessed .cnt files from older folder logic based on trial.
    (Niet critical voor nu, maar maken we ook portable.)
    """
    recovered_files = []
    base_data_path = Path(getattr(args, "data_root", "D:/"))

    folders_to_check = []
    if args.trial == "t0":
        folders_to_check = ["T0_T1", "T0_T2"]
    elif args.trial == "t1":
        folders_to_check = ["T0_T2"]
    elif args.trial == "t2":
        folders_to_check = ["T0_T1"]

    for folder_name in folders_to_check:
        folder_path = base_data_path / folder_name / "t0"
        if folder_path.exists():
            for f in folder_path.glob("*.cnt"):
                patient_id = re.match(r"(VEP\d+)", f.name)
                if patient_id and patient_id.group(1) not in processed_ids:
                    recovered_files.append(f)

    return sorted(recovered_files)


def sort_results(source_folder: str, power_folder: str, plv_folder: str) -> None:
    """
    Sorts .pkl files from a mixed folder into power and PLV subfolders based on filename.
    """
    source = Path(source_folder)
    power_dest = Path(power_folder)
    plv_dest = Path(plv_folder)
    power_dest.mkdir(parents=True, exist_ok=True)
    plv_dest.mkdir(parents=True, exist_ok=True)

    for f in source.glob("*.pkl"):
        name = f.name
        if "_power.pkl" in name:
            dest = power_dest / name
        elif "_plv_stim.pkl" in name or "_plv_base.pkl" in name:
            dest = plv_dest / name
        else:
            print(f"Skipping unknown file: {name}")
            continue

        if not dest.exists():
            shutil.move(str(f), str(dest))
