"""
Module patients selects patientfiles, loads their EEG data and filters them.
"""
import re
import argparse
from pathlib import Path
DATA_ROOT = Path(r"D:/")

TIME_TO_FOLDER = {
    "t0": "cenobamate_eeg_1",
    "t1": "cenobamate_eeg_2",
    "t2": "cenobamate_eeg_3",
}
from collections import defaultdict
import shutil
import numpy as np
import pandas as pd
import mne
from mne.io.base import BaseRaw
mne.set_log_level('error')

def parse_args()-> tuple[dict[str, str], argparse.Namespace]:
    """
    Parse command-line arguments for trial and time selection. 
    It assumes that patientdata is ordened in a time map, within a trial map.

    Returns
    -------
    :trial_map: dict of str
        Mapping from trial keys ("t0", "t1", "t2") to their corresponding folder names.
    :time_map: dict[str, list[str]]
        Mapping from trial keys to theis valid timepoints.
    :args: argparse.Namespace
        Parsed arguments with attributes:
        - trial : str
            Selected trial key.
        - time : str
            Selected time key (validated against the trial).
    """
    parser = argparse.ArgumentParser()

    trial_map = {
        "t0": "T0_T1_T2",
        "t1": "T0_T1",
        "t2": "T0_T2"}
    parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t2", help = "Choose between t0, t1, t2")

    time_map = {
        "t0": ["t0", "t1", "t2"],
        "t1": ["t0", "t1"],
        "t2": ["t0", "t2"]
    }
    parser.add_argument("-t", "--time", choices = list(time_map.keys()), default="all",
                        help = "Choose point in time: t0, t1, t2 or all")

    args = parser.parse_args()
    if args.time == "all":
        args.time = time_map[args.trial]
    elif args.time not in time_map[args.trial]:
        raise ValueError(f"Error: timepoint in {args.time} is not valid vor trial {args.trial}")

    return trial_map, time_map, args

from pathlib import Path

def patient_files(trial_map: dict[str, str], args: argparse.Namespace) -> list[Path]:
    """
    Returns a list of patient files for a given trial/time combo.
    Aangepast voor jouw Windows USB-structuur:

        D:/cenobamate_eeg_1  -> t0
        D:/cenobamate_eeg_2  -> t1
        D:/cenobamate_eeg_3  -> t2
    """
    timepoints = args.time if isinstance(args.time, list) else [args.time]

    files: list[Path] = []

    for tp in timepoints:
        folder_name = TIME_TO_FOLDER.get(tp)
        if folder_name is None:
            continue

        folder = DATA_ROOT / folder_name
        print(f"Zoek in map: {folder}")  # debug

        if not folder.exists():
            print(f"  LET OP: map bestaat niet.")
            continue

        # Pak gewoon alle .cnt-bestanden (ongeacht precieze naam)
        new_files = sorted(folder.glob("*.cnt"))
        print(f"  Gevonden .cnt-bestanden: {len(new_files)}")

        files.extend(new_files)

    print(f"Totaal aantal .cnt-bestanden dat verwerkt gaat worden: {len(files)}")
    return files


def eeg(src, passband, notch = 50, occi: bool = False, plot: bool = False)-> BaseRaw:
    """
    Loads the EEG data.

    Parameters
    ----------
    :src: Path
        Pathway to EEG-data file.
    :passband: 1x2 list
        List containing the lower and upper frequency boundary of the passband filter.
    :notch: float
        Line frequency during the measurement to notch-filter for.
    :occi: bool, optional
            Option to choose either all channels or only the occipital ones.
    :plot: bool, optional
        Option to plot the filtered raw EEG data with basic line- and passbandfiltering.

    Returns
    -------
    :raw: mne.BaseRaw
        Raw EEG data with basic line- and passbandfiltering.
    """
    raw = mne.io.read_raw_ant(src, preload=True, verbose='ERROR')
    raw.filter(l_freq=passband[0], h_freq=passband[1], picks="eeg", verbose='ERROR')

    line_freq = notch if (freq := raw.info["line_freq"]) is None else freq
    lowpass = np.arange(line_freq, raw.info["lowpass"]+1, line_freq)
    raw.notch_filter(freqs=(lowpass), notch_widths=(lowpass)/line_freq, picks=["eeg"], verbose='ERROR')

    if "EOG" in raw.ch_names:
        raw.drop_channels("EOG")
    if occi:
        raw = raw.copy().pick(["O1", "O2", "Oz"])

    threshold = 1e-6
    channels_dropped = set([ch for ch in raw.ch_names if np.all(np.abs(raw.get_data(picks=ch))<threshold)])
    if channels_dropped:
        raw.drop_channels([ch for ch in channels_dropped if ch in raw.ch_names])
        print(f"Dropped channels: {channels_dropped}")

    if plot:
        raw.plot(scalings = "auto", title="Filtered EEG data", show=True, block=True)

    return raw

def save_pickle_results(data: pd.DataFrame, pt_file: str, folder_name: str, feat: str = "power"):
    """
    Saves dataframes (singles or multiple) as pickle files in designated pathway.

    Parameters
    ----------
    :data: pd.DataFrame | tuple | list
        Name of the patient file: VEPxx_T, xx = patient ID, T = timepoint.
    :pt_file: str
        Dataframe containing powers at different frequencies. 
    :folder_name: str
        Name for the folder to save the pickle files. 
    :feat: str
        Type of feature that is being saved. Either "power" or "plv".
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

def filter_files(folder: str, time_map: dict[str, list[str]], args: argparse.Namespace,
                 feat: str = "power") -> list[str]:
    """
    Filters patients with complete data across required timepoints for either power or PLV results.
    Moves incomplete patient files to ./results_incomplete.

    Parameters
    :folder: str
        Name of the folder containing the power results.
    :trial_map: dict
        Dictionary mapping trial arguments to folder names.
    :args: argparse
        Parsed command-line arguments, must contain `trial` and `time`.
    :feat: str
        Feature type: "power" or "plv

    Returns 
    -------
        List of patient IDs with complete data across required timepoints.  
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

    if args.time == "all":
        args.time = time_map[args.trial]
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
            # resultaat staat al in results_incomplete -> verwijder de dubbel in de bronmap
            f.unlink()
        else:
            f.rename(dest)


    return sorted(complete)

def sync(folder_power: str, folder_plv: str, folder_incomplete: str) -> None:
    """
    Ensures only patients with both power and PLV (stim only) files for the same timepoint remain.
    Any unmatched files are moved to ./results_incomplete.

    Parameters
    ----------
    :folder_power: str
        Path to the folder containing *_power.pkl files.
    :folder_plv: str
        Path to the folder containing *_plv_stim.pkl and *_plv_base.pkl files.
    :folder_incomplete: str
        Folder to move incomplete results into.
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

    # Collect patient-time combinations for power
    power_files = list(power_path.glob("*_power.pkl"))
    power_map = defaultdict(list)
    for f in power_files:
        pid, t = extract_patient_time(f.name, "power")
        if pid and t:
            power_map[(pid, t)].append(f)

    # Collect patient-time combinations for PLV
    plv_files = list(plv_path.glob("*_plv_stim.pkl"))
    plv_map = defaultdict(list)
    for f in plv_files:
        pid, t = extract_patient_time(f.name, "plv")
        if pid and t:
            plv_map[(pid, t)].append(f)

    # Find matching sets (patient, time) that exist in BOTH modalities
    valid_keys = set(power_map.keys()) & set(plv_map.keys())

    # Move any file whose (patient, time) is NOT valid
    for (pid, t), files in list(power_map.items()):
        if (pid, t) not in valid_keys:
            for f in files:
                dest = trash / f.name
                shutil.move(str(f), str(dest))

    for (pid, t), files in list(plv_map.items()):
        if (pid, t) not in valid_keys:
            for f in files:
                dest = trash / f.name
                shutil.move(str(f), str(dest))

def add_patients(args: argparse.Namespace, processed_ids: set[str]) -> list[Path]:
    """
    Recover unprocessed .cnt files from T0-only folders based on trial logic.
    
    Parameters
    ----------
    :args: argpare.Namespace
        Parsed arguments with attributes trial & time.
    :processed_ids: set of str
        Patient files that have been used as train data.

    Returns
    -------
    :sorted(recovered_files): list
        Patient files that will be used as unseen test data.
    """
    recovered_files = []
    base_data_path = Path("D:/")  # <-- USB-stick, evt. aanpassen als jouw letter anders is
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

    Parameters
    ----------
    :source_folder: str
        Path to the folder containing mixed .pkl files.
    :power_folder: str
        Destination folder for power files.
    :plv_folder: str
        Destination folder for PLV files.
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
