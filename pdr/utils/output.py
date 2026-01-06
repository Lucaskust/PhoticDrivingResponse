from pathlib import Path
from datetime import datetime
import shutil

def make_run_dir(outputs_root: str | Path, run_name: str = "run") -> Path:
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_root / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # standaard subfolders
    for sub in ["features/power", "features/plv", "features/specparam", "stats", "ml", "figures", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir

def copy_config(config_path: str | Path, run_dir: str | Path) -> None:
    config_path = Path(config_path)
    run_dir = Path(run_dir)
    shutil.copy2(config_path, run_dir / "config_used.toml")
