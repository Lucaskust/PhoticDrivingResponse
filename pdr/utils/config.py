from pathlib import Path
import tomllib

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)
