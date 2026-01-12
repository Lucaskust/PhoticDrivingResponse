from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config.toml>")
        sys.exit(2)

    cfg = load_toml(Path(sys.argv[1]))

    problems, warns = [], []

    out = cfg.get("outputs", {})
    if out and not isinstance(out, dict):
        problems.append("[outputs] must be a table")

    labels = cfg.get("labels", {})
    if labels and not isinstance(labels, dict):
        problems.append("[labels] must be a table")

    ml = cfg.get("ml", {})
    if ml and not isinstance(ml, dict):
        problems.append("[ml] must be a table")

    feats = cfg.get("features", {})
    if feats and not isinstance(feats, dict):
        problems.append("[features] must be a table")

    sp = cfg.get("specparam", {})
    if sp and not isinstance(sp, dict):
        problems.append("[specparam] must be a table")

    if problems:
        print("[FAIL]")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    print("[OK] Config parsed.")
    if warns:
        print("[WARN]")
        for w in warns:
            print(" -", w)


if __name__ == "__main__":
    main()
