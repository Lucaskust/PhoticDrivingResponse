from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
# Python 3.11+ heeft tomllib standaard
import tomllib


def load_config(path: Path) -> Dict[str, Any]:
    """Load a TOML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    # defaults (zodat ontbrekende keys niet meteen alles slopen)
    cfg.setdefault("pipeline", {})
    cfg.setdefault("data", {})
    cfg.setdefault("preprocess", {})
    cfg.setdefault("power", {})
    cfg.setdefault("phase", {})
    cfg.setdefault("specparam", {})
    cfg.setdefault("output", {})

    # pipeline defaults
    cfg["pipeline"].setdefault("use_power", True)
    cfg["pipeline"].setdefault("use_plv", True)
    cfg["pipeline"].setdefault("use_specparam", True)

    # output defaults
    cfg["output"].setdefault("root", "outputs")
    cfg["output"].setdefault("run_tag", "dev")

    return cfg


def resolve_specparam_config(cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    """
    Resolve SpecParam settings:
    - neemt preset uit cfg["specparam"]["presets"][preset_name]
    - past daarna cfg["specparam"]["override"] toe
    - zet peak_width_limits om naar tuple
    Returns: (fmin, fmax, model_params_dict)
    """
    sp = cfg.get("specparam", {}) or {}

    fmin = float(sp.get("fmin", 2.0))
    fmax = float(sp.get("fmax", 45.0))
    aperiodic_mode = sp.get("aperiodic_mode", "fixed")

    preset_name = sp.get("preset", "baseline")
    presets = sp.get("presets", {}) or {}
    preset_params = presets.get(preset_name, {}) or {}

    override = sp.get("override", {}) or {}

    # merge (override wint)
    model_params: Dict[str, Any] = {**preset_params, **override}

    # altijd meenemen
    model_params["aperiodic_mode"] = aperiodic_mode

    # TOML lists -> tuple voor specparam
    if "peak_width_limits" in model_params:
        pwl = model_params["peak_width_limits"]
        if isinstance(pwl, list) and len(pwl) == 2:
            model_params["peak_width_limits"] = (float(pwl[0]), float(pwl[1]))

    # cast ints waar logisch
    for k in ["max_n_peaks"]:
        if k in model_params:
            model_params[k] = int(model_params[k])

    for k in ["min_peak_height", "peak_threshold"]:
        if k in model_params:
            model_params[k] = float(model_params[k])

    return fmin, fmax, model_params

def resolve_specparam_config(spec_cfg: dict, preset: str, fallback_presets: dict) -> Tuple[float, float, dict]:
    """
    Leest jouw TOML schema:

    [specparam]
    fmin, fmax, aperiodic_mode, override = {}
    [specparam.presets.<presetname>]  (bv baseline/conservative/harmonics)

    Returns:
      fmin_fit, fmax_fit, model_params (voor SpectralModel)
    """
    spec_cfg = spec_cfg or {}

    fmin = float(spec_cfg.get("fmin", 2.0))
    fmax = float(spec_cfg.get("fmax", 45.0))

    model_params = dict(fallback_presets.get(preset, {}))

    # aperiodic_mode uit config heeft prioriteit
    if "aperiodic_mode" in spec_cfg and spec_cfg["aperiodic_mode"] is not None:
        model_params["aperiodic_mode"] = str(spec_cfg["aperiodic_mode"])

    # preset-specifieke settings uit TOML
    toml_presets = spec_cfg.get("presets", {}) or {}
    if preset in toml_presets and isinstance(toml_presets[preset], dict):
        model_params.update(toml_presets[preset])

    # globale override (overschrijft alles)
    override = spec_cfg.get("override", {}) or {}
    if isinstance(override, dict):
        model_params.update(override)

    # peak_width_limits: TOML list -> tuple
    if "peak_width_limits" in model_params and isinstance(model_params["peak_width_limits"], (list, tuple)):
        model_params["peak_width_limits"] = tuple(float(x) for x in model_params["peak_width_limits"])

    return fmin, fmax, model_params
