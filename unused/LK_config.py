from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_POWER = PROJECT_ROOT / "results_POWER"
RESULTS_PLV = PROJECT_ROOT / "results_PLV"

DERIVED_DIR = PROJECT_ROOT / "LK_derived"
FIG_DIR = PROJECT_ROOT / "LK_figures"
DERIVED_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Mapping van bestandsnaam VEPxx_1/2/3 naar timepoint
TIME_NUM_TO_TP = {"1": "t0", "2": "t1", "3": "t2"}

# Specparam fitting range (laat 0.5 Hz en hele hoge freq vaak weg voor stabielere fits)
SPEC_F_RANGE = (1, 45)

# Responders (zoals in main.py)
RESPONDER_IDS = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}

# Instellingen die vaak werken (wordt in code “veilig” gefilterd op wat de class accepteert)
SPEC_SETTINGS = dict(
    peak_width_limits=(2,12),
    max_n_peaks=6,
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
)

# LK_config.py



# -----------------------------
# SpecParam presets (FOOOF-like)
# -----------------------------
SPEC_PRESETS: Dict[str, Dict[str, Any]] = {
    # Jullie huidige baseline (zoals je beschreef)
    "baseline": dict(
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.10,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
    ),

    # Stim/harmonics sneller detecteren (gevoeliger)
    "harmonics_sensitive": dict(
        peak_width_limits=(1, 6),
        max_n_peaks=8,
        min_peak_height=0.05,
        peak_threshold=1.5,
        aperiodic_mode="fixed",
    ),

    # Minder ruispeaks / minder overfit
    "conservative": dict(
        peak_width_limits=(2, 8),
        max_n_peaks=4,
        min_peak_height=0.15,
        peak_threshold=2.5,
        aperiodic_mode="fixed",
    ),

    # Alleen gebruiken als je echt brede freq-range fit (bv. tot 70 Hz) en knee verwacht
    "knee_wideband": dict(
        peak_width_limits=(1, 8),
        max_n_peaks=6,
        min_peak_height=0.10,
        peak_threshold=2.0,
        aperiodic_mode="knee",
    ),
}

DEFAULT_PRESET = "baseline"

# Default freq-range per preset (optioneel; kun je ook gewoon overal 1–45 houden)
SPEC_RANGE_PRESET: Dict[str, Tuple[float, float]] = {
    "baseline": (1.0, 45.0),
    "harmonics_sensitive": (1.0, 45.0),
    "conservative": (1.0, 45.0),
    "knee_wideband": (1.0, 70.0),
}


def get_spec_settings(preset: str = DEFAULT_PRESET) -> Dict[str, Any]:
    """Return a copy of preset settings."""
    if preset not in SPEC_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(SPEC_PRESETS)}")
    return dict(SPEC_PRESETS[preset])


def get_spec_freq_range(preset: str = DEFAULT_PRESET) -> Tuple[float, float]:
    if preset not in SPEC_RANGE_PRESET:
        # fallback: jouw oude globale range als je die al had
        return (1.0, 45.0)
    return SPEC_RANGE_PRESET[preset]


def apply_overrides(settings: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """
    overrides: list of "key=value" strings.
    Supported keys are whatever SpecParam model accepts.
    Tries to parse numbers/tuples automatically.
    """
    if not overrides:
        return settings

    def _parse_value(v: str):
        v = v.strip()
        # tuple like "(1, 6)" or "1,6"
        if v.startswith("(") and v.endswith(")"):
            inner = v[1:-1]
            parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
            return tuple(float(p) if "." in p else int(p) for p in parts)
        if "," in v and all(p.strip().replace(".", "", 1).isdigit() for p in v.split(",")):
            parts = [p.strip() for p in v.split(",")]
            return tuple(float(p) if "." in p else int(p) for p in parts)
        # bool
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        # int/float
        if v.replace(".", "", 1).isdigit():
            return float(v) if "." in v else int(v)
        # string
        return v

    out = dict(settings)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = _parse_value(v)
    return out


def resolve_specparam_config(
    preset: str,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    overrides: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    Combine preset settings + optional manual overrides + optional fmin/fmax override.
    """
    settings = get_spec_settings(preset)
    settings = apply_overrides(settings, overrides)

    fr = get_spec_freq_range(preset)
    if fmin is not None:
        fr = (float(fmin), fr[1])
    if fmax is not None:
        fr = (fr[0], float(fmax))

    return settings, fr
