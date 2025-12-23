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