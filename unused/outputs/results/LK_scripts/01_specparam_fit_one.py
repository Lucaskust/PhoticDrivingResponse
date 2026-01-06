from pathlib import Path
import re
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PhoticDrivingResponse.unused.LK_config import RESULTS_POWER, FIG_DIR, SPEC_F_RANGE, SPEC_SETTINGS, TIME_NUM_TO_TP

# Probeer specparam, anders fallback naar fooof (handig als importnaam anders blijkt)
def get_model_class():
    try:
        from specparam import SpectralModel
        return SpectralModel
    except Exception:
        try:
            from fooof import FOOOF
            return FOOOF
        except Exception as e:
            raise ImportError("Kan specparam of fooof niet importeren. Check je install.") from e

PATTERN = re.compile(r"(VEP\d+)_([123])_power\.pkl$")

def extract_freqs_psd(df: pd.DataFrame):
    # 1) freq uit index of kolom
    freqs = None
    if df.index is not None:
        try:
            freqs = df.index.to_numpy(dtype=float)
        except Exception:
            freqs = None

    num = df.select_dtypes(include="number").copy()

    # Als er een frequentiekolom is, gebruik die
    for c in ["freq", "freqs", "frequency", "frequencies", "Hz"]:
        if c in num.columns:
            freqs = num[c].to_numpy(dtype=float)
            num = num.drop(columns=[c])
            break

    if freqs is None:
        raise ValueError("Kon geen frequenties vinden (niet in index, niet in freq-kolom).")

    # 2) PSD als gemiddelde over numerieke kolommen
    if num.shape[1] == 0:
        raise ValueError("Geen numerieke PSD-kolommen gevonden.")
    psd = num.mean(axis=1).to_numpy(dtype=float)

    # 3) cleanup
    mask = np.isfinite(freqs) & np.isfinite(psd)
    freqs, psd = freqs[mask], psd[mask]

    if psd.size == 0:
        raise ValueError("PSD is leeg na opschonen (NaN/Inf filtering).")

    # Specparam/FOOOF verwachten lineaire PSD (>0). Niet zelf loggen.
    # Als er negatieve waarden in zitten, dan is dit waarschijnlijk al log10 of dB.
    if np.nanmin(psd) < 0:
        # Heuristiek: grote negatieve waarden zijn vaak dB
        if np.nanmedian(psd) < -50 or np.nanmin(psd) < -200:
            psd = 10 ** (psd / 10.0)   # dB -> lineair
        else:
            psd = 10 ** psd            # log10 -> lineair

    # Zorg dat alles strikt > 0 is (anders gaat specparam log10() stuk)
    eps = np.finfo(float).tiny
    psd = np.clip(psd, eps, None)

    return freqs, psd

def safe_init_model(ModelCls):
    sig = inspect.signature(ModelCls)
    kwargs = {k: v for k, v in SPEC_SETTINGS.items() if k in sig.parameters}
    return ModelCls(**kwargs)

def main():
    files = sorted(RESULTS_POWER.glob("*_power.pkl"))
    if not files:
        raise SystemExit(f"Geen power pickles gevonden in {RESULTS_POWER}")

    f = files[0]  # pak de eerste, je kan dit later hardcoderen
    m = PATTERN.search(f.name)
    pid, tnum = (m.group(1), m.group(2)) if m else ("UNKNOWN", "X")
    tp = TIME_NUM_TO_TP.get(tnum, "unknown")

    df = pd.read_pickle(f)
    freqs, psd = extract_freqs_psd(df)

    # fit range
    fmin, fmax = SPEC_F_RANGE
    fr_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_fit, psd_fit = freqs[fr_mask], psd[fr_mask]

    ModelCls = get_model_class()
    model = safe_init_model(ModelCls)
    model.fit(freqs_fit, psd_fit, freq_range=SPEC_F_RANGE)

    # plot
    plt.figure()
    if hasattr(model, "plot"):
        model.plot()
        plt.title(f"{pid} {tp} specparam fit")
    else:
        plt.plot(freqs_fit, psd_fit)
        plt.title(f"{pid} {tp} PSD (geen model.plot beschikbaar)")

    out = FIG_DIR / f"specparam_fit_{pid}_{tp}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out}")

if __name__ == "__main__":
    main()
