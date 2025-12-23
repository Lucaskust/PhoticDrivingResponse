import argparse
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

def main(cnt_path: str, out_dir: str = "quicklook"):
    os.makedirs(out_dir, exist_ok=True)

    raw = mne.io.read_raw_ant(cnt_path, preload=True, verbose="error")
    raw.pick_types(eeg=True)

    # Interactieve viewer: scroll door de volledige raw opname
    # Sluit het venster om het script door te laten gaan
    raw.plot(n_channels=32, duration=10, scalings="auto", show=True, block=True)

    # korte info
    sfreq = raw.info["sfreq"]
    n_samp = raw.n_times
    dur = n_samp / sfreq
    print(f"Loaded: {cnt_path}")
    print(f"Channels: {len(raw.ch_names)} | sfreq: {sfreq:.2f} Hz | duration: {dur:.2f} s")

    # 1) tijdsignaal (eerste 10s of hele recording)
    tmax = min(10.0, dur)
    data, times = raw[:, :int(tmax*sfreq)]
    fig1 = plt.figure()
    plt.plot(times, data[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title(f"Timeseries (1st channel) - {os.path.basename(cnt_path)}")
    fig1.savefig(os.path.join(out_dir, os.path.basename(cnt_path).replace(".cnt","_timeseries.png")), dpi=200)
    plt.close(fig1)

    # 2) PSD (Welch)
    psd = raw.compute_psd(method="welch", fmin=1, fmax=45, n_fft=int(sfreq*2), n_overlap=int(sfreq), n_per_seg=int(sfreq*2), verbose="error")
    freqs = psd.freqs
    p = psd.get_data()  # shape: (n_channels, n_freqs)

    fig2 = plt.figure()
    plt.plot(freqs, 10*np.log10(np.mean(p, axis=0)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"Mean PSD (dB) - {os.path.basename(cnt_path)}")
    fig2.savefig(os.path.join(out_dir, os.path.basename(cnt_path).replace(".cnt","_psd.png")), dpi=200)
    plt.close(fig2)

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnt", required=True, help="Path to .cnt file")
    ap.add_argument("--out", default="quicklook", help="Output directory")
    args = ap.parse_args()
    main(args.cnt, args.out)
