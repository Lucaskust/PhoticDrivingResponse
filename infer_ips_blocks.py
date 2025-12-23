import numpy as np
import mne

PATH = r"D:\cenobamate_eeg_1\VEP02_1.cnt"
TARGETS = np.array([6.0, 10.0, 20.0])           # Hz die je verwacht
TOL_HZ = 1.0                                    # tolerantiebereik per interval
MIN_PULSES_PER_BLOCK = 20                       # filter kleine rommelblokken

raw = mne.io.read_raw_ant(PATH, preload=False, verbose="error")
onsets = np.array(raw.annotations.onset, dtype=float)

# Sorteren voor zekerheid
onsets = np.sort(onsets)

# Intervals en instantane freq
dt = np.diff(onsets)
freq = 1.0 / dt

# Classificeer elk interval naar dichtstbijzijnde target
nearest_idx = np.argmin(np.abs(freq[:, None] - TARGETS[None, :]), axis=1)
nearest_hz = TARGETS[nearest_idx]
ok = np.abs(freq - nearest_hz) <= TOL_HZ

# Maak runs van opeenvolgende intervals met dezelfde class (en ok=True)
blocks = []
i = 0
while i < len(dt):
    if not ok[i]:
        i += 1
        continue
    hz = nearest_hz[i]
    start_i = i
    while i < len(dt) and ok[i] and nearest_hz[i] == hz:
        i += 1
    end_i = i  # intervals [start_i, end_i-1]

    # Vertaal interval-index naar onset-index (pulsjes)
    start_pulse = start_i
    end_pulse = end_i  # inclusive puls index is end_i
    n_pulses = (end_pulse - start_pulse) + 1

    if n_pulses >= MIN_PULSES_PER_BLOCK:
        t_start = onsets[start_pulse]
        t_end = onsets[end_pulse]
        dur = t_end - t_start
        blocks.append((hz, t_start, t_end, dur, n_pulses))

# Print samenvatting
print(f"Total triggers: {len(onsets)}")
print(f"Total blocks found: {len(blocks)}\n")
for j, (hz, t0, t1, dur, n) in enumerate(blocks, 1):
    print(f"Block {j:02d}: {hz:>4.0f} Hz | start {t0:8.3f}s | end {t1:8.3f}s | dur {dur:6.2f}s | pulses {n}")
