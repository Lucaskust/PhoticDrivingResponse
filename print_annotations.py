import mne

path = r"D:\cenobamate_eeg_1\VEP02_1.cnt"

raw = mne.io.read_raw_ant(path, preload=False, verbose="error")

print("sfreq:", raw.info["sfreq"])
print("n_times:", raw.n_times)
print("duration (s):", raw.n_times / raw.info["sfreq"])

print("\nUnique annotation descriptions (first 50):")
descs = list(raw.annotations.description)
print("n unique:", len(set(descs)))
print(sorted(set(descs))[:50])

print("\nFirst 30 annotations (onset, duration, description):")
for i in range(30):
    print(raw.annotations.onset[i], raw.annotations.duration[i], raw.annotations.description[i])


print("\nTotal annotations:", len(raw.annotations))
