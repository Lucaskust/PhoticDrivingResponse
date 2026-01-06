import os
import re
import glob
import pandas as pd

import PhoticDrivingResponse.pdr.features.specparam_blocks as spb

# === PAS DIT AAN ===
IN_DIRS = [
    r"D:\cenobamate_eeg_1",
    r"D:\cenobamate_eeg_2",
    r"D:\cenobamate_eeg_3",
]
OUT_DIR = r"specparam_out_blocks_all"
# ===================

os.makedirs(OUT_DIR, exist_ok=True)

# verzamel alle cnt files uit alle mappen
cnt_files = [] 
for d in IN_DIRS:
    cnt_files += glob.glob(os.path.join(d, "*.cnt"))
cnt_files = sorted(set(cnt_files))

print(f"Found {len(cnt_files)} .cnt files total")

all_rows = []
failed = []

for f in cnt_files:
    base = os.path.basename(f).replace(".cnt", "")
    try:
        # run analyse -> schrijft summary csv weg
        spb.main(f, OUT_DIR, fmin=2, fmax=45, aperiodic_mode="fixed")

        summ_path = os.path.join(OUT_DIR, f"{base}_specparam_blocks_summary.csv")
        df = pd.read_csv(summ_path)

        # patient/timepoint uit naam: VEP02_1 -> patient=VEP02, timepoint=1
        m = re.match(r"^(VEP\d+)[_\-](\d+)$", base)
        if m:
            df.insert(0, "patient", m.group(1))
            df.insert(1, "timepoint", int(m.group(2)))
        else:
            df.insert(0, "patient", base)
            df.insert(1, "timepoint", None)

        all_rows.append(df)

    except Exception as e:
        failed.append((f, str(e)))
        print(f"[FAIL] {base}: {e}")

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(os.path.join(OUT_DIR, "all_specparam_blocks_summary.csv"), index=False)
    print(f"\nWrote: {os.path.join(OUT_DIR, 'all_specparam_blocks_summary.csv')}")

if failed:
    pd.DataFrame(failed, columns=["file", "error"]).to_csv(os.path.join(OUT_DIR, "failed_files.csv"), index=False)
    print(f"Wrote failed list: {os.path.join(OUT_DIR, 'failed_files.csv')}")
