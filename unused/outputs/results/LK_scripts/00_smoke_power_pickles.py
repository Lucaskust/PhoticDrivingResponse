from pathlib import Path
import re
import pandas as pd
import numpy as np

from PhoticDrivingResponse.unused.LK_config import RESULTS_POWER

PATTERN = re.compile(r"(VEP\d+)_([123])_power\.pkl$")

def main():
    files = sorted(RESULTS_POWER.glob("*_power.pkl"))
    print(f"POWER pickles gevonden: {len(files)} in {RESULTS_POWER}")

    for f in files[:5]:
        m = PATTERN.search(f.name)
        print(f"\nFile: {f.name}, match: {bool(m)}")
        df = pd.read_pickle(f)
        print("Type:", type(df))
        if isinstance(df, pd.DataFrame):
            print("Shape:", df.shape)
            print("Columns (eerste 10):", list(df.columns)[:10])
            # Print snelle stats op numerieke data
            num = df.select_dtypes(include="number")
            if num.size:
                arr = num.to_numpy().astype(float)
                print("Min/Max (numeric):", np.nanmin(arr), np.nanmax(arr))
        else:
            print("Niet-DataFrame content, inspecteer even handmatig.")

if __name__ == "__main__":
    main()