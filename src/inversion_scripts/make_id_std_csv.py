import pandas as pd
import xarray as xr
import glob

allfiles = glob.glob(f"../obspack_data/GEOSChem.ObsPack*")
unique_ids = []

for f in allfiles:
    print(f)
    data = xr.open_dataset(f)
    for x in data["obspack_id"].values:
        x = x.decode().strip()
        x = x.split("_")[0]
        if x not in unique_ids:
            unique_ids.append(x)

df = pd.DataFrame({"obspack_id":unique_ids})
df["std"] = 15.0

df.to_csv("../ObsPack_std.csv", index=False)
