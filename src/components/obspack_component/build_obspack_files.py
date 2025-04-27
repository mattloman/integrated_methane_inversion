import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime


def filtered_paths(obspack_config):
    filepaths = []
    for f in [os.path.join(obspack_config["data_path"],x) for x in os.listdir(obspack_config["data_path"])]:

        if os.path.isdir(f):
            print(f"Dropping {f} as it is a directory")
            continue

        elif (".NC" not in f.upper()) and (".NC4" not in f.upper()):
            print(f"Dropping {f} as it does not have a netcdf prefix")
            continue

        else:
            filepaths.append(f)

    return filepaths

def convert_dt(dt_str):
    dt = datetime.strptime(dt_str, "%Y%m%d")
    return np.datetime64(dt)


if __name__ == "__main__":
    obspack_config = {
        "data_path": sys.argv[1],
        "lat_min": float(sys.argv[2]),
        "lat_max": float(sys.argv[3]),
        "lon_min": float(sys.argv[4]),
        "lon_max": float(sys.argv[5]),
        "start_date": np.datetime64(convert_dt(sys.argv[6])),
        "end_date": np.datetime64(convert_dt(sys.argv[7])),
    }

    # For testing
    # obspack_config = {
    #     "data_path": "/scratch/mloman/data/obspack_all",
    #     "lat_min": 32.5,
    #     "lat_max": 52.75,
    #     "lon_min": -87.8125,
    #     "lon_max": -63.75,
    #     "start_date": np.datetime64(convert_dt("20200501")),
    #     "end_date": np.datetime64(convert_dt("20200601")),
    # }

    filepaths = filtered_paths(obspack_config)
    filepaths.sort()

    if not filepaths:
        raise ValueError(f"{obspack_config['data_path']} does not contain any valid ObsPack files.")


    day_dict = {x: [] for x in np.arange(obspack_config["start_date"],
                                         obspack_config["end_date"],
                                         dtype='datetime64[D]')}

    for path in filepaths:
        with xr.open_dataset(path) as obs_data:

            # Obspack_id isn't acutally used for anything in GC just has to exist, subsittude with file name.
            if "obspack_id" in obs_data.variables:
                df = obs_data[["time", "obs", "latitude", "longitude", "altitude", "obspack_id"]].to_pandas()
            else:
                df = obs_data[["time", "obs", "latitude", "longitude", "altitude"]].to_pandas()
                df["obspack_id"] = path.replace(obspack_config["data_path"], "")
                df["obspack_id"] = df["obspack_id"].astype("S200")

            df = df[df["time"].between(obspack_config["start_date"], obspack_config["end_date"])]
            df = df[df["latitude"].between(obspack_config["lat_min"], obspack_config["lat_max"])]
            df = df[df["longitude"].between(obspack_config["lon_min"], obspack_config["lon_max"])]

            if df.size == 0:
                #print(f"{path} dropped as outside of time or grid range.")
                continue

            # Currently we just treat every sample as hourly averaged.
            df["CT_sampling_strategy"] = 2

            for day in day_dict.keys():
                df_day = df[df["time"].dt.floor("d") == day]

                day_dict[day].append(df_day)


    for day in day_dict.keys():
        day_dict[day] = pd.concat(day_dict[day])
        day_ds = day_dict[day].to_xarray()

        day_str = day.astype("O").strftime("%Y%m%d")
        write_path = os.path.join(f"obspack_data/GEOSChem.ObsPack.{day_str}_0000z.nc4")

        day_ds.to_netcdf(write_path)
