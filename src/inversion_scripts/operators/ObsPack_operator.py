import xarray as xr
import os
import numpy as np
import pandas as pd

def get_std(GC_ObsPacks):
    csv = pd.read_csv("../ObsPack_std.csv", index_col='station')
    mapping = csv.to_dict()["sd"]

    GC_ObsPacks["std"] = GC_ObsPacks["obspack_id"].astype(str)
    GC_ObsPacks["std"] = xr.apply_ufunc(lambda x: x.strip().split("_")[0], GC_ObsPacks["std"], vectorize=True)
    GC_ObsPacks["std"] = xr.apply_ufunc(lambda x: mapping[x[:3]], GC_ObsPacks["std"], vectorize=True)

    return GC_ObsPacks["std"]

def apply_obspack_operator(
    filename,
    n_elements,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    build_jacobian,
    period_i,
    config,
    use_water_obs=False,
    csv_std=True,
):
    """
    Apply the averaging tropomi operator to map GEOS-Chem methane data to TROPOMI observation space.

    Arguments
        filename       [str]        : TROPOMI netcdf data file to read
        BlendedTROPOMI [bool]       : if True, use blended TROPOMI+GOSAT data
        n_elements     [int]        : Number of state vector elements
        gc_startdate   [datetime64] : First day of inversion period, for GEOS-Chem and TROPOMI
        gc_enddate     [datetime64] : Last day of inversion period, for GEOS-Chem and TROPOMI
        xlim           [float]      : Longitude bounds for simulation domain
        ylim           [float]      : Latitude bounds for simulation domain
        gc_cache       [str]        : Path to GEOS-Chem output data
        build_jacobian [log]        : Are we trying to map GEOS-Chem sensitivities to TROPOMI observation space?
        period_i       [int]        : kalman filter period
        config         [dict]       : dict of the config file
        use_water_obs  [bool]       : if True, use observations over water

    Returns
        output         [dict]       : Dictionary with:
                                        - obs_GC : GEOS-Chem and TROPOMI methane data
                                        - TROPOMI methane
                                        - GEOS-Chem methane
                                        - TROPOMI lat, lon
                                        - TROPOMI lat index, lon index
                                          If build_jacobian=True, also include:
                                            - K      : Jacobian matrix
    """

    # # Tempoary variables for testing
    # filename = "../obspack_data/GEOSChem.ObsPack.20180501_0000z.nc4"
    # config = {}
    # period_i = 1
    # config["RunName"] = "NY_CT_final_all_2018"
    # config["OutputPath"] = "../../"
    # build_jacobian = True
    # n_elements = 261
    # ntracers = 2
    # config["OptimizeBCs"] = True
    # config["PerturbValueBCs"] = 10
    # config["OptimizeBCs"] = True
    # config["isRegional"] = True
    # config["LognormalErrors"] = True
    # Read dataded

    print(f"Working up: {filename}")

    opt_BC = config["OptimizeBCs"]
    is_Regional = config["isRegional"]

    ObsPack = xr.open_dataset(filename)
    # add 3-character ID for station filtering
    ObsPack["id3char"] = ObsPack["obspack_id"].astype(str)
    ObsPack["id3char"] = xr.apply_ufunc(lambda x: x[:3], ObsPack["id3char"], vectorize=True)

    ObsPack_name = filename.split("/")[-1] #Would be nice to get the file name and path seperate. This is back to front.

    # Number of ObsPack observations
    n_obs = len(ObsPack["obs"])
    print("Found", n_obs, "ObsPack observations.")

    # At a later date include a gc_cache like work up but this is ok for now.
    jacobian_dir = "../jacobian_runs"
    GC_dirs = sorted([os.path.join(jacobian_dir,x) for x in os.listdir(jacobian_dir) if config["RunName"] in x])
    GC_dirs.sort()

    output_path = os.path.join("OutputDir", ObsPack_name)
    GC_paths = [os.path.join(x, output_path) for x in GC_dirs]

    output_0000 = os.path.join(GC_dirs[0], "OutputDir")
    grid_data = xr.open_dataset([os.path.join(output_0000, x) for x in os.listdir(output_0000) if "LevelEdge"][0])

    GC_ObsPacks = []
    for i, path in enumerate(GC_paths[:2]):
        print(path)
        pack = xr.open_dataset(path)
        pack = pack[["CH4"]]
        pack = pack.rename({"CH4": f"CH4_{'base' if i == 0 else 'base_emis'}"})
        GC_ObsPacks.append(pack)

    #GC_ObsPacks.append(ObsPack[["value", "time", "latitude", "longitude", "obspack_id"]]) # Merge the observation values.
    GC_ObsPacks.append(ObsPack[["value", "time", "latitude", "longitude", "obspack_id", "id3char"]]) # Merge the observation values.
    GC_ObsPacks = xr.merge(GC_ObsPacks)

    grid_lats = grid_data["lat"].values
    grid_lons = grid_data["lon"].values

    GC_ObsPacks["latitude"] = xr.apply_ufunc(lambda x: grid_lats[np.argmin(abs(grid_lats - x))], GC_ObsPacks["latitude"], vectorize=True)
    GC_ObsPacks["longitude"] = xr.apply_ufunc(lambda x: grid_lons[np.argmin(abs(grid_lons - x))], GC_ObsPacks["longitude"], vectorize=True)

    jacobian_ObsPack = []
    for path in GC_paths[1:]:
        print(path)
        pack = xr.open_dataset(path)
        pack = pack[[x for x in pack if "CH4_" in x]]
        pack_vars = [x for x in pack]
        for e in pack_vars:
            pack = pack.rename({e: e[-4:]})
        jacobian_ObsPack.append(pack)

    if config["OptimizeBCs"]:
        for i, path in enumerate(GC_paths[-4:]):
            print(path)
            pack = xr.open_dataset(path)
            pack = pack[[x for x in pack if "CH4" in x]]
            pack = pack.rename({"CH4": str(n_elements-(3-i)).zfill(4)})
            jacobian_ObsPack.append(pack)

    # add ID so we can subset by ID
    jacobian_ObsPack.append(ObsPack[["value", "time", "latitude", "longitude"]])

    jacobian_ObsPack = xr.merge(jacobian_ObsPack)

    # Time might want to be shifted to midpoints.
    # This would need two days open at a time, as 23.30 gets shifted to the next day.
    #GC_ObsPacks["time"] = GC_ObsPacks["time"] + (GC_ObsPacks["averaging_interval"]/2)

    # Output
    output = {}

    # Hour filter
    #hour_filter = pd.to_datetime(GC_ObsPacks["time"].values).hour == hour
    #GC_ObsPacks_hour = GC_ObsPacks.isel(obs=hour_filter)
    #jacobian_ObsPack_hour = jacobian_ObsPack.isel(obs=hour_filter)

    # Nan / zero filter
    nan_filter = np.logical_and(~np.isnan(GC_ObsPacks["value"].values), GC_ObsPacks["value"].values > 0)
    GC_ObsPacks = GC_ObsPacks.isel(obs=nan_filter)
    jacobian_ObsPack = jacobian_ObsPack.isel(obs=nan_filter)

    # measurement ID filter
    # remove observations with the obspack IDs listed
    id_filter = ~np.isin(GC_ObsPacks["id3char"].values,
                         ["ACT", "CON", "DNH", "DWS", "ECO", "HFM", "IAG", "NHA", "PSP", "TMD", "UNY", "WHT"]) # planeflights + pos mean bias
    #                     ["CON", "DNH", "DWS", "HFM", "IAG", "NHA", "PSP", "TMD", "UNY", "WHT"]) # sites with positive mean bias (model > obs)
    #                     ["ACT", "CON", "ECO", "IAG", "NHA"]) # planeflights
    GC_ObsPacks = GC_ObsPacks.isel(obs=id_filter)
    jacobian_ObsPack = jacobian_ObsPack.isel(obs=id_filter)

    if csv_std:
        GC_ObsPacks["std"] = get_std(GC_ObsPacks)

    # Average timestep. Take the average of Observations that happen in the same gridbox
    # In the same timestep. There is alot of redundancy here as the GC outputs will all be the
    # same but currently the aim is to minimise work up prior to the runs.


    #print("jacobian_ObsPack:", jacobian_ObsPack)
    #print("\nGC_ObsPacks:", GC_ObsPacks)

    pert_jacobian_matrix = jacobian_ObsPack.drop_vars(["value", "time", "latitude", "longitude"]).to_array().values
    emis_base_vector = GC_ObsPacks["CH4_base_emis"].values
    base_vector = GC_ObsPacks["CH4_base"].values

    n_obs_filtered = len(emis_base_vector)
    if build_jacobian:
        # Initialize Jacobian K
        jacobian_K = np.zeros([n_obs_filtered, n_elements], dtype=np.float32)
        jacobian_K.fill(np.nan)

        pertf = os.path.expandvars(
            f'{config["OutputPath"]}/{config["RunName"]}/'
            f"archive_perturbation_sfs/pert_sf_{period_i}.npz"
        )

        emis_perturbations_dict = np.load(pertf)
        emis_perturbations = emis_perturbations_dict["effective_pert_sf"]

    # Initialize array with n_gridcells rows and 5 columns. Columns are
    # TROPOMI CH4, GEOSChem CH4, longitude, latitude, observation counts
    obs_GC = np.zeros([n_obs_filtered, 6], dtype=np.float32)
    obs_GC.fill(np.nan)

    # For each gridcell dict with tropomi obs:
    for i in range(n_obs_filtered):

        # If building Jacobian matrix from GEOS-Chem perturbation simulation sensitivity data:
        if build_jacobian:
            base_xch4 = base_vector[i]
            emis_base_xch4 = emis_base_vector[i]
            pert_jacobian_xch4 = pert_jacobian_matrix[:,i]

            # if config["OptimizeOH"]:
            #     vars_to_xch4 = ["jacobian_ch4", "emis_base_ch4", "oh_base_ch4"]
            # else:
            #     vars_to_xch4 = ["jacobian_ch4", "emis_base_ch4"]


            # separate variables for convenience later


            # Not implemented
            # if config["OptimizeOH"]:
            #     oh_base_xch4 = xch4["oh_base_ch4"]

            # Calculate sensitivities and save in K matrix
            # determine which elements are for emis,
            # BCs, and OH
            is_oh = np.full(n_elements, False, dtype=bool)
            is_bc = np.full(n_elements, False, dtype=bool)
            is_emis = np.full(n_elements, False, dtype=bool)

            # for e in range(n_elements):
            #     i_elem = e + 1
            #     # booleans for whether this element is a
            #     # BC element or OH element
            #     # is_OH_element = check_is_OH_element(
            #     #     i_elem, n_elements, config["OptimizeOH"], config["isRegional"]
            #     # )
            #
            #     # is_BC_element = check_is_BC_element(
            #     #     i_elem,
            #     #     n_elements,
            #     #     config["OptimizeOH"],
            #     #     config["OptimizeBCs"],
            #     #     is_OH_element,
            #     #     config["isRegional"],
            #     # )
            #
            #     # is_oh[e] = is_OH_element
            #     # is_bc[e] = is_BC_element
            # is_emis = ~np.equal(is_oh | is_bc, True)

            # get perturbations and calculate sensitivities
            perturbations = np.full(n_elements, 1.0, dtype=float)

            # fill pert base array with values
            # array contains 1 entry for each state vector element
            # fill array with nans
            base_xch4 = np.full(n_elements, np.nan)

            if config["OptimizeBCs"]:
                is_emis[:-4] = True

            # fill emission elements with the base value
            base_xch4 = np.where(is_emis, emis_base_xch4, base_xch4)

            # emissions perturbations
            perturbations[0 : is_emis.sum()] = emis_perturbations

            # OH perturbations Not Implemented
            # if config["OptimizeOH"]:
            #     # fill OH elements with the OH base value
            #     base_xch4 = np.where(is_oh, oh_base_xch4, base_xch4)
            #
            #     # compute BC perturbation for jacobian construction
            #     oh_perturbation = float(config["PerturbValueOH"]) - 1.0
            #
            #     # update perturbations array to include OH perturbations
            #     perturbations = np.where(is_oh, oh_perturbation, perturbations)

            # BC perturbations
            if config["OptimizeBCs"]:
                is_bc[-4:] = True
                # fill BC elements with the base value, which is same as emis value
                base_xch4 = np.where(is_bc, emis_base_xch4, base_xch4)

                # compute BC perturbation for jacobian construction
                bc_perturbation = config["PerturbValueBCs"]

                # update perturbations array to include OH perturbations
                perturbations = np.where(is_bc, bc_perturbation, perturbations)

            # calculate difference
            delta_xch4 = pert_jacobian_xch4 - base_xch4

            # calculate sensitivities
            sensi_xch4 = delta_xch4 / perturbations

            # fill jacobian array
            jacobian_K[i, :] = sensi_xch4

        # Save actual and virtual TROPOMI data
        obs_GC[i, 0] = GC_ObsPacks["value"].values[i]*1E9 # Observation
        obs_GC[i, 1] = GC_ObsPacks["CH4_base"].values[i]*1E9  # Virtual TROPOMI methane column observation
        obs_GC[i, 2] = GC_ObsPacks["longitude"].values[i] # longitude
        obs_GC[i, 3] = GC_ObsPacks["latitude"].values[i] # latitude
        obs_GC[i, 4] = 1  # observation counts
        obs_GC[i, 5] = GC_ObsPacks["std"].values[i]
        if obs_GC.shape[0] > 0:
            output[f"obs_GC"] = obs_GC

            # Optionally return the Jacobian
            if build_jacobian:
                output[f"K"] = jacobian_K

    return output
