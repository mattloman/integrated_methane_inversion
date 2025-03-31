import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from src.inversion_scripts.utils import (
    filter_stationary,
    get_strdate,
    check_is_OH_element,
    check_is_BC_element,
)

from src.inversion_scripts.operators.operator_utilities import (
    get_gc_lat_lon,
    get_gc_p,
    get_gc_lev,
    read_all_geoschem,
    merge_pressure_grids,
    remap,
    remap_sensitivities,
    get_gridcell_list,
    get_3dgrid_list,
    nearest_loc,
)


def apply_average_stationary_operator(
    filename,
    n_elements,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    gc_cache,
    build_jacobian,
    period_i,
    config,
):
    """
    Apply the averaging stationary operator to map GEOS-Chem methane data to stationary observation space.

    Arguments
        filename       [str]        : obspack netcdf data file to read
        n_elements     [int]        : Number of state vector elements
        gc_startdate   [datetime64] : First day of inversion period, for GEOS-Chem and observations
        gc_enddate     [datetime64] : Last day of inversion period, for GEOS-Chem and observations
        xlim           [float]      : Longitude bounds for simulation domain
        ylim           [float]      : Latitude bounds for simulation domain
        gc_cache       [str]        : Path to GEOS-Chem output data
        build_jacobian [log]        : Are we trying to map GEOS-Chem sensitivities to stationary observation space?
        period_i       [int]        : kalman filter period
        config         [dict]       : dict of the config file

    Returns
        output         [dict]       : Dictionary with:
                                        - obs_GC : GEOS-Chem and observational methane data
                                        - observed methane
                                        - GEOS-Chem methane
                                        - observation lat, lon
                                        - observation lat index, lon index
                                          If build_jacobian=True, also include:
                                            - K      : Jacobian matrix
    """
    # Read observation data
    OBSPACK = read_stationary(filename, gc_startdate, gc_enddate)
    if OBSPACK == None:
        print(f"Skipping {filename} due to file processing issue.")
        return OBSPACK
    elif pd.isnull(OBSPACK): # read_stationary returns pd.NaT if {filename} has no observations in inversion period 
        OBSPACK = None
        return OBSPACK

    obs_ind = filter_stationary(
        OBSPACK, xlim, ylim, gc_startdate, gc_enddate
    )

    # Number of observations
    n_obs = len(obs_ind[0])
    print("Found", n_obs, " observations.")

    # get the lat/lons of gc gridcells
    gc_lat_lon_lev = get_gc_lat_lon(gc_cache, gc_startdate)

    # add gc gridcell levels to lat_lon_lev dictionary
    date = pd.to_datetime(gc_startdate).strftime("%Y%m%d_%H")
    file_species = f"GEOSChem.SpeciesConc.{date}00z.nc4"
    filename = f"{gc_cache}/{file_species}"
    with xr.open_dataset(filename) as gc_data:
        gc_lat_lon_lev["lev"] = gc_data["lev"].values

    # Define time threshold (hour 00 after the inversion period)
    date_after_inversion = str(gc_enddate + np.timedelta64(1, "D"))[:10].replace(
        "-", ""
    )
    time_threshold = f"{date_after_inversion}_00"

    # map obs into gridcells and average the observations
    # into each gridcell. Only returns gridcells containing observations
    obs_mapped_to_gc = average_obspack_observations(
        OBSPACK, gc_lat_lon_lev, obs_ind, time_threshold, gc_cache
    )
    n_gridcells = len(obs_mapped_to_gc)

    if build_jacobian:
        # Initialize Jacobian K
        jacobian_K = np.zeros([n_gridcells, n_elements], dtype=np.float32)
        jacobian_K.fill(np.nan)

        pertf = os.path.expandvars(
            f'{config["OutputPath"]}/{config["RunName"]}/'
            f"archive_perturbation_sfs/pert_sf_{period_i}.npz"
        )

        emis_perturbations_dict = np.load(pertf)
        emis_perturbations = emis_perturbations_dict["effective_pert_sf"]

    # create list to store the dates/hour of each gridcell
    all_strdate = [gridcell["time"] for gridcell in obs_mapped_to_gc]
    all_strdate = list(set(all_strdate))

    # Read GEOS_Chem data for the dates of interest
    all_date_gc = read_all_geoschem(
        all_strdate, gc_cache, n_elements, config, build_jacobian
    )

    # Initialize array with n_gridcells rows and 6 columns. Columns are
    # observed CH4, GEOSChem CH4, longitude, latitude, GEOSChem pressure, observation counts
    obs_GC = np.zeros([n_gridcells, 6], dtype=np.float32)
    obs_GC.fill(np.nan)

    # For each gridcell dict with TROPOMI obs:
    for i, gridcell_dict in enumerate(obs_mapped_to_gc):

        # Get GEOS-Chem data for the date of the observation:
        p_obs = gridcell_dict["p_obs"]
        dry_air_subcolumns = gridcell_dict["dry_air_subcolumns"]  # mol m-2
        apriori = gridcell_dict["apriori"]  # mol m-2
        avkern = gridcell_dict["avkern"]
        strdate = gridcell_dict["time"]
        GEOSCHEM = all_date_gc[strdate]

        # Get GEOS-Chem pressure edges for the cell
        p_gc = GEOSCHEM["PEDGE"][gridcell_dict["iGC"], gridcell_dict["jGC"], :]
        # Get GEOS-Chem methane for the cell
        gc_CH4 = GEOSCHEM["CH4"][gridcell_dict["iGC"], gridcell_dict["jGC"], :]
        # Get merged GEOS-Chem/TROPOMI pressure grid for the cell
        merged = merge_pressure_grids(p_obs, p_gc)
        # Remap GEOS-Chem methane to TROPOMI pressure levels
        obs_CH4 = remap(
            gc_CH4,
            merged["data_type"],
            merged["p_merge"],
            merged["edge_index"],
            merged["first_gc_edge"],
        )  # ppb
        # Convert ppb to mol m-2
        sat_CH4_molm2 = sat_CH4 * 1e-9 * dry_air_subcolumns  # mol m-2
        # Derive the column-averaged XCH4 that TROPOMI would see over this ground cell
        # using eq. 46 from TROPOMI Methane ATBD, Hasekamp et al. 2019
        virtual_tropomi = (
            sum(apriori + avkern * (sat_CH4_molm2 - apriori))
            / sum(dry_air_subcolumns)
            * 1e9
        )  # ppb

        # If building Jacobian matrix from GEOS-Chem perturbation simulation sensitivity data:
        if build_jacobian:

            # TODO: Eliminate redundant code mapping GC to
            #       TROPOMI when build_jacobian=True

            if config["OptimizeOH"]:
                vars_to_xch4 = ["jacobian_ch4", "emis_base_ch4", "oh_base_ch4"]
            else:
                vars_to_xch4 = ["jacobian_ch4", "emis_base_ch4"]

            xch4 = {}

            for v in vars_to_xch4:
                # Get GEOS-Chem jacobian ch4 at this lat/lon, for all vertical levels and state vector elements
                jacobian_lonlat = GEOSCHEM[v][
                    gridcell_dict["iGC"], gridcell_dict["jGC"], :, :
                ]
                # Map the sensitivities to TROPOMI pressure levels
                sat_deltaCH4 = remap_sensitivities(
                    jacobian_lonlat,
                    merged["data_type"],
                    merged["p_merge"],
                    merged["edge_index"],
                    merged["first_gc_edge"],
                )  # mixing ratio, unitless
                # Tile the TROPOMI averaging kernel
                avkern_tiled = np.transpose(np.tile(avkern, (n_elements, 1)))
                # Tile the TROPOMI dry air subcolumns
                dry_air_subcolumns_tiled = np.transpose(
                    np.tile(dry_air_subcolumns, (n_elements, 1))
                )  # mol m-2
                # Derive the change in column-averaged XCH4 that TROPOMI would see over this ground cell
                xch4[v] = np.sum(
                    avkern_tiled * sat_deltaCH4 * dry_air_subcolumns_tiled, 0
                ) / sum(
                    dry_air_subcolumns
                )  # mixing ratio, unitless

            # separate variables for convenience later
            pert_jacobian_xch4 = xch4["jacobian_ch4"]
            emis_base_xch4 = xch4["emis_base_ch4"]
            if config["OptimizeOH"]:
                oh_base_xch4 = xch4["oh_base_ch4"]

            # Calculate sensitivities and save in K matrix
            # determine which elements are for emis,
            # BCs, and OH
            is_oh = np.full(n_elements, False, dtype=bool)
            is_bc = np.full(n_elements, False, dtype=bool)
            is_emis = np.full(n_elements, False, dtype=bool)

            for e in range(n_elements):
                i_elem = e + 1
                # booleans for whether this element is a
                # BC element or OH element
                is_OH_element = check_is_OH_element(
                    i_elem, n_elements, config["OptimizeOH"], config["isRegional"]
                )

                is_BC_element = check_is_BC_element(
                    i_elem,
                    n_elements,
                    config["OptimizeOH"],
                    config["OptimizeBCs"],
                    is_OH_element,
                    config["isRegional"],
                )

                is_oh[e] = is_OH_element
                is_bc[e] = is_BC_element
            is_emis = ~np.equal(is_oh | is_bc, True)

            # fill pert base array with values
            # array contains 1 entry for each state vector element
            # fill array with nans
            base_xch4 = np.full(n_elements, np.nan)
            # fill emission elements with the base value
            base_xch4 = np.where(is_emis, emis_base_xch4, base_xch4)
            # fill BC elements with the base value, which is same as emis value
            base_xch4 = np.where(is_bc, emis_base_xch4, base_xch4)
            if config["OptimizeOH"]:
                # fill OH elements with the OH base value
                base_xch4 = np.where(is_oh, oh_base_xch4, base_xch4)

            # get perturbations and calculate sensitivities
            perturbations = np.full(n_elements, 1.0, dtype=float)

            if config["OptimizeOH"]:
                oh_perturbation = config["PerturbValueOH"]
            else:
                oh_perturbation = 1.0
            if config["OptimizeBCs"]:
                bc_perturbation = config["PerturbValueBCs"]
            else:
                bc_perturbation = 1.0

            # fill perturbation array with OH and BC perturbations
            perturbations[0 : is_emis.sum()] = emis_perturbations
            perturbations = np.where(is_oh, oh_perturbation, perturbations)
            perturbations = np.where(is_bc, bc_perturbation, perturbations)

            # calculate difference
            delta_xch4 = pert_jacobian_xch4 - base_xch4

            # calculate sensitivities
            sensi_xch4 = delta_xch4 / perturbations

            # fill jacobian array
            jacobian_K[i, :] = sensi_xch4

        # Save actual and virtual TROPOMI data
        obs_GC[i, 0] = gridcell_dict[
            "methane"
        ]  # Actual TROPOMI methane column observation
        obs_GC[i, 1] = virtual_tropomi  # Virtual TROPOMI methane column observation
        obs_GC[i, 2] = gridcell_dict["lon_sat"]  # TROPOMI longitude
        obs_GC[i, 3] = gridcell_dict["lat_sat"]  # TROPOMI latitude
        obs_GC[i, 4] = gridcell_dict["observation_count"]  # observation counts

    # Output
    output = {}

    # Always return the coincident TROPOMI and GEOS-Chem data
    output["obs_GC"] = obs_GC

    # Optionally return the Jacobian
    if build_jacobian:
        output["K"] = jacobian_K

    return output


def apply_stationary_operator(
    filename,
    n_elements,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    gc_cache,
    build_jacobian,
    period_i,
    config,
):
    """
    Apply the observation operator to map GEOS-Chem methane data to stationary observation space.

    Arguments
        filename       [str]        : obspack netcdf data file to read
        n_elements     [int]        : Number of state vector elements
        gc_startdate   [datetime64] : First day of inversion period, for GEOS-Chem and observations
        gc_enddate     [datetime64] : Last day of inversion period, for GEOS-Chem and observations
        xlim           [float]      : Longitude bounds for simulation domain
        ylim           [float]      : Latitude bounds for simulation domain
        gc_cache       [str]        : Path to GEOS-Chem output data
        build_jacobian [log]        : Are we trying to map GEOS-Chem sensitivities to observation space?
        period_i       [int]        : kalman filter period
        config         [dict]       : dict of the config file

    Returns
        output         [dict]       : Dictionary with one or two fields:
                                                        - obs_GC : GEOS-Chem and stationary methane data
                                                    - stationary methane
                                                    - GEOS-Chem methane
                                                    - stationary lat, lon
                                                    - stationry lat index, lon index
                                                      If build_jacobian=True, also include:
                                                        - K      : Jacobian matrix
    """

    # Read observation data
    OBSPACK = read_stationary(filename, gc_startdate, gc_enddate)
    if OBSPACK == None:
        print(f"Skipping {filename} due to file processing issue.")
        return OBSPACK
    elif pd.isnull(OBSPACK): # read_stationary returns pd.NaT if {filename} has no observations in inversion period 
        OBSPACK = None
        return OBSPACK

    obs_ind = filter_stationary(
        OBSPACK, xlim, ylim, gc_startdate, gc_enddate
    )

    # Number of TROPOMI observations
    n_obs = len(obs_ind[0])

    # If need to build Jacobian from GEOS-Chem perturbation simulation sensitivity data:
    if build_jacobian:
        # Initialize Jacobian K
        jacobian_K = np.zeros([n_obs, n_elements], dtype=np.float32)
        jacobian_K.fill(np.nan)

    # Initialize a list to store the dates we want to look at
    all_strdate = []

    # Define time threshold (hour 00 after the inversion period)
    date_after_inversion = str(gc_enddate + np.timedelta64(1, "D"))[:10].replace(
        "-", ""
    )
    time_threshold = f"{date_after_inversion}_00"

    # For each observation
    for k in range(n_obs):
        # Get the date and hour
        iObs = obs_ind[0][k]  # lat index
        jObs = obs_ind[1][k]  # lon index
        time = pd.to_datetime(str(OBSPACK["time"][iObs, jObs]))
        strdate = get_strdate(time, time_threshold)
        all_strdate.append(strdate)
    all_strdate = list(set(all_strdate))

    # Read GEOS_Chem data for the dates of interest
    all_date_gc = read_all_geoschem(
        all_strdate, gc_cache, n_elements, config, build_jacobian
    )

    # Initialize array with n_obs rows and 6 columns. Columns are TROPOMI CH4, GEOSChem CH4, longitude, latitude, II, JJ
    obs_GC = np.zeros([n_obs, 6], dtype=np.float32)
    obs_GC.fill(np.nan)

    # For each observation:
    for k in range(n_obs):

        # Get GEOS-Chem data for the date of the observation:
        iObs = obs_ind[0][k]
        jObs = obs_ind[1][k]
       # p_sat = TROPOMI["pressures"][iSat, jSat, :]
       # dry_air_subcolumns = TROPOMI["dry_air_subcolumns"][iSat, jSat, :]  # mol m-2
       # apriori = TROPOMI["methane_profile_apriori"][iSat, jSat, :]  # mol m-2
       # avkern = TROPOMI["column_AK"][iSat, jSat, :]
        time = pd.to_datetime(str(TROPOMI["time"][iSat, jSat]))
        strdate = get_strdate(time, time_threshold)
        GEOSCHEM = all_date_gc[strdate]
        dlon = np.median(np.diff(GEOSCHEM["lon"]))  # GEOS-Chem lon resolution
        dlat = np.median(np.diff(GEOSCHEM["lat"]))  # GEOS-Chem lon resolution

        # =======================================================
        #       Map GEOS-Chem to TROPOMI observation space
        # =======================================================

        # Otherwise, initialize tropomi virtual xch4 and virtual sensitivity as zero
        area_weighted_virtual_tropomi = 0  # virtual tropomi xch4
        area_weighted_virtual_tropomi_sensitivity = 0  # virtual tropomi sensitivity

        # Find GEOS-Chem lats & lons closest to the observation coordinates
        iGC = nearest_loc(OBSPACK["longitude"][iObs, jObs, :], gc_lons, tolerance=max(dlon, 0.5))
        jGC = nearest_loc(OBSPACK["latitude"][iObs, jObs, :], gc_lats, tolerance=max(dlat, 0.5))

        # If the tolerance in nearest_loc() is not satisfied, skip the observation
        if np.nan in iGC + jGC:
            continue
        
        

        # For each GEOS-Chem grid cell that touches the TROPOMI pixel:
        for gridcellIndex in range(len(gc_coords)):

            # Get GEOS-Chem lat/lon indices for the cell
            iGC, jGC = ij_GC[gridcellIndex]

            # Get GEOS-Chem pressure edges for the cell
            p_gc = GEOSCHEM["PEDGE"][iGC, jGC, :]

            # Get GEOS-Chem methane for the cell
            gc_CH4 = GEOSCHEM["CH4"][iGC, jGC, :]

            # Get merged GEOS-Chem/TROPOMI pressure grid for the cell
            merged = merge_pressure_grids(p_sat, p_gc)

            # Remap GEOS-Chem methane to TROPOMI pressure levels
            sat_CH4 = remap(
                gc_CH4,
                merged["data_type"],
                merged["p_merge"],
                merged["edge_index"],
                merged["first_gc_edge"],
            )  # ppb

            # Convert ppb to mol m-2
            sat_CH4_molm2 = sat_CH4 * 1e-9 * dry_air_subcolumns  # mol m-2

            # Derive the column-averaged XCH4 that TROPOMI would see over this ground cell
            # using eq. 46 from TROPOMI Methane ATBD, Hasekamp et al. 2019
            virtual_tropomi_gridcellIndex = (
                sum(apriori + avkern * (sat_CH4_molm2 - apriori))
                / sum(dry_air_subcolumns)
                * 1e9
            )  # ppb

            # Weight by overlapping area (to be divided out later) and add to sum
            area_weighted_virtual_tropomi += (
                overlap_area[gridcellIndex] * virtual_tropomi_gridcellIndex
            )  # ppb m2

            # If building Jacobian matrix from GEOS-Chem perturbation simulation sensitivity data:
            if build_jacobian:

                # Get GEOS-Chem perturbation sensitivities at this lat/lon, for all vertical levels and state vector elements
                sensi_lonlat = GEOSCHEM["jacobian_ch4"][iGC, jGC, :, :]

                # Map the sensitivities to TROPOMI pressure levels
                sat_deltaCH4 = remap_sensitivities(
                    sensi_lonlat,
                    merged["data_type"],
                    merged["p_merge"],
                    merged["edge_index"],
                    merged["first_gc_edge"],
                )  # mixing ratio, unitless

                # Tile the TROPOMI averaging kernel
                avkern_tiled = np.transpose(np.tile(avkern, (n_elements, 1)))

                # Tile the TROPOMI dry air subcolumns
                dry_air_subcolumns_tiled = np.transpose(
                    np.tile(dry_air_subcolumns, (n_elements, 1))
                )  # mol m-2

                # Derive the change in column-averaged XCH4 that TROPOMI would see over this ground cell
                tropomi_sensitivity_gridcellIndex = np.sum(
                    avkern_tiled * sat_deltaCH4 * dry_air_subcolumns_tiled, 0
                ) / sum(
                    dry_air_subcolumns
                )  # mixing ratio, unitless

                # Weight by overlapping area (to be divided out later) and add to sum
                area_weighted_virtual_tropomi_sensitivity += (
                    overlap_area[gridcellIndex] * tropomi_sensitivity_gridcellIndex
                )  # m2

        # Compute virtual TROPOMI observation as weighted mean by overlapping area
        # i.e., need to divide out area [m2] from the previous step
        virtual_tropomi = area_weighted_virtual_tropomi / sum(overlap_area)

        # For global inversions, area of overlap should equal area of TROPOMI pixel
        # This is because the GEOS-Chem grid is continuous
        if dlon > 2.0:
            assert (
                abs(sum(overlap_area) - polygon_tropomi.area) / polygon_tropomi.area
                < 0.01
            ), f"ERROR: overlap area ({sum(overlap_area)}) /= satellite pixel area ({polygon_tropomi.area})"

        # Save actual and virtual TROPOMI data
        obs_GC[k, 0] = TROPOMI["methane"][
            iSat, jSat
        ]  # Actual TROPOMI methane column observation
        obs_GC[k, 1] = virtual_tropomi  # Virtual TROPOMI methane column observation
        obs_GC[k, 2] = TROPOMI["longitude"][iSat, jSat]  # TROPOMI longitude
        obs_GC[k, 3] = TROPOMI["latitude"][iSat, jSat]  # TROPOMI latitude
        obs_GC[k, 4] = iSat  # TROPOMI index of longitude
        obs_GC[k, 5] = jSat  # TROPOMI index of latitude

        if build_jacobian:
            # Compute TROPOMI sensitivity as weighted mean by overlapping area
            # i.e., need to divide out area [m2] from the previous step
            jacobian_K[k, :] = area_weighted_virtual_tropomi_sensitivity / sum(
                overlap_area
            )

    # Output
    output = {}

    # Always return the coincident TROPOMI and GEOS-Chem data
    output["obs_GC"] = obs_GC

    # Optionally return the Jacobian
    if build_jacobian:
        output["K"] = jacobian_K

    return output


def read_stationary(filename, gc_startdate, gc_enddate):
    """
    Read obspack data and save important variables to dictionary.
    Arguments
        filename [str]  : ObsPack netcdf data file to read
        gc_startdate   [datetime64] : First day of inversion period, for GEOS-Chem and TROPOMI
        gc_enddate     [datetime64] : Last day of inversion period, for GEOS-Chem and TROPOMI
    Returns
        dat      [dict] : Dictionary of important variables from stationary data:
                            - CH4 (mol/mol)
                            - Latitude
                            - Longitude
                            - Altitude (m)
                            - Time (s since Jan 1 1970)
                            - CH4 measurement uncertainty (mol/mol)
    """
    assert (
        "ch4" in filename
    ), f"ch4 not in filename {filename}, but values are being read in as methane measurements"

    try:
        # Initialize dictionary for stationary data
        dat = {}

        # Extract data from netCDF file to our dictionary
        with xr.open_dataset(filename) as obs_data:
            obs_time = obs_data["time"].values[:]
            # check for data in time range before reading in any other data
            time_inrange = any(list(filter(lambda x: x >= gc_startdate and x <= gc_enddate, obs_time)))
            if time_inrange: 
                dat["time"] = obs_time
                dat["methane"] = obs_data["value"].values[:]
                dat["longitude"] = obs_data["longitude"].values[:]
                dat["latitude"] = obs_data["latitude"].values[:]
                dat["altitude"] = obs_data["altitude"].values[:]
                dat["std_dev"] = obs_data["value_std_dev"].values[:]
                dat["n"] = obs_data["nvalue"].values[:]
                # Add an axis here to mimic the (scanline, groundpixel) format of operational TROPOMI data
                # This is so the blended data will be compatible with the TROPOMI operators
                for key in dat.keys():
                    dat[key] = np.expand_dims(dat[key], axis=0)

            else: # if none, return pd.NaT
                dat = pd.NaT

    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    return dat


def average_obspack_observations(OBSPACK, gc_lat_lon_lev, obs_ind, time_threshold, gc_cache):
    """
    Map observations into appropriate gc gridcells. Then average all
    observations within a gridcell for processing.

    Arguments
        OBSPACK        [dict]   : Dict of observation data
        gc_lat_lon     [list]   : list of dictionaries containing gc gridcell info
        obs_ind        [int]    : index list of stationary data that passes filters
        gc_cache       [str]    : Path to GEOS-Chem output data

    Returns
        output         [dict[]]   : flat list of dictionaries the following fields:
                                    - lat                 : gridcell latitude
                                    - lon                 : gridcell longitude
                                    - iGC                 : longitude index value
                                    - jGC                 : latitude index value
                                    - lat_obs             : averaged observation latitude
                                    - lon_obs             : averaged observation longitude
                                    - time                : averaged time
                                    - methane             : averaged methane
                                    - std_dev             : standard deviation of averaged methane
                                    - observation_count   : number of observations averaged in cell

    """
    n_obs = len(obs_ind[0])
    gc_lats = gc_lat_lon_lev["lat"]
    gc_lons = gc_lat_lon_lev["lon"]
    gc_levs = gc_lat_lon_lev["lev"]
    dlon = np.median(np.diff(gc_lat_lon_lev["lon"]))  # GEOS-Chem lon resolution
    dlat = np.median(np.diff(gc_lat_lon_lev["lat"]))  # GEOS-Chem lon resolution
    gridcell_dicts = get_3dgrid_list(gc_lons, gc_lats, gc_levs)

    for k in range(n_obs):
        iObs = obs_ind[0][k]  # empty index (the one added in read_stationary) 
        jObs = obs_ind[1][k]  # actual index of observation that passed filter_stationary

        # Find GEOS-Chem 3d coords closest to the observation coordinates
        iGC = nearest_loc(OBSPACK["longitude"][iObs, jObs], gc_lons, tolerance=max(dlon, 0.5))
        jGC = nearest_loc(OBSPACK["latitude"][iObs, jObs], gc_lats, tolerance=max(dlat, 0.5))
        
        # If the tolerance in nearest_loc() is not satisfied, skip the observation
        if np.isnan(iGC + jGC):
            continue

        # convert times to epoch time to make taking the mean easier
        time_obs = int(pd.to_datetime(str(OBSPACK["time"][iObs, jObs])).strftime("%s"))
        kGC = get_gc_lev(gc_cache, time_obs, OBSPACK["altitude"][iObs, jObs], [iGC, jGC])

        # Add obs info to gridcell_dicts
        gridcell_dict = gridcell_dicts[iGC][jGC][kGC]
        gridcell_dict["lat_obs"].append(OBSPACK["latitude"][iObs, jObs])
        gridcell_dict["lon_obs"].append(OBSPACK["longitude"][iObs, jObs])
        #gridcell_dict["p_obs"].append(  # convert obs altitude to GC level and then to pressure
        #    get_gc_p(gc_cache, time_obs, [iGC, jGC, kGC])
        #)
        #gridcell_dict["lev_obs"].append(kGC)
        gridcell_dict["time"].append(time_obs) 
        gridcell_dict["methane"].append(
            OBSPACK["methane"][iObs, jObs]
        )  # Actual methane observation
        gridcell_dict["std_dev"].append(OBSPACK["std_dev"][iObs, jObs])
        # increment the observation count
        gridcell_dict["observation_count"] += 1

    # filter out gridcells without any observations
    gridcell_dicts = [
        item for item in gridcell_dicts.flatten() if item["observation_count"] > 0
    ]
    # average observation values for each gridcell
    for gridcell_dict in gridcell_dicts:

        gridcell_dict["lat_obs"] = np.average(
            gridcell_dict["lat_obs"],
        )
        gridcell_dict["lon_obs"] = np.average(
            gridcell_dict["lon_obs"],
        )
        gridcell_dict["p_obs"] = np.average(
            gridcell_dict["p_obs"]
        )
        gridcell_dict["methane"] = np.average(
            gridcell_dict["methane"],
        )
        gridcell_dict["std_dev"] = np.sqrt(
            np.square(gridcell_dict["std_dev"])
        ) / len(gridcell_dict["std_dev"]) 
        # take mean of epoch times and then convert gc filename time string
        time = pd.to_datetime(
            datetime.datetime.fromtimestamp(int(np.mean(gridcell_dict["time"])))
        )
        gridcell_dict["time"] = get_strdate(time, time_threshold)
        
        # for level just return unique values along 0 axis
        # gridcell_dict["lev_obs"] = np.unique(
        #     gridcell_dict["lev_obs"],
        #     axis=0,
        # )
    return gridcell_dicts
