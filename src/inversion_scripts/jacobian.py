#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
import re
import os
import datetime
import yaml
import gc
from src.inversion_scripts.utils import save_obj
from src.inversion_scripts.operators.TROPOMI_operator import (
    apply_average_tropomi_operator,
    apply_tropomi_operator,
)
from src.inversion_scripts.operators.ObsPack_operator import apply_obspack_operator
from joblib import Parallel, delayed


def apply_operator(operator, params, config):
    """
    Run the chosen operator based on selected instrument

    Arguments
        operator [str]    : Data conversion operator to use
        params   [dict]   : parameters to run the given operator
    Returns
        output   [dict]   : Dictionary with:
                            - obs_GC : GEOS-Chem and TROPOMI methane data
                            - TROPOMI methane
                            - GEOS-Chem methane
                            - TROPOMI lat, lon
                            - TROPOMI lat index, lon index
                              If build_jacobian=True, also include:
                                - K      : Jacobian matrix
    """
    if operator == "TROPOMI_average":
        return apply_average_tropomi_operator(
            params["filename"],
            params["BlendedTROPOMI"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["build_jacobian"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    elif operator == "TROPOMI":
        return apply_tropomi_operator(
            params["filename"],
            params["BlendedTROPOMI"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["build_jacobian"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    else:
        raise ValueError("Error: invalid operator selected.")


def get_tropomi(tropomi_cache, gc_startdate, gc_enddate):
    # Get TROPOMI data filenames for the desired date range
    allfiles = glob.glob(f"{tropomi_cache}/*.nc")
    sat_files = []
    for index in range(len(allfiles)):
        filename = allfiles[index]
        shortname = re.split(r"\/", filename)[-1]
        shortname = re.split(r"\.", shortname)[0]
        strdate = re.split(r"\.|_+|T", shortname)[4]
        strdate = datetime.datetime.strptime(strdate, "%Y%m%d")
        if (strdate >= gc_startdate) and (strdate <= gc_enddate):
            sat_files.append(filename)
    sat_files.sort()
    print("Found", len(sat_files), "TROPOMI data files.")

    return sat_files

def get_obspack(gc_startdate, gc_enddate):
    allfiles = glob.glob(f"../obspack_data/GEOSChem.ObsPack*")
    obs_files = []
    for filename in allfiles:
        strdate = filename[-18:-10]
        strdate = datetime.datetime.strptime(strdate, "%Y%m%d")
        if (strdate >= gc_startdate) and (strdate <= gc_enddate):
            obs_files.append(filename)
    obs_files.sort()
    print("Found", len(obs_files), "ObsPack files.")

    return obs_files


if __name__ == "__main__":

    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    startday = sys.argv[2]
    endday = sys.argv[3]
    lonmin = float(sys.argv[4])
    lonmax = float(sys.argv[5])
    latmin = float(sys.argv[6])
    latmax = float(sys.argv[7])
    n_elements = int(sys.argv[8])
    tropomi_cache = sys.argv[9]
    BlendedTROPOMI = sys.argv[10].lower() == "true"
    use_water_obs = sys.argv[11].lower() == "true"
    isPost = sys.argv[12]
    period_i = int(sys.argv[13])
    build_jacobian = sys.argv[14]
    viz_prior = sys.argv[15]
    use_obspack = sys.argv[16].lower() == "true"
    obspack_cache = sys.argv[17]

    # Reformat start and end days for datetime in configuration
    start = f"{startday[0:4]}-{startday[4:6]}-{startday[6:8]} 00:00:00"
    end = f"{endday[0:4]}-{endday[4:6]}-{endday[6:8]} 23:59:59"

    # Configuration
    workdir = "."
    if build_jacobian.lower() == "true":
        build_jacobian = True
    else:
        build_jacobian = False
    if isPost.lower() == "false":  # if sampling prior simulation
        gc_cache = f"{workdir}/data_geoschem"
        outputdir = f"{workdir}/data_converted"
        vizdir = f"{workdir}/data_visualization"

        # for lognormal, we also sample the prior simulation in a
        # separate call to jacobian.py solely for visualization purposes
        if viz_prior.lower() == "true":
            gc_cache = f"{gc_cache}_prior"
            outputdir = f"{outputdir}_prior"
            vizdir = f"{vizdir}_prior"

    else:  # if sampling posterior simulation
        gc_cache = f"{workdir}/data_geoschem_posterior"
        outputdir = f"{workdir}/data_converted_posterior"
        vizdir = f"{workdir}/data_visualization_posterior"
    xlim = [lonmin, lonmax]
    ylim = [latmin, latmax]
    gc_startdate = np.datetime64(datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S"))
    gc_enddate = np.datetime64(
        datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=1)
    )
    print("Start:", start)
    print("End:", end)

    if use_obspack:
        files = get_obspack(gc_startdate, gc_enddate)
    else:
        files = get_tropomi(tropomi_cache, gc_startdate, gc_enddate)

    # Map GEOS-Chem to TROPOMI observation space
    # Also return Jacobian matrix if build_jacobian=True
    def process_tropomi(filename):

        # Check if TROPOMI file has already been processed
        print("========================")
        shortname = re.split(r"\/", filename)[-1]
        print(shortname)
        date = re.split(r"\.", shortname)[0]

        # If not yet processed, run apply_average_tropomi_operator()
        if not os.path.isfile(f"{outputdir}/{date}_GCtoTROPOMI.pkl"):
            print("Applying TROPOMI operator...")

            output = apply_operator(
                "TROPOMI_average",
                {
                    "filename": filename,
                    "BlendedTROPOMI": BlendedTROPOMI,
                    "n_elements": n_elements,
                    "gc_startdate": gc_startdate,
                    "gc_enddate": gc_enddate,
                    "xlim": xlim,
                    "ylim": ylim,
                    "gc_cache": gc_cache,
                    "build_jacobian": build_jacobian,
                    "period_i": period_i,
                    "use_water_obs": use_water_obs,
                },
                config,
            )

            # we also save out the unaveraged tropomi operator for visualization purposes
            viz_output = apply_operator(
                "TROPOMI",
                {
                    "filename": filename,
                    "BlendedTROPOMI": BlendedTROPOMI,
                    "n_elements": n_elements,
                    "gc_startdate": gc_startdate,
                    "gc_enddate": gc_enddate,
                    "xlim": xlim,
                    "ylim": ylim,
                    "gc_cache": gc_cache,
                    "build_jacobian": False,
                    "period_i": period_i,
                    "use_water_obs": use_water_obs,
                },
                config,
            )

            if output == None:
                return 0
        else:
            return 0

        if output["obs_GC"].shape[0] > 0:
            print("Saving .pkl file")
            save_obj(output, f"{outputdir}/{date}_GCtoTROPOMI.pkl")
            save_obj(viz_output, f"{vizdir}/{date}_GCtoTROPOMI.pkl")
        
        #Clean up to reduce memory use
        del output, viz_output
        gc.collect()

        return 0


    def process_obspack(filename):
        # Check if obspack file has already been processed
        print("========================")
        print(filename)
        date = filename[-18:-10]

        # If not yet processed, run apply_average_tropomi_operator()
        if not os.path.isfile(f"{outputdir}/{date}_GCtoObsPack.pkl"):
            print("Applying ObsPack operator...")
            output = apply_obspack_operator(filename,
                                            n_elements,
                                            gc_startdate,
                                            gc_enddate,
                                            xlim,
                                            ylim,
                                            build_jacobian,
                                            period_i,
                                            config,
                                            use_water_obs,
                                            )

            if output == None:
                return 0
        else:
            return 0

        if len(output) > 0:
            print("Saving .pkl file")
            save_obj(output, f"{outputdir}/{date}_GCtoObsPack.pkl")
        return 0


    if use_obspack:
        results = Parallel(n_jobs=-1)(delayed(process_obspack)(filename) for filename in files)
    else:
        results = Parallel(n_jobs=-1)(delayed(process_tropomi)(filename) for filename in files)

    print(f"Wrote files to {outputdir}")
