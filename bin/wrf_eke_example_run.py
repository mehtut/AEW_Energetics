#!/usr/bin/env python

import argparse
import datetime
import logging
import os
import time

import xarray as xr

from wrf_eke_example.runners import run_calc_averages_serial, run_calc_averages, run_calc_averages_cuda


def main(args):
    years = [y for y in range(args.start_year, args.end_year)]
    chunk_sizes = {"lev": args.chunksize_lev, "x": args.chunksize_x, "y": args.chunksize_y}
    chunks = {"time": -1}
    for dim in chunk_sizes:
        if chunk_sizes[dim] is not None:
            chunks[dim] = chunk_sizes[dim]
    lat_lon_path = os.path.join(args.data_path, args.scenario, args.dataset)
    file_locations = []
    file_suffixes = []
    for year in years:
        file_locations.append(
            os.path.join(
                args.data_path,
                args.scenario,
                "{}-{}".format(year, year + 1), "Variables"))
        file_suffixes.append(f'{args.scenario}_{year}.nc')
    fig = None

    results_name = "WRF_TCM_M-O_{}-{}_avg_{}_EKE_{}".format(
        args.start_year, args.end_year, args.scenario, args.backend)

    if args.backend == "dask":
        eke_avg, total_eke_avg, fig = run_calc_averages(file_locations, file_suffixes, lat_lon_path,
                                                        years=years, chunks=chunks, title=results_name,
                                                        scheduler_file=args.scheduler_file)
    elif args.backend == "dask_cuda":
        eke_avg, total_eke_avg, fig = run_calc_averages_cuda(file_locations, file_suffixes, lat_lon_path,
                                                             years=years, chunks=chunks, title=results_name,
                                                             scheduler_file=args.scheduler_file)
    elif args.backend == "serial":
        eke_avg, total_eke_avg = run_calc_averages_serial(args.data_path, args.scenario, lat_lon_path,
                                                          start_year=args.start_year, end_year=args.end_year,
                                                          chunks=chunks)
    else:
        raise RuntimeError("Unknown backend: {}".format(args.backend))

    # save figure
    if fig is not None:
        fig.savefig(results_name + ".pdf")

    # save output averages
    u_filenames = []
    v_filenames = []
    for i in range(len(file_locations)):
        u_filenames.append(os.path.join(file_locations[i], "ua_" + file_suffixes[i]))
        v_filenames.append(os.path.join(file_locations[i], "va_" + file_suffixes[i]))

    results = xr.Dataset(
        data_vars={
            "years": xr.DataArray(data=years, dims=("year",)),
            "eke_avg": xr.DataArray(data=eke_avg, dims=("lev", "lat")),
            "total_eke_avg": xr.DataArray(data=[total_eke_avg]),
            "u_source_files": xr.DataArray(data=u_filenames, dims=("year")),
            "v_source_files": xr.DataArray(data=v_filenames, dims=("year"))
            },
        attrs={
            "created": datetime.datetime.now().astimezone().isoformat()
            })
    results.to_netcdf(results_name + ".nc")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate Average Eddy Kinetic Energy for a set of vectors with LEV')
    parser.add_argument('--data_path', type=str,
                    help='Path to the input vector data files', required=True)
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='Dask scheduler file to connect to an existing cluster')
    parser.add_argument('--scenario', type=str, default="Historical",
                      help='default is Historical')
    parser.add_argument('--start_year', type=int, default=2001, help='First year to process')
    parser.add_argument('--end_year', type=int, default=2011, help='Last year to process')
    parser.add_argument('--dataset', type=str, default="wrfout_d01_2008-07-01_00_00_00",
                        help='Name of the WRF data file for coordinates')
    parser.add_argument('--chunksize_lev', type=int, default=1, help='Number of LEV dimension per chunk')
    parser.add_argument('--chunksize_y', type=int, default=None, help='Number of Y (LAT) dimension per chunk')
    parser.add_argument('--chunksize_x', type=int, default=None, help='Number of X (LON) dimension per chunk')
    parser.add_argument('--backend', type=str, default="dask",
                    help='Select a backend to run the example, "serial", "dask", "dask_cuda"')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise OSError("Not found: {}".format(args.data_path))

    if args.backend not in ["serial", "dask", "dask_cuda"]:
        raise RuntimeError("backend not found: {}".format(args.backend))

    logger = logging.Logger(__name__)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    start = time.time()
    main(args)
    end = time.time()
    total = end - start
    logger.info("Wall clock time: {} minutes {} seconds".format(int(total/60), total%60))