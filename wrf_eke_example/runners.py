from .eke import scatter_data, crop_lat_lon
from .plot import plot_eke_avg


def run_calc_averages_cuda(file_locations, file_suffixes, lat_lon_path,
                           years=None, chunks=None, title=None, local_dir="/tmp", scheduler_file=None):
    from .eke_cuda import calc_averages

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    if scheduler_file is None:
        cluster = LocalCUDACluster(local_directory=local_dir)
        dask_client = Client(cluster)
    else:
        cluster = None
        dask_client = Client(scheduler_file=scheduler_file)

    if chunks is None:
        chunks = {"time": -1, "lev": 1}
    if years is None:
        years = [y for y in range(2001, 2011)]
    yearly_chunks = [chunks for year in years]
    data_futures = dask_client.map(scatter_data, file_locations, file_suffixes, yearly_chunks)
    data = [(x[0].result(), x[1].result()) for x in dask_client.gather(data_futures)]

    # get cropped lat and lon
    lat, lon, lat_index_north, lat_index_south, lon_index_west, lon_index_east = crop_lat_lon(lat_lon_path)
    fut = dask_client.submit(
        calc_averages,
        data,
        lat_index_north,
        lat_index_south,
        lon_index_west,
        lon_index_east)
    eke_avg, total_eke_avg = fut.result()
    fig, ax = plot_eke_avg(eke_avg, lat, title=title + "_dask_cuda")
    if cluster is not None:
        dask_client.shutdown()
    dask_client.close()

    return eke_avg, total_eke_avg, fig


def run_calc_averages(file_locations, file_suffixes, lat_lon_path,
                      num_workers=2, years=None, chunks=None, title=None, local_dir="/tmp", scheduler_file=None):
    from .eke import calc_averages

    from dask.distributed import LocalCluster, Client

    if scheduler_file is None:
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, local_directory=local_dir)
        dask_client = Client(cluster)
    else:
        cluster = None
        dask_client = Client(scheduler_file=scheduler_file)

    if chunks is None:
        chunks = {"time": -1, "lev": 1}
    if years is None:
        years = [y for y in range(2001, 2011)]
    yearly_chunks = [chunks for year in years]
    data_futures = dask_client.map(scatter_data, file_locations, file_suffixes, yearly_chunks)
    data = [(x[0].result(), x[1].result()) for x in dask_client.gather(data_futures)]

    # get cropped lat and lon
    lat, lon, lat_index_north, lat_index_south, lon_index_west, lon_index_east = crop_lat_lon(lat_lon_path)
    fut = dask_client.submit(
        calc_averages,
        data,
        lat_index_north,
        lat_index_south,
        lon_index_west,
        lon_index_east)
    eke_avg, total_eke_avg = fut.result()
    fig, ax = plot_eke_avg(eke_avg, lat, title=title + "_dask")
    if cluster is not None:
        dask_client.shutdown()
    dask_client.close()

    return eke_avg, total_eke_avg, fig


def run_calc_averages_serial(input_path, scenario_type, lat_lon_path, start_year, end_year,
                             chunks=None, local_dir="/tmp"):
    from .serial import calc_averages, create_plot

    eke_avg, total_eke_avg = calc_averages(
        input_path,
        scenario_type,
        lat_lon_path,
        start_year,
        end_year,
        chunks=chunks)
    create_plot(eke_avg, lat_lon_path, "serial")

    return eke_avg, total_eke_avg
