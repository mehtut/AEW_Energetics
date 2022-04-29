# stdlib imports
import os

# data read/write/process imports
import dask
import dask.graph_manipulation
from dask.distributed import get_client
from netCDF4 import Dataset
import numpy as np
from scipy import signal
import wrf
import xarray as xr


def get_data(file_location, file_suffix, chunks):
    """Return the vector DataArrays"""
    ua = xr.open_dataset(os.path.join(file_location, 'ua_' + file_suffix), chunks=chunks)["ua"]
    va = xr.open_dataset(os.path.join(file_location, 'va_' + file_suffix), chunks=chunks)["va"]
    return ua, va


def scatter_data(file_location, file_suffix, chunks):
    """Convenience function for scattering data references to the workers."""

    dask_client = get_client()
    
    with xr.open_dataset(os.path.join(file_location, 'ua_' + file_suffix), chunks=chunks) as u_data:
        u = dask_client.scatter(u_data["ua"])
    with xr.open_dataset(os.path.join(file_location, 'va_' + file_suffix), chunks=chunks) as v_data:    
        v = dask_client.scatter(v_data["va"])
    
    return u, v


def get_lat_lon(dataset_path):
    """Return the lat/lon values uncropped"""

    with Dataset(dataset_path) as data:
        # get lat and lon values
        # get the latitude and longitude at a single time (since they don't change with time)
        lat = wrf.getvar(data, "lat")  # ordered lat, lon
        lon = wrf.getvar(data, "lon")  # ordered lat, lon

    return lat, lon


def crop_lat_lon(dataset_path, region=((-40,-10),(30,40)), zoom=((-20,0),(20,20))):
    """Apply a bounding box to the lat/lon values and return the indices of the extent."""
    with Dataset(dataset_path) as data:
        # get lat and lon values
        # get the latitude and longitude at a single time (since they don't change with time)
        lat = wrf.getvar(data, "lat")  # ordered lat, lon
        lon = wrf.getvar(data, "lon")  # ordered lat, lon

        if region is None:
            region = ((float(lon.min()),float(lat.min())),(float(lon.max()),float(lat.max())))

        # get the cropping indices since we don't want the lat/lon for the entire domain
        lon_index_west, lat_index_south = wrf.ll_to_xy(data, region[0][1], region[0][0], meta=False)  # 10S, 40W
        lon_index_east, lat_index_north = wrf.ll_to_xy(data, region[1][1], region[1][0], meta=False)  # 40N, 30E

        # cover the entire longitude range
        if lon_index_west >= lon_index_east:
            lon_index_west = 0
            lon_index_east = lon.sizes['west_east']

        lat_crop = lat[lat_index_south:lat_index_north + 1, lon_index_west:lon_index_east + 1]
        lon_crop = lon[lat_index_south:lat_index_north + 1, lon_index_west:lon_index_east + 1]

        if zoom is not None:
            # get more zoomed in cropping indices
            lon_west = zoom[0][0]
            lon_east = zoom[1][0]
            lat_north = zoom[1][1]
            lat_south = zoom[0][1]

            lat_index_north = (np.abs(lat_crop - lat_north)).argmin(axis=0)[0]
            lat_index_south = (np.abs(lat_crop - lat_south)).argmin(axis=0)[0]
            lon_index_west = (np.abs(lon_crop - lon_west)).argmin()
            lon_index_east = (np.abs(lon_crop - lon_east)).argmin()

    return lat_crop, lon_crop, int(lat_index_north), int(lat_index_south), int(lon_index_west), int(lon_index_east)


def get_butterworth_filter(order=6, fs=(1/6), big_period_day=5.0, small_period_day=3.0):
    """
    Create a butterworth bandpass filter to remove noise from the WRF_EKE data.

    Default values:
      order = 6   order of filtering
      fs = (1/6)  sampling rate is 1 sample every 6 hours
      big_period_day = 5.0   band start
      small_period_day = 3.0   band end
    """
    nyq = .5 * fs  # Nyquist frequency is 1/2 times the sampling rate
    big_period_hr = big_period_day * 24.0  # convert the days to hours
    small_period_hr = small_period_day * 24.0
    low_frequency = (1 / big_period_hr) / nyq  # 1 over the period to get the frequency and then
    high_frequency = (1 / small_period_hr) / nyq  # divide by the Nyquist frequency to normalize
    b, a = signal.butter(order, [low_frequency, high_frequency], btype='bandpass')
    return b, a


def _lfilter(x, *args, **kwargs):
    """
    Wrap the call to scipy.signal.lfilter with xarray.apply_ufunc.
    This allows for argument remapping to work with map_blocks expecting that x is the first arguemnt.
    It also allows for maintaining an xarray/dask object, since apply_ufunc returns a DataArray,
    which supports chunking, etc.  A transpose is used because apply_ufunc moves the first axis to the last.
    """
    return xr.apply_ufunc(
        signal.lfilter, args[0], args[1], x, kwargs=kwargs).transpose("time", "lev", "y", "x")


def get_filtered(x, *args, **kwargs):
    """Compute the filter for every array chunk and aggregate the result."""
    return x.map_blocks(_lfilter, args=args, kwargs=kwargs, template=x)


def calc_uu_vv_2d(u, v, lon_index_west, lon_index_east):
    """Compute the squared mean of the vectors within the longitude limits."""
    # zonal average, order will be lev, lat)
    uu_mean = dask.array.square(u).mean(axis=0)[:,:,lon_index_west:lon_index_east+1]
    vv_mean = dask.array.square(v).mean(axis=0)[:,:,lon_index_west:lon_index_east+1]
    return (uu_mean + vv_mean).mean(axis=2)


def calc_eke_1d(uu_vv_2d, lat_index_north, lat_index_south):
    """Compute the trapezoidal integral approximation from the meridional average."""
    g = 9.8
    uu_vv_1d = 0.5 * (uu_vv_2d[:,lat_index_south:lat_index_north+1]/g).mean(axis=1)
    return np.trapz(np.asarray(uu_vv_1d), dx=5000.0, axis=0)


def apply_filters(u, v, time=slice(0,-120), butterworth=True, butterworth_kwargs=None):
    """Apply time and linear filtering to the vectors."""

    # filter/reduce the data by time
    u_time_filtered = u[time,:,:,:]
    v_time_filtered = v[time,:,:,:]

    if butterworth:
        # collect the bandpass filtering inputs
        if butterworth_kwargs is None:
            butterworth_kwargs = {}
        b, a = get_butterworth_filter(**butterworth_kwargs)

        # apply the linear filter
        u_bandpass_filtered = get_filtered(u_time_filtered, b, a, axis=0)
        v_bandpass_filtered = get_filtered(v_time_filtered, b, a, axis=0)

        u_out = u_bandpass_filtered
        v_out = v_bandpass_filtered
    else:
        u_out = u_time_filtered
        v_out = v_time_filtered

    return u_out, v_out


def calc_averages(data, lat_index_north, lat_index_south, lon_index_west, lon_index_east,
                  time_slice=slice(0,-120), butterworth=True, butterworth_kwargs=None):
    """Compute the average Eddy Kinetic Energy (EKE) and total EKE for a given extent."""
    uu_vv_2d = []
    for d in data:
        if butterworth:
            u, v = apply_filters(
                d[0],
                d[1],
                time=time_slice,
                butterworth=butterworth,
                butterworth_kwargs=butterworth_kwargs)
        else:
            u = d[0]
            v = d[1]

        uv = calc_uu_vv_2d(
            u,
            v,
            lon_index_west,
            lon_index_east).persist()
        uu_vv_2d.append(uv)

    eke_2d = [0.5 * uv for uv in uu_vv_2d]
    eke_avg = np.mean(eke_2d, axis=0)

    eke_1d = [calc_eke_1d(uv, lat_index_north, lat_index_south) for uv in uu_vv_2d]
    total_eke_avg = np.mean(eke_1d, axis=0)

    return eke_avg, total_eke_avg
