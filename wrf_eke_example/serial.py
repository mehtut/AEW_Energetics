# stdlib imports
import gc
import os

# 3rd party imports
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import wrf
import xarray as xr
import scipy.signal as signal
import scipy.ndimage as ndimage

# This is a function to get the map projection and the lat and lon values from the WRF data
def lat_lon(dataset_path):
    with Dataset(dataset_path) as data:
        # get lat and lon values
        # get the latitude and longitude at a single time (since they don't change with time)
        lat = wrf.getvar(data, "lat")  # ordered lat, lon
        lon = wrf.getvar(data, "lon")  # ordered lat, lon
        # get the cropping indices since we don't want the lat/lon for the entire domain
        lon_index_west, lat_index_south = wrf.ll_to_xy(data, -10., -40., meta=False)  # 10S, 40W
        lon_index_east, lat_index_north = wrf.ll_to_xy(data, 40., 30., meta=False)  # 40N, 30E
        lat_crop = lat.values[lat_index_south:lat_index_north + 1, lon_index_west:lon_index_east + 1]
        lon_crop = lon.values[lat_index_south:lat_index_north + 1, lon_index_west:lon_index_east + 1]
        # get more zoomed in cropping indices
        lon_west = -20.
        lon_east = 20.
        lat_north = 20.  # 25.
        lat_south = 0.  # 5.

        lat_index_north = (np.abs(lat_crop - lat_north)).argmin(axis=0)[0]
        lat_index_south = (np.abs(lat_crop - lat_south)).argmin(axis=0)[0]
        lon_index_west = (np.abs(lon_crop - lon_west)).argmin()
        lon_index_east = (np.abs(lon_crop - lon_east)).argmin()

    return lat_crop, lon_crop, lat_index_north, lat_index_south, lon_index_west, lon_index_east


# filtering function to filter variables for waves with periods between 3-5 days
# takes a variable where the first index (farthest to the left) must be time
# returns a variable of the same dimensions as the original that has been temporally
# filtered for 3-5 days using a Butterworth bandpass filter
def butter_bandpass_filter(variable):
    order = 6  # order of filtering, 6 is common
    fs = 1 / 6  # sampling rate is 1 sample every six hours
    nyq = .5 * fs  # Nyquist frequency is 1/2 times the sampling rate
    big_period_day = 5.0  # band start (longer period)
    small_period_day = 3.0  # band end
    big_period_hr = big_period_day * 24.0  # convert the days to hours
    small_period_hr = small_period_day * 24.0
    low_frequency = (1 / big_period_hr) / nyq  # 1 over the period to get the frequency and then
    high_frequency = (1 / small_period_hr) / nyq  # divide by the Nyquist frequency to normalize
    b, a = signal.butter(order, [low_frequency, high_frequency], btype='bandpass')
    # works on axis 0 (time)
    filtered_variable = signal.lfilter(b, a, variable, axis=0)
    return filtered_variable


def calc_averages(input_path, scenario_type, lat_lon_path, start_year=2001, end_year=2011, chunks=None):
    # set scenario type
    # scenario_type = 'Historical'   #'Historical' # 'late_century'

    g = 9.8  # m/s^2 gravity acceleration

    # get cropped lat and lon
    lat, lon, lat_index_north, lat_index_south, lon_index_west, lon_index_east = lat_lon(lat_lon_path)

    # define two lists for holding EKE values over the ten years
    eke_list = []
    total_eke_list = []
    # loop through years
    for year in range(start_year, end_year):
        print('Year =', year)
        # location of WRF file
        file_location = os.path.join(input_path, scenario_type, "{}-{}".format(year, year + 1), "Variables")

        # get u and v
        file_suffix = f'{scenario_type}_{year}.nc'

        # this is for May-November, ordered time, lev, lat, lon
        # don't include November, so go only to -120 (30 days * 4 times each day)
        with xr.open_dataset(os.path.join(file_location, 'ua_' + file_suffix), chunks=chunks) as u_dset:
            u_data = u_dset.ua.data[:-120 ,: ,: ,:]
        u_temp_filt = butter_bandpass_filter(u_data)
        uu = np.square(u_temp_filt)
        uu_3d = np.mean(uu, axis=0)
        del u_data
        del u_temp_filt
        del uu

        # this is for May-November, ordered time, lev, lat, lon
        # don't include November, so go only to -120 (30 days * 4 times each day)
        with xr.open_dataset(os.path.join(file_location, 'va_' + file_suffix), chunks=chunks) as v_dset:
            v_data = v_dset.va.data[:-120 ,: ,: ,:]
        v_temp_filt = butter_bandpass_filter(v_data)
        vv = np.square(v_temp_filt)
        vv_3d = np.mean(vv, axis=0)
        del v_data
        del v_temp_filt
        del vv

        # calculate u'^2 + v'^2
        uu_vv_3d = uu_3d + vv_3d
        del uu_3d
        del vv_3d

        # zonal average
        uu_vv_2d = np.mean(uu_vv_3d[:, :, lon_index_west:lon_index_east + 1], axis=2)  # order will be lev, lat
        del uu_vv_3d

        # multiply by 0.5; then append eke_2d to the eke_list
        eke_2d = 0.5 * uu_vv_2d
        eke_list.append(eke_2d)
        del eke_2d

        # meridional average and multiply by 0.5 and divide by g
        eke_1d = 0.5 * np.mean(uu_vv_2d[:, lat_index_south:lat_index_north + 1] / g, axis=1)  # order will be lev
        del uu_vv_2d

        # integrate in pressure dim; append the total_eke to the total_eke_list
        total_eke = np.trapz(eke_1d, dx = 5000., axis=0)
        total_eke_list.append(total_eke)
        del eke_1d
        del total_eke
        gc.collect()

    # average the lists
    total_eke_avg = np.mean(total_eke_list, axis=0)
    print('10 year total EKE for', scenario_type, '=', total_eke_avg, 'J/m^2')

    eke_avg = np.mean(eke_list, axis=0)
    return eke_avg, total_eke_avg


def create_plot(eke_avg, lat_lon_path, scenario_type, suffix=""):
    lat, lon, lat_index_north, lat_index_south, lon_index_west, lon_index_east = lat_lon(lat_lon_path)

    # smooth array for plotting
    eke_smooth = ndimage.gaussian_filter(eke_avg, sigma=0.9, order=0)

    # create plot
    fig, ax = plt.subplots()
    clevels = np.linspace(-8, 8, 33)  # np.linspace(-14,14,29)  # contour levels
    cmap = plt.cm.PuOr
    units = '$\mathrm{m}^{2}$ $\mathrm{s}^{-2}$'
    # define pressure levels for vertical cross setion
    p_levels = np.linspace(1000, 100, 19)
    # mesh the lat and vertical level values
    X, Y = np.meshgrid(lat[:, 0], p_levels)

    plt.xlabel("Latitude")
    plt.ylabel("Pressure (hPa)")
    contours_fill = plt.contourf(X, Y, eke_avg, cmap=cmap, levels=clevels, extend="both")
    ax.set_xlim(-5, 25)
    plt.minorticks_on()
    plt.gca().invert_yaxis()  # invert y axis, gca stands for "get current axis" and is a helper function

    cbar = plt.colorbar(contours_fill, shrink=.58, orientation='horizontal')  # , pad=.09) #shrink = .75,

    # square up plot axes
    ax.set_aspect(1. / ax.get_data_ratio())

    if len(suffix) > 0:
        suffix = "_" + suffix

    # save figure as a PDF
    fig.savefig('WRF_TCM_M-O_2001-2010avg_' + scenario_type + '_EKE' + suffix + '.pdf', bbox_inches='tight')
