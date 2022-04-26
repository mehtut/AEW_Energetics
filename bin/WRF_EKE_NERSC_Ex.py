from __future__ import division  # makes division not round with integers 
import os
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import signal
import scipy.ndimage as ndimage

# This program calculates the eddy kinetic energy and plots the meridional
# distribution. The equation used is EKE = [u'^2+v'^2]/2g
# [ ] indicates a zonal average 


# This is a function to get the map projection and the lat and lon values from the WRF data
def lat_lon():
	file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/Historical/wrfout_d01_2008-07-01_00_00_00'
	data = Dataset(file_location)
	# get lat and lon values
	# get the latitude and longitude at a single time (since they don't change with time)
	lat = wrf.getvar(data, "lat") # ordered lat, lon
	lon = wrf.getvar(data, "lon") # ordered lat, lon
	# get the cropping indices since we don't want the lat/lon for the entire domain
	lon_index_west, lat_index_south = wrf.ll_to_xy(data,-10.,-40., meta=False) # 10S, 40W
	lon_index_east, lat_index_north = wrf.ll_to_xy(data,40.,30., meta=False) # 40N, 30E
	lat_crop = lat.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
	lon_crop = lon.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
	# get more zoomed in cropping indices
	lon_west = -20. # 20W
	lon_east = 20. # 20E
	lat_north = 20. # 20N
	lat_south = 0. # 0N
	lat_index_north = np.argmin((np.abs(lat_crop - lat_north)), axis=0)[0]
	lat_index_south = np.argmin((np.abs(lat_crop - lat_south)), axis=0)[0] 
	lon_index_west = (np.abs(lon_crop - lon_west)).argmin() 
	lon_index_east = (np.abs(lon_crop - lon_east)).argmin() 

	return lat_crop, lon_crop, lat_index_north, lat_index_south, lon_index_west, lon_index_east


# filtering function to filter variables for waves with periods between 3-5 days 
# takes a variable where the first index (farthest to the left) must be time
# returns a variable of the same dimensions as the original that has been temporally 
# filtered for 3-5 days using a Butterworth bandpass filter
def butter_bandpass_filter(variable):
	print("Temporal filter...")
	order = 6  # order of filtering, 6 is common
	fs = 1/6  # sampling rate is 1 sample every six hours
	nyq = .5 * fs  # Nyquist frequency is 1/2 times the sampling rate
	big_period_day = 5.0    # band start (longer period)
	small_period_day = 3.0  # band end
	big_period_hr = big_period_day*24.0  # convert the days to hours
	small_period_hr = small_period_day*24.0
	low_frequency = (1/big_period_hr) / nyq     # 1 over the period to get the frequency and then
	high_frequency = (1/small_period_hr) / nyq  # divide by the Nyquist frequency to normalize
	print(low_frequency)
	print(high_frequency)
	b, a = signal.butter(order, [low_frequency, high_frequency], btype='bandpass')
	# works on axis 0 (time)
	filtered_variable = signal.lfilter(b, a, variable, axis=0)
	return filtered_variable


def main():
	# set scenario type
	scenario_type = 'Historical'   #'Historical' # 'late_century'

	g = 9.8  # m/s^2 gravity acceleration

	# get cropped lat and lon
	lat, lon, lat_index_north, lat_index_south, lon_index_west, lon_index_east = lat_lon() 
	
	# define two lists for holding EKE values over the ten years
	eke_list = []
	total_eke_list = []
	# loop through years
	for year in range(2001,2011):
		print('Year =', year)
		# location of WRF file
		file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/' + scenario_type + '/' + str(year) + '/'

		# get u and v
		u_data = xr.open_dataset(file_location + 'ua_' + scenario_type + '_' + str(year) + '.nc')
		v_data = xr.open_dataset(file_location + 'va_' + scenario_type + '_' + str(year) + '.nc')
		u_4d_full = u_data.ua # this is for May-November, ordered time, lev, lat, lon
		v_4d_full = v_data.va # this is for May-November, ordered time, lev, lat, lon
		print(u_4d_full.shape)
		u_4d = u_4d_full[:-120,:,:,:] # don't include November, so go only to -120 (30 days * 4 times each day)
		v_4d = v_4d_full[:-120,:,:,:] # don't include November, so go only to -120 (30 days * 4 times each day)
		print(u_4d.shape)
		
		# spatially filter u and v
		u_temp_filt = butter_bandpass_filter(u_4d)
		v_temp_filt = butter_bandpass_filter(v_4d)

		# square u' and v'
		uu = np.square(u_temp_filt)
		vv = np.square(v_temp_filt)

		# time average (on dim 0), order will then be lev, lat, lon
		uu_3d = np.mean(uu, axis = 0) 
		vv_3d = np.mean(vv, axis = 0) 
		
		# calculate u'^2 + v'^2 
		uu_vv_3d = uu_3d + vv_3d

		# zonal average
		uu_vv_2d = np.mean(uu_vv_3d[:,:,lon_index_west:lon_index_east+1], axis = 2) # order will be lev, lat

		# multiply by 0.5; then append eke_2d to the eke_list
		eke_2d = 0.5*uu_vv_2d
		eke_list.append(eke_2d)
		del eke_2d

		# meridional average and multiply by 0.5 and divide by g
		eke_1d = 0.5*np.mean(uu_vv_2d[:,lat_index_south:lat_index_north+1]/g, axis = 1) # order will be lev

		# integrate in pressure dim; append the total_eke to the total_eke_list
		total_eke = np.trapz(eke_1d, dx = 5000., axis=0)
		total_eke_list.append(total_eke)
		del total_eke

	# average the lists
	total_eke_avg = np.mean(total_eke_list, axis=0)
	print('10 year total EKE for', scenario_type, '=', total_eke_avg, 'J/m^2')

	eke_avg = np.mean(eke_list, axis=0)
	print(eke_avg.shape)

	# smooth array for plotting
	eke_smooth = ndimage.gaussian_filter(eke_avg, sigma=0.9, order=0)

	# create plot
	fig, ax = plt.subplots()
	clevels = np.linspace(-8,8,33) #np.linspace(-14,14,29)  # contour levels
	cmap = plt.cm.PuOr 
	units = '$\mathrm{m}^{2}$ $\mathrm{s}^{-2}$'
	# define pressure levels for vertical cross setion
	p_levels = np.linspace(1000,100,19)
	# mesh the lat and vertical level values
	X,Y = np.meshgrid(lat[:,0],p_levels)  
	plt.xlabel("Latitude")
	plt.ylabel("Pressure (hPa)")
	contours_fill = plt.contourf(X,Y,eke_avg, cmap=cmap, levels=clevels, extend="both")
	ax.set_xlim(-5,25)
	plt.minorticks_on()
	plt.gca().invert_yaxis()  # invert y axis, gca stands for "get current axis" and is a helper function

	cbar = plt.colorbar(contours_fill,  shrink = .58, orientation='horizontal') #, pad=.09) #shrink = .75,

	# square up plot axes
	ax.set_aspect(1./ax.get_data_ratio())

	# save figure as a PDF
	fig.savefig('WRF_TCM_M-O_2001-2010avg_' + scenario_type + '_EKE.pdf', bbox_inches='tight')



if __name__ == '__main__':
	main()
