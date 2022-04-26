# stdlib imports
import gc
import os
import shutil
import traceback

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import dask
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import wrf
import xarray as xr


def plot_eke_avg(eke_avg, lat, title=None, size=None, fig=None, ax=None, smooth=False):
    """
    Created for use with an interactive environment, with the ability to insert this plot into
    an existing figure as a subplot.

    If a figure object and axes object are not supplied, a figure will be created.
    """

    if size is None:
        size = plt.rcParams.get('figure.figsize')

    # create plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=size)
    clevels = np.linspace(-8, 8, 33)  # np.linspace(-14,14,29)  # contour levels
    cmap = plt.cm.PuOr
    units = '$\mathrm{m}^{2}$ $\mathrm{s}^{-2}$'
    # define pressure levels for vertical cross setion
    p_levels = np.linspace(1000, 100, 19)
    # mesh the lat and vertical level values
    X, Y = np.meshgrid(lat[:, 0], p_levels)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Pressure (hPa)")

    if smooth:
        # smooth array for plotting
        eke_data = ndimage.gaussian_filter(eke_avg, sigma=0.9, order=0)
    else:
        eke_data = eke_avg

    contours_fill = ax.contourf(X, Y, eke_data, cmap=cmap, levels=clevels, extend="both")

    ax.set_xlim(-5, 25)

    # set the minor ticks
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    # invert the y axis
    ax.invert_yaxis()

    cbar = fig.colorbar(contours_fill, shrink=.58, orientation='horizontal', ax=ax)  # , pad=.09) #shrink = .75,

    # square up plot axes
    ax.set_aspect(1. / ax.get_data_ratio())

    return fig, ax


def plot_coords(lat, lon, bbox=None):
    extent = (float(lon.min()) - 10, float(lon.max()) + 10, float(lat.min()) - 5, float(lat.max()) + 5)

    figsize = (20, 15)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)

    ax.set_extent(extent, crs=proj)
    ax.set_title('Lat/Lon values', fontsize=16)
    ax.coastlines('50m', linewidth=0.8)
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.scatter(lon, lat, color='black', s=1)

    crop_rectangle = (
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]))
    crop_x = [p[0] for p in crop_rectangle]
    crop_y = [p[1] for p in crop_rectangle]

    ax.plot(crop_x, crop_y, color='orange', linewidth=3)

    return fig, ax


def save_quiverplot_jpg(u, v, lat, lon, years, t=0, lev=0, figsize=(19.20,10.80),
                        title=None, suptitle=None, name=None, color=None, extent=None):
    try:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)

        if extent is not None:
            ax.set_extent(extent, crs=proj)
        if title is None:
            ax.set_title('AEW Energetics U and V components {}'.format(years), fontsize=16)
        if suptitle is None:
            suptitle = "LEV: {}   Time: {}".format(lev, t)

        ax.text(
            0.5,
            1.25,
            suptitle,
            fontsize=18,
            ha="center",
            transform=ax.transAxes)
        ax.coastlines('50m', linewidth=0.8)
        ax.add_feature(cfeature.OCEAN)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        for item in ([ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)

        if color is None:
            color = (u ** 2 + v ** 2) ** 0.5

        q = ax.quiver(
            lon,
            lat,
            u,
            v,
            color,
            transform=proj
            )
        norm = mpl.colors.Normalize()
        norm.autoscale(color)
        sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
        plt.colorbar(sm)
        fig.savefig(name, dpi=100)
        return name
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        # release memory

        # circular references exist in the matplotlib code, closing all plot figures and
        # running an explicit garbage collection is needed to reclaim memory
        try:
            plt.cla()
            plt.close(fig)
        except:
            pass

        # may still be needed
        #gc.collect()


def get_colorspace(data):
    if os.path.exists('colorspace.nc'):
        color = xr.open_dataarray('colorspace.nc')
        return color

    min_magnitudes = []
    max_magnitudes = []
    for i in range(len(data)):
        mag = (dask.array.square(data[i][0]) + dask.array.square(data[i][1])) ** 0.5
        min_magnitudes.append(mag.min())
        max_magnitudes.append(mag.max())
    num_elements = np.prod(data[0][0][0 , 0 ,::5 ,::5].shape)
    color = np.linspace(min(min_magnitudes), max(max_magnitudes), num_elements)
    return color


def _write_video(filename, frames, codec='mp4v', fps=5, size=(1920, 1080)):
    writer = cv2.VideoWriter(os.path.abspath(filename), cv2.VideoWriter_fourcc(*codec), fps, size)
    for fname in frames:
        img = cv2.imread(fname)
        if img.shape[0] != size[0] or img.shape[1] != size[1]:
            img = cv2.resize(img, size)
        writer.write(img)
    writer.release()
    return True


def create_quiver_anim(data, lat, lon, years, lev=0, color=None, filename=None):
    if color is None:
        color = get_colorspace(data)

    if filename is None:
        filename = 'quivers_{}_5fps.mp4'.format(years)
    imgs_path = 'quiverplot_frames_{}'.format(years[0])
    if os.path.exists(imgs_path):
        shutil.rmtree(imgs_path)
    os.mkdir(imgs_path)

    figsize = (20, 15)
    plt.ioff()

    extent = (int(lon.min()) - 10, int(lon.max()) + 10, int(lat.min()) - 5, int(lat.max()) + 5)
    sparse_lat = wrf.to_np(lat[::5, ::5]),
    sparse_lon = wrf.to_np(lon[::5, ::5]),
    sizes = [d[0].sizes["time"] for d in data]
    saves = []
    for i in range(len(data)):
        for t in range(sizes[i]):
            pq = os.path.join(os.path.abspath(imgs_path), 'quivers_{}_{:03d}.jpg'.format(years[i], t))
            imgsave = dask.delayed(save_quiverplot_jpg)(
                data[i][0][t, lev, ::5, ::5],
                data[i][1][t, lev, ::5, ::5],
                sparse_lat,
                sparse_lon,
                years,
                t=t,
                lev=0,
                name=pq,
                figsize=figsize,
                color=color,
                extent=extent)
            saves.append(imgsave)

    names = dask.compute(saves)[0]
    assert _write_video(filename, names)

    return  filename
