
import cartopy.crs as ccrs
import os
import json
from scipy.stats import gaussian_kde
import mome_geo
import matplotlib.pyplot as plt
import sys
from scipy.stats import gaussian_kde, binned_statistic_2d
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import griddata
plt.style.use('dark_background')



def my_test(): 

    plot_heatmaps_for_fields(os.path.join(os.path.dirname(__file__), "tmp_us_cities.json"), ["jobs_directly_supported", "annual_average_wage"], 
                             lat_field= "main_city_lat", long_field= "main_city_long")
    plt.show()



def plot_world_map(lat_min, lat_max, lon_min, lon_max):
    """
    Returns a matplotlib ax of the world zoomed to the specified latitude and longitude boundaries,
    with a black background and white borders.
    """
    # Create a figure and axis with PlateCarree projection
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_facecolor('black')

    # Set the extent: [west, east, south, north]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add white coastlines
    ax.coastlines(color='white')

    # Add white gridlines and set label colors to white
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    # gl.xlabel_style = {'color': 'white'}
    # gl.ylabel_style = {'color': 'white'}

    return ax


def overlay_heatmap_on_map(
    data,
    value_field,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    cmap='hot',
    alpha=0.5,
    gridsize=200,
    lat_field="lat",
    long_field="long",
    scatter=False,
    scatter_kwargs=None,
    method='linear'
):
    """
    Overlays an interpolated heatmap of a user-selected field on a world map,
    using scipy.interpolate.griddata for true interpolation.

    Args:
        data (list of dict): List of data points, each with 'lat', 'long', and the value_field.
        value_field (str): The key in each dict whose value is to be overlaid as a heatmap.
        lat_min (float): Minimum latitude (south boundary, -90 to 90).
        lat_max (float): Maximum latitude (north boundary, -90 to 90).
        lon_min (float): Minimum longitude (west boundary, -180 to 180).
        lon_max (float): Maximum longitude (east boundary, -180 to 180).
        cmap (str): Matplotlib colormap for the heatmap.
        alpha (float): Transparency of the heatmap overlay.
        gridsize (int): Resolution of the heatmap grid.
        lat_field (str): Key for latitude in data dicts.
        long_field (str): Key for longitude in data dicts.
        scatter (bool): If True, overlay a scatter plot of the data points.
        scatter_kwargs (dict): Additional keyword arguments for plt.scatter.
        method (str): Interpolation method: 'linear', 'cubic', or 'nearest'.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes with the map and heatmap overlay.
    """
    # Extract coordinates and values, filtering out missing data
    lats = np.array([d[lat_field] for d in data if d[lat_field] is not None and d[long_field] is not None])
    lons = np.array([d[long_field] for d in data if d[lat_field] is not None and d[long_field] is not None])
    values = np.array([d[value_field] for d in data if d[lat_field] is not None and d[long_field] is not None])

    # Prepare grid
    xi = np.linspace(lon_min, lon_max, gridsize)
    yi = np.linspace(lat_min, lat_max, gridsize)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate values onto the grid
    zi = griddata(
        points=(lons, lats),
        values=values,
        xi=(xi, yi),
        method=method
    )

    # Set color scale to actual value range
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    # Plot the base map using the provided function
    ax = mome_geo.plot_world_map(lat_min, lat_max, lon_min, lon_max)

    # Overlay the heatmap
    im = ax.imshow(
        zi,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin='lower',
        cmap=cmap,
        alpha=alpha,
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    # Optionally overlay scatter plot of data points
    if scatter:
        if scatter_kwargs is None:
            scatter_kwargs = {'c': 'red', 's': 10, 'marker': 'o', 'edgecolor': 'k', 'alpha': 0.7}
        ax.scatter(lons, lats, **scatter_kwargs, label='Data Points')

    plt.colorbar(im, ax=ax, label=value_field)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f"Interpolated Heatmap of '{value_field}'")
    if scatter:
        ax.legend()

    return ax


def plot_heatmaps_for_fields(
    data_file,
    target_fields,
    cmap='viridis',
    lat_field = "lat", 
    long_field = "long", 
    alpha=0.6,
    gridsize=100, 
    margin_lat = 0.5, 
    margin_long = 0.23
):
    """
    Entry point to plot heatmaps for one or more fields from a data file on a world map.

    This function loads a list of dicts from the specified data file, determines the bounding
    box for the map based on the min/max latitudes and longitudes in the data, and overlays
    a heatmap for each specified field using the overlay_heatmap_on_map function.

    Args:
        data_file (str): Path to the source data file (should contain a list of dicts with 'lat', 'long', and value fields).
        target_fields (list of str): List of field names to visualize as heatmaps.
        cmap (str, optional): Matplotlib colormap for the heatmap. Default is 'viridis'.
        alpha (float, optional): Transparency of the heatmap overlay. Default is 0.6.
        gridsize (int, optional): Resolution of the heatmap grid. Default is 100.

    Returns:
        list of matplotlib.axes.Axes: List of matplotlib axes with the map and heatmap overlays for each field.
    """

    # Load data from file
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract latitudes and longitudes for bounding box
    lats = [d[lat_field] for d in data if d[lat_field] is not None]
    lons = [d[long_field] for d in data if d[long_field] is not None]
    margin_lat = (max(lats) - min(lats)) * margin_lat if lats else 0
    margin_lon = (max(lons) - min(lons)) * margin_long if lons else 0
    lat_min = min(lats) - margin_lat
    lat_max = max(lats) + margin_lat
    lon_min = min(lons) - margin_lon
    lon_max = max(lons) + margin_lon

    axes_list = []
    for value_field in target_fields:
        ax = overlay_heatmap_on_map(
            data=data,
            value_field=value_field,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_field = lat_field, 
            long_field = long_field, 
            cmap=cmap,
            alpha=alpha,
            gridsize=gridsize, 
            scatter = True
        )
        ax.set_title(f"Heatmap of '{value_field}'")
        axes_list.append(ax)
    #     print('o k for {}'.format(value_field))
    # plt.show()
    return axes_list