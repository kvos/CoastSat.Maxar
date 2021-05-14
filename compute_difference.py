"""
Extract shorelines from Maxar/DigitalGlobe imagery
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import skimage.filters as filters
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
import skimage.measure as measure

# other modules
from osgeo import gdal
from osgeo import osr
from pylab import ginput
import pickle
import geopandas as gpd
from shapely import geometry

# own modules
import SDS_tools, SDS_preprocess, SDS_shoreline, SDS_transects

# plotting params
plt.style.use('default')
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['hatch.linewidth'] = 0.5
fontsize_text = 10
plt.ion()

#%%

# load shorelines 
sl1 = gpd.read_file('20Jul2020.geojson', driver='GeoJSON')
sl2 = gpd.read_file('3Aug2020.geojson', driver='GeoJSON')

geom1 = sl1.iloc[0].geometry
sl1_coords = np.empty((len(geom1),2))
for i in range(len(geom1)):
    sl1_coords[i,0] = list(geom1[i].coords)[0][0]
    sl1_coords[i,1] = list(geom1[i].coords)[0][1]
    
geom2 = sl2.iloc[0].geometry
sl2_coords = np.empty((len(geom2),2))
for i in range(len(geom2)):
    sl2_coords[i,0] = list(geom2[i].coords)[0][0]
    sl2_coords[i,1] = list(geom2[i].coords)[0][1]

fig,ax = plt.subplots(1,1)
ax.axis('equal')
ax.plot(sl1_coords[:,0],sl1_coords[:,1],'r.')
ax.plot(sl2_coords[:,0],sl2_coords[:,1],'b.')


# load transects
geojson_file = os.path.join(os.getcwd(), 'beaches_shapefile', 'transects.geojson')
gdf = gpd.read_file(geojson_file, driver='GeoJSON')
transects = dict([])
for i in gdf.index:
    transects[str(i+1)] = np.array(gdf.loc[i,'geometry'].coords)[[1,0],:]
 
for i,key in enumerate(list(transects.keys())):
    ax.plot(transects[key][0,0],transects[key][0,1], 'ko', ms=5)
    ax.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
    ax.text(transects[key][0,0]-100, transects[key][0,1]+100, key,
                va='center', ha='right')

settings_transects = { # intersections of 2D shorelines
                      'along_dist':         5,             # along-shore distance to use for intersection
                      'max_std':            15,             # max std for points around transect
                      'max_range':          30,             # max range for points around transect
                      'min_val':            -100,           # largest negative value along transect (landwards of transect origin)
                      # outlier removal
                      'nan/max':            'max',         # mode for removing outliers ('auto', 'nan', 'max')
                      'prc_std':            0.1,            # percentage to use in 'auto' mode to switch from 'nan' to 'max'
                      'plot_fig':           True,          # True to save the figure of the transects
                      }
cross_distance = SDS_transects.compute_intersection([sl1_coords, sl2_coords], transects, settings_transects) 

gdf2 = gdf.copy()
for i in range(len(gdf2)):
    gdf2.at[i,'geometry'] = gdf2.iloc[i]['geometry'].centroid
    gdf2.at[i,'July'] = cross_distance[str(i+1)][0]
    gdf2.at[i,'Aug'] = cross_distance[str(i+1)][1]
    gdf2.at[i,'shoreline change [m]'] = cross_distance[str(i+1)][1] - cross_distance[str(i+1)][0]
    
gdf2.to_file('shoreline_change.shp')