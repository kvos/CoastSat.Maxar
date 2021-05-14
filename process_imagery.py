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
import SDS_tools, SDS_preprocess, SDS_shoreline

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

 

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
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
#%% 1. Use shapefile with beaches to crop the images (better to do it on QGIS)
# beaches = gpd.read_file(os.path.join(os.getcwd(),'beaches_shapefile','beaches.shp'))

# # make 200m around shoreline and take the minimum rectangle
# beach = beaches.loc[beaches['beach_name'] == 'Narrabeen']
# beach_rect = np.array(beach.iloc[0]['geometry'].buffer(200).envelope.exterior.coords)

# # load the image
# image_path = os.path.join(os.getcwd(),'order1',
#                           '20JUL19000828-S2AS_R2C1-200000742456_01_P001.tif')
# data = gdal.Open(image_path, gdal.GA_ReadOnly)

# # # clip the image
# left = np.min(beach_rect[:,0])
# right = np.max(beach_rect[:,0])
# upper = np.max(beach_rect[:,1])
# lower = np.min(beach_rect[:,1])
# window = [left,upper,right,lower]
# gdal.Translate(os.path.join(os.getcwd(),'cropped_images',
#                             beach.iloc[0]['beach_name']+'.tif'),
#                image_path, projWin=window)

#%% load the clipped image (in the cropped images folder)
# image_name = 'Narrabeen_Jul_clipped.tif'
image_name = 'Narrabeen_03Aug_clipped_60cm.tif'
epsg_out = 28356
image_path = os.path.join(os.getcwd(),'cropped',image_name)
data = gdal.Open(image_path, gdal.GA_ReadOnly)
# get image EPSG code
proj = osr.SpatialReference(wkt=data.GetProjection())
epsg = int(proj.GetAttrValue('AUTHORITY',1))
# get georef
georef = np.array(data.GetGeoTransform())
# get all bands
bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
im_ms = np.stack(bands, 2).astype(float)

# no cloud mask (all False to be compatible with coastsat functions)
cloud_mask = np.zeros((im_ms.shape[0],im_ms.shape[1])).astype(bool)
for k in range(im_ms.shape[2]):
    im_zero = np.isin(im_ms[:,:,k], 0)
    cloud_mask = np.logical_or(cloud_mask, im_zero)

# compute RGB for visualisation
im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 98)
# NDWI (NIR - Green normalised)
im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
# NIR - Blue normalised
# im_nir_blue_norm = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)
# Red - Blue normalised
# im_red_blue_norm = SDS_tools.nd_index(im_ms[:,:,2], im_ms[:,:,0], cloud_mask)

#%% extract shoreline
# mask pixels and flatten to vector
vec_ndwi = im_ndwi.reshape(im_ndwi.shape[0] * im_ndwi.shape[1])
vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
vec = vec_ndwi[~vec_mask]
# apply otsu's threshold
vec = vec[~np.isnan(vec)]
# t_otsu = filters.threshold_multiotsu(vec)
t_otsu1 = -0.25312
# t_otsu1 = -0.32312
t_otsu2 =  0.145676
# use Marching Squares algorithm to detect contours on ndwi image
contours = measure.find_contours(im_ndwi,t_otsu2)
# remove contours that contain NaNs (due to cloud pixels in the contour)
contours = SDS_shoreline.process_contours(contours)
# convert pixel coordinates to world coordinates
contours_world = SDS_tools.convert_pix2world(contours, georef)
# convert world coordinates to desired spatial reference system
contours_epsg = SDS_tools.convert_epsg(contours_world, epsg, epsg_out)
# remove contours that have a perimeter < min_length_sl (provided in settings dict)
contours_long = []
contours_length = []
for l, wl in enumerate(contours_epsg):
    coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
    a = geometry.LineString(coords) # shapely LineString structure
    if a.length >= 500:
        # deltaX = np.abs(np.nanmax(wl[:,0])-np.nanmin(wl[:,0]))
        # deltaY = np.abs(np.nanmax(wl[:,1])-np.nanmin(wl[:,1]))
        # sq_param = a.length / (deltaX + deltaY)
        # print(sq_param)
        # if sq_param <= 3:
        contours_length.append(a.length)
        contours_long.append(wl)
# choose the longest contour (comment to use more contours)
# contours_long = [contours_epsg[np.argmax(contours_length)]]
# format points into np.array
x_points = np.array([])
y_points = np.array([])
for k in range(len(contours_long)):
    x_points = np.append(x_points,contours_long[k][:,0])
    y_points = np.append(y_points,contours_long[k][:,1])
contours_array = np.transpose(np.array([x_points,y_points]))
# convert to pixel coordinates
sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(contours_array,
                                     epsg_out,epsg)[:,[0,1]], georef)
# get most easterly point for each row
sl = np.round(sl_pix)
sl_offshore = []
for row in np.unique(sl[:,1]):
    idx = np.where(sl[:,1]==row)[0]
    if len(idx) > 1:
        eastings = sl[idx,0]
        idx_max = idx[np.argmax(eastings)]
        sl_offshore.append(sl[idx_max,:])
    else:
        sl_offshore.append(sl[idx[0],:])
sl_offshore = np.array([(_[0],_[1]) for _ in sl_offshore])

#%% make figure with RGB and indices
cmap = plt.cm.coolwarm
cmap.set_bad(color='k')

fig = plt.figure()
fig.set_size_inches([13, 9])
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
gs = gridspec.GridSpec(2, 3, height_ratios=[4,1])
gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[1,:])

ax1.imshow(im_RGB)
ax1.axis('off')
ax1.set_title('RGB')
cax = ax2.imshow(im_ndwi,cmap=cmap, 
                 norm=MidpointNormalize(midpoint=t_otsu2,vmin=-0.8,vmax=0.8))
cbar = fig.colorbar(cax,ax=ax2, orientation='vertical', shrink=0.75, pad=0.02)
ax2.axis('off')
ax2.set_title('NDWI')

# divide in 3 classes
im_labels = np.empty((im_ms.shape[0],im_ms.shape[1],3))
im_labels[:,:,0] = im_ndwi <= t_otsu1
im_labels[:,:,1] = np.logical_and(im_ndwi > t_otsu1, im_ndwi <= t_otsu2)
im_labels[:,:,2] = im_ndwi > t_otsu2

im_labels = im_labels.astype(bool)

vec1 = vec[vec <= t_otsu1]
vec2 = vec[np.logical_and(vec > t_otsu1, vec <= t_otsu2)]
vec3 = vec[vec > t_otsu2]

# make color map
cmap = cm.get_cmap('tab20c')
colorpalette = cmap(np.arange(0,13,1))
colours = np.zeros((3,4))
colours[2,:] = colorpalette[5]
colours[1,:] = np.array([204/255,1,1,1])
colours[0,:] = np.array([0,91/255,1,1])

# make classified image
im_class = im_RGB.copy()
for k in range(0,im_labels.shape[2]):
    im_class[im_labels[:,:,k],0] = colours[k,0]
    im_class[im_labels[:,:,k],1] = colours[k,1]
    im_class[im_labels[:,:,k],2] = colours[k,2]
    
# plot classified image
ax3.imshow(im_RGB)
ax3.imshow(im_class, alpha=1)
ax3.axis('off')
orange_patch = mpatches.Patch(color=colours[0,:], label='class 1')
white_patch = mpatches.Patch(color=colours[1,:], label='class 2')
blue_patch = mpatches.Patch(color=colours[2,:], label='class 3')
black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
ax3.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
            bbox_to_anchor=(1.5, 0.6), fontsize=12)
ax3.set_title('Segmented image')
# plot the contours
ax2.plot(sl_offshore[:,0], sl_offshore[:,1], 'k.', markersize=1)
ax1.plot(sl_offshore[:,0], sl_offshore[:,1], 'k.', markersize=1)
ax3.plot(sl_offshore[:,0], sl_offshore[:,1], 'k.', markersize=1)

# plot histogram
binwidth = 0.01
ax4.set_facecolor('0.75')
ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
ax4.set(ylabel='PDF',yticklabels=[], xlim=[-1,1], xlabel='NDWI pixel values')
bins = np.arange(np.nanmin(vec1), np.nanmax(vec1) + binwidth, binwidth)
ax4.hist(vec1,bins=bins,density=True,color=colours[0,:],label='class 1',alpha=0.75)
bins = np.arange(np.nanmin(vec2), np.nanmax(vec2) + binwidth, binwidth)
ax4.hist(vec2,bins=bins,density=True,color=colours[1,:],label='class 2',alpha=0.75)
bins = np.arange(np.nanmin(vec3), np.nanmax(vec3) + binwidth, binwidth)
ax4.hist(vec3,bins=bins,density=True,color=colours[2,:],label='class 3',alpha=0.75)
ax4.axvline(x=t_otsu1,ls='--',c='r',label='Otsu 1')
ax4.axvline(x=t_otsu2,ls='--',c='k',label='Otsu 2')
ax4.legend()

# store shoreline
shoreline = SDS_tools.convert_pix2world(sl_offshore[:,[1,0]], georef)
coords = shoreline
geom = geometry.MultiPoint([(coords[_,0], coords[_,1]) for _ in range(coords.shape[0])])
# save into geodataframe with attributes
gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
gdf.index = [0]
gdf.loc[0,'date'] = '20 July 2020'
gdf.crs = "EPSG:32756"
# save GEOJSON layer to file
# gdf.to_file('20Jul2020.geojson',driver='GeoJSON', encoding='utf-8')

fig.savefig('final_fig_aug2.jpg', dpi=300)