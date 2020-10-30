"""
Extract shorelines from Maxar/DigitalGlobe imagery
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure

# other modules
from osgeo import gdal
from osgeo import osr
from pylab import ginput
import pickle
import geopandas as gpd
from shapely import geometry

# own modules
import SDS_tools, SDS_preprocess

#%% 1. Use shapefile with beaches to crop the images (better to do it on QGIS)
beaches = gpd.read_file(os.path.join(os.getcwd(),'beaches_shapefile','beaches.shp'))

# make 200m around shoreline and take the minimum rectangle
beach = beaches.loc[beaches['beach_name'] == 'Narrabeen']
beach_rect = np.array(beach.iloc[0]['geometry'].buffer(200).envelope.exterior.coords)

# load the image
image_path = os.path.join(os.getcwd(),'order1',
                          '20JUL19000828-S2AS_R2C1-200000742456_01_P001.tif')
data = gdal.Open(image_path, gdal.GA_ReadOnly)

# # clip the image
left = np.min(beach_rect[:,0])
right = np.max(beach_rect[:,0])
upper = np.max(beach_rect[:,1])
lower = np.min(beach_rect[:,1])
window = [left,upper,right,lower]
gdal.Translate(os.path.join(os.getcwd(),'cropped_images',
                            beach.iloc[0]['beach_name']+'.tif'),
               image_path, projWin=window)


#%% load the clipped image
image_path = os.path.join(os.getcwd(),'cropped_images',
                          'Narrabeen_20Jul_clipped.tif')
data = gdal.Open(image_path, gdal.GA_ReadOnly)
# get image EPSG code
proj = osr.SpatialReference(wkt=data.GetProjection())
epsg = proj.GetAttrValue('AUTHORITY',1)
# get georef
georef = np.array(data.GetGeoTransform())
# get bands
bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
im_ms = np.stack(bands, 2).astype(float)

# plot RGB and NIR
cloud_mask = np.zeros((im_ms.shape[0],im_ms.shape[1])).astype(bool)
im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99)
im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
im_ndwi_b = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)

plt.ioff()
fig,ax = plt.subplots(1,4,figsize=(15,10),sharex=True,sharey=True,tight_layout=True)
ax[0].imshow(im_RGB)
ax[0].axis('off')
ax[0].set_title('RGB 20 July')
ax[1].imshow(im_ms[:,:,3],cmap='seismic')
ax[1].axis('off')
ax[1].set_title('NIR')
ax[2].imshow(im_ndwi,cmap='seismic')
ax[2].axis('off')
ax[2].set_title('NDWI')
ax[3].imshow(im_ndwi_b,cmap='seismic')
ax[3].axis('off')
ax[3].set_title('NIR minus Blue normalised')
fig.savefig('indices5.jpg',dpi=300)
plt.close(fig)


