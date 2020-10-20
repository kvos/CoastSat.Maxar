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
import SDS_tools

# load beaches shapefile
beaches = gpd.read_file(os.path.join(os.getcwd(),'beaches_shapefile','beaches.shp'))

beach = beaches.loc[beaches['beach_name'] == 'Narrabeen']
beach_rect = np.array(beach.iloc[0]['geometry'].envelope.exterior.coords)

# load image
image_path = os.path.join(os.getcwd(),'order1',
                          '20JUL19000828-S2AS_R2C1-200000742456_01_P001.tif')
data = gdal.Open(image_path, gdal.GA_ReadOnly)


# # crop image
# left = np.min(beach_rect[:,0])
# right = np.max(beach_rect[:,0])
# upper = np.max(beach_rect[:,1])
# lower = np.min(beach_rect[:,1])
# window = [left,upper,right,lower]
# gdal.Translate( os.path.join(os.getcwd(),'cropped','test.tif'),image_path,
#                projWin=window)


# get image EPSG code
proj = osr.SpatialReference(wkt=data.GetProjection())
epsg = proj.GetAttrValue('AUTHORITY',1)
# get georef
georef = np.array(data.GetGeoTransform())

# convert beach rectangle to pixel coordinates
beach_rect_pix = SDS_tools.convert_world2pix(beach_rect,georef)

# get first band
band1 = data.GetRasterBand(1).ReadAsArray()
plt.figure()
plt.imshow(band1, cmap='gray')
plt.plot(beach_rect_pix[:,0],beach_rect_pix[:,1],'ro-')






bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
im_ms = np.stack(bands, 2)

