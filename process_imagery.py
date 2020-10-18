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
from pylab import ginput
import pickle
import geopandas as gpd
from shapely import geometry

# load one image
image_path = os.path.join(os.getcwd(),'order1',
                          '20JUL19000828-S2AS_R1C1-200000742456_01_P001.tif')
data = gdal.Open(image_path, gdal.GA_ReadOnly)


