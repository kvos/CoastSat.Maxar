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

def initialise():
    image_name = 'Narrabeen_20Jul_clipped.tif'
    image_path = os.path.join(os.getcwd(),image_name)
    data = gdal.Open(image_path, gdal.GA_ReadOnly)
    # get image EPSG code
    proj = osr.SpatialReference(wkt=data.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    # get georef
    georef = np.array(data.GetGeoTransform())
    # get all bands
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im_ms = np.stack(bands, 2).astype(float)
    
    # no cloud mask (all False to be compatible with coastsat functions)
    cloud_mask = np.zeros((im_ms.shape[0],im_ms.shape[1])).astype(bool)
    
    
    im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99)

    
    return georef, epsg, im_ndwi, im_RGB

georef, epsg, index, im_RGB = initialise()


#%% compute RGB for visualisation


# im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99)
# # NDWI (NIR - Green normalised)
# im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
# # NIR - Blue normalised
# im_nir_blue_norm = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)
# # Red - Blue normalised
# im_red_blue_norm = SDS_tools.nd_index(im_ms[:,:,2], im_ms[:,:,0], cloud_mask)

# # to not try to plot images interactively, they are too big
# plt.ioff()
# cmap = 'coolwarm'
# make figure 
# fig,ax = plt.subplots(1,5,figsize=(15,10),sharex=True,sharey=True,tight_layout=True)
# ax[0].imshow(im_RGB)
# ax[0].axis('off')
# ax[0].set_title(image_name)
# ax[1].imshow(im_ms[:,:,3],cmap=cmap)
# ax[1].axis('off')
# ax[1].set_title('NIR')
# ax[2].imshow(im_ndwi,cmap=cmap)
# ax[2].axis('off')
# ax[2].set_title('NDWI')
# ax[3].imshow(im_nir_blue_norm,cmap=cmap)
# ax[3].axis('off')
# ax[3].set_title('NIR minus Blue normalised')
# ax[4].imshow(im_red_blue_norm,cmap=cmap)
# ax[4].axis('off')
# ax[4].set_title('Red minus Blue normalised')
# # save figure as .jpg and close it
# fp_figure = os.path.join(os.getcwd(),'figures',image_name.split('.tif')[0]+'.jpg')
# fig.savefig(fp_figure,dpi=300)
# plt.close(fig)





#%% Convert to vec

import skimage.filters as filters

def otsu_hist(index):
    #vec_im = np.copy(im_nir_blue_norm)
    vec_im = np.copy(index)
    vec = vec_im.reshape(vec_im.shape[0] * vec_im.shape[1])
    vec = vec[~np.isnan(vec)]
    
    #t_otsu = filters.multi_otsu(vec)
    t_otsu = filters.threshold_multiotsu(vec)


    # Plot histogram

    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([8, 8])
    
    # according to the image shape, decide whether it is better to have the images
    # in vertical subplots or horizontal subplots
    ax = fig.add_subplot(111)
        
    # Set labels
    ax.set_title('NDWI Normalised Pixel Value Histogram Thresholding',
                  fontsize = 10)
    ax.set_xlabel('NDWI' + ' Pixel Value', fontsize = 10)
    #ax.set_ylabel("Pixel Count", fontsize= 10)
    ax.set_ylabel("Pixel Class PDF", fontsize= 10)
    ax.axes.yaxis.set_ticks([])
    
    # Plot threshold value(s)
    ax.axvline(x = t_otsu[0], color = 'k', linestyle = '--', label = 'Threshold Value')
    ax.axvline(x = t_otsu[1], color = 'k', linestyle = '--', label = 'Threshold Value')
    
    
    # Add legend
    ax.legend(bbox_to_anchor = (1,1), loc='lower right', framealpha = 1,
              fontsize = 8) #, fontsize = 'xx-small')
    
    # Plot histogram
    ax.hist(vec, 150, color='blue', alpha=0.8, density=True)
    
    plt.show()
    
    return t_otsu



t_otsu = otsu_hist(index)


#%%

def pseudo_classify(index_in):
    index = np.copy(index_in)
    index[index > t_otsu[1]] = 3
    index[np.logical_and(index < t_otsu[1], index > t_otsu[0])] = 2
    index[index < t_otsu[0]] = 1
    
    plt.imshow(index)
    plt.show()

pseudo_classify(index)


#%%

def ref_sl_buffer(index_in):
    # Mask index
    index = np.copy(index_in)
    index = index < t_otsu[1]
    
    # Remove small water
    elem = morphology.square(25)
    index = morphology.binary_opening(index,elem)
    index = morphology.remove_small_objects(index, 
                                        min_size=1000*1000, 
                                        connectivity=1)
    index = index == 0
    
    # Remove small sand
    index = morphology.remove_small_objects(index, 
                                    min_size=1000*1000, 
                                    connectivity=1)
    index = morphology.binary_opening(index,elem)
    
    plt.imshow(index)
    plt.show()
    
    contours = measure.find_contours(index, 0.5)
    
    print('finding contours')
    
    idx = 0
    for i, sl in enumerate(contours):
        print(len(sl))
        if len(sl) >= len(contours[idx]):
            ref_sl_contour = sl
            idx = i
    
    
    
    # Create buffer
    im_shape = index.shape
    
    ref_sl_pix_rounded = np.round(ref_sl_contour).astype(int)

    # make sure that the pixel coordinates of the reference shoreline are inside the image
    idx_col = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[0])
    idx_row = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[1])
    idx_inside = np.logical_and(idx_row, idx_col)
    ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

    # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
    im_binary = np.zeros(im_shape)
    for j in range(len(ref_sl_pix_rounded)):
        im_binary[ref_sl_pix_rounded[j,0], ref_sl_pix_rounded[j,1]] = 1
    im_binary = im_binary.astype(bool)

    # dilate the binary image to create a buffer around the reference shoreline
    se = morphology.disk(30)
    im_buffer = morphology.binary_dilation(im_binary, se)

    #plt.imshow(im_buffer)
    #plt.show()
    
    return im_buffer

im_buffer = ref_sl_buffer(index)


#%% Get contours

def cont(index, im_buffer):
    
    index_in = np.copy(index)
    
    index_in[~im_buffer] = np.nan

    contours = measure.find_contours(index_in, t_otsu[1])
    
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = SDS_shoreline.process_contours(contours)
    # remove contours that are too short
    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, int(epsg), 28356)
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = geometry.LineString(coords) # shapely LineString structure
        if a.length >= 1000:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(contours_array,
                                                                28356,
                                                                int(epsg))[:,[0,1]], georef)
    return sl_pix

sl_pix = cont(index, im_buffer)



#%% Plot index im


import matplotlib.colors as colors


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


def plot_im(index_in):
    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([4, 8])
    
    # according to the image shape, decide whether it is better to have the images
    # in vertical subplots or horizontal subplots
    ax = fig.add_subplot(111)
    
    # Mask index
    index = np.copy(index_in)
    #index = mask()

    # Find index limits
    min = np.nanmin(index)
    max = np.nanmax(index)
    
    # Plot colourised index
    cmap = plt.cm.coolwarm  # red to blue
    cmap.set_bad(color='0.3')
    cax = ax.imshow(index, 
                     cmap=cmap, 
                     clim=(min, max), 
                     norm=MidpointNormalize(midpoint = t_otsu[1],
                                            vmin=min, vmax=max))
    
    # Overlay shoreline
    ax.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize = 0.1)     

    # Add colourbar
    cbar = fig.colorbar(cax, ax = ax, orientation='vertical', shrink=0.65)
    cbar.set_label('NDWI Pixel Value', rotation=270, labelpad=10)
    
    # Figure settings
    ax.axis('off')
    ax.set_title('NDWI', fontsize=10)        
    
    fig.savefig('/Users/Yarran/Desktop/ndwi.png', dpi=400)#, bbox_inches='tight', pad_inches=0.7) 

    plt.show()

plot_im(index)


#%%
def rgb_plot(im_RGB):
    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([4, 8])
    
    # according to the image shape, decide whether it is better to have the images
    # in vertical subplots or horizontal subplots
    ax = fig.add_subplot(111)

    # Set nan colour
    im_RGB = np.where(np.isnan(im_RGB), 0.3, im_RGB)
    
    # Plot background RGB im
    ax.imshow(im_RGB)
    
    # Overlay shoreline
    ax.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize = 0.3)     

    # Figure settings
    ax.axis('off')
    ax.set_title('RGB', fontsize=10) 
    
    fig.savefig('/Users/Yarran/Desktop/rgb.png', dpi=400)#, bbox_inches='tight', pad_inches=0.7) 
    
    plt.show()


rgb_plot(im_RGB)





