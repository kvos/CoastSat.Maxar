''' 

Functions for Maxar image shoreline extraction
    Yarran Doherty & Kilian Vos 2020
    
'''

from osgeo import gdal
from osgeo import osr

import numpy as np

import skimage.filters as filters
import skimage.morphology as morphology
import skimage.exposure as exposure
import skimage.measure as measure
import skimage.transform as transform

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from shapely import geometry


#%% Shoreline Extraction 

def initialise(image_path):
    
    ''' 
    Import image file, calculate water index and extract geospatial features
    Yarran Doherty & Kilian Vos 2020
    '''
    
    # Open image
    data = gdal.Open(image_path, gdal.GA_ReadOnly)
    
    # get image EPSG code
    proj = osr.SpatialReference(wkt=data.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    
    # get georef data
    georef = np.array(data.GetGeoTransform())
    
    # get all bands
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im_ms = np.stack(bands, 2).astype(float)
    
    # Create fake cloud mask (all False to be compatible with coastsat functions)
    cloud_mask = np.zeros((im_ms.shape[0],im_ms.shape[1])).astype(bool)
    
    # Calculate water index
    im_ndwi = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)

    # Process RGB image for plotting
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99)

    return georef, epsg, im_ndwi, im_RGB



def threshold(index):
    
    ''' 
    Caluclates shoreline value threshold from water index image and plots histogram 
    Yarran Doherty & Kilian Vos 2020
    '''
    
    # Convert to 1d array and ermove nan values
    vec_im = np.copy(index)
    vec = vec_im.reshape(vec_im.shape[0] * vec_im.shape[1])
    vec = vec[~np.isnan(vec)]
    
    # Perform multi otsu thresholding
    t_otsu = filters.threshold_multiotsu(vec)

    # Plot histogram
    plt.ioff()
    
    # create figure
    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches([8, 8])
    ax = fig.add_subplot(111)
        
    # Set labels
    ax.set_title('NDWI Pixel Value Histogram Thresholding',
                  fontsize = 10)
    ax.set_xlabel('NDWI Pixel Value', fontsize = 10)
    #ax.set_ylabel("Pixel Count", fontsize= 10)
    ax.set_ylabel("Pixel Class PDF", fontsize= 10)
    ax.axes.yaxis.set_ticks([])
    
    # Plot threshold value(s)
    ax.axvline(x = t_otsu[0], color = 'k', linestyle = '--', label = 'Class Threshold')
    ax.axvline(x = t_otsu[1], color = 'k', linestyle = '--', label = 'Shoreline Threshold')
    
    # Add legend
    ax.legend(bbox_to_anchor = (1,1), loc='lower right', framealpha = 1,
              fontsize = 8) #, fontsize = 'xx-small')
    
    # Plot histogram
    ax.hist(vec, 150, color='blue', alpha=0.8, density=True)
    
    plt.show()
    
    return t_otsu


def pseudo_classify(index_in, t_otsu):
    
    '''
    Classify image using multi otsu values
    Yarran Doherty & Kilian Vos 2020
    '''
    
    index = np.copy(index_in)
    index[index > t_otsu[1]] = 3
    index[np.logical_and(index < t_otsu[1], index > t_otsu[0])] = 2
    index[index < t_otsu[0]] = 1
    
    plt.imshow(index)
    plt.show()
    
    return index


def ref_sl_buffer(index_in, t_otsu):
    
    '''
    Create buffer region around shoreline to prevent contouring on non
        sand/water regions
        
    Yarran Doherty & Kilian Vos 2020
    '''
    
    # Mask index
    index = np.copy(index_in)
    index = index < t_otsu[1]
    
    # Remove small water regions
    elem = morphology.square(25)
    index = morphology.binary_opening(index,elem)
    index = morphology.remove_small_objects(index, 
                                        min_size=1000*1000, 
                                        connectivity=1)
    index = index == 0
    
    # Remove small sand regions
    index = morphology.remove_small_objects(index, 
                                    min_size=1000*1000, 
                                    connectivity=1)
    index = morphology.binary_opening(index,elem)
    
    # Plot image
    #plt.imshow(index)
    #plt.show()
    
    # Contour sand/water region
    contours = measure.find_contours(index, 0.5)
    
    # print('finding contours')
    
    # Extract longest contour
    idx = 0
    for i, sl in enumerate(contours):
        #print(len(sl))
        if len(sl) >= len(contours[idx]):
            ref_sl_contour = sl
            idx = i
    
    # Convert contour into buffer
    im_shape = index.shape
    ref_sl_pix_rounded = np.round(ref_sl_contour).astype(int)
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


def extract_shoreline(index, im_buffer, t_otsu, georef, epsg):
    
    '''
    Extract shoreline contour
    Yarran Doherty & Kilian Vos 2020
    '''
    
    index_in = np.copy(index)
    
    index_in[~im_buffer] = np.nan

    contours = measure.find_contours(index_in, t_otsu[1])
    
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)
    # remove contours that are too short
    # convert pixel coordinates to world coordinates
    contours_world = convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = convert_epsg(contours_world, int(epsg), 28356)
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
    sl_pix = convert_world2pix(convert_epsg(contours_array,
                                                                28356,
                                                                int(epsg))[:,[0,1]], georef)
    return sl_pix


#%% Plotting

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


def index_plot(index_in, t_otsu, sl_pix):
    
    '''
    Plot index image with shorelines
    Yarran Doherty & Kilian Vos 2020
    '''
    
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



def rgb_plot(im_RGB, sl_pix):
    
    '''
    Plot RGB image with shorelines
    Yarran Doherty & Kilian Vos 2020
    '''
    
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



#%% COASTSAT FUNCTIONS

def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj


def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact 
    with clouds.
    
    KV WRL 2020
    
    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours    
    
    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs) 
        
    """
    
    # initialise variable
    contours_nonans = []
    # loop through contours and only keep the ones without NaNs
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    
    return contours_nonans


def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

    return points_converted


def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


