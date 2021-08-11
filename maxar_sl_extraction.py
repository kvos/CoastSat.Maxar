"""

Extract shorelines from Maxar/DigitalGlobe imagery
    Yarran Doherty & Kilian Vos 2020

"""

import sl_functions
import os


#%% 0. Crop downloaded imagery to area of interest (AOI)

# Processing run time is heavily impacted by file size so the smaller the AOI, 
    # the faster the processing time

# Cropped image file should be saved as a new file. It is recomended this 
    # step  be performed in ArcGIS, QGIS or similar



#%% 1. Load image data and calculate water index

# Cropped AOI image .tif filepath
image_path = os.path.join(os.getcwd(), 'Narrabeen_20Jul_clipped.tif')

# Import data
georef, epsg, index, im_RGB = sl_functions.initialise()



#%% 2. Calculate water index shoreline threshold and plot histogram

t_otsu = sl_functions.threshold(index)



#%% 3. Create buffer region around shoreline to ensure only sand/water region is contoured

im_buffer = sl_functions.ref_sl_buffer(index, t_otsu)



#%% 4. Extract shoreline

sl_pix = sl_functions.extract_shoreline(index, im_buffer, t_otsu, georef, epsg)



#%% 5. Plot shoerline on index and rgb images

sl_functions.index_plot(index, t_otsu, sl_pix)
sl_functions. rgb_plot(im_RGB, sl_pix)



