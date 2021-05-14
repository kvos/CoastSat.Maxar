"""This module contains functions to analyze the shoreline data along transects' 
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# other modules
import skimage.transform as transform
from pylab import ginput
import geopandas as gpd
from datetime import datetime, timedelta

# own modules
import SDS_tools

def create_transect(origin, orientation, length):
    """
    Create a 2D transect of points with 1m interval. 
    
    Arguments:
    -----------
        origin: np.array
            contains the X and Y coordinates of the origin of the transect
        orientation: int
            angle of the transect (anti-clockwise from North) in degrees
        length: int
            length of the transect in metres
        
    Returns:    
    -----------
        transect: np.array
            contains the X and Y coordinates of the transect
        
    """     
    x0 = origin[0]
    y0 = origin[1]
    # orientation of the transect
    phi = (90 - orientation)*np.pi/180 
    # create a vector with points at 1 m intervals
    x = np.linspace(0,length,length+1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y 
    # translate and rotate the vector using the origin and orientation
    tf = transform.EuclideanTransform(rotation=phi, translation=(x0,y0))
    transect = tf(coords)
                
    return transect

def draw_transects(output, settings):
    """
    Allows the user to draw shore-normal transects over the mapped shorelines.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        settings: dict
            contains the inputs
    Returns:    
    -----------
        transects: dict
            contains the X and Y coordinates of all the transects drawn. These are also saved
             as a .geojson (+ a .jpg figure showing the location of the transects)
        
    """    
    sitename = settings['inputs']['sitename']
    filepath = os.path.join(settings['inputs']['filepath'], sitename)

    # plot all shorelines
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis('equal')
    ax1.set_xlabel('Eastings [m]')
    ax1.set_ylabel('Northings [m]')
    ax1.grid(linestyle=':', color='0.5')
    for i in range(len(output['shorelines'])):
        sl = output['shorelines'][i]
        date = output['dates'][i]
        ax1.plot(sl[:, 0], sl[:, 1], '.', markersize=3, label=date.strftime('%d-%m-%Y'))
#    ax1.legend()
    fig1.set_tight_layout(True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    ax1.set_title('Click two points to define each transect (first point is the origin of the transect).\n'+
              'When all transects have been defined, click on <ENTER>', fontsize=16)
    
    # initialise variable
    transects = dict([])
    counter = 0
    # loop until user breaks it by click <enter>
    while 1:
        # let user click two points
        pts = ginput(n=2, timeout=1e9)
        if len(pts) > 0:
            origin = pts[0]
        else:
            fig1.gca().set_title('Transect locations', fontsize=16)
            fig1.savefig(os.path.join(filepath, 'jpg_files', sitename + '_transect_locations.jpg'), dpi=200)
            plt.title('Transect coordinates saved as ' + sitename + '_transects.geojson')
            plt.draw()
            ginput(n=1, timeout=3, show_clicks=True)
            plt.close(fig1)
            break
        counter = counter + 1
        transect = np.array([pts[0], pts[1]])
        
        # alternative of making the transect the origin, orientation and length
#        temp = np.array(pts[1]) - np.array(origin)
#        phi = np.arctan2(temp[1], temp[0])
#        orientation = -(phi*180/np.pi - 90)
#        length = np.linalg.norm(temp)
#        transect = create_transect(origin, orientation, length)
        
        transects[str(counter)] = transect
        
        # plot the transects on the figure
        ax1.plot(transect[:,0], transect[:,1], 'b-', lw=2.5)
        ax1.plot(transect[0,0], transect[0,1], 'rx', markersize=10)
        ax1.text(transect[-1,0], transect[-1,1], str(counter), size=16,
                 bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        
    # save as transects.geojson (for GIS)
    gdf = SDS_tools.transects_to_gdf(transects)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson    
    gdf.to_file(os.path.join(filepath, sitename + '_transects.geojson'), driver='GeoJSON', encoding='utf-8')
    print('Transect locations saved in ' + filepath)
        
    return transects

def compute_intersection(shorelines, transects, settings):
    """
    Computes the intersection between the 2D mapped shorelines and the transects, to generate
    time-series of cross-shore distance along each transect.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        settings: dict
                along_dist: alongshore distance to caluclate the intersection (median of points 
                within this distance).   
                nan/max:
                max_std:
                max_range:
                min_val:
                prc_std:
                        
    Returns:    
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally 
            corrected.
        
    """      
    
    # initialise dictionary with intersections for each transect
    cross_dist = dict([])
    
    along_dist = settings['along_dist']

    # loop through each transect
    for key in transects.keys():
        
        # initialise variables
        std_intersect = np.zeros(len(shorelines))
        med_intersect = np.zeros(len(shorelines))
        max_intersect = np.zeros(len(shorelines))
        min_intersect = np.zeros(len(shorelines))
        n_intersect = np.zeros(len(shorelines))
        
        # loop through each shoreline
        for i in range(len(shorelines)):

            sl = shorelines[i]
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
#            temp_sl = sl - np.array(transects[key][0,:])
#            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
#            diff_angle = (phi - phi_sl)
#            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
#            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]
            
            idx_close = np.where(idx_dist)[0]
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
                xy_rot[0, xy_rot[0,:] < settings['min_val']] = np.nan
                
                # compute std, median, max, min of the intersections
                std_intersect[i] = np.nanstd(xy_rot[0,:])
                med_intersect[i] = np.nanmedian(xy_rot[0,:])
                max_intersect[i] = np.nanmax(xy_rot[0,:])
                min_intersect[i] = np.nanmin(xy_rot[0,:])
                n_intersect[i] = len(xy_rot[0,:])
                
        # quality control the intersections using dispersion metrics (std and range)
        condition1 = std_intersect <= settings['max_std']
        condition2 = (max_intersect - min_intersect) <= settings['max_range']
        condition3 = n_intersect > 2
        idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)
        
        # save copy for QA plot later
        med_intersect_temp = med_intersect.copy()
        max_intersect_temp = max_intersect.copy()
        
        # decide what to do with the intersections with high dispersion 
        if settings['nan/max'] == 'auto':
            # compute the percentage of data points where the std is larger than the user-defined max
            prc_over = np.sum(std_intersect > settings['max_std'])/len(std_intersect)
            # if more than a certain percentage is above, use the maximum intersection
            if prc_over > settings['prc_std']:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            # otherwise put a nan
            else:
                med_intersect[~idx_good] = np.nan
                
        elif settings['nan/max'] == 'max':
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan
            prc_over = 0
            
        elif settings['nan/max'] == 'nan':
            med_intersect[~idx_good] = np.nan
            prc_over = 0
        
        else:
            raise Exception('the nan/max parameter can only be: nan, max or auto')

        # store in dict
        cross_dist[key] = med_intersect

    return cross_dist

def identify_outliers(chainage, dates, cross_change):
    """
    Remove outliers based on despiking [iterative method]
    
    Arguments:
    -----------
    chainage: list
        time-series of shoreline change
    dates: list of datetimes
        correspondings dates
    cross_change: float
        threshold distance to identify a point as an outlier
        
    Returns:    
    ----------- 
    chainage_temp: list
        time-series of shoreline change without outliers
    dates_temp: list of datetimes
        dates without outliers
        
    """
    
    # make a copy of the inputs
    chainage_temp = chainage.copy()
    dates_temp = dates.copy()
    
    # loop through the time-series always starting from the start
    # when an outlier is found, remove it and restart
    # repeat until no more outliers are found in the time-series
    k = 0
    while k < len(chainage_temp):
        
        for k in range(len(chainage_temp)):
            
            # check if the first point is an outlier
            if k == 0:
                # difference between 1st and 2nd point in the time-series
                diff = chainage_temp[k] - chainage_temp[k+1]
                if np.abs(diff) > cross_change:
                    chainage_temp.pop(k)  
                    dates_temp.pop(k)
                    break
                
            # check if the last point is an outlier
            elif k == len(chainage_temp)-1:
                # difference between last and before last point in the time-series
                diff = chainage_temp[k] - chainage_temp[k-1]
                if np.abs(diff) > cross_change:
                    chainage_temp.pop(k)  
                    dates_temp.pop(k) 
                    break
                
            # check if a point is an isolated outlier or in a group of 2 consecutive outliers
            else:  
                # calculate the difference with the data point before and after
                diff_m1 = chainage_temp[k] - chainage_temp[k-1]
                diff_p1 = chainage_temp[k] - chainage_temp[k+1]
                # remove point if isolated outlier, distant from both neighbours
                condition1 = np.abs(diff_m1) > cross_change
                condition2 = np.abs(diff_p1) > cross_change
                # check that distance from neighbours has the same sign 
                condition3 = np.sign(diff_p1) == np.sign(diff_m1)
                if np.logical_and(np.logical_and(condition1,condition2),condition3):
                    chainage_temp.pop(k)  
                    dates_temp.pop(k) 
                    break
                
                # check for 2 consecutive outliers in the time-series
                if k >= 2 and k < len(chainage_temp)-2:
                    
                    # calculate difference with the data around the neighbours of the point
                    diff_m2 = chainage_temp[k-1] - chainage_temp[k-2]
                    diff_p2 = chainage_temp[k+1] - chainage_temp[k+2]
                    # remove if there are 2 consecutive outliers (see conditions below)
                    condition4 = np.abs(diff_m2) > cross_change
                    condition5 = np.abs(diff_p2) > cross_change
                    condition6 = np.sign(diff_m1) == np.sign(diff_p2)
                    condition7 = np.sign(diff_p1) == np.sign(diff_m2)
                    # check for both combinations (1,5,6 and ,2,4,7)
                    if np.logical_and(np.logical_and(condition1,condition5),condition6):
                        chainage_temp.pop(k)  
                        dates_temp.pop(k) 
                        break
                    elif np.logical_and(np.logical_and(condition2,condition4),condition7):
                        chainage_temp.pop(k)  
                        dates_temp.pop(k) 
                        break
                    
                    # also look for clusters of 3 outliers
                    else:
                        # increase the distance to make sure these are really outliers
                        condition4b = np.abs(diff_m2) > 1.5*cross_change
                        condition5b = np.abs(diff_p2) > 1.5*cross_change
                        condition8 = np.sign(diff_m2) == np.sign(diff_p2)
                        # if point is close to immediate neighbours but 
                        # the neighbours are far from their neighbours, point is an outlier
                        if np.logical_and(np.logical_and(np.logical_and(condition4b,condition5b),
                                                         np.logical_and(~condition1,~condition2)),
                                                         condition8):
                            print('*', end='')
                            chainage_temp.pop(k)  
                            dates_temp.pop(k) 
                            break                            
        
        # if one full loop is completed (went through all the time-series without removing outlier)
        # then increment k to get out of the loop
        k = k + 1
            
     
    # return the time-series where the outliers have been removed           
    return chainage_temp, dates_temp

def reject_outliers(cross_distance, output, settings):
    """
    
    Arguments:
    -----------
        cross_distance: dict
            time-series of shoreline change
        output: dict
            mapped shorelines with metadata
        settings: dict
        
    Returns:    
    ----------- 
        chain_dict: dict
            contains the updated time-series of cross-shore distance with the corresponding dates
        
    """
    
    chain_dict = dict([])
    
    for i,key in enumerate(list(cross_distance.keys())):
        
        chainage = cross_distance[key].copy()
        if sum(np.isnan(chainage)) == len(chainage):
            continue

        # 1. remove nans and negative chainages
        idx_nonan = np.where(~np.isnan(chainage))[0]
        chainage1 = [chainage[k] for k in idx_nonan]
        dates1 = [output['dates'][k] for k in idx_nonan]
#        satnames1 = [output['satname'][k] for k in idx_nonan]

        # 2. remove points where the MNDWI threshold is above a certain value (max_threshold)
        if np.isnan(settings['otsu_threshold'][0]):
            chainage2 = chainage1
            dates2 = dates1
        else:
            threshold1 = [output['threshold'][k] for k in idx_nonan]
            idx_thres = np.where(np.logical_and(np.array(threshold1) <= settings['otsu_threshold'][1],
                                                np.array(threshold1) >= settings['otsu_threshold'][0]))[0]
            chainage2 = [chainage1[k] for k in idx_thres]
            dates2 = [dates1[k] for k in idx_thres]
            if len(chainage2) < 30:
                continue
        
        # 3. remove outliers based on despiking [iterative method]
        chainage3, dates3 = identify_outliers(chainage2, dates2, settings['max_cross_change'])          
        
        # remove outliers both ends of the time-series (for high tide range sites only)
        if settings['max_cross_change'] > settings['cross_change_vals']:
            if not np.isnan(settings['max_threshold']): # if before tidal correction
                # check if any of the last 3 points is an outlier
                # these may affect the beach slope calculations
                diffs = np.abs([chainage3[-1] - chainage3[-2],
                                chainage3[-2] - chainage3[-3],
                                chainage3[-3] - chainage3[-4]])
                bool_diff =  diffs >= 0.8*settings['max_cross_change']
                if np.any(bool_diff):
                    idx_last = -(np.max(np.where(bool_diff))+1)
                    dates3 = dates3[:idx_last]
                    chainage3 = chainage3[:idx_last]
                    print('|%s#%d|'%(key[-4:], idx_last), end='')
            else: # after tidal correction
                # check if any of the first 3 points is an outlier
                diffs = np.abs([chainage3[0] - chainage3[1],
                                chainage3[1] - chainage3[2],
                                chainage3[2] - chainage3[3]])
                bool_diff =  diffs >= 1.5*settings['max_cross_change']
                if np.any(bool_diff):
                    idx_first = np.max(np.where(bool_diff))+1
                    # check that the datapoints are less than 2 years apart
                    if dates3[idx_first]-dates3[idx_first-1] < timedelta(days=730):
                        dates3 = dates3[idx_first:]
                        chainage3 = chainage3[idx_first:]   
                        print('|%d#%s|'%(idx_first,key[-4:]), end='')

        # fill with nans the indices to be removed from cross_distance
        idx_kept = []
        for date in output['dates']: idx_kept.append(date in dates3)
        chainage[~np.array(idx_kept)] = np.nan
        # store in chain_dict
        chain_dict[key] = chainage
                
        # figure for QA
        if settings['plot_fig']:
            fig,ax=plt.subplots(2,1,figsize=[12,6], sharex=True)
            fig.set_tight_layout(True)
            ax[0].grid(linestyle=':', color='0.5')
            ax[0].set(ylabel='distance [m]',
                   title= 'Transect %s    -    %d points' % (key, len(chainage3)))
            mean_cross_dist = np.nanmedian(chainage3)
            # plot the data points
            ax[0].plot(dates1, chainage1-mean_cross_dist, 'C0-')
            ax[0].plot(dates1, chainage1-mean_cross_dist, 'C2o', ms=4, mec='k', mew=0.7)
            # plot the indices removed because of the threshold
            ax[0].plot(dates2, chainage2-mean_cross_dist, 'C3o', ms=4, mec='k', mew=0.7)
            # plot the final time-series
            ax[0].plot(dates3, chainage3-mean_cross_dist, 'C0o', ms=4, mfc='w', mec='C0')
            ax[1].grid(linestyle=':', color='0.5') 
            ax[1].plot(dates3, chainage3-mean_cross_dist, 'C0-o', ms=4, mfc='w', mec='C0')
            print('%s  - removed %d'%(key, len(dates1) - len(dates3)))

    return chain_dict