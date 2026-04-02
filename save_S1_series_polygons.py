#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:49:34 2019

@author: juanma
"""
import os
import numpy as np
from spectral import envi

import matplotlib.pyplot as plt
import matplotlib.path as pth

from scipy.ndimage import binary_erosion

import shapefile as shp
import pickle as pk

#####################
# Auxiliary functions
#####################

def read_dates_from_stack( folder ):
    
    # List all files in the folder

    image_list = [ os.path.basename(f.path) for f in os.scandir(folder) ]
   
    # Loop for all different dates in the input folder

    date_list = []

    for img in image_list:
        
        # Dates in YYYYMMDD format are the first 8 characters
        
        date = img[:8]

        # Add date to list if it was not yet on the list
        
        if date not in date_list:
            date_list.append( date)
            
    # Return the list of dates, sorted in ascending (chronological) order
            
    return sorted(date_list)
        
def axes_from_metadata( file_with_metadata ):
   
    # Read the metadata of the Envi header file
    
    file_hdr = envi.open( file_with_metadata + '.hdr',  file_with_metadata + '.bin')
    meta = file_hdr.metadata
   
    # Size of the images
    
    nlines = int(meta['lines'])
    ncols = int(meta['samples'])
        
    # Extension of the study area in the images
    
    meta_xref = float(meta['map info'][3])
    meta_yref = float(meta['map info'][4])
    meta_xref_col = float(meta['map info'][1])
    meta_yref_lin = float(meta['map info'][2])
    meta_dx = float(meta['map info'][5])
    meta_dy = float(meta['map info'][6])
    
    # Axis limits (min,max) along both coordinates
    
    meta_x_min = meta_xref - meta_dx*meta_xref_col
    meta_x_max = meta_xref + meta_dx*(int(meta['samples'])-meta_xref_col)
    
    meta_y_min = meta_yref - meta_dy*(int(meta['lines'])-meta_yref_lin)
    meta_y_max = meta_yref + meta_dy*meta_yref_lin
        
    # Generate axis along both coordinates. The Y axis (North-South) must be inverted (flipped)
    
    y_axis = np.flip( np.linspace( meta_y_min, meta_y_max, nlines, endpoint=False) + meta_dy)
    x_axis = np.linspace( meta_x_min, meta_x_max, ncols, endpoint=False)
    
    return x_axis, y_axis

        
######################################################
# PARAMETERS 
######################################################

# Input products
main_folder = '/home/juanma/MyStore/Users/Juanma/Processed_Data/Sentinel1/Flevopolder/results/UASP/Geocoded_Products'

products_folder = 'C2/BOXCAR_4x19'

# base_product = 'sigma0_dB'
# ml_txt = '4x19'

input_folder = os.path.join( main_folder, products_folder )

# # Reference data
groundtruth_folder = '/home/juanma/MyStore/Users/Juanma/GroundData_Flevopolder/Flevoland_data/Data_25_fields/Feloveland-fields-Shapefiles'

# Output file
# output_file = main_folder + '/S1_' + orbit + '_all_polygons' + '.pkl'

######################################################
# MAIN PROGRAM #######################################
######################################################

# Example with one parcel

file_polygon_example = 'AKW-G1_C.shp'

groundtruth_file = os.path.join( groundtruth_folder, file_polygon_example )
sf = shp.Reader(groundtruth_file)

records = sf.records()

# Dates with images

dates = read_dates_from_stack( input_folder )
ndates = len(dates)

# Read metadata of one image file

file_for_metadata = os.path.join( input_folder, dates[0] + '.sen1', 'C11' )

X_geo, Y_geo = axes_from_metadata( file_for_metadata  )

# Make a canvas with coordinates

X_geogrid, Y_geogrid = np.meshgrid( X_geo, Y_geo ) 
X_geogrid, Y_geogrid = X_geogrid.flatten(), Y_geogrid.flatten()
allpoints_geo = np.vstack((X_geogrid, Y_geogrid)).T

# Read the polygon

s = sf.shape(0)
geoj = s.__geo_interface__
    
pp = geoj['coordinates']
 
mypoly = np.squeeze( np.array(pp) )

mypath = pth.Path(mypoly)
    
# Select points inside the polygon and generate mask

mygrid = mypath.contains_points(allpoints_geo)
mymask = mygrid.reshape(len(X_geo), len(Y_geo), order='F').T 

# Erode the mask to avoid edge effects (due to multilook and imprecise geocoding)

binary_erosion(mymask, structure=np.ones((3,3))).astype(mymask.dtype)

# Extract values inside the masked area for each date, and save mean and standard deviation

VH_dB_mean = np.zeros( ndates ) 
VH_dB_std = np.zeros( ndates ) 
VV_dB_mean = np.zeros( ndates ) 
VV_dB_std = np.zeros( ndates ) 

for idate, date in enumerate( dates ):
    
    print(date)
    
    # Sigma0 at VV channel is C11
    
    data = envi.open( os.path.join( input_folder, date + '.sen1', 'C11' + '.hdr' ), os.path.join( input_folder, date + '.sen1', 'C11' + '.bin' )).read_band(0)
    datavalues_inside_polygon = 10*np.log10( data[mymask] + 1e-12 )
    
    VV_dB_mean[idate] = np.mean( datavalues_inside_polygon )
    VV_dB_std[idate] = np.std( datavalues_inside_polygon )
    
    # Sigma0 at VH channel is C22
    
    data = envi.open( os.path.join( input_folder, date + '.sen1', 'C22' + '.hdr' ), os.path.join( input_folder, date + '.sen1', 'C22' + '.bin' ) ).read_band(0)
    datavalues_inside_polygon = 10*np.log10( data[mymask] + 1e-12 )
    
    VH_dB_mean[idate] = np.mean( datavalues_inside_polygon )
    VH_dB_std[idate] = np.std( datavalues_inside_polygon )


