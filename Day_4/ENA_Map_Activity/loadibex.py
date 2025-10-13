# -*- coding: utf-8 -*-
"""
loadibex.py: this file contains functions to load IBEX data from files *txt , *.csv

Version: v1.8 (2024-10-16)

recent changes:
    240604: add function load_incafile()
    240815: add json file data load.
    240816: add load option based on ebin, year, qty.
    241016: filename parameter in load_json

Created on Tue Mar 12 10:28:27 2024

@author: jgasser
"""

import numpy as np
import json

''' Load JSON file containing the default config variables (path,...)'''
def load_json(filename='ibex_data'):
    rout = __file__.replace('\\','/')
    this_path = rout[0: 1+rout.rindex('/')]
    with open(this_path +filename +'.json', 'r') as fid:
        jsn = json.load(fid)
    return jsn

'''
load data file from IBEX data in folder, based on filename & path.
Input:
    filename: the <subfolder and> filename of datafile.
    path: the data containing directory path. if None, the correct path for Lo or Hi 
        data is looked up in the JSON file.
Output: returns lon, lat, data.
    lon: Ecl. Longitude 1d array
    lat: Ecl. Latitude 1d array
    data: 2d array with 'qty' data.
'''
def load_ibexfile(filename, path=None):
    if path is None:
        if '-hi-' in filename:
            path = load_json()['path_ibex_hi']
        elif '-lo-' in filename:
            path = load_json()['path_ibex_lo']
        else:
            path ='' #runtime path
    
    if path and path[-1] != '/':
        path +='/'
    fid = open(path +filename, 'r')
    
    hdline= fid.readline()
    # hdr_num= int(hdline[2:4])
    lat_num= int(hdline[5:7])
    lon_num= int(hdline[8:10])
    
    buf =[]
    for line in fid:
        if line[0] == '#':
            continue
        nums= [ float(s) for s in line.split(' ') if s]
        buf.append(nums)
    fid.close()
    
    data = np.array(buf) # flux data set from file
    lon= np.linspace(0, 360, lon_num+1) # ecl.longitude bin edges
    lat= np.linspace(-90, 90, lat_num+1) # ecl.latitude bin edges
    
    # --- Removing inconsistency in Lo-maps (2015) between flux <--> fexp, fvar
    #     on whether data available or not. :
    if filename[:-11].endswith('_2015/lv60.lohb-trp-flux100-lo') and (
            filename[:-4].endswith('fexp') or filename[:-4].endswith('fvar') ):
        data[ 1:29, 13:17 ] =0
    # --- enabling the following line causes some 'nan holes' in some flux maps
    # data[ np.where(data==0) ] = np.nan
    
    return lon, lat, data
# END OF FUNCTION


'''
load data file from IBEX data release folder:
this function will load "standard" data file from Lo or Hi: noSP, noCG, ram data.
Input:
    is_hi: True= IBEX-Hi data (False= IBEX-Lo data.)
    ebin: energy bin no. (1 to 8 for Lo; 2 to 6 for Hi)
    year: year of annual data map. 2009 to 2019 for Lo, to 2022 for Hi (as of DR17/18)
    qty: which quantity data to load: 'flux', 'fvar', 'fexp'.
Output: returns lon, lat, data.
    lon: Ecl. Longitude 1d array
    lat: Ecl. Latitude 1d array
    data: 2d array with 'qty' data.
( added: 24-08-16 )
'''
def load_ibexfile_std( is_hi=True, ebin=3, year=2009, qty='flux'):
    if is_hi:
        path = load_json()['path_ibex_hi']
        file0 = load_json()['file0_ibex_hi']
        ebin0str= '-2-'
    else:
        path = load_json()['path_ibex_lo']
        file0 = load_json()['file0_ibex_lo']
        ebin0str= '-1-'
        
    file1 = file0.replace( ebin0str, '-'+str(ebin)+'-').replace(
                '2009',str(year) ).replace('flux.', qty+'.')
    return load_ibexfile(file1, path)
# END OF FUNCTION


'''
load data file exported from routine "ibex_postprocess_export.py"
    containing combined IBEX maps.
'''
def load_combifile(filename, path=None):
    if path is None:
        path = load_json()['path_combined_data']
    if path[-1] != '/':
        path +='/'
    fid = open(path +filename, 'r')
    
    lon, lat, buf = [],[],[]
    for line in fid:
        if line[0] == '#':
            continue
        elif lon ==[]:
            lon= [ float(s) for s in line.split(',')]
        elif lat ==[]:
            lat= [ float(s) for s in line.split(',')]
        else:
            nums= [ float(s) for s in line.split(',') if s]
            buf.append(nums)
    fid.close()
    
    data = np.array(buf)
    return np.array(lon), np.array(lat), data
# END OF FUNCTION


'''
load data file from Cassini/INCA data.
'''
def load_incafile(filename, path=None):
    if not path:
        path = load_json()['path_inca']
    if path[-1] != '/':
        path +='/'
    fid = open(path +filename, 'r')
    
    buf = []
    for line in fid:
        line = line.replace('  ',' ').replace('  ',' ')
        nums= [ float(s) for s in line.split(' ') if s]
        buf.append(nums)
    fid.close()
    
    data = np.array(buf) # flux data set from file
    lon = np.linspace(0,360, data.shape[0]+1)
    lat = np.linspace(-90,90, data.shape[1]+1)
    return lon, lat, data.T
# END OF FUNCTION

'''
load data file from Cassini/INCA yearly data as processed by JG.
'''
def load_incafile_yearly(filename, path=None):
    if not path:
        path = load_json()['path_inca']
    if path[-1] != '/':
        path +='/'
    fid = open(path +filename, 'r')
    
    buf = []
    for line in fid:
        if line.startswith('#'):
            buf =[]
            continue
        line = line.replace('\n','')
        nums= [ float(s) for s in line.split(',') if s]
        buf.append(nums)
    fid.close()
    
    data = np.array(buf) # flux data set from file
    lon = np.linspace(0,360, data.shape[1]+1)
    lat = np.linspace(-90,90, data.shape[0]+1)
    return lon, lat, data
# END OF FUNCTION

'''
load IBEX data file postprocessed and exported by jsokol routine.
'''
def load_sokolfile(filename, path='Y:/Sokol/IBEX-Lo/Ribbon/figs/SC-RamOnly/flux/files/EclReg/'):
    
    if path[-1] != '/':
        path +='/'
    #filename= 'lvset_h_noSP_ram_2018_ESA7_ecl.txt'
    fid = open(path +filename, 'r')
    buf =[]
    for line in fid:
        nums = line.replace('\n','').split('\t') # list
        buf.append( [float(n) for n in nums])
    fid.close()
    
    data= np.array(buf)
    ll= np.where( data[0:,1] != data[0,1] )[0][0]
    lon= data[ 0:ll, 0]
    lat= data[ 0::ll, 1]
    flux= np.reshape( data[0:, 2], [lat.size, lon.size])
    
    # translate lon,lat from points to bins
    lon0= (3*lon[0] -lon[1])/2
    lonF= (3*lon[-1] -lon[-2])/2
    lon_bins= (lon[1:] +lon[0:-1])/2
    lon = np.append( np.append(np.array(lon0), lon_bins), lonF)
    
    lat0= (3*lat[0] -lat[1])/2
    latF= (3*lat[-1] -lat[-2])/2
    lat_bins= (lat[1:] +lat[0:-1])/2
    lat = np.append( np.append(np.array(lat0), lat_bins), latF)
    
    return lon, lat, flux
# END OF FUNCTION
