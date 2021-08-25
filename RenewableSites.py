# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:22:07 2021

@author: bhavrathod
"""
'''
ijbd
4/2/2021
This is a complementary script providing an interface for accessing the capacity factors from `powGen-wtk-nsrdb.py` output.
'''
import numpy as np
import pandas as pd
import os

def _getAnnualCF(filename, cf_only):
    re = pd.read_csv(filename,index_col=0)

    # get locations
    lats = [float(c.split()[0]) for c in re.columns]
    lons = [float(c.split()[1]) for c in re.columns]
    
    cf = pd.DataFrame()
    if not cf_only:
        cf['Latitude'] = lats
        cf['Longitude'] = lons
    cf['Annual CF'] = np.average(re.values.T,axis=1)

    return cf

def getAnnualCF(solar_filename, wind_filename, cf_only=False):
    '''
    Args
    -------
        `solar_filename` (str) : Absolute path to the solar capacity factor file.
        `wind_filename` (wind) : Absolute path to the wind capacity factor file.
    Returns
    -------
        `renewableSites` (pd.Series) : Series of lat/lons
    '''
    solar = _getAnnualCF(solar_filename, cf_only)
    if not cf_only:
        solar['Technology'] = 's'

    wind = _getAnnualCF(wind_filename, cf_only)
    if not cf_only:
        wind['Technology'] = 'w'
    return solar.append(wind)
    '''
    SMR = wind.copy(deep = True) #_getSMRCF() For later use of making the SMRs located at each decomissioned coal plant
    SMR['Technology'] = 'smr'
    SMR['Annual CF'] = 0.95
    # 
    
    RE1 = solar.append(wind)
    return RE1.append(SMR)
    '''
def _getHourlyCF(filename, cf_only):
    re = pd.read_csv(filename,index_col=0)

    # get locations
    lats = [float(c.split()[0]) for c in re.columns]
    lons = [float(c.split()[1]) for c in re.columns]
    
    cf = pd.DataFrame()
    if not cf_only:
        cf['Latitude'] = lats
        cf['Longitude'] = lons
    gen = re.values.T

    for i in range(gen.shape[1]):
        cf['Hr {}'.format(i)] = gen[:,i]

    return cf

def getHourlyCF(solar_filename, wind_filename, cf_only=False):
    '''
    Args
    -------
        `solar_filename` (str) : Absolute path to the solar capacity factor file.
        `wind_filename` (wind) : Absolute path to the wind capacity factor file.
    Returns
    -------
        `renewableSites` (pd.Series) : Series of lat/lons
    '''
    solar = _getHourlyCF(solar_filename,cf_only)
    if not cf_only:
        solar['Technology'] = 's'

    wind = _getHourlyCF(wind_filename,cf_only)
    if not cf_only:
        wind['Technology'] = 'w'

    return solar.append(wind)