# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:07:08 2021

@author: bhavrathod
"""
import numpy as np
import pandas as pd 
import os, sys 

MODULE_PATH     = os.path.dirname(__file__)
# List of all plants (coal plants only?).
EIA_PLANT_FILE = os.path.join(MODULE_PATH,'data/eia8602019/2___Plant_Y2019.xlsx')
# List of all individual generators in powerplants.
EIA_GENERATOR_FILE  = os.path.join(MODULE_PATH,'data/eia8602019/3_1_Generator_Y2019.xlsx')
# Emissions from plants.
EIA_ENVIRO_FILE = os.path.join(MODULE_PATH,'data/eia8602019/6_2_EnviroEquip_Y2019.xlsx')
EGRID_FOLDER = os.path.join(MODULE_PATH,'data/egrid')
# not sure what this does yet.
EASIUR_FILE = os.path.join(MODULE_PATH,'data/easiur/msc_per_ton_by_plant.csv')
EIA_923_FILE = os.path.join(MODULE_PATH,'data/eia923/EIA923GenFuel.csv')

DEFAULT_STACK_HEIGHT = 150 # none, 0, 150, or 300
INFLATION_RATE = 1.2 # 2010 USD to 2020 USD

COAL_PFT = ['BIT','RC','SC','SGC','LIG','SUB','WC']

def _getStackHeight(plantCodes):

    plantCodes = plantCodes.values if isinstance(plantCodes,pd.Series) else plantCodes

    # get stack height (m)
    stackHeight = pd.Series(data=np.ones(len(plantCodes))*DEFAULT_STACK_HEIGHT,
                            index=plantCodes)

    plants = pd.read_excel(EIA_ENVIRO_FILE,skiprows=1,sheet_name='Stack Flue',usecols=['Plant Code','Stack Height (Feet)'])

    # convert to meters
    plants['Stack Height (Feet)'].where(plants['Stack Height (Feet)'].astype(str) != ' ', -1, inplace=True)
    plants['Stack Height'] = plants['Stack Height (Feet)'].astype(float)*.3048
    plants.drop(columns=['Stack Height (Feet)'],inplace=True)
    plants['Stack Height'].where(plants['Stack Height'] >= 0, DEFAULT_STACK_HEIGHT, inplace=True)

    # fill missing
    plants = plants.groupby(['Plant Code']).mean()
    stackHeight.loc[plants.index.intersection(plantCodes)] = plants.loc[plants.index.intersection(plantCodes),'Stack Height']

    return stackHeight

def _getEmissions(plantCodes):

    plantCodes = plantCodes.values if isinstance(plantCodes,pd.Series) else plantCodes

    # read data
    emissionsFilename = os.path.join(EGRID_FOLDER,'egrid2019_data.xlsx')
    plants = pd.read_excel(  emissionsFilename,
                                sheet_name='UNT19',
                                skiprows=  1, # different file formatting for 2012
                                usecols=['ORISPL','FUELU1','NOXAN','SO2AN'])

    # filter
    plants.dropna(inplace=True)
    plants = plants[plants['FUELU1'].isin(COAL_PFT)]

    # conversions tons -> metric tonnes
    plants['NOXAN'] *= .907
    plants['SO2AN'] *= .907

    # aggregate and filter
    plants = plants.groupby(['ORISPL']).sum()
    plants = plants.loc[plants.index.intersection(plantCodes)]
    
    return pd.Series(data=plants['SO2AN'].values,index=plants.index.values), pd.Series(data=plants['NOXAN'].values,index=plants.index.values)

def _getEasiur(plantCodes, stackHeight, season):
    
    assert(season == 'Annual' or season == 'Spring' or season == 'Summer' or season == 'Fall' or season == 'Winter')
    
    stackHeight = stackHeight.values if isinstance(stackHeight,pd.Series) else stackHeight


    # Filter
    df = pd.read_csv(EASIUR_FILE,usecols=['Plant Code','SO2 {} Ground'.format(season),'SO2 {} 150m'.format(season),
                                                'SO2 {} 300m'.format(season),'NOX {} Ground'.format(season),
                                                'NOX {} 150m'.format(season),'NOX {} 300m'.format(season)])
    df.set_index('Plant Code',inplace=True)

    margCostPerTonSO2 = np.zeros(len(plantCodes))
    margCostPerTonNOx = np.zeros(len(plantCodes))

    for i in range(len(plantCodes)):
        if not plantCodes[i] in df.index:
            margCostPerTonSO2[i] = np.nan 
            margCostPerTonNOx[i] = np.nan
        elif stackHeight[i] < 75:
            margCostPerTonSO2[i] = df.at[plantCodes[i],'SO2 {} Ground'.format(season)]
            margCostPerTonNOx[i] = df.at[plantCodes[i],'NOX {} Ground'.format(season)]
        elif stackHeight[i] < 225:
            margCostPerTonSO2[i] = df.at[plantCodes[i],'SO2 {} 150m'.format(season)]
            margCostPerTonNOx[i] = df.at[plantCodes[i],'NOX {} 150m'.format(season)]
        else: 
            margCostPerTonSO2[i] = df.at[plantCodes[i],'SO2 {} 300m'.format(season)]
            margCostPerTonNOx[i] = df.at[plantCodes[i],'NOX {} 300m'.format(season)]
        
    return margCostPerTonSO2 * INFLATION_RATE, margCostPerTonNOx * INFLATION_RATE

def getMarginalHealthCosts(plantCodes,season='Annual'):
    '''Get marginal health costs ($/MWh) for plants across the United States. All data processing should be done separately from this function call. This function provides an abstraction for accessing m.h.c. data from this module's underlying csv.
    
    Arguments:
    ----------
    `plantCodes` (ndarray or pd.Series) : Numpy array of integer plant codes
    `season` (str) : Season of underlying marginal health costs provided by EASIUR [`Annual`|`Spring`|`Summer`|`Fall`|`Winter`]
    `years` (int or list): OPTIONAL: Year(s) of generation/emissions data. Generation and emissions averaged over each year. Must be in {2010, 2012, 2014, 2016, 2018, 2019}.
    Returns:
    ----------
    `marginalHealthCosts` (series) : pandas series of health damages ($) per generation (MWh) at a certain plant code. Plants with incomplete data return 'na' cells. Series is indexed by plant code.
    '''
    assert(season in ['Annual','Spring','Summer','Fall','Winter'])

    if isinstance(plantCodes, pd.Series):
        plantCodes = plantCodes.values
    if isinstance(plantCodes,int):
        plantCodes = np.array([plantCodes])

    plants = pd.DataFrame(index=plantCodes)
    # plant data
    plants['stack height'] = _getStackHeight(plantCodes)
    # emissions and generation data [MWh], [m.ton], [m.ton]
    plants['generation'] = getPlantGeneration(plantCodes)
    plants['SO2'], plants['NOx'] = _getEmissions(plantCodes)
    # marginal emissions [m.ton] / [MWh] = [m.ton / MWh]
    plants['SO2 rate'] = plants['SO2']/plants['generation']
    plants['NOx rate'] = plants['NOx']/plants['generation']
    # marginal emissions costs [$ / m.ton]
    plants['SO2 cost'], plants['NOx cost'] = _getEasiur(plantCodes, plants['stack height'], 'Annual')
    # marginal health costs [$ / m.ton] * [m.ton / Mwh] = [$ / MWh]
    plants['marginal health cost'] = plants['SO2 cost']*plants['SO2 rate'] + plants['NOx cost']*plants['NOx rate']
            
    return pd.Series(data=plants['marginal health cost'].values, index=plantCodes)

def getCoalPlants(regions='ALL'):
    ''' getCoalPlants: Get the coal plant ORIS codes, latitudes, longitude, and coal capacity.
    Args:
    --------
    `regions` (str or list): all NERC regions, balancing authorities, or states to include e.g. 'PJM', 'WECC', 'NY
    `all_thermal` (bool): All thermal generators (not just coal).
    Return:
    --------
    `plants` (DataFrame): Dataframe of plant codes, locations, balancing authority, and NERC region; indexed by plant code.
    '''
    # handle single str region
    regions = [regions] if isinstance(regions,str) else regions
    
    # open file 
    plants = pd.read_excel(EIA_PLANT_FILE,skiprows=1,usecols=['Plant Code', 'State', 'Latitude', 'Longitude','NERC Region','Balancing Authority Code'])
    generators = pd.read_excel(EIA_GENERATOR_FILE,skiprows=1,usecols=["Plant Code","Technology","Status","Nameplate Capacity (MW)"])

    # filter plants
    plants = plants if 'ALL' in regions else plants[plants['NERC Region'].isin(regions) | plants['Balancing Authority Code'].isin(regions) |plants['State'].isin(regions)]

    # drop unnecessary
    plants.drop(columns=['NERC Region','Balancing Authority Code','State'],inplace=True)

    # filter generators
    generators = generators[generators['Status'] == 'OP']
    coalGenerators = generators[generators['Technology'].str.contains('Coal')]   

    # final filter; should include only plants with coal generators in the correct region
    plants = plants[plants['Plant Code'].isin(coalGenerators['Plant Code'])]

    # capacity
    plants['Coal Capacity (MW)'] = [np.sum(coalGenerators["Nameplate Capacity (MW)"][coalGenerators['Plant Code'] == pc].values) for pc in plants['Plant Code']]

    plants.set_index(plants['Plant Code'].values,inplace=True)
    
    #plants.to_csv('plantCoords.csv')

    return plants

def getPlantGeneration(plantCodes):
    ''' getCoalGeneration: Get 2019 plant generation (MWh). Return as a pandas dataframe indexed by plant code.
    Args:
    --------
    `plantCodes` (list, ndarray, or series): If Dataframe, must have column 'Plant Code'.
    `years` (int or list): OPTIONAL: Year(s) of data. If a list, annual generation is found as the mean of each year. Must be in {2010, 2012, 2014, 2016, 2018, 2019}.
    Return:
    --------
    `generation` (DataFrame): Dataframe of annual generation indexed by plant.
    '''

    if isinstance(plantCodes, pd.Series):
        plantCodes = plantCodes.values
    elif isinstance(plantCodes,list):
        plantCodes = np.array(plantCodes)

    # FROM DYLAN

    coalPlants = pd.read_csv(EIA_923_FILE,usecols=['Plant Id','AER Fuel Type Code','Net Generation (Megawatthours)','Total Fuel Consumption MMBtu'])
    coalPlants = coalPlants[coalPlants['AER Fuel Type Code'].isin(['COL','WOC'])]
    coalPlants = coalPlants[coalPlants['Total Fuel Consumption MMBtu'] !=0]
    coalPlants.drop(columns=['AER Fuel Type Code','Total Fuel Consumption MMBtu'],inplace=True)
    coalPlants = coalPlants.groupby(['Plant Id']).sum()
    coalPlants = coalPlants.loc[coalPlants.index.intersection(plantCodes)]
    coalPlants = coalPlants[coalPlants['Net Generation (Megawatthours)'] > 0]

    return pd.Series(data=coalPlants['Net Generation (Megawatthours)'],index=coalPlants.index.values)


# x = getCoalPlants(['NY'])
# print(x)