import numpy as np
import pandas as pd 
import os, sys


def getCoalEFs(coalPlantList):
    """returns the employment factor for each coal plant in panda dataframe
    At state level demension

    Args:
        coalPlantList ([list]): [includes the coal plants slated for retirement- should be from EIA 860]
    """ 
    '''
    #read in coal plant data EF_data\coalEFs.xlsx
    coalPlantLocs = pd.read_excel("EF_data/coalPlantLocs.xlsx")
    
    #reads in state level coal EF
    stateEFs = pd.read_excel("EF_data/coalEFs.xlsx")
    
    #creation of  coal state dict, will change into EFs after loop
    coalEFDataframe = dict()
    
    #run through tech list to make sure plant is coal and then find state plant is in
    for plantName in coalPlantList:
        
        #finds which row that coal plant is in the excel EIA dataset
        plantRowLocation = np.where(coalPlantLocs["Plant Code"] == plantName)[0]
        
        #if multiple generators are called select the first row
        if len(plantRowLocation) > 1:
            plantRowLocation = plantRowLocation[0]

        #gets state and assigns into a dict with key as plant name
        coalEFDataframe[plantName] = coalPlantLocs["State"][plantRowLocation]
    
    #final coal dict set up
    coalEFPandaDf = dict()
    #run through the EF list assigning EF values for each state
    for plantName in coalEFDataframe.keys():
        
        #finds which row that state is in the EF dataset
        plantRowLocation = np.where(stateEFs["State"] == coalEFDataframe[plantName])[0][0]

        #updating coalEFDataFrame to for it to carry the respective EF of that state
        coalEFPandaDf[plantName] = stateEFs["Coal EFs"][plantRowLocation]
    
    #transforming coal EF dict into panda dataframe 
    pandaCoalEF = pd.DataFrame.from_dict(coalEFPandaDf,orient='index',columns=['Plant EF'])
    
    return pandaCoalEF
    '''
    #simple EF factor: source in README, unable to implement state level jobs simply due to no resolution of job numbers at a state level basis
    return .14


def getCoalDecom():
    """returns the employment factor for decommissioning a coal plant- right now fixed as static value
    source: Job creation during the global energy transition towards 100% renewable power system by 2050 
    """
    return 1.65