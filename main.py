# All imports
import pandas as pd
import numpy as np
from getReEFs import batchReEFs
from Cplex_main import test_cplex
import CoalPlants
import RenewableSites
from haversine import haversine, Unit
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import branca
import branca.colormap as cm
import os
import Optimization_Landscape as OL

import argparse


parser = argparse.ArgumentParser(description='Import the input variables for the model run')
parser.add_argument('--csvFile', type=str, help='Data year. Must be in 2007-2014 (inclusive).',required=True)
args = parser.parse_args()

import_CSV = pd.read_csv(args.csvFile)

if import_CSV['Value'][0] == 'MidA':
    solFileName = 'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' # 
    winFileName = 'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' #
    cont = False
else:
    solFileName = 'solar_cf_CONTINENTAL_0.5_2014.csv' # 
    winFileName = 'wind_cf_CONTINENTAL_0.5_2014.csv' #
    cont = True

# How many years will the analysis run for?
numYears = int(import_CSV['Value'][1])

# Region of coal plants under analysis
if import_CSV['Value'][2] == 'MidA':
    region = ['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ']
elif import_CSV['Value'][2] == 'Cont':
    region = 'ALL'
else:
    region = import_CSV['Value'][2].split('_') 


# Threshold distance within which all RE investments must be located
threshDist = int(import_CSV['Value'][3])

# Discount Rate which is applied in the objective calculation portion
DiscRate = float(import_CSV['Value'][7])

if import_CSV['Value'][4] == 'TRUE':
    SMR_bool = True
else: 
    SMR_bool = False
    
if import_CSV['Value'][5] == 'TRUE':
    SMRONLY = True
else: 
    SMRONLY = False

if import_CSV['Value'][6] == 'Low':
    SMR_Values = [728000, 7000, 9.46]
elif import_CSV['Value'][6] == 'Med': 
    SMR_Values = [3248000, 25000, 9.46]
elif import_CSV['Value'][6] == 'High': 
    SMR_Values = [10556000, 95000, 9.46]
else:
    SMR_Values = import_CSV['Value'][6].split('_')
    for v in SMR_Values:
        v = float(v)
        
scenarios = OL.InitialValues(a_steps = int(import_CSV['Value'][10]),b_steps = int(import_CSV['Value'][10]),g_steps = int(import_CSV['Value'][10]))

CONEF, REOMEF, EFType, MAXCAP,SITEMAXCAP,reSites,plants, mCapDF,coalPlants,folderName, RED_indexes,MASK, REV_INDS = OL.PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate,SMR_Values, solFileName, winFileName,getNewEFs = import_CSV['Value'][5],SMROnly = SMRONLY, Nation = cont)

if import_CSV['Value'][8] == 'TRUE':
    scen = []
    for s in import_CSV['Value'][9].split('_'):
        scen.append(float(s))
    obj, plants2, model = test_cplex(scen[0],scen[1],scen[2],numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,DiscRate, SMR_Values[0],SMR_Values[1],SMR_Values[2],RED_indexes,MASK,REV_INDS ,CO2Limits = 'Linear2030')

    NEW_RES = OL.SummarizeResults(obj, plants2, model, scen, region, threshDist,SMR_bool, reSites, numYears,folderName,DiscRate,EFType,SMR_Values, prints = True)

    OL.PostProcess(obj,model,numYears,region,coalPlants,reSites,scen, SMR_bool,folderName,NEW_RES)
else:
    for scen in scenarios:
        obj, plants2, model = test_cplex(scen[0],scen[1],scen[2],numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,DiscRate, SMR_Values[0],SMR_Values[1],SMR_Values[2],RED_indexes,MASK,REV_INDS ,CO2Limits = 'Linear2030')

        NEW_RES = OL.SummarizeResults(obj, plants2, model, scen, region, threshDist,SMR_bool, reSites, numYears,folderName,DiscRate,EFType,SMR_Values, prints = True)

        OL.PostProcess(obj,model,numYears,region,coalPlants,reSites,scen, SMR_bool,folderName,NEW_RES)