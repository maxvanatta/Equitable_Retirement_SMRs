# All imports
import pandas as pd
import numpy as np
from getReEFs import batchReEFs
from main import test_cplex
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

solFileName = 'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'
winFileName = 'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'

# How many years will the analysis run for?
numYears = 3

# Region of coal plants under analysis
region = ['NY']#['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']#['VA']#['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ']  #['DE']# 


# Threshold distance within which all RE investments must be located
threshDist = 100

SMR_bool = True
SMR_CAPEX = 10556000 # $/MW High: 10556000, Low: 728000, median: 3248000
SMR_FOPEX = 95000 # $/MW High: 95000, Low: 7000, Median: 25000
SMR_VOPEX = 8.71 # $/MWh NuScale: 9.46, An: 8.71
SMR_Values = [SMR_CAPEX, SMR_FOPEX, SMR_VOPEX]

DiscRate = 0.05

scenarios = OL.InitialValues(a_steps = 3,b_steps = 3,g_steps = 3)

CONEF, REOMEF, EFType, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants,folderName = OL.PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate,SMR_Values, solFileName, winFileName,getNewEFs = False)

Results = OL.Initial3DSet(scenarios,numYears,region,CONEF,REOMEF,EFType,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,mCapDF,threshDist,coalPlants,folderName,DiscRate, SMR_Values,solFileName, winFileName)

Results.to_csv('Summary_'+folderName+'_Level_0'+'.csv')

Steps = 2
n = 1
while n <= Steps:
    Results = OL.StepDown(Results,CONEF,REOMEF,EFType, numYears,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,mCapDF,threshDist,coalPlants,region,SMR_bool,folderName,solFileName, winFileName,DiscRate, PartNumber = 2, criteria_tolerance = 0.99, SMRs = SMR_Values)
    Results.to_csv('Summary_'+folderName+'_Level_'+str(n)+'.csv')
    n +=1