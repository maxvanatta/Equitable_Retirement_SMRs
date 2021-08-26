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
region = ['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ']  #['VA']#['NJ']#['DE']# 

# Threshold distance within which all RE investments must be located
threshDist = 100

SMR_bool = True

DiscRate = 0.05

#scen = [0,0,1]

scenarios = OL.InitialValues(a_steps = 1,b_steps = 1,g_steps = 1)

CONEF, REOMEF, EFType, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants,folderName = OL.PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate,getNewEFs = False)

Results = OL.Initial3DSet(scenarios,numYears,region,CONEF,REOMEF,EFType,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,mCapDF,threshDist,coalPlants,folderName,DiscRate)

Results.to_csv('Initial_Set_SMR.csv')

Steps = 0
n = 1
while n <= Steps:
    Results = OL.StepDown(Results,CONEF,REOMEF,EFType, numYears,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,mCapDF,threshDist,coalPlants,region,SMR_bool,folderName,DiscRate, PartNumber = 2, criteria_Series = 'Unweighted Objective', criteria_tolerance = 0.20)
    Results.to_csv('Summary_'+str(region)+'_'+str(SMR_bool)+'_Level_'+str(n)+'.csv')
    n +=1

