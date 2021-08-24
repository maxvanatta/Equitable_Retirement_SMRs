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
region = ['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ']  #['NJ']#['DE']# 

# Threshold distance within which all RE investments must be located
threshDist = 100

SMR_bool = False

scen = [0,0,1]

CONEF, REOMEF, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants,folderName = OL.PrepareModel(numYears,region,threshDist,SMR_bool,getNewEFs = False)

obj, model, df = OL.SingleModel(scen,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,coalPlants,threshDist,folderName)

