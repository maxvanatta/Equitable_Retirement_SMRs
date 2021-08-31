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
import csv
import Optimization_Landscape as OL
import warnings
warnings.filterwarnings("ignore")

solFileName = 'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' # 'solar_cf_AL_AZ_AR_CA_CO_CT_DE_FL_GA_ID_IL_IN_IA_KS_KY_LA_ME_MD_MA_MI_MN_MS_MO_MT_NE_NV_NH_NJ_NM_NY_NC_ND_OH_OK_OR_PA_RI_SC_SD_TN_TX_UT_VT_VA_WA_WV_WI_WY_0.5_2014.csv'
#'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' # 
winFileName = 'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' #'wind_cf_AL_AZ_AR_CA_CO_CT_DE_FL_GA_ID_IL_IN_IA_KS_KY_LA_ME_MD_MA_MI_MN_MS_MO_MT_NE_NV_NH_NJ_NM_NY_NC_ND_OH_OK_OR_PA_RI_SC_SD_TN_TX_UT_VT_VA_WA_WV_WI_WY_0.5_2014.csv'
#'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv' #


# How many years will the analysis run for?
numYears = 7

# Region of coal plants under analysis
region = ['NY','PA','OH','WV','KY','TN','VA','MD','DE','NC','NJ']  #['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'] #['CO'] #['NJ']#['DE']
#
# Threshold distance within which all RE investments must be located
threshDist = 100

SMR_bool = True
SMR_CAPEX = 2526001 # $/MW
SMR_FOPEX = 25000 # $/MW
SMR_VOPEX = 9.46 # $/MWh
SMR_Values = [SMR_CAPEX, SMR_FOPEX, SMR_VOPEX]

DiscRate = 0.05

scen = [1,0,0]

CONEF, REOMEF,EFType, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants,folderName = OL.PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate,SMR_Values, solFileName, winFileName,getNewEFs = False)

obj, model, df = OL.SingleModel(scen,numYears,solFileName,winFileName,region,CONEF,REOMEF,EFType,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,coalPlants,threshDist,folderName,DiscRate, SMRs = SMR_Values)

os.chdir(folderName)
w = csv.writer(open(folderName+'_SingleRun.csv', 'w'))
Labels = []
Values = []
for key, val in df.items():
    Labels.append(key)
    Values.append(val)
w.writerow(Labels)
w.writerow(Values)

#cd Documents\UM\ASSET\EquitableRetirement_Project\model_729\model
#AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY