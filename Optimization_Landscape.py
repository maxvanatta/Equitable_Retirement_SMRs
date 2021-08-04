# -*- coding: utf-8 -*-
"""
This file runs a series of scenarios (objective values) and outputs:
    a text file for each optimization
    a map for each optimization
    a csv for the coalData, reData, MAXCAP for each objective
    
    For the entire landscape of optimizations, it outputs a csv for each step of refining.
    
"""

# All imports
import pandas as pd
import numpy as np
from getREefs import batchReEFs
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

# Create file name using states we pulled data for OR use data points for all states in mid-atlantic, all mid-Atlantic states eligible for RE sites in files below
solFileName = 'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'
winFileName = 'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'


def PrepareModel(numYears,region,threshDist,SMR_bool):
    plants = CoalPlants.getCoalPlants(region)
    plants['HISTGEN'] = CoalPlants.getPlantGeneration(plants['Plant Code'])
    plants['HD'] = CoalPlants.getMarginalHealthCosts(plants['Plant Code'])
    plants.dropna(inplace=True)
    coalData = pd.read_excel('3_1_Generator_Y2019.xlsx',header=1,index_col='Plant Code',sheet_name='Operable',usecols='B:F')
    coalPlants = plants.merge(coalData, left_on='Plant Code', right_index=True)
    coalPlants = coalPlants.drop_duplicates()
    print(coalPlants)
    
    reSites = RenewableSites.getAnnualCF(solFileName,winFileName)


    if SMR_bool == True:
        for index,row in coalPlants.iterrows():
            df = {'Technology':'smr','Latitude':row['Latitude'],'Longitude':row['Longitude'],'Annual CF': 0.70}
            reSites = reSites.append(df, ignore_index = True)
    
    reSitesL = list(reSites['Latitude'].astype(str)+','+reSites['Longitude'].astype(str)+','+reSites['Technology'].astype(str))

    # Get construction EFs and RE O&M EFs for sites in csv files from cell above
    # CONEF,REOMEF = batchReEFs(solFileName,winFileName,numYears)
    # np.savetxt('CONEF_'+str(numYears)+'.csv', CONEF, delimiter=',')
    # np.savetxt('REOMEF_'+str(numYears)+'.csv', REOMEF, delimiter=',')
    
    # OR load the information from csv files saved from prior runs for above regions/numYears to save time.
    CONEF = np.loadtxt('CONEF_'+str(numYears)+'.csv', delimiter=',')
    REOMEF = np.loadtxt('REOMEF_'+str(numYears)+'.csv',  delimiter=',')
    
    if SMR_bool == True:
        for index,row in coalPlants.iterrows():
            CONEF = np.append(CONEF,[1.67]*numYears)
            REOMEF = np.append(REOMEF,[0.42]*numYears)
    
    MAXCAP = np.zeros((len(reSites),len(coalPlants)))
    SITEMAXCAP = np.zeros(len(reSites))
    
    
    reSites['Eligible'] = 0
    
    # for each coal plant, use its lat lon to calculate distance between RE sites and the plant. if distance is more than X then make capacity 0
    
    for c in range(MAXCAP.shape[1]):
        coalCord = (coalPlants.iloc[c,1],coalPlants.iloc[c,2])
        for s in range(MAXCAP.shape[0]):
            reCord = (reSites.iloc[s,0],reSites.iloc[s,1])
            dist = haversine(coalCord,reCord, unit=Unit.MILES)
            # if distance > threshold then set MAXCAP = 0. Else MAXCAP is 1000
            if dist<threshDist:
                MAXCAP[s,c] = 1000
                reSites.iloc[s,-1] = 1
    
    ind = reSites['Latitude'].astype(str)+reSites['Longitude'].astype(str)
    mCapDF = pd.DataFrame(MAXCAP,index=ind,columns=list(coalPlants['Plant Name']))
    
    mCapDF['S'] = mCapDF[list(mCapDF.columns)].sum(axis=1)
    
    
    count = 0
    for s in mCapDF['S']:
        if s == 0:
            SITEMAXCAP[count]=0
        else:
            SITEMAXCAP[count]=1000
        count += 1
    
        
    SITEMINCAP = []
    for x in SITEMAXCAP:
        if x > 0:
            SITEMINCAP.append(10.)
        else:
            SITEMINCAP.append(0.)
    SITEMINCAP  = np.array(SITEMINCAP)
            
    reSites = reSites.reset_index(drop=True)
    
    return CONEF, REOMEF, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants
    
    
# Function Definitions for the adaptive optimization

def MultiLevelABG(PDF, SeriesToInclude = ['Weighted Objective','Unweighted Objective','A','B','G','Renewables','First Year Coal Retire']):
    arrays = [PDF['a'].tolist(),PDF['b'].tolist(),PDF['g'].tolist()]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["a","b","g"])
    adv_PD = pd.DataFrame()
    for i in SeriesToInclude:
        adv_PD[i] = PDF[i]
    adv_PD.index = index
    print(adv_PD.shape)
    return adv_PD

def StepDown(pdf,CONEF, REOMEF, numYears ,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,mCapDF,threshDist,coalPlants,region, SMR_bool, PartNumber = 2, criteria_Series = 'Unweighted Objective'):
    ind_vals = pdf.index.values.tolist()
    
    a_vals = []
    g_vals = []
    b_vals = []
    
    for i in ind_vals:
        a_vals.append(i[0])
        b_vals.append(i[1])
        g_vals.append(i[2])
    
    a_diff_min = MinDiff(a_vals)
    b_diff_min = MinDiff(b_vals)
    g_diff_min = MinDiff(g_vals)
    
    a_max,a_min = max(a_vals), min(a_vals)
    b_max,b_min = max(b_vals), min(b_vals)
    g_max,g_min = max(g_vals), min(g_vals)
    
    a_test = a_min
    b_test = b_min
    g_test = g_min
    
    new_objectives = []
    
    while a_test < a_max:
        while b_test < b_max:
            while g_test < g_max:
                p1 = (a_test, b_test, g_test)
                p2 = (a_test+a_diff_min, b_test, g_test)
                p3 = (a_test, b_test, g_test+g_diff_min)
                p4 = (a_test+a_diff_min, b_test, g_test+g_diff_min)
                p5 = (a_test, b_test+b_diff_min ,g_test)
                p6 = (a_test+a_diff_min,b_test+b_diff_min, g_test)
                p7 = (a_test,b_test+b_diff_min,g_test+g_diff_min)
                p8 = (a_test+a_diff_min,b_test+b_diff_min,g_test+g_diff_min)
                if (p1 in ind_vals) and (p2 in ind_vals) and (p3 in ind_vals) and (p4 in ind_vals) and (p5 in ind_vals) and (p6 in ind_vals) and (p7 in ind_vals) and (p8 in ind_vals):
                    
                    if float(pdf.loc[p1,criteria_Series]) == float(pdf.loc[p2,criteria_Series]) == float(pdf.loc[p3,criteria_Series]) == float(pdf.loc[p4,criteria_Series]) == float(pdf.loc[p5,criteria_Series]) == float(pdf.loc[p6,criteria_Series]) == float(pdf.loc[p7,criteria_Series]) == float(pdf.loc[p8,criteria_Series]):
                        pass
                    else:
                        a_val_new = np.arange(p1[0],p2[0]+(abs(p1[0]-p8[0])/PartNumber),(abs(p1[0]-p8[0])/PartNumber))
                        b_val_new = np.arange(p1[1],p8[1]+(abs(p8[1]-p1[1])/PartNumber),(abs(p8[1]-p1[1])/PartNumber))
                        g_val_new = np.arange(p1[2],p8[2]+(abs(p8[2]-p1[2])/PartNumber),(abs(p8[2]-p1[2])/PartNumber))
                        for i in a_val_new:
                            for j in b_val_new:
                                for z in g_val_new:
                                    if (i,j,z) not in ind_vals:
                                        ind_vals.append((i,j,z))
                                        new_objectives.append((i,j,z))
                                    else:
                                        pass
                                        
                g_test +=g_diff_min
            b_test += b_diff_min
            g_test = g_min
        b_test = b_min
        a_test += a_diff_min
    
    #Evaluates the objective functions
    temp_pd = pd.DataFrame()
    print(len(new_objectives))
    n = 1
    for obj_vals in new_objectives:
        print(n,'of',len(new_objectives))
        n+=1
        i,j,z = obj_vals[0],obj_vals[1], obj_vals[2]
        obj, plants2, model = test_cplex(i,j,z,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool)
        Results_df = SummarizeResults(obj, plants2, model, [i,j,z], region, threshDist,SMR_bool, reSites, numYears)
        temp_pd = temp_pd.append(Results_df,ignore_index= True)
        PostProcess(obj,numYears,region,coalPlants,reSites,[i,j,z], SMR_bool)
    new_pdf_multi = MultiLevelABG(temp_pd)
    return pd.concat([pdf,new_pdf_multi]).sort_index()

def MinDiff(vals):
    diff = 1000000
    for i in vals:
        for j in vals:
            if i-j != 0:
                if abs(i-j) < diff:
                    diff = abs(i-j)
    return diff

def InitialValues(A_MIN =0, A_MAX=1, B_MIN=0, B_MAX=1, G_MIN=0, G_MAX=1, a_steps=4, b_steps=4, g_steps=4):
    output_list = []
    a_diff = (A_MAX-A_MIN)/a_steps
    b_diff = (B_MAX-B_MIN)/b_steps
    g_diff = (G_MAX-G_MIN)/g_steps
    
    a_tests = np.arange(A_MIN,A_MAX+a_diff,a_diff)
    b_tests = np.arange(B_MIN,B_MAX+b_diff,b_diff)
    g_tests = np.arange(G_MIN,G_MAX+g_diff,g_diff)
    
    for a in a_tests:
        for b in b_tests:
            for g in g_tests:
                output_list.append([a,b,g])
    return output_list

def SummarizeResults(obj, plants, model, scenario, region, threshDist,SMR_bool, reSites, numYears):

    FileWrite = open('Objective_Record_'+str(region)+'_'+str(scenario[0])+'_'+str(scenario[1])+'_'+str(scenario[2])+'_'+str(threshDist)+'_'+str(SMR_bool)+'.txt','a+')

    print('System cost component:')
    FileWrite.write('System cost component:')

    # Validate System Costs
    SMR_num = plants.index.size 
    RE_num = len(reSites) - SMR_num
    Coal_first_bool = False
    Ren_Bool = False
    aC = 0
    bC = 0
    dC = 0
    for y in range(numYears):
        for c in range(len(plants)):
            aC += model.Params.COALFOPEX[c]*plants['Coal Capacity (MW)'].values[c]*obj.coalOnline[c,y] *(1.05**(y-1))
            bC += model.Params.COALVOPEX[c]*obj.coalGen[c,y]*(1.05**(y-1))
            if y ==1:
                if bC == 0:
                    Coal_first_bool = True
            for r in range(len(reSites)):
                dC += model.Params.REFOPEX[r]*obj.reCap[r,c,y]+model.Params.RECAPEX[r]*obj.capInvest[r,c,y] + model.Params.REVOPEX[r]*obj.reGen[r,c,y] *(1.05**(y-1))
            if dC>0:
                Ren_Bool = True
    print('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX+REVOPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))
    FileWrite.write('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))


    # Health damage component
    print('\nHealth damage component:')
    FileWrite.write('\nHealth damage component:')

    hd = 0
    for y in range(numYears):
        for c in range(len(plants)):
            hd += plants['HD'].values[c]*obj.coalOnline[c,y] *(1.15**(y-1))
    print('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))
    FileWrite.write('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))

    # Jobs component
    FileWrite.write('\nJobs component')
    print('\nJobs component')
    sumCoalEF = 0
    sumREEF = 0
    for y in range(numYears):
        a = 0
        for c in range(len(plants)):
            a += model.Params.RETEF[c]*obj.capRetire[c,y]+model.Params.COALOMEF[c]*obj.coalGen[c,y]
        sumCoalEF += a
        print('\tYear {} RETEF + COALOMEF = {}.'.format(y,round(a)))
        FileWrite.write('\n\tYear {} RETEF + COALOMEF = {}.'.format(y,round(a)))

    for y in range(numYears):
        b = 0
        for c in range(len(plants)):
            for r in range(len(reSites)):
                b += model.Params.CONEF[r,y]*obj.capInvest[r,c,y]+model.Params.REOMEF[r,y]*obj.reGen[r,c,y]
        print('\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        FileWrite.write('\n\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        sumREEF += b
    print('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))
    FileWrite.write('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))

    objS = (aC+bC+dC)*scenario[0]+hd*scenario[1]-(sumREEF+sumCoalEF)*scenario[2]
    print('\nSum of objective components = {}'.format(round(objS)))
    FileWrite.write('\nSum of objective components = {}'.format(round(objS)))
    FileWrite.close()
    
    
    
    #'Weighted Objective','Unweighted Objective','A','B','G'
    df = {'a':scenario[0],'b':scenario[1],'g':scenario[2],'Weighted Objective':(aC+bC+dC)*scenario[0]+hd*scenario[1]-(sumREEF+sumCoalEF)*scenario[2],'Unweighted Objective':(aC+bC+dC+hd-(sumREEF+sumCoalEF)),'A':round(aC+bC+dC,2),'B':hd,'G':(sumREEF+sumCoalEF),'Renewables':Ren_Bool,'First Year Coal Retire':Coal_first_bool}
    return df

def PostProcess(obj,numYears,region,coalPlants,reSites,scenario, SMR_bool):
    cLat = []
    cLon = []
    pNam = []
    coalRetire = []
    coalOnline = []
    capRetire = []
    coalGen = []
    coalYr = []

    reOnline = []
    reInvest = []
    cpInvest = []
    totReCap = []
    renGenrn = []
    yr = []
    cPlant = []
    Lat = []
    Lon = []
    Typ = []
    CF = []
    elg = []

    # RE investment Lat/Lon/Type
    for y in range(numYears):
        cYr = y+2020

        for c in range(coalPlants.shape[0]):
            cLat.append(coalPlants.iloc[c,1])
            cLon.append(coalPlants.iloc[c,2])
            pNam.append(coalPlants.iloc[c,7])
            coalRetire.append(obj.coalRetire[c,y])
            coalOnline.append(obj.coalOnline[c,y])
            capRetire.append(obj.capRetire[c,y])
            coalGen.append(obj.coalGen[c,y])
            coalYr.append(cYr)

            for s in range(reSites.shape[0]):
                # If reOnline flag is set for site s for plant c and year y then add flags
                if obj.reOnline[s,c,y]==1:
                    reOnline.append(1)
                else:
                    reOnline.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reInvest[s,c,y]==1:
                    reInvest.append(1)
                else:
                    reInvest.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.capInvest[s,c,y]>0:
                    cpInvest.append(obj.capInvest[s,c,y])
                else:
                    cpInvest.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reCap[s,c,y]>0:
                    totReCap.append(obj.reCap[s,c,y])
                else:
                    totReCap.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reGen[s,c,y]>0:
                    renGenrn.append(obj.reGen[s,c,y])
                else:
                    renGenrn.append(0)
                yr.append(cYr)
                cPlant.append(coalPlants.iloc[c,7])
                Lat.append(reSites.iloc[s,0])
                Lon.append(reSites.iloc[s,1])
                Typ.append(reSites.iloc[s,3])
                CF.append(reSites.iloc[s,2])
                elg.append(reSites.iloc[s,-1])

    # Create coal data CSV file.
    dat = {'Year':coalYr,'Lat':cLat,'Lon':cLon,'coalOnline':coalOnline,'coalGen':coalGen,'coalRetire':coalRetire,\
           'capRetire':capRetire}
    coalData = pd.DataFrame(dat)
    coalData.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_coalData.csv')

    dat = {'Year':yr,'Lat':Lat,'Lon':Lon,'Type':Typ,'Ann.CF':CF,'EligibleSite':elg,'Online':reOnline,'Investment':reInvest,\
           'Invested MW':cpInvest,'Total MW Cap.':totReCap,'Tot MWh Gen':renGenrn,'Repl. Plant':cPlant}
    reData = pd.DataFrame(dat)
    reData.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_reData.csv')
    
    # centered on Onion Maiden restaurant in Pittsburgh PA
    m = folium.Map(
        location=[40.42185334811013, -79.99594457857727],
        tiles="Cartodb positron",
        zoom_start=4
    )
    def detCol(arg):
        if arg=='s':
            return 'orange'
        elif arg=='w':
            return 'blue'
        else:
            return 'green'

    # container for coal plant locations.
    coalFG = folium.FeatureGroup(name='Coal plant locations')
    df = coalPlants
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', '+str(df.iloc[c,7])+', '+str(df.iloc[c,3])+' MW, HD '+str(round(df.iloc[c,5],2))
        folium.Circle(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            radius=3.0,
            color='red'
        ).add_to(coalFG)
    coalFG.add_to(m)

    # show all eligible coal plant locations.
    elSitesFG = folium.FeatureGroup(name='Eligible RE locations', show=False)
    df = reData.loc[(reData['EligibleSite']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', '+str(df.iloc[c,7])
        folium.CircleMarker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            weight=0.5,
            color='grey'
        ).add_to(elSitesFG)
    elSitesFG.add_to(m)

    # locate where RE Investments happened in year 2020
    reInvestFG = folium.FeatureGroup(name='Sites w/ Investments',show=False)
    marker_cluster = MarkerCluster().add_to(reInvestFG)
    df = reData.loc[(reData['Investment']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.CircleMarker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            color=detCol(df.iloc[c,3]),
            radius=4,
        ).add_to(marker_cluster)
    reInvestFG.add_to(m)

    # locate online RE plants in year 2020
    onlineFG = folium.FeatureGroup(name='Sites online 2020',show=False)
    marker_cluster = MarkerCluster().add_to(onlineFG)
    df = reData.loc[(reData['Online']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.Circle(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            color='green',
            radius=1.5,
        ).add_to(marker_cluster)
    onlineFG.add_to(m)

    # locate where RE Investments happened in year 2020
    validInvFG = folium.FeatureGroup(name='0+ MW RE capacity',show=False)
    marker_cluster = MarkerCluster().add_to(validInvFG)
    df = reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['Invested MW']!=0)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.Marker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            icon=folium.Icon(color=detCol(df.iloc[c,3]))
        ).add_to(marker_cluster)
    validInvFG.add_to(m)

    def detRad(arg):
        if arg<50:
            return 5
        elif arg<5000:
            return 10
        elif arg<1200000:
            return 15
        else:
            return 20

    # locate where RE Investments happened in year 2020
    reGenFG = folium.FeatureGroup(name='RE Gen Magnitude',show=False)
    df['Loc']=df['Lat'].astype(str)+df['Lon'].astype(str)
    totGen = []
    totMW = []
    types = []
    Lat = []
    Lon = []
    for l in df['Loc'].unique():
        tDF = df.loc[df['Loc']==l]
        types.append('_'.join(list(set(tDF.Type.values))))
        totGen.append(tDF['Tot MWh Gen'].sum())
        totMW.append(tDF['Total MW Cap.'].sum())
        Lat.append(tDF.iloc[0,1])
        Lon.append(tDF.iloc[0,2])
    d = {'Lt':Lat,'Ln':Lon,'totGen':totGen,'totMW':totMW,'types':types}
    df1 = pd.DataFrame(d)
    for c in range(df1.shape[0]):
        popText = str(df1.iloc[c,0])+str(df1.iloc[c,1])+', Type:'+str(df1.iloc[c,4])\
        +', '+str(df1.iloc[c,2])+' MWh'
        folium.CircleMarker(
            location=[df1.iloc[c,0],df1.iloc[c,1]],
            tooltip=popText,
            popup=popText,
            radius = detRad(df1.iloc[c,2]),
        ).add_to(reGenFG)
    reGenFG.add_to(m)
    folium.LayerControl().add_to(m)
    m.save('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_map.html')
    print('Generated a map with all information as well.')
    
def Constraints(obj,plants, numYears, reSites,coalPlants,MAXCAP,SITEMAXCAP):
    # Constraint 1 validation
    for y in range(numYears):
        for c in range(len(plants)):
            if obj.coalGen[c,y] == plants.HISTGEN.values[c]*obj.coalOnline[c,y]:
                pass
            else:
                print('\tConstraint 1 failed')
    print('\n')
    # Constraint 2 validation
    for y in range(numYears):
        for c in range(len(plants)):
            genSum = 0
            for s in range(len(reSites)):
                genSum += obj.reGen[s,c,y]
            if genSum == plants.HISTGEN.values[c]-obj.coalGen[c,y]:
                pass
            else:
                print('\tConstraint 2 failed for year {}, plant {}'.format(y,coalPlants['Plant Name'].values[c]))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            for s in range(len(reSites)):
                if obj.reGen[s,c,y]<=reSites['Annual CF'].values[s]*obj.reCap[s,c,y]*8760:
                    pass
                else:
                    print('\tConstraint 3 failed for year {}, plant {} ({}), site {}'.format(y,coalPlants['Plant Name'].values[c],c,s))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            for s in range(len(reSites)):
                if obj.reCap[s,c,y]<=MAXCAP[s,c]*obj.reOnline[s,c,y]:
                    pass
                else:
                    print('\tConstraint 4 failed for year {}, plant {} ({}), site {}'.format(y,coalPlants['Plant Name'].values[c],c,s))

    print('\n')
    for y in range(numYears):
        for s in range(len(reSites)):
            sumCap = 0
            for c in range(len(plants)):
                # Get the sum of RE capacity at a site for all coal plants
                sumCap += obj.reCap[s,c,y]
            if sumCap <= SITEMAXCAP[s]:
                pass
            else:
                print('\tConstraint 5 failed for year {}, at site {}'.format(y,s))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            for s in range(len(reSites)):
                if y != 0:
                    if obj.capInvest[s,c,y]==obj.reCap[s,c,y]-obj.reCap[s,c,y-1]:
                        pass
                    else:
                        print('\tConstraint 6 failed for year {}, plant {} ({}), site {} by {} MW'.format(y,coalPlants['Plant Name'].values[c],c,s,obj.reCap[s,c,y]-obj.reCap[s,c,y-1]))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            for s in range(len(reSites)):
                if obj.capInvest[s,c,y]<=MAXCAP[s,c]*obj.reInvest[s,c,y]:
                    pass
                else:
                    print('\tConstraint 7 failed for year {}, plant {} ({}), site {}'.format(y,coalPlants['Plant Name'].values[c],c,s))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            if obj.capRetire[c,y]==plants['Coal Capacity (MW)'].values[c]*obj.coalRetire[c,y]:
                pass
            else:
                print('\tConstraint 8 failed for year {}, plant {} ({})'.format(y,coalPlants['Plant Name'].values[c],c))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            for s in range(len(reSites)):
                if y!=0:
                    if obj.reInvest[s,c,y]==obj.reOnline[s,c,y]-obj.reOnline[s,c,y-1]:
                        pass
                    else:
                        print('\tConstraint 9 failed for year {}, plant {} ({}), site {}'.format(y,coalPlants['Plant Name'].values[c],c,s))
    # site limits
    limits = {'MAXSITES' : np.ones(len(plants))*10}
    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            indSum = 0
            for s in range(len(reSites)):
                indSum += obj.reInvest[s,c,y]
            if indSum <= limits['MAXSITES'][c]*obj.coalRetire[c,y]:
                pass
            else:
                print('\tConstraint 10 failed for year {}, plant {} ({})'.format(y,coalPlants['Plant Name'].values[c],c))

    print('\n')
    for y in range(numYears):
        for c in range(len(plants)):
            if y!=0:
                if obj.coalRetire[c,y]==obj.coalOnline[c,y-1]-obj.coalOnline[c,y]:
                    pass
                else:
                    print('\tConstraint 11 failed for year {}, plant {} ({})'.format(y,coalPlants['Plant Name'].values[c],c))
                    
                    
                    
def Initial3DSet(scenarios,numYears,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,mCapDF,threshDist,coalPlants):
    temp_pd = pd.DataFrame()
    for scenario in scenarios:
        mCapDF.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_MAXCAP.csv')
    
        a = scenario[0]
        b = scenario[1]
        g = scenario[2]
    
        obj, plants, model = test_cplex(a,b,g,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool)
        
        
        Results_df = SummarizeResults(obj, plants, model, scenario, region, threshDist,SMR_bool, reSites, numYears)
        temp_pd = temp_pd.append(Results_df,ignore_index= True)
        PostProcess(obj,numYears,region,coalPlants,reSites,scenario, SMR_bool)
        
        print(Results_df)
        
    new_pdf_multi = MultiLevelABG(temp_pd)
    return new_pdf_multi