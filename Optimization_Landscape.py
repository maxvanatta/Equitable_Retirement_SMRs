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



def PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate, SMRs, solFileName, winFileName, getNewEFs = False, SMROnly = False, Nation = False):
    plants = CoalPlants.getCoalPlants(region)
    plants['HISTGEN'] = CoalPlants.getPlantGeneration(plants['Plant Code'])
    plants['HD'] = CoalPlants.getMarginalHealthCosts(plants['Plant Code'])
    plants.dropna(inplace=True)
    coalData = pd.read_excel('3_1_Generator_Y2019.xlsx',header=1,index_col='Plant Code',sheet_name='Operable',usecols='B:F')
    coalPlants = plants.merge(coalData, left_on='Plant Code', right_index=True)
    coalPlants = coalPlants.drop_duplicates()
    #coalPlants = coalPlants.loc[coalPlants['Plant Code']==6002]
    print(coalPlants)
    
    if SMR_bool == True:
        folderName = ('_'.join(region)+'_'+str(numYears)+'years_'+str(threshDist)+'miles_'+str(DiscRate)+'_SMR_'+str(SMR_bool)+'_'+str(SMRs[0])+'_'+str(SMRs[1])+'_'+str(SMRs[2]))
    else:
        folderName = ('_'.join(region)+'_'+str(numYears)+'years_'+str(threshDist)+'miles_'+str(DiscRate)+'_SMR_'+str(SMR_bool))
    
    reSites = RenewableSites.getAnnualCF(solFileName,winFileName)
    
    if SMROnly == True:
        reSites['Annual CF'] = 0

    if SMR_bool == True:
        for index,row in coalPlants.iterrows():
            df = {'Technology':'smr','Latitude':row['Latitude'],'Longitude':row['Longitude'],'Annual CF': 0.90}
            reSites = reSites.append(df, ignore_index = True)
    
    reSitesL = list(reSites['Latitude'].astype(str)+','+reSites['Longitude'].astype(str)+','+reSites['Technology'].astype(str))

    
    if getNewEFs == True:
        
    # Get construction EFs and RE O&M EFs for sites in csv files from cell above
        CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears)
        np.savetxt('CONEF_'+str(numYears)+'.csv', CONEF, delimiter=',')
        np.savetxt('REOMEF_'+str(numYears)+'.csv', REOMEF, delimiter=',')
    
    # OR load the information from csv files saved from prior runs for above regions/numYears to save time.
    else:
        if Nation:
            res = pd.read_csv('reEFs_cont.csv')
        else:
            res = pd.read_csv('reEFs.csv')
        res = res.loc[res['Year'] < (2020+numYears)]
        CONEF = np.array(res['Con/Instl EF'])
        REOMEF = np.array(res['O&M EF'])
    
    EFType = []
    for i in res['Unnamed: 0']:
        EFType.append(i.split(',')[-1])
    
    if SMR_bool == True:
        for index,row in coalPlants.iterrows():
            CONEF = np.append(CONEF,[1.67]*numYears)
            REOMEF = np.append(REOMEF,[0.42]*numYears)
            EFType = np.append(EFType,['SMR']*numYears)
    
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

    listFiles = os.listdir()
    if folderName in listFiles:
        pass
    else:
        os.mkdir(folderName)
    
    print(folderName)
    
    return CONEF, REOMEF, EFType, MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP, mCapDF,coalPlants, folderName
    

def SingleModel(scen,numYears,solFileName,winFileName,region,CONEF,REOMEF,EFType,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,coalPlants,threshDist,folderName,DiscRate, SMRs):
    
    obj, plants2, model = test_cplex(scen[0],scen[1],scen[2],numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SITEMINCAP,SMR_bool,DiscRate, SMRs[0],SMRs[1],SMRs[2])
    SummarizeResults(obj, plants2, model, [scen[0],scen[1],scen[2]], region, threshDist,SMR_bool, reSites, numYears,folderName,DiscRate,EFType,SMRs, prints = True)
    PostProcess(obj,model,numYears,region,coalPlants,reSites,[scen[0],scen[1],scen[2]], SMR_bool,folderName)
    
    return obj, model

def InitialValues(A_MIN =0, A_MAX=1, B_MIN=0, B_MAX=1, G_MIN=0, G_MAX=1, a_steps=2, b_steps=2, g_steps=2):
    output_list = []
    testerLista = []
    testerListb = []
    testerListg = []
    OutOfOne = []
    a_diff = (A_MAX-A_MIN)/a_steps
    b_diff = (B_MAX-B_MIN)/b_steps
    g_diff = (G_MAX-G_MIN)/g_steps
    
    a_tests = np.arange(A_MIN,A_MAX+a_diff,a_diff)
    b_tests = np.arange(B_MIN,B_MAX+b_diff,b_diff)
    g_tests = np.arange(G_MIN,G_MAX+g_diff,g_diff)
    
    for a in a_tests:
        for b in b_tests:
            for g in g_tests:
                if (a + b+ g)==0:
                    pass
                else:
                    testerLista.append(a)
                    testerListb.append(b)
                    testerListg.append(g)
                    output_list.append([a,b,g])
                    OutOfOne.append('_'.join([str(a/(a+b+g)),str(b/(a+b+g)),str(g/(a+b+g))]))
    scen_pd = pd.DataFrame({'Scens':output_list,'OfOne':OutOfOne})
    scen_pd.drop_duplicates(subset = 'OfOne',inplace = True)
    return scen_pd['Scens'].to_list()

def SummarizeResults(obj, plants, model, scenario, region, threshDist,SMR_bool, reSites, numYears,folderName,DiscRate,EFType,SMRs, prints = False):
    os.chdir(folderName)
    
    InputFileWrite = open('Input_Values_'+'_'.join(region)+'_'+str(scenario[0])+'_'+str(scenario[1])+'_'+str(scenario[2])+'_'+str(threshDist)+'_'+str(SMR_bool)+'.txt','w')
    InputFileWrite.write('Region Modeled:\n')
    InputFileWrite.write(','.join(region)+'\n')
    InputFileWrite.write('Scenario Modeled:\n')
    print(scenario)
    InputFileWrite.write(str(scenario[0])+','+str(scenario[1])+','+str(scenario[2])+'\n')
    InputFileWrite.write('RE Site Radius (mi):\n')
    InputFileWrite.write(str(threshDist)+' mi\n')
    InputFileWrite.write('Number of Years:\n')
    InputFileWrite.write(str(numYears)+'\n')
    InputFileWrite.write('SMR usage?:\n')
    InputFileWrite.write(str(SMR_bool)+'\n')
    InputFileWrite.write('SMR Costs:\n')
    InputFileWrite.write('CAPEX ($/MW): '+str(SMRs[0])+' FOPEX ($/MW): '+str(SMRs[1])+' VOPEX ($/MWh): '+str(SMRs[2])+'\n')
    InputFileWrite.write('Discount Rate:\n')
    InputFileWrite.write(str(DiscRate)+'\n')
    InputFileWrite.close()
    
    
    FileWrite = open('Objective_Record_'+'_'.join(region)+'_'+str(scenario[0])+'_'+str(scenario[1])+'_'+str(scenario[2])+'_'+str(threshDist)+'_'+str(SMR_bool)+'.txt','w')

    if prints == True:
        print('System cost component:')
    FileWrite.write('System cost component:')
    
    
    CostCoalOM = []
    CostCoalRet = [] # do we need to find values for this aspect?
    CostRECons = []
    CostREOM = []
    
    HealthObj = []
    
    JobsCoalRet = []
    JobsCoalOM = []
    JobsREOM = []
    JobsRECONS = []
    
    # Validate System Costs
    SMR_num = plants.index.size 
    RE_num = len(reSites) - SMR_num
    Coal_first_bool = False
    Ren_Bool = False
    aC = 0
    bC = 0
    dC = 0
    
    
    for y in range(numYears):
        RECons = 0
        REOM = 0
        a2 = 0
        b2 = 0
        
        for c in range(len(plants)):
            aC += model.Params.COALFOPEX[c] * model.Params.COALCAP[c] * obj.coalOnline[c,y]
            bC += model.Params.COALVOPEX[c]*obj.coalGen[c,y]
            if y ==1:
                if bC == 0:
                    Coal_first_bool = True
            for r in range(len(reSites)):
                
                RECons += model.Params.RECAPEX[r]*obj.capInvest[r,c,y]
                REOM +=  (model.Params.REFOPEX[r]*obj.reCap[r,c,y]+ model.Params.REVOPEX[r] * obj.reGen[r,c,y])
                
            if dC>0:
                Ren_Bool = True
                
        dC = (RECons + REOM)/((1+DiscRate)**(y+1))
        CostCoalOM.append(aC+bC)
        CostRECons.append(RECons)
        CostREOM.append(REOM)
        
    if prints == True:
        print('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX+REVOPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))
    FileWrite.write('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))


    # Health damage component
    if prints == True:
        print('\nHealth damage component:')
    FileWrite.write('\nHealth damage component:')

    hd = 0
    for y in range(numYears):
        h = 0
        for c in range(len(plants)):
            h += plants['HD'].values[c]*obj.coalGen[c,y] # This was formerly the HD * capOnline (which is a boolean not the gen which it should be) MV 9/17/2021
        hd += h
        HealthObj.append(h)
    if prints == True:
        print('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))
    FileWrite.write('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))

    # Jobs component
    FileWrite.write('\nJobs component')
    if prints == True:
        print('\nJobs component')
    sumCoalEF = 0
    for y in range(numYears):
        C_OM = 0
        C_RET = 0
        for c in range(len(plants)):
            C_RET += model.Params.RETEF[c]*obj.capRetire[c,y]
            C_OM += +model.Params.COALOMEF[c]*obj.coalGen[c,y]
        
        a = C_RET+C_OM
        JobsCoalRet.append(C_RET)
        JobsCoalOM.append(C_OM)
        if prints == True:
            print('\tYear {} RETEF + COALOMEF = {}.'.format(y,round(a)))
        FileWrite.write('\n\tYear {} RETEF + COALOMEF = {}.'.format(y,round(a)))
        sumCoalEF+=a
    
    sumREEF = 0
    for y in range(numYears):
        b = 0
        RE_Cap = 0
        RE_OM = 0
        for c in range(len(plants)):
            for r in range(len(reSites)):
                RE_Cap += model.Params.CONEF[r,y]*obj.capInvest[r,c,y]
                RE_OM += model.Params.REOMEF[r,y]*obj.reCap[r,c,y]# reGen turned to reCap to MV 08092021
        b = RE_Cap + RE_OM       
        if prints == True:
            print('\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        FileWrite.write('\n\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        sumREEF += b
        JobsREOM.append(RE_OM)
        JobsRECONS.append(RE_Cap)
    if prints == True:
        print('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))
    FileWrite.write('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))

    objS = (aC+bC+dC)*scenario[0]+hd*scenario[1]-(sumREEF+sumCoalEF)*scenario[2]
    if prints == True:
        print('\nSum of objective components = {}'.format(round(objS)))
    FileWrite.write('\nSum of objective components = {}'.format(round(objS)))
    FileWrite.close()
    os.chdir('..')

def PostProcess(obj,model,numYears,region,coalPlants,reSites,scenario, SMR_bool,folderName):
    for i in scenario:
        i = float(i) # done to prevent multiple formats of scenario values which get REALLY frustrating in data analysis MV 9/15/2021
    cLat = []
    cLon = []
    pNam = []
    coalRetire = []
    coalOnline = []
    capRetire = []
    coalGen = []
    coalYr = []
    
    # increased metrics for the summarizing portion  MV 9/15/21
    coalOM_Cost = []
    coalOM_Jobs = []
    coalRet_Jobs = []
    coalHealth = []
    capOnline = []
    
    reOnline = []
    reInvest = []
    cpInvest = []
    totReCap = []
    renGenrn = []
    
    # increased metrics for the summarizing portion  MV 9/15/21
    reCons_cost = []
    reCons_Jobs = []
    reOM_cost = []
    reOM_Jobs = []
    
    
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
            
            coalOM_Cost.append((model.Params.COALFOPEX[c] * model.Params.COALCAP[c] * obj.coalOnline[c,y]) + (model.Params.COALVOPEX[c] * obj.coalGen[c,y]))  # MV 9/17/2021
            coalOM_Jobs.append(model.Params.COALOMEF[c]*obj.coalGen[c,y]) # MV 9/17/2021
            coalRet_Jobs.append(model.Params.RETEF[c]*obj.capRetire[c,y]) # MV 9/17/2021
            coalHealth.append(model.Params.HD[c]*obj.coalGen[c,y]) # MV 9/17/2021
            capOnline.append(model.Params.COALCAP[c]-obj.capRetire[c,y])  # MV 9/17/2021

            for r in range(reSites.shape[0]):
                # If reOnline flag is set for site s for plant c and year y then add flags
                if obj.reOnline[r,c,y]==1:
                    reOnline.append(1)
                    reOM_cost.append(model.Params.REVOPEX[r] * obj.reGen[r,c,y] + model.Params.REFOPEX[r] * obj.reCap[r,c,y]) # MV 9/17/2021
                    reOM_Jobs.append(model.Params.REOMEF[r,y]*obj.reCap[r,c,y]) # MV 9/17/2021
                else:
                    reOnline.append(0)
                    reOM_cost.append(0)
                    reOM_Jobs.append(0)
                
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reInvest[r,c,y]==1:
                    reInvest.append(1)
                    reCons_cost.append(model.Params.RECAPEX[r] * obj.capInvest[r,c,y]) # MV 9/17/2021
                    reCons_Jobs.append(model.Params.CONEF[r,y]*obj.capInvest[r,c,y]) # MV 9/17/2021
                else:
                    reInvest.append(0)
                    reCons_cost.append(0)
                    reCons_Jobs.append(0)
                
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.capInvest[r,c,y]>0:
                    cpInvest.append(obj.capInvest[r,c,y])
                else:
                    cpInvest.append(0)
                
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reCap[r,c,y]>0:
                    totReCap.append(obj.reCap[r,c,y])
                else:
                    totReCap.append(0)
                
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reGen[r,c,y]>0:
                    renGenrn.append(obj.reGen[r,c,y])
                else:
                    renGenrn.append(0)
                yr.append(cYr)
                cPlant.append(coalPlants.iloc[c,7])
                Lat.append(reSites.iloc[r,0])
                Lon.append(reSites.iloc[r,1])
                Typ.append(reSites.iloc[r,3])
                CF.append(reSites.iloc[r,2])
                elg.append(reSites.iloc[r,-1])
                
    os.chdir(folderName)

    # Create coal data CSV file.
    dat = {'Year':coalYr,'Lat':cLat,'Lon':cLon,'coalOnline':coalOnline,'coalGen':coalGen,'capOnline':capOnline,'coalOM_Jobs':coalOM_Jobs,'coalRetire':coalRetire,'capRetire':capRetire,'coalRet_Jobs':coalRet_Jobs,'coalHealth':coalHealth}
    coalData = pd.DataFrame(dat)
    coalData.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_coalData.csv')

    dat = {'Year':yr,'Lat':Lat,'Lon':Lon,'Type':Typ,'Ann.CF':CF,'EligibleSite':elg,'Online':reOnline,'Investment':reInvest,'RE_Cons_Jobs':reCons_Jobs,'RE_Cons_Cost':reCons_cost,'Invested MW':cpInvest,'Total MW Cap.':totReCap,'Tot MWh Gen':renGenrn, 'RE_OM_Jobs':reOM_Jobs, 'reOM_cost':reOM_cost, 'Repl. Plant':cPlant}
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
    os.chdir('..')
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
                    
                    
                    
