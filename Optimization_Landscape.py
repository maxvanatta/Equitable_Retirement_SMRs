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



def PrepareModel(numYears,region,threshDist,SMR_bool,DiscRate, SMRs, solFileName, winFileName, getNewEFs = False, SMROnly = False, Nation = False, jobCoeff = 1):
    plants = CoalPlants.getCoalPlants(region)
    plants['HISTGEN'] = CoalPlants.getPlantGeneration(plants['Plant Code'])
    plants['HD'] = CoalPlants.getMarginalHealthCosts(plants['Plant Code'])
    print('checkpoint1')
    plants.dropna(inplace=True)
    coalData = pd.read_excel('3_1_Generator_Y2019.xlsx',header=1,index_col='Plant Code',sheet_name='Operable',usecols='B:F')
    coalPlants = plants.merge(coalData, left_on='Plant Code', right_index=True)
    coalPlants = coalPlants.drop_duplicates()
    print('checkpoint12')
    EM_Data = pd.read_csv('adjCO2.csv')
    EM_Data
    
    ids = coalPlants['Plant Code'].to_list()
    
    CO2s = []
    lower_limit = []
    for i in ids:
        h = (EM_Data.loc[EM_Data['Plant Code']== i]['Adj_emissions']).to_list()
        CO2s.append(h[0])
        if (EM_Data.loc[EM_Data['Plant Code']== i]['Elec_Percent']).to_list()[0] >= 0.2:
            lower_limit.append(i)
    
    plants['Marginal Emissions USt/MWh'] = CO2s
    coalPlants['Marginal Emissions USt/MWh'] = CO2s
    print('pre-cogen limit: ',plants.shape)
    
    plants = plants.loc[plants['Plant Code'].isin(lower_limit)]
    coalPlants = coalPlants.loc[coalPlants['Plant Code'].isin(lower_limit)]
    print('post-cogen limit: ',plants.shape)
    print(coalPlants)
    coalPlants.to_csv('coalplants_cp.csv')
    
    
    if SMR_bool == True:
        folderName = ('_'.join(region)+'_'+str(numYears)+'years_'+str(threshDist)+'miles_'+str(DiscRate)+'_DR_'+str(jobCoeff)+'JC_SMR_'+str(SMR_bool)+'_'+str(SMRs[0])+'_'+str(SMRs[1])+'_'+str(SMRs[2]))
    else:
        folderName = ('_'.join(region)+'_'+str(numYears)+'years_'+str(threshDist)+'miles_'+str(DiscRate)+'_DR_'+str(jobCoeff)+'JC_SMR_'+str(SMR_bool))
    
    NREL_Supply_Data = pd.read_csv('NREL_SupplyCurve.csv')
    NREL_CFs = NREL_Supply_Data.loc[NREL_Supply_Data['Year']==2020][['Site','Year','CF']]
    NREL_CFs
    NREL_CAPEX = NREL_Supply_Data.loc[NREL_Supply_Data['Year']==2020][['Site','Year','CAPEX_Mult']]
    reSites = NREL_Supply_Data.loc[NREL_Supply_Data['Year']==2020]
    reSites
    
    
    EFs_new = pd.read_csv('reEFs_cont_update.csv')
    EFs_new
    CONEF = EFs_new['Con/Instl EF']
    REOMEF = EFs_new['O&M EF']

    EFType = []
    for i in EFs_new['Unnamed: 0.1']:
        EFType.append(i.split(',')[-1])
    C_h = []
    R_h = []

    ReSite_Helper = int(len(CONEF)/21) #numYears)
    EFType = EFType[:ReSite_Helper]
    i = 0
    while i < ReSite_Helper:
        inds = list(np.arange(i,len(CONEF),ReSite_Helper))
        #print(inds)
        for x in inds:
            C_h.append(CONEF[x])
            R_h.append(REOMEF[x])
        i+=1

    CONEF = np.reshape(np.array(C_h),(len(reSites),21)) #numYears))
    REOMEF = np.reshape(np.array(R_h),(len(reSites),21)) #numYears))
    MAXCAP = np.zeros((len(reSites),len(coalPlants)))
    Site_Maxs = reSites['total_cap'].tolist()

    SITEMAXCAP = np.array(Site_Maxs)


    # for each coal plant, use its lat lon to calculate distance between RE sites and the plant. if distance is more than X then make capacity 0
    reSites['Eligible'] = 0
    RESITE_REDUCED = []
    ns = []
    for c in range(MAXCAP.shape[1]):
        #print('step1')
        c_REDUCED = []
        site = 0
        coalPlants.iloc[c,1]
        coalPlants.loc[coalPlants.index.tolist()[c],'Latitude']
        coalCord = (coalPlants.iloc[c,1],coalPlants.iloc[c,2])
        for s in range(MAXCAP.shape[0]):
            reCord = (reSites.loc[s,'Lat'],reSites.loc[s,'Lon'])
            dist = haversine(coalCord,reCord, unit=Unit.MILES)
            if dist<threshDist:
                #print(s)
                c_REDUCED.append(s)
                if EFType[s] == 'S':
                    MAXCAP[s,c] = 1
                elif EFType[s] =='W':
                    MAXCAP[s,c] = 1
                elif EFType[s] == 'smr':
                    MAXCAP[s,c] = 1
                reSites.loc[s,'Eligible'] = 1
            #site+=1
        RESITE_REDUCED.append(c_REDUCED)
        ns.append(len(c_REDUCED))

    SITEMAXCAP#*=reSites['Eligible']

    #print(SITEMAXCAP)

    
    #print(RESITE_REDUCED)
    
    #print(ns)

    MASK_base = []
    NP_base = []
    for i in RESITE_REDUCED:
        MASK_base.append([1]*len(i)+[0]*(max(ns)-len(i)))
        i = i + [0]*max(ns)
        NP_base.append(i[:max(ns)])
    RED_indexes = np.array(NP_base)
    MASK = np.array(MASK_base)

    REV_INDS = np.zeros((MAXCAP.shape[0],MAXCAP.shape[1]),dtype = int)
    for rr_r in range(RED_indexes.shape[1]):
        for rr_c in range(RED_indexes.shape[0]):
            r_val = RED_indexes[rr_c,rr_r]
            c_val = rr_c
            REV_INDS[r_val,c_val] = int(rr_r)




    ind = reSites['Lat'].astype(str)+reSites['Lon'].astype(str)
    mCapDF = pd.DataFrame(MAXCAP,index=ind,columns=list(coalPlants['Plant Name']))
    mCapDF.to_csv('MAXCAP_test.csv')

    mCapDF['S'] = mCapDF[list(mCapDF.columns)].sum(axis=1)


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
    
    return CONEF, REOMEF, EFType, MAXCAP,SITEMAXCAP,reSites,plants, mCapDF,coalPlants, folderName, RED_indexes, MASK, REV_INDS, NREL_CAPEX    

def SingleModel(scen,numYears,solFileName,winFileName,region,CONEF,REOMEF,EFType,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,coalPlants,threshDist,folderName,DiscRate, SMRs,RED_indexes,MASK,REV_INDS,CO2_Limits = 'Linear2030'):
    
    obj, plants2, model = test_cplex(scen[0],scen[1],scen[2],numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,DiscRate, SMRs[0],SMRs[1],SMRs[2],RED_indexes,MASK,REV_INDS,CO2Limits = CO2_Limits)
    NEW_RES = SummarizeResults(obj, plants2, model, [scen[0],scen[1],scen[2]], region, threshDist,SMR_bool, reSites, numYears,folderName,DiscRate,EFType,SMRs, prints = True)
    PostProcess(obj,model,numYears,region,coalPlants,reSites,[scen[0],scen[1],scen[2]], SMR_bool,folderName,NEW_RES)
    
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
    '''
    InputFileWrite = open('Input_Values_.txt','w')
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
    '''
    #FileWrite = open('Objective_Record_'+'_'.join(region)+'_'+str(scenario[0])+'_'+str(scenario[1])+'_'+str(scenario[2])+'_'+str(threshDist)+'_'+str(SMR_bool)+'.txt','w')

    if prints == True:
        print('System cost component:')
    #FileWrite.write('System cost component:')
    
    NEW_capInvest = np.zeros((len(reSites),len(plants),numYears))
    NEW_reInvest = np.zeros((len(reSites),len(plants),numYears))
    NEW_reOnline = np.zeros((len(reSites),len(plants),numYears))
    NEW_reGen = np.zeros((len(reSites),len(plants),numYears))
    NEW_reCap = np.zeros((len(reSites),len(plants),numYears))
    
    for y in range(numYears):
        for c in range(len(plants)):
            for r in range(model.Params.MASK.shape[1]):
                r_index = model.Params.RED_INDEXES[c,r]
                NEW_capInvest[r_index,c,y] += float(obj.capInvest[r,c,y])
                NEW_reGen[r_index,c,y] += obj.reGen[r,c,y]
                NEW_reCap[r_index,c,y] += obj.reCap[r,c,y]
                if NEW_reInvest[r_index,c,y] == 0:
                    NEW_reInvest[r_index,c,y] += int(obj.reInvest[r,c,y])
                if NEW_reOnline[r_index,c,y] == 0:
                    NEW_reOnline[r_index,c,y] += int(obj.reOnline[r,c,y])
        
    
    CostCoalOM = []
    CostCoalRet = [] # do we need to find values for this aspect?

    
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
            aC += model.Params.COALFOPEX[c,y] * model.Params.COALCAP[c] * obj.coalOnline[c,y]/((1+DiscRate)**(y))
            bC += model.Params.COALVOPEX[c,y]*obj.coalGen[c,y]/((1+DiscRate)**(y))
            if y ==1:
                if bC == 0:
                    Coal_first_bool = True
            for r in range(len(reSites)):
                
                RECons += model.Params.RECAPEX[r,y]*NEW_capInvest[r,c,y]
                
                REOM +=  (model.Params.REFOPEX[r,y]*NEW_reCap[r,c,y]+ model.Params.REVOPEX[r,y] * NEW_reGen[r,c,y])
                
            if dC>0:
                Ren_Bool = True
                
        dC += (RECons + REOM)/((1+DiscRate)**(y))
        print(aC+bC+dC)
        
    if prints == True:
        print('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX+REVOPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))
    #FileWrite.write('\tCOALFOPEX = {}\n\tCOALVOPEX = {}\n\tREFOPEX+RECAPEX = {}\n\t\tTotal = {}\n\t\tAlpha = {}\n\t\tTotal = {}'.format(aC,bC,dC,round(aC+bC+dC,2),scenario[0],round(aC+bC+dC,2)*scenario[0]))


    # Health damage component
    if prints == True:
        print('\nHealth damage component:')
    #FileWrite.write('\nHealth damage component:')

    hd = 0
    for y in range(numYears):
        h = 0
        for c in range(len(plants)):
            h += plants['HD'].values[c]*obj.coalGen[c,y]/((1+DiscRate)**(y)) # This was formerly the HD * capOnline (which is a boolean not the gen which it should be) MV 9/17/2021
        hd += h
        HealthObj.append(h)
    if prints == True:
        print('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))
    #FileWrite.write('\tHealth damage sum: {}\n\tBeta = {}\n\tTotal = {}'.format(hd, scenario[1], hd*scenario[1]))

    # Jobs component
    #FileWrite.write('\nJobs component')
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
        #FileWrite.write('\n\tYear {} RETEF + COALOMEF = {}.'.format(y,round(a)))
        sumCoalEF+=a
    
    sumREEF = 0
    for y in range(numYears):
        b = 0
        RE_Cap = 0
        RE_OM = 0
        for c in range(len(plants)):
            for r in range(len(reSites)):
                RE_Cap += model.Params.CONEF[r,y]*NEW_capInvest[r,c,y]
                RE_OM += model.Params.REOMEF[r,y]*NEW_reCap[r,c,y]# reGen turned to reCap to MV 08092021
        b = RE_Cap + RE_OM       
        if prints == True:
            print('\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        #FileWrite.write('\n\tYear {} CONEF + REOMEF = {}.'.format(y,b))
        sumREEF += b
        JobsREOM.append(RE_OM)
        JobsRECONS.append(RE_Cap)
    if prints == True:
        print('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))
    #FileWrite.write('\t\tGamma = -{}\n\t\tTotal = {}'.format(scenario[2],(sumREEF+sumCoalEF)*scenario[2]))

    objS = (aC+bC+dC)*scenario[0]+hd*scenario[1]-(sumREEF+sumCoalEF)*scenario[2]
    if prints == True:
        print('\nSum of objective components = {}'.format(round(objS)))
    #FileWrite.write('\nSum of objective components = {}'.format(round(objS)))
    #FileWrite.close()
    os.chdir('..')
    
    NEW_RES = [NEW_capInvest ,NEW_reInvest ,NEW_reOnline,NEW_reGen,NEW_reCap]
    return NEW_RES

def PostProcess(obj,model,numYears,region,coalPlants,reSites,scenario, SMR_bool,folderName,NEW_RES):
    
    NEW_capInvest = NEW_RES[0]
    NEW_reInvest = NEW_RES[1]
    NEW_reOnline = NEW_RES[2]
    NEW_reGen = NEW_RES[3]
    NEW_reCap = NEW_RES[4]
        
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
    
    yrCoalGen = []
    yrCoalCap = []
    yrCoalOM = []
    yrCoalCost = []
    yrSolarCap = []
    yrSolarGen = []
    yrSolarOM = []
    yrSolarCons = []
    yrSolarCost_c = []
    yrSolarCost_om = []
    yrWindCap = []
    yrWindGen = []
    yrWindOM = []
    yrWindCons = []
    yrWindCost_c = []
    yrWindCost_om = []
    yrSMRCap = []
    yrSMRGen = []
    yrSMRCons = []
    yrSMROM = []
    yrSMRCost_c = []
    yrSMRCost_om = []
    
    yr2 = []
    
    
    CO2perPlant = []

    # RE investment Lat/Lon/Type
    for y in range(numYears):
        cYr = y+2020
        
        cg_h = 0
        cc_h = 0
        com_h = 0
        ccost_h = 0
        sc_h = 0
        sg_h = 0
        sco_h = 0
        som_h = 0
        scc_h = 0 #cons cost
        scoo_h = 0 # om cost
        wc_h = 0 
        wg_h = 0
        wco_h = 0
        wom_h = 0
        wcc_h = 0 #cons cost
        wcoo_h = 0 # om cost
        smrc_h = 0
        smrg_h = 0
        smrco_h = 0
        smrom_h = 0
        smrcc_h = 0 #cons cost
        smrcoo_h = 0 # om cost
        
        yr2.append(cYr)

        for c in range(coalPlants.shape[0]):
            cLat.append(coalPlants.loc[coalPlants.index.tolist()[c],'Latitude'])
            cLon.append(coalPlants.loc[coalPlants.index.tolist()[c],'Longitude'])
            pNam.append(coalPlants.loc[coalPlants.index.tolist()[c],'Plant Name'])
            coalRetire.append(obj.coalRetire[c,y])
            coalOnline.append(obj.coalOnline[c,y])
            capRetire.append(obj.capRetire[c,y])
            coalGen.append(obj.coalGen[c,y])
            coalYr.append(cYr)
            
            coalOM_Cost.append((model.Params.COALFOPEX[c,y] * model.Params.COALCAP[c] * obj.coalOnline[c,y]) + (model.Params.COALVOPEX[c,y] * obj.coalGen[c,y]))  # MV 9/17/2021
            coalOM_Jobs.append(model.Params.COALOMEF[c]*obj.coalGen[c,y]) # MV 9/17/2021
            coalRet_Jobs.append(model.Params.RETEF[c]*obj.capRetire[c,y]) # MV 9/17/2021
            coalHealth.append(model.Params.HD[c]*obj.coalGen[c,y]) # MV 9/17/2021
            capOnline.append(model.Params.COALCAP[c]*obj.coalOnline[c,y])  # MV 9/17/2021
            
            CO2perPlant.append(model.Params.CO2ME[c]*obj.coalGen[c,y]) # MV 10/3/2021
            
            cg_h += obj.coalGen[c,y]
            cc_h += model.Params.COALCAP[c]*obj.coalOnline[c,y]
            com_h += model.Params.COALOMEF[c]*obj.coalGen[c,y]
            ccost_h += (model.Params.COALFOPEX[c,y] * model.Params.COALCAP[c] * obj.coalOnline[c,y]) + (model.Params.COALVOPEX[c,y] * obj.coalGen[c,y])

            for r in range(reSites.shape[0]):
                if NEW_reCap[r,c,y]>0:
                    reOnline.append(1)
                else:
                    reOnline.append(0)
                reOM_cost.append(model.Params.REVOPEX[r,y] * NEW_reGen[r,c,y] + model.Params.REFOPEX[r,y] * NEW_reCap[r,c,y]) # MV 9/17/2021
                reOM_Jobs.append(model.Params.REOMEF[r,y]*NEW_reCap[r,c,y]) # MV 9/17/2021
                reInvest.append(NEW_capInvest[r,c,y])
                reCons_cost.append(model.Params.RECAPEX[r,y] * NEW_capInvest[r,c,y]) # MV 9/17/2021
                reCons_Jobs.append(model.Params.CONEF[r,y]*NEW_capInvest[r,c,y]) # MV 9/17/2021
                cpInvest.append(NEW_capInvest[r,c,y])
                totReCap.append(NEW_reCap[r,c,y])
                renGenrn.append(NEW_reGen[r,c,y])
                yr.append(cYr)
                cPlant.append(coalPlants.loc[coalPlants.index.tolist()[c],'Plant Name'])
                Lat.append(reSites.loc[r,'Lat'])
                Lon.append(reSites.loc[r,'Lon'])
                Typ.append(reSites.loc[r,'Type'])
                CF.append(reSites.loc[r,'CF'])
                elg.append(reSites.loc[r,'Eligible'])
                if reSites.loc[r,'Type'] =='S':
                    sc_h += NEW_reCap[r,c,y]
                    sg_h += NEW_reGen[r,c,y]
                    sco_h += model.Params.CONEF[r,y]*NEW_capInvest[r,c,y]
                    som_h += model.Params.REOMEF[r,y]*NEW_reCap[r,c,y]
                    scc_h += model.Params.RECAPEX[r,y] * NEW_capInvest[r,c,y]
                    scoo_h += model.Params.REVOPEX[r,y] * NEW_reGen[r,c,y] + model.Params.REFOPEX[r,y] * NEW_reCap[r,c,y]
                if reSites.loc[r,'Type'] =='W':
                    wc_h += NEW_reCap[r,c,y]
                    wg_h += NEW_reGen[r,c,y]
                    wco_h += model.Params.CONEF[r,y]*NEW_capInvest[r,c,y]
                    wom_h += model.Params.REOMEF[r,y]*NEW_reCap[r,c,y]
                    wcc_h += model.Params.RECAPEX[r,y] * NEW_capInvest[r,c,y]
                    wcoo_h += model.Params.REVOPEX[r,y] * NEW_reGen[r,c,y] + model.Params.REFOPEX[r,y] * NEW_reCap[r,c,y]
                if reSites.loc[r,'Type'] =='smr':
                    smrc_h += NEW_reCap[r,c,y]
                    smrg_h += NEW_reGen[r,c,y]
                    smrco_h += model.Params.CONEF[r,y]*NEW_capInvest[r,c,y]
                    smrom_h += model.Params.REOMEF[r,y]*NEW_reCap[r,c,y]
                    smrcc_h += model.Params.RECAPEX[r,y] *NEW_capInvest[r,c,y]
                    smrcoo_h += model.Params.REVOPEX[r,y] * NEW_reGen[r,c,y] + model.Params.REFOPEX[r,y] * NEW_reCap[r,c,y]
        yrCoalGen.append(cg_h)
        yrCoalCap.append(cc_h)
        yrCoalOM.append(com_h)
        yrCoalCost.append(ccost_h)
        yrSolarCap.append(sc_h)
        yrSolarGen.append(sg_h)
        yrSolarOM.append(som_h)
        yrSolarCons.append(sco_h)
        yrSolarCost_c.append(scc_h)
        yrSolarCost_om.append(scoo_h)
        
        yrWindCap.append(wc_h)
        yrWindGen.append(wg_h)
        yrWindOM.append(wom_h)
        yrWindCons.append(wco_h)
        yrWindCost_c.append(wcc_h)
        yrWindCost_om.append(wcoo_h)
        
        yrSMRCap.append(smrc_h)
        yrSMRGen.append(smrg_h)
        yrSMRCons.append(smrco_h)
        yrSMROM.append(smrom_h)
        yrSMRCost_c.append(smrcc_h)
        yrSMRCost_om.append(smrcoo_h)
        
    os.chdir(folderName)

    # Create coal data CSV file.
    dat = {'Year':coalYr,'Lat':cLat,'Lon':cLon,'coalOnline':coalOnline,'coalGen':coalGen,'capOnline':capOnline,'coalOM_Jobs':coalOM_Jobs,'CO2 emissions (lb)':CO2perPlant,'coalRetire':coalRetire,'capRetire':capRetire,'coalRet_Jobs':coalRet_Jobs,'coalHealth':coalHealth}
    coalData = pd.DataFrame(dat)
    #coalData.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_coalData.csv')
    coalData.to_csv('_coalData.csv')

    dat = {'Year':yr,'Lat':Lat,'Lon':Lon,'Type':Typ,'Ann.CF':CF,'EligibleSite':elg,'Online':reOnline,'Investment':reInvest,'RE_Cons_Jobs':reCons_Jobs,'RE_Cons_Cost':reCons_cost,'Invested MW':cpInvest,'Total MW Cap.':totReCap,'Tot MWh Gen':renGenrn, 'RE_OM_Jobs':reOM_Jobs, 'reOM_cost':reOM_cost, 'Repl. Plant':cPlant}
    reData = pd.DataFrame(dat)
    #reData.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_reData.csv')
    reData.to_csv('_reData.csv')
    
    #reData.loc[reData['Invested MW']>0].to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_reData_FILTERED.csv')
    reData.loc[reData['Tot MWh Gen']>0].to_csv('_reData_FILTERED.csv')

    Yearly = {'Year':yr2,'Yearly CO2 Limit':list(model.Params.CO2Limits),'CO2 Emissions (USt/yr)':list(np.sum(obj.CO2Emissions,axis=0)),'Coal Generation (MWh)':yrCoalGen,'Coal Capacity (MW)':yrCoalCap,'Coal O&M Jobs':yrCoalOM,'Coal O&M Cost':yrCoalCost,'Solar Capacity (MW)':yrSolarCap,'Solar Generation (MWh)':yrSolarGen,'Solar O&M Jobs':yrSolarOM,'Solar Construction Jobs':yrSolarCons,'Solar O&M Cost':yrSolarCost_om,'Solar Construction Cost':yrSolarCost_c,'Wind Capacity (MW)':yrWindCap,'Wind Generation (MWh)':yrWindGen,'Wind O&M Jobs':yrWindOM,'Wind Construction Jobs':yrWindCons,'Wind O&M Cost':yrWindCost_om,'Wind Construction Cost':yrWindCost_c,'SMR Capacity (MW)':yrSMRCap,'SMR Generation (MWh)':yrSMRGen,'SMR Construction Jobs':yrSMRCons,'SMR O&M Jobs':yrSMROM,'SMR O&M Cost':yrSMRCost_om,'SMR Construction Cost':yrSMRCost_c}
    #,'CO2 Emissions (lb/yr)':np.sum(obj.CO2Emissions,axis=0)
    
    Sum_Data = pd.DataFrame(Yearly)
    #Sum_Data.to_csv('.'.join(list(map(str,scenario)))+'_'+'_'.join(region)+'_'+str(SMR_bool)+'_Summary.csv')
    Sum_Data.to_csv('_Summary.csv')
          
