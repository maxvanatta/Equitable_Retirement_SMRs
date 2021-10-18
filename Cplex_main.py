# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:03:40 2021

@author: bhavrathod
"""
from EquitableRetirement_CO2 import EquitableRetirement
import CoalPlants
import RenewableSites
from getReEFs import batchReEFs

import numpy as np
import pandas as pd




def test_cplex(alp,bet,gam,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,DiscRate, SMR_CAPEX, SMR_FOPEX, SMR_VOPEX,RED_indexes,MASK, REV_INDS, CO2Limits= 'Linear2030', jobCoeff = 1.0, RE_Case = 'Moderate'):
    ''' use sample data to test runtime and large-scale functionality of formulation '''
    print('TEST_CPLEX:')
    print('\t','getting data...')

    # costs
    if SMR_bool == True:
        SMR_num = plants.index.size 
    else: 
        SMR_num = 0
    RE_num = len(reSites) - SMR_num
    print(RE_num)
    
    ATB_data = pd.read_csv('ATB_Data.csv')
    
    if RE_Case =='Conservative':
        Solar_CAPEX = ATB_data['PV_Conservative_Cap'].to_list()
        Wind_CAPEX = ATB_data['Wind_Conservative_Cap'].to_list()
        Solar_OM = ATB_data['PV_Conservative_OM'].to_list()
        Wind_OM = ATB_data['Wind_Conservative_OM'].to_list()
        Coal_VOPEX = ATB_data['Coal O&M $/MWhr'].to_list()
        Coal_FOPEX = ATB_data['Coal O&M $/kW-yr'].to_list()
    
    elif RE_Case =='Advanced':
        Solar_CAPEX = ATB_data['PV_Advanced_Cap'].to_list()
        Wind_CAPEX = ATB_data['Wind_Advanced_Cap'].to_list()
        Solar_OM = ATB_data['PV_Advanced_OM'].to_list()
        Wind_OM = ATB_data['Wind_Advanced_OM'].to_list()
        Coal_VOPEX = ATB_data['Coal O&M $/MWhr'].to_list()
        Coal_FOPEX = ATB_data['Coal O&M $/kW-yr'].to_list()
    
    else:
        Solar_CAPEX = ATB_data['PV_Moderate_Cap'].to_list()
        Wind_CAPEX = ATB_data['Wind_Moderate_Cap'].to_list()
        Solar_OM = ATB_data['PV_Moderate_OM'].to_list()
        Wind_OM = ATB_data['Wind_Moderate_OM'].to_list()
        Coal_VOPEX = ATB_data['Coal O&M $/MWhr'].to_list()
        Coal_FOPEX = ATB_data['Coal O&M $/kW-yr'].to_list()
    
    RECAPEX = []
    REFOPEX = []
    REVOPEX = []
    COALVOPEX = []
    COALFOPEX = []

    for r in range(int(RE_num/2)):
        RECAPEX_h = []
        REFOPEX_h = []
        REVOPEX_h = []
        for y in range(numYears):
            RECAPEX_h.append(Solar_CAPEX[y]*1000)
            REFOPEX_h.append(Solar_OM[y]*1000)
            REVOPEX_h.append(0)
        RECAPEX.append(RECAPEX_h)
        REFOPEX.append(REFOPEX_h)
        REVOPEX.append(REVOPEX_h)
    for r in range(int(RE_num/2)):
        RECAPEX_h = []
        REFOPEX_h = []
        REVOPEX_h = []
        for y in range(numYears):
            RECAPEX_h.append(Wind_CAPEX[y]*1000)
            REFOPEX_h.append(Wind_OM[y]*1000)
            REVOPEX_h.append(0)
        RECAPEX.append(RECAPEX_h)
        REFOPEX.append(REFOPEX_h)
        REVOPEX.append(REVOPEX_h)
    if SMR_bool:
        for r in range(SMR_num):
            RECAPEX_h = []
            REFOPEX_h = []
            REVOPEX_h = []
            for y in range(numYears):
                RECAPEX_h.append(SMR_CAPEX)
                REFOPEX_h.append(SMR_FOPEX)
                REVOPEX_h.append(SMR_VOPEX)
            RECAPEX.append(RECAPEX_h)
            REFOPEX.append(REFOPEX_h)
            REVOPEX.append(REVOPEX_h)

    for c in range(len(plants)):
        COALVOPEX_h = []
        COALFOPEX_h = [] 
        for y in range(numYears):
            COALVOPEX_h.append(Coal_VOPEX[y])
            COALFOPEX_h.append(Coal_FOPEX[y]*1000)

        COALVOPEX.append(COALVOPEX_h)
        COALFOPEX.append(COALFOPEX_h)

    ATB_data

    costs = { 'RECAPEX' : np.array(RECAPEX),
                  'REFOPEX' : np.array(REFOPEX),
                  'REVOPEX' : np.array(REVOPEX),
                  'COALVOPEX' : np.array(COALVOPEX),
                  'COALFOPEX' : np.array(COALFOPEX)
            }
    
    # site limits
    limits = { 'MAXCAP' : MAXCAP,
                'SITEMAXCAP' : SITEMAXCAP,
                'MAXSITES' : np.ones(len(plants))*100
              }#MAXSITES set to 100 to reduce constraint pressure until many tiny plants shows up as an issue. # MV 10/04/2021
    
    
    ef = { 'RETEF' : np.ones(len(plants))*1.65,
           'CONEF' : CONEF,
           'COALOMEF' : np.ones(len(plants))* (plants['Coal Capacity (MW)'].values*0.14/plants['HISTGEN'].values), 
           'REOMEF' : REOMEF/30
         }
    
    ### BUILD MODEL
    m = EquitableRetirement()

    m.Y = np.arange(numYears)+2020
    m.R = reSites.index.values
    m.C = plants.index.values
    m.R_RED = np.array(range(RED_indexes.shape[1]))
    
    
    
    m.Params.HISTGEN = plants['HISTGEN'].values
    m.Params.CO2ME = np.array(plants['Marginal Emissions USt/MWh']) #np.full(len(m.C),1400) # THis will be replaced on a per plant basis. 1400lb/MWh currrently
    CO2_Limits = [sum(m.Params.CO2ME*m.Params.HISTGEN)] #this keeps first year as unrestricted by CO2 and the second year is purely economic closing.
    if CO2Limits == 'Linear2030':
        endYear = 2030
        years = 10
        total_initial = sum(m.Params.CO2ME*m.Params.HISTGEN)
        for i in range(years-1): #-1 from the first value being input in the initial list forulation above
            CO2_Limits.append(total_initial-(total_initial/(years-1))*i)
        CO2_Limits = CO2_Limits+([0]*(numYears))         
    if CO2Limits == 'Linear2035':
        endYear = 2035
        years = 15
        total_initial = sum(m.Params.CO2ME*m.Params.HISTGEN)
        for i in range(years-1):
            CO2_Limits.append(total_initial-(total_initial/(years-1))*i)
        CO2_Limits = CO2_Limits+([0]*(numYears))
    
    print(CO2_Limits[:numYears])
    
    m.Params.CO2Limits = np.array(CO2_Limits[:numYears])

    
    m.Params.COALCAP = plants['Coal Capacity (MW)'].values
    m.Params.CF = reSites['Annual CF'].values * reSites['Eligible']
    m.Params.RECAPEX = costs['RECAPEX']
    m.Params.REFOPEX = costs['REFOPEX']
    m.Params.REVOPEX = costs['REVOPEX']
    m.Params.COALVOPEX = costs['COALVOPEX']
    m.Params.COALFOPEX = costs['COALFOPEX']
    m.Params.MAXCAP = limits['MAXCAP']
    m.Params.SITEMAXCAP = limits['SITEMAXCAP']
    m.Params.MAXSITES = limits['MAXSITES']
    m.Params.HD = plants['HD'].values
    m.Params.RETEF = ef['RETEF']
    m.Params.CONEF = ef['CONEF']
    m.Params.COALOMEF = ef['COALOMEF']
    m.Params.REOMEF = ef['REOMEF']
    m.Params.JOBCEOFF = jobCoeff
    m.Params.DiscRate = DiscRate
    
    m.Params.RED_INDEXES = RED_indexes
    m.Params.MASK = MASK
    m.Params.REV_INDS = REV_INDS
    

    '''
    ### CHECK DIMS
    print('\t','Y\t',len(m.Y))
    print('\t','R\t',len(m.R))
    print('\t','C\t',len(m.C))
    print('\t','HISTGEN\t',m.Params.HISTGEN.shape)
    print('\t','COALCAP\t',m.Params.COALCAP.shape)
    print('\t','CF\t',m.Params.CF.shape)
    print('\t','RECAPEX\t',m.Params.RECAPEX.shape)
    print('\t','REFOPEX\t',m.Params.REFOPEX.shape)
    print('\t','REVOPEX\t',m.Params.REVOPEX.shape)
    print('\t','COALVOPEX\t',m.Params.COALVOPEX.shape)
    print('\t','COALFOPEX\t',m.Params.COALFOPEX.shape)
    print('\t','MAXCAP\t',m.Params.MAXCAP.shape)
    print('\t','SITEMAXCAP\t',m.Params.SITEMAXCAP.shape)
    print('\t','SITEMINCAP\t',m.Params.SITEMINCAP.shape)
    print('\t','MAXSITES\t',m.Params.MAXSITES.shape)
    print('\t','HD\t',m.Params.HD.shape)
    print('\t','RETEF\t',m.Params.RETEF.shape)
    print('\t','CONEF\t',m.Params.CONEF.shape)
    print('\t','COALOMEF\t',m.Params.COALOMEF.shape)
    print('\t','REOMEF\t',m.Params.REOMEF.shape)

    print('\t','')
    '''
    print('\t','solving...')
    
    m.solve(alp,bet,gam,DiscRate,jobCoeff,solver='cplex')

    print('\t',m.Output.Z)
    
    return m.Output, plants, m
