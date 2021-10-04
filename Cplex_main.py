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




def test_cplex(alp,bet,gam,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites,plants,SMR_bool,DiscRate, SMR_CAPEX, SMR_FOPEX, SMR_VOPEX, CO2Limits= 'Linear2030', jobCoeff = 1):
    ''' use sample data to test runtime and large-scale functionality of formulation '''
    print('TEST_CPLEX:')
    print('\t','getting data...')

    # costs
    if SMR_bool == True:
        SMR_num = plants.index.size 
    else: 
        SMR_num = 0
    RE_num = len(reSites) - SMR_num
    
    costs = { 'RECAPEX' : np.array([1600000]*(RE_num//2) + [1700000]*(RE_num//2) + [SMR_CAPEX]*(SMR_num)),
              'REFOPEX' : np.array([19000]*(RE_num//2) + [43000]*(RE_num//2) + [SMR_FOPEX]*(SMR_num)),
              'REVOPEX' : np.array([0]*(RE_num//2) + [0]*(RE_num//2) + [SMR_VOPEX]*(SMR_num)),
              'COALVOPEX' : np.ones(len(plants))*(4.+11*2.2),
              'COALFOPEX' : np.ones(len(plants))*40000.
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
    
    
    
    m.Params.HISTGEN = plants['HISTGEN'].values
    m.Params.CO2ME = np.full(len(m.C),1400) # THis will be replaced on a per plant basis. 1400lb/MWh currrently
    CO2_Limits = []
    if CO2Limits == 'Linear2030':
        endYear = 2030
        years = 10
        total_initial = sum(m.Params.CO2ME*m.Params.HISTGEN)
        for i in range(numYears):
            CO2_Limits.append(total_initial-(total_initial/years)*i)
        if numYears> years+1:
            CO2_Limits = CO2_Limits+ ([0]*(numYears-(years+1)))         
    if CO2Limits == 'Linear2035':
        endYear = 2035
        years = 15
        total_initial = sum(m.Params.CO2ME*m.Params.HISTGEN)
        for i in range(numYears):
            CO2_Limits.append(total_initial-(total_initial/years)*i)
        if numYears> years+1:
            CO2_Limits = CO2_Limits+ ([0]*(numYears-(years+1)))
            
    
    m.Params.CO2Limits = np.array(CO2_Limits)

    
    m.Params.COALCAP = plants['Coal Capacity (MW)'].values
    m.Params.CF = reSites['Annual CF'].values
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
