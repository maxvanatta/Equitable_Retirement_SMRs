import numpy as np
import pandas as pd 
from geopy.geocoders import Nominatim
from stateAbrevationsMap import us_state_abbrev
import math


def getDeclineFactors(plantDataset,year, declineFactor = "avg"):
    """returns a list that contains the decline factor for each plant in that year

    Args:
        plantDataset (panda): plants considered
        year (int): year of study for EFs
        declineFactor (optional): base case set to true where we average the utility and residential solar decline factors (can also input "avg"), if we only want one or the other simply put in "res" or "util"
    """
    if year <= 2020:
        #just return a list of decline factors that are zero (no decline) if year <= 2020
        return [[0,0]] * len(plantDataset)
    #read in decline factors into main dataset, contains capital expenditure declines (installation/construction) & operational expenditures (O&M)
    declineFactorDataset = pd.read_excel(("EF_data/declineFactors.xlsx"))
        
    #round to the nearest multiple of value below (in this case it is 5, so we have values for 2020, 2025, 2030....2050)
    roundOn = 5
    
    #getting lower decline factor- from this lower decline year we also know the upper decline year as each column is only in 5 year increments
    roundDownYear = roundOn * round(math.floor(year/roundOn))
    roundUpYear = roundDownYear + 5
    
    #*** since all plants are treated as getting the same decline factor we really only need to compute the decline factors for wind & solar once
     
    ############# SOLAR DECLINE FACTOR CALCULATION START #############
    
    #setting bool values for either res, utility, or average the decline factors based on input parameter
    if declineFactor == "avg":
        resBool = True
        utilBool = True
    elif declineFactor == "res":
        resBool = True
        utilBool = False
    elif declineFactor == "util":
        utilBool = True
        resBool = False
    else:
        raise ValueError("decline factor input parameter should either be: 'avg', 'res', or 'util'")
    
    
    
    #if solar plant we will be getting the bottom two rows which contain utility and residential solar- leading to an average
    
    #get out capital expenditures average of utility(row 1) and residential (row 2) for solar bottom year
    capexBottomDeclineEF = (declineFactorDataset["CAPEX " +(str(roundDownYear))][1]*utilBool + declineFactorDataset["CAPEX " +(str(roundDownYear))][2]*resBool)/(utilBool+resBool)
    
    #repeat for rounded up year
    capexTopDeclineEF = (declineFactorDataset["CAPEX " +(str(roundUpYear))][1]*utilBool + declineFactorDataset["CAPEX " +(str(roundUpYear))][2]*resBool)/(utilBool+resBool)
    
    #gives us the annual change in decline factor from previous rounded year to next, uses linear interpolation
    annualCapexDecline = (capexTopDeclineEF-capexBottomDeclineEF)/roundOn

    #calculate final CAPEX decline factor, will be the lower bound + (years from bottom)*annualDecline
    solarCapexDeclineFactor = capexBottomDeclineEF + annualCapexDecline* (year-roundDownYear)
    
    #repeat the same process as above but with OPEX-operational expenditures
    opexBottomDeclineEF = (declineFactorDataset["OPEX " +(str(roundDownYear))][1]*utilBool + declineFactorDataset["OPEX " +(str(roundDownYear))][2]*resBool)/(utilBool+resBool)
    
    opexTopDeclineEF = (declineFactorDataset["OPEX " +(str(roundUpYear))][1]*utilBool + declineFactorDataset["OPEX " +(str(roundUpYear))][2]*resBool)/(utilBool+resBool)
    
    annualOpexDecline = (opexTopDeclineEF-opexBottomDeclineEF)/roundOn

    #calculate final OPEX decline factor, will be the lower bound + (years from bottom)*annualDecline
    solarOpexDeclineFactor = opexBottomDeclineEF + annualOpexDecline* (year-roundDownYear)
    
    ############# SOLAR DECLINE FACTOR CALCULATION END #############


    ############# WIND DECLINE FACTOR CALCULATION START #############
    
    #if wind plant we will be getting only the top row of data
    #get out capital expenditures 
    capexBottomDeclineEF = declineFactorDataset["CAPEX " +(str(roundDownYear))][0]
    
    #repeat for rounded up year
    capexTopDeclineEF = declineFactorDataset["CAPEX " +(str(roundUpYear))][0]
    
    #gives us the annual change in decline factor from previous rounded year to next, uses linear interpolation
    annualCapexDecline = (capexTopDeclineEF-capexBottomDeclineEF)/roundOn

    #calculate final CAPEX decline factor, will be the lower bound + (years from bottom)*annualDecline
    windCapexDeclineFactor = capexBottomDeclineEF + annualCapexDecline* (year-roundDownYear)

    #repeat same process as above for OPEX
    opexBottomDeclineEF = declineFactorDataset["OPEX " +(str(roundDownYear))][0]
    
    opexTopDeclineEF = declineFactorDataset["OPEX " +(str(roundUpYear))][0]
    
    annualOpexDecline = (opexTopDeclineEF-opexBottomDeclineEF)/roundOn
    
    windOpexDeclineFactor = opexBottomDeclineEF + annualOpexDecline* (year-roundDownYear)        
    
    ############# WIND DECLINE FACTOR CALCULATION END #############

    #create final list that mirrors the plant dataset but holds [CAPEX decline factor, OPEX decline factor
    declineFactorList = []
    #run through each of the plants technology
    for plantDetails in plantDataset:
        #should either be a "S" for solar or "W" for wind
        plantTech = plantDetails[2]
        
        if plantTech == "S": 
            #return the solar decline factors, first the capital then the operational
            declineFactorList.append([solarCapexDeclineFactor,solarOpexDeclineFactor])
        else:
            #return the wind decline factors, first the capital then the operational
            declineFactorList.append([windCapexDeclineFactor,windOpexDeclineFactor])  
        
    #once done we are left with a list of the respective decline factors
    return declineFactorList
 
 
            
def getReEFs(rePlantList,year,optionalDeclineFactor = "avg"):
    """returns the employment factor for each lat long in panda dataframe, need to specify wind or solar
    At state level dimension

    Args:
        rePlantList ([list]): [[lat,long, and either a "S" or "W" for wind or solar]]
    """ 
    # initialize Nominatim API  
    geolocator = Nominatim(user_agent="renewableEnergyEFs")
    
    #read in solar and wind EF data for each state
    renewableEFsDataset = pd.read_excel("EF_data/reEFs.xlsx")

    
    #getting the total decline factors for that year and for all plants
    declineFactorList = getDeclineFactors(rePlantList,year,optionalDeclineFactor)
    #creation of final renewable EF dict, will convert to panda dataframe at end
    reEFDict = dict()
    
    #run through each of the lat longs and RE points
    for plantDetails,declineFactor in zip(rePlantList,declineFactorList):
        lat = plantDetails[0]
        long = plantDetails[1]
        #get out the full details of location
        location = geolocator.reverse(f"{lat},{long}")
        
        state = location.raw['address'].get('state', '')
        
        #state Abreviations
        stateAbrev = us_state_abbrev[state]
        
        plantRowLocation = np.where(renewableEFsDataset["State"] == stateAbrev)[0][0]
        #if asking for solar plant, returns construction and then O&M EFs, else if wind then return single EF
        if plantDetails[2] == "S":
            #multiply by (1-declineFactor) of either Capex-0 position or OPEX-1 position
            reEFDict[lat+","+long+","+plantDetails[2]] = [renewableEFsDataset["PV Con/Instl EF"][plantRowLocation]* (1-declineFactor[0]), renewableEFsDataset["PV O&M EF"][plantRowLocation]*(1-declineFactor[1])]
        else:
            #for wind right now I am taking the average of the two decline factors-WILL MOST LIKELY CHANGE
            reEFDict[lat+","+long+","+plantDetails[2]] = [renewableEFsDataset["WT Con/Instl EF"][plantRowLocation] * (1-(declineFactor[0])),renewableEFsDataset["WT O&M EF"][plantRowLocation] * (1-(declineFactor[1]))]

    #transforming renewable EF dict into panda dataframe 
    pandaReEF = pd.DataFrame.from_dict(reEFDict,orient='index',columns=['Con/Instl EF','O&M EF'])
    
    return pandaReEF


#Updated with the older batchReEFs function- MV 08092021

def batchReEFs(solarFile,windFile,year):
    
    s = pd.read_csv(solarFile,index_col=0)
    sCols = s.columns
    
    w = pd.read_csv(windFile,index_col=0)
    wCols = w.columns
    
    # input arg format [["42.360081","-71.058884", "S"],["42.360081","-71.058884", "W"]]
    
    # print(sCols[0].split()[0])
    
    reEFs = []
    
    for y in range(year):
        print('\tFetching RE EFs for year ', 2020+y)
        mList = []
        for cell in range(sCols.shape[0]):
            l_s = []
            l_s.append(sCols[cell].split()[0])
            l_s.append(sCols[cell].split()[1])
            l_s.append('S')
            mList.append(l_s)
        
        for cell in range(wCols.shape[0]):
            l_w = []
            l_w.append(wCols[cell].split()[0])
            l_w.append(wCols[cell].split()[1])
            l_w.append('W')
            mList.append(l_w)
        renewableEFs = getReEFs(mList,2020+y)
        renewableEFs['Year'] = 2020+y
        reEFs.append(renewableEFs)
    
    res = pd.concat(reEFs)
    res.to_csv('reEFs.csv')
    
    CONEF = np.array(res['Con/Instl EF'])
    REOMEF = np.array(res['O&M EF'])
    
    return CONEF, REOMEF