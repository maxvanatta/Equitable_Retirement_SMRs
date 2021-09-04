# All imports
import pandas as pd
import numpy as np
from getReEFs import batchReEFs

solFileName = 'solar_cf_CONTINENTAL_0.5_2014.csv' # 
winFileName = 'wind_cf_CONTINENTAL_0.5_2014.csv' 


# How many years will the analysis run for?
numYears = 1

CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2022)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2023)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2024)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2025)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2026)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2027)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2028)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2029)
CONEF,REOMEF,res = batchReEFs(solFileName,winFileName,numYears, startYear = 2030)