# Equitable_Retirement_SMRs
Code and resources for equitable retirement of coal plants to renewables and SMRs building upon work from ijbd, julflores000, CodeSmith92, and rathod-b

Updated 9/18/2021 MV

# Introduction:
Download all .py files included in this Repo including:
- Cplex_main.py : This file sets up the optimization objects and includes the cost values! (From ijbd, julflore000, rathod-b, CodeSmith92). This file was previously known as main.py, but was changed for specificity as the new main.py included is what actually coordinates all the files for the model.
- CoalPlants.py : This file sets up the coal plants (From ijbd, julflore000, rathod-b, CodeSmith92)
- stateAbrevationsMap.py : (From ijbd, julflore000, rathod-b, CodeSmith92)
- RenewableSites.py : This file sets up the renewable  (From ijbd, julflore000, rathod-b, CodeSmith92)
- getCoalEFs : Updated by julflore000
- getReEFs.py : Updated by julflore000
- EquitableRetirement.py : OLD, Not used
- EquitableRetirement_CO2.py : This file is where the optimization, objective, constraints, etc exists. Updated from version by ijbd, julflore000, Bhavesh Rathod, Dylan Smith by MaxV.  NOW includes the CO2 limitation 
- Optimization_Landscape : This file contains the functions which access all others to build the model and to do recursive model runs.
- main.py : This file is runs the entire model and takes a csv input using cli.  This file should be formatted as the Inputs.csv included here.
- reEFs.csv : This file contains the employment data for the RE sites for 20 years for the midAtlantic. The full continental US will be updated soon.

Download the supplemental materials folder (not here due to size) at this link: 
https://drive.google.com/file/d/1NNVT_x_v16EArvWSdI9LvS6XqzJaUwKw/view?usp=sharing
Ensure the directory after unzipping this folder includes: 
- data (folder)
- EF_data (folder)
- EqSystemCosts-main (folder)
- 3_1_Generator_Y2019.xlsx
- solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv
- wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv
- reEFs.csv
- reEFs_cont.csv
- solar_cf_CONTINENTAL_0.5_2014.csv
- wind_cf_CONTINENTAL_0.5_2014.csv

# Environment Setup:
Ensure all dependancies are install in the current environment:
pip install pyomo
pip install cplex 
pip install folium
pip install numpy
pip install pandas
pip install plotly
pip install matplotlib
pip install kaleido
pip install geopy
pip install haversine
pip install openpyxl

Cplex requires additional installation through IBM.

# To run the model(s):
Open the 'Inputs.csv' file and edit for the given parameters.
This file should be formatted:

| Parameter | Value | Type | Hint |
| --- | --- | --- | --- |
| Region File | *MidA* or *Cont* | String | This dfeines which resource file will be accessed.  If the model is only for Mid-Atlantic, using the MidA file will be much faster |
| Years | *11* | int | This is number of years the model will be run for. 11 is the typical long range currently |
| Region | *OH_NY_NJ* of *MidA* for example | String | This can be MidA, Cont, or a list of state abbreviations seperated by _ |
| Distance | *100* | int | Radius of acceptable renewable site replacements from the Coalplants |
| SMR_Bool | *TRUE* or *FALSE* | Bool (but string due to import process) | This boolean dictates whether the SMRs can be included in the model for replacement generation.  These replacements are built on the site of the retired coal plant. |
| SMR_Only | *TRUE* or *FALSE* | Bool (but string due to import process) | Should the model only include SMRs and no REs? |
| SMR_Cost | *Low*, *Med*, *High*, or *CAPEX_FOPEX_VOPEX* | String | This can either send the Low, Median or High cost from the range of SMRs/Microreactors. Alternatively a new custom list of values can be included. |
| Discount Rate | *0.05* | float | Discount rate as a decimal |
| Single Model | *TRUE* or *FALSE* | bool | Should this model run only a single set of objectives or should it be a landscape of objective values. |
| Scenario | *0.0_0.0_1.0* | list of floats | list of three floats for the a,b,g objective weights without brackets and using _ |
| Number of Steps | *1* | int | if Single Model == FALSE, number of subdivisions for the landscape in each axis |

Important: You can adjust the values within these columns, but the Values column must retain its header of *Values* and the order of the values must remain the same.
Run the file through command line, for example:
`python main.py --csvFile Inputs.csv`

model output:
- __*FileName_*coalData.csv : which describes the coal plants and includes:
  - yearly cost of O&M [$]
  - yearly O&M Jobs [job-years]
  - Retirement Jobs [job-years]
  - yearly health damages [$]
  - yearly Generation [MWh]
  - Capacity online [MW]
  - Capacity retired [MW]
  - Additionally: Lat, Long, Name, Retirment Bool
- __*FileName_*reData.csv : which describes the renewable generation results of the model.
  - yearly cost of O&M [$]
  - Consturction Costs [$]
  - yearly O&M Jobs [job-years]
  - Construction Jobs [job-years]
  - yearly Generation [MWh]
  - Capacity online [MW]
  - Capacity retired [MW]
  - Type of generator: s: Solar, w:Wind, smr, SMR
  - Annual Capacity Factor
  - Additionally: Lat, Long, Site eligibility, Online Bool, Investment Bool
- __*FileName_*MAXCAP.csv : which is a helper file
- __*FileName_*map.html : which graphically shows the location of the changes and the RE sites.

# Getting compiled results out: 
Script and information for this coming soon! (written script, but haven't made it pretty yet/made it bug-free yet)

# Issues outstanding
- Currently National model does not go past 11 years.  Will produce more years for 20 year model runs later.

# Notes from prior iterations:
The adapting landscape method which would refine the results based upon the difference between results, hence investigating more specific areas of variation without havign to go through and edit the code too dramatically was not reliable (or at least it found the other issues within the model) and therefore is currently not available. 

The output print from prior iterations was not correctly printing the health damages. It had been printing the sum of marginal health damages for each plant rather than the total health dmamages from each plant. For example, Plant A has a $50/MWh marginal health damage and was online, therefore the value used in the sum would be $50. Now it is more accurately reporting the total damages such as Plant A generating 100 MWh and therefore causing $5,000 of health damages. This second way where the total damages are found is what the objective function already output.

