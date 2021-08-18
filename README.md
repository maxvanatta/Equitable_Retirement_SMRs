# Equitable_Retirement_SMRs
This is the updated Equitable Retirement project post the addition of the SMRs.  More to come (MV 8/3/2021)

# Introduction:
Download all .py files included in this Repo including:
- main.py : This file sets up the optimization objects and includes the cost values! (From ijbd, julflore000, Bhavesh Rathod, Dylan Smith)
- CoalPlants.py : This file sets up the coal plants (From ijbd, julflore000, Bhavesh Rathod, Dylan Smith)
- stateAbrevationsMap.py : (From ijbd, julflore000, Bhavesh Rathod, Dylan Smith)
- RenewableSites.py : This file sets up the renewable  (From ijbd, julflore000, Bhavesh Rathod, Dylan Smith)
- getCoalEFs : Updated by julflore000
- getReEFs.py : Updated by julflore000
- EquitableRetirement.py : This file is where the optimization, objective, constraints, etc exists. Updated from version by ijbd, julflore000, Bhavesh Rathod, Dylan Smith by MaxV.
- Optimization_Landscape : This file contains the functions which access all others to build the model and to do recursive model runs.
- Eq_Ret_Main.py : This file will run through many iterations of the model and find when there are large enough differences between results to investigate further.
- Eq_Ret_Main_Single.py : This file runs a single optimization for a gven scenario.

Download the supplemental materials folder (not here due to size) at this link: 
https://drive.google.com/file/d/1Om8A8gX3tqSgJervvPJzNeAvljGMHI7X/view?usp=sharing
Ensure the directory after unzipping this folder includes: 
- data (folder)
- EF_data (folder)
- EqSystemCosts-main (folder)
- 3_1_Generator_Y2019.xlsx
- solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv
- wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv

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

# To run the model RECURSIVELY:
Open the Eq_Ret_Main.py file in an editor

Line 36: For the first run of the file (or the first run at a new yearly duration), ensure the getNewEFs = True. After the CONEF & REOMEF files have been written for the given duration, this can be False.

Choose region, number of years (numYears), renewable radius in miles (threshDist), and whether to include SMRs (SMR_bool)

Line 38: Create initial results to refine from. This will be based upon the scenarios created in line 34 using the OL.InitialValues function.
- Outputs 'Initial_Set.csv' of the initial results in a folder created for these conditions. 

Lines 42- 47: Choose how much adaptive refining occurs:
- Steps is the number of times the refining script repeats 
- line 45: PartNumber is the number of subdivisions areas of interest are broken into. (2 means that a middle value between existing is found)
- Each step outputs the updated high level results to a csv labeled for the conditions.

Criteria tolerance is a value from 0-1 which is the percent in which ALL (costs, health, and jobs) values are judged against eachother. For example with a 0.40 value, the lowest cost value of the 8 tested points, must be within 40% of the highest cost value. This must also be true for jobs and health values as well.  A value of 0 requires all values be exactly the same, and a value of 1 means no refining will occur.

All scenarios modeled output:
- ___coalData.csv : which describes the coal plants
- ___map.html : which graphically shows the location of the changes and the RE sites.
- ___reData.csv : which describes the renewable generation results of the model.
- ___MAXCAP.csv : which is a helper file

# To run SINGLE SCENARIO:
Open the Eq_Ret_Main_Single.py file in an editor

Line 34: For the first run of the file (or the first run at a new yearly duration), ensure the getNewEFs = True. After the CONEF & REOMEF files have been written for the given duration, this can be False.

Choose scenario (objective weighting, scen), region, number of years (numYears), renewable radius in miles (threshDist), and whether to include SMRs (SMR_bool)

model output:
- ___coalData.csv : which describes the coal plants
- ___map.html : which graphically shows the location of the changes and the RE sites.
- ___reData.csv : which describes the renewable generation results of the model.
- ___MAXCAP.csv : which is a helper file

# Issues outstanding
- Jobs are still not optimal across the entire multi-state region (580 job-years over the entire model)
- Model will not run on Great Lakes due to memory and time limit


