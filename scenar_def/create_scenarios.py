from pyDOE import *
from pandas import read_csv,DataFrame,read_stata
import numpy as np
import os

scenar_folder=os.getcwd()
model=scenar_folder.strip('scenar_def')

# numCases = 500
# ranges=read_csv("scenarios_ranges.csv")
# numUncertainties=len(ranges)
# lhsample= lhs(numUncertainties,samples=numCases,criterion="corr")
# lhsample= read_csv("lhs-table-200.csv")
# lhssample=lhsample.drop("Unnamed: 0",axis=1)
# scenarios=lhssample.values*np.diff(ranges[['min','max']].values).T+ranges['min'].values
# scenarios=DataFrame(scenarios,columns=ranges['variable'])
# scenarios.to_csv("scenarios-200.csv")

# for wbreg in list(['LAC','MNA','ECA','SSA','SAS','EAP']):
    # ranges=read_csv("scenarios_"+wbreg+"_ranges.csv")
	# scenarios.to_csv("scenarios_"+wbreg+".csv")
	
uncertainties=['shareag','sharemanu','shareemp','grserv','grag','grmanu','skillpserv','skillpag','skillpmanu','p','b','voice']
numUncertainties=len(uncertainties)
lhsample= lhs(numUncertainties,samples=300,criterion="corr")
# lhsample= read_csv("lhs-table-200.csv")
# lhssample=lhsample.drop("Unnamed: 0",axis=1)
# scenarios=lhsample*np.diff(ranges[['min','max']]).T+ranges['min'].values
scenarios=DataFrame(lhsample,columns=uncertainties)
# scenarios.to_csv("impact-scenarios-300.csv",index=False)
scenarios.to_csv("lhs-table-300-12uncertainties.csv",index=False)