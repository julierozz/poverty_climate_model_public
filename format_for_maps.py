from pandas import read_csv,isnull
import matplotlib.pyplot as plt
import os
from lib_for_growth_model import *
from lib_for_analysis import change_name

nameofthisround = 'withskillp_jan23'

model=os.getcwd()
wdata="{}/world_results_{}/".format(model,nameofthisround)
gisfolder="C:\\Users\\julierozenberg\\Documents\\gis\\"

giscodes=read_csv("giscodes.csv")

impacts=read_csv(wdata+"cc_impacts.csv")

forthemap=giscodes.copy()
			
for id in impacts.index:
	cc=impacts.ix[id,'countrycode']
	if sum(giscodes['ISO 3166-1 A3']==cc)==1:
		forthemap.ix[forthemap['ISO 3166-1 A3']==cc,'opt']=impacts.ix[id,'below125pcopt']
		forthemap.ix[forthemap['ISO 3166-1 A3']==cc,'pess']=impacts.ix[id,'below125pcpess']
	elif sum(giscodes['ISO 3166-1 A3']==cc)==0:
		print("No correspondence for "+cc)
	elif sum(giscodes['ISO 3166-1 A3']==cc)>1:
		print("More than one correspondence for "+cc)
		
forthemap["opt"].fillna(-1, inplace=True)
forthemap["pess"].fillna(-1, inplace=True)
		
forthemap.to_csv(gisfolder+"forthemap_cc_impacts.csv")
