import sys
from pandas import Series,DataFrame,read_csv
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

data2day      = 30
year          = 2030
ini_year      = 2007

nameofthisround = 'may2016'

model             = os.getcwd()
data              = model+'/data/'
scenar_folder     = model+'/scenar_def/'
finalhhdataframes = model+'/finalhhdataframes_new/'
ssp_folder        = model+'/ssp_data/'
pik_data          = model+'/pik_data/'
iiasa_data        = model+'/iiasa_data/'
data_gidd_csv     = model+'/data_gidd_csv_v4/'

codes         = read_csv('wbccodes2014.csv')
codes_tables  = read_csv(ssp_folder+"ISO3166_and_R32.csv")
ssp_pop       = read_csv(ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
ssp_gdp       = read_csv(ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
hhcat         = read_csv(scenar_folder+"hhcat_v2.csv")
industry_list = read_csv(scenar_folder+"list_industries.csv")



list_csv=os.listdir(data_gidd_csv)
list_countries  = [re.search('(.*)_GIDD.csv', s).group(1) for s in list_csv]


for countrycode in list_countries:
	# if not os.path.isfile("finalhhdataframes/"+countrycode+"_finalhhframe.csv"):
	wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
	if wbreg == 'YHI':
		continue
	finalhhframe = create_correct_data(countrycode,data_gidd_csv,hhcat,industry_list)
	if finalhhframe["idh"].dtype=='O':
		finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
		finalhhframe["idh"]=finalhhframe["idh"].astype(float)
	finalhhframe.to_csv("finalhhdataframes/"+countrycode+"_finalhhframe.csv",encoding='utf-8',index=False)