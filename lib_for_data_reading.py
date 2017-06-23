from pandas import read_excel,concat,Series,DataFrame,read_csv,isnull,notnull
import numpy as np
from scipy import interpolate
import os
from perc import wp
import re

#########################################################################################################################################
## This library was created to deal with the I2D2 surveys database. It reads the data and converts the surveys into the finalhhframe.
#########################################################################################################################################

def some(rawdataframe, n):
    return rawdataframe.ix[np.random.choice(rawdataframe.index, n)]
	
def split_big_dframe(finalhhframe,hhcat):
	a=finalhhframe[['weight','reg02']].groupby('reg02').apply(lambda x:x['weight'].count())
	b=DataFrame(a,columns=['count'])
	c=b.sort(columns=['count'],ascending=False)
	bool1 = c.cumsum()<c.cumsum()['count'].iloc[-1]/2
	list_reg = list(c[bool1].dropna().index)
	finalhhframe1=finalhhframe.ix[finalhhframe['reg02'].isin(list_reg),:]
	finalhhframe2=finalhhframe.ix[~finalhhframe['reg02'].isin(list_reg),:]
	int_columns=['children','old','decile']+['cat{}workers'.format(thecat) for thecat in hhcat['hhcat'].unique()]
	finalhhframe1=merges_rows_bis(int_columns,finalhhframe1)
	finalhhframe2=merges_rows_bis(int_columns,finalhhframe2)
	return finalhhframe1,finalhhframe2
	
def convert_int_to_float(rawdataframe):
	if rawdataframe["idh"].dtype=='O':
		rawdataframe[['wgthh2007','Y']]=rawdataframe[['wgthh2007','Y']].astype(float)
	else:
		rawdataframe[['idh','wgthh2007','Y']]=rawdataframe[['idh','wgthh2007','Y']].astype(float)
	return rawdataframe
	
def del_missing_Y(rawdataframe):
	"delete rows with missing income and distribute weights"
	rawdataframe=rawdataframe.drop(rawdataframe.ix[isnull(rawdataframe["wgthh2007"]),:].index)
	missing=isnull(rawdataframe["Y"])
	add2wgt=rawdataframe.ix[missing,'wgthh2007'].sum()
	scal=rawdataframe['wgthh2007'].sum()/(rawdataframe['wgthh2007'].sum()-add2wgt)
	rawdataframe=rawdataframe.drop(rawdataframe.ix[missing,:].index)
	rawdataframe['wgthh2007']=rawdataframe['wgthh2007']*scal
	rawdataframe['skilled'].fillna(0, inplace=True)
	rawdataframe['skilled']=rawdataframe['skilled'].astype(bool)
	return rawdataframe
	
def find_indus(rawdataframe,industry_list):
	rawdataframe["newindus"]=np.nan
	if rawdataframe["industry"].dtype=='int64':
		rawdataframe["industry"]=rawdataframe["industry"].astype(float)
	if rawdataframe["industry"].dtype=='float64':
		rawdataframe.ix[notnull(rawdataframe["industry"]),"industry"]=rawdataframe.ix[notnull(rawdataframe["industry"]),"industry"].apply(lambda x:str(int(x)))
	for ii in industry_list.index:
		rawdataframe.ix[rawdataframe['industry']==industry_list.loc[ii,'indata'],'newindus']=industry_list.loc[ii,'industrycode']
	return rawdataframe
	
def create_age_col(rawdataframe):
	"sort children from adults and old people"
	rawdataframe['isold']=(rawdataframe['age']>64)
	rawdataframe['isanadult']=(rawdataframe['age']>14)&(rawdataframe['age']<65)
	rawdataframe['isachild']=(rawdataframe['age']<15)
	return rawdataframe
	
def create_gender_col(rawdataframe):
	"identifies gender for adults only"
	rawdataframe['isawoman']=((rawdataframe['isanadult'])&(rawdataframe['gender']=='Female'))
	rawdataframe['isaman']=((rawdataframe['isanadult'])&(rawdataframe['gender']=='Male'))
	return rawdataframe

def associate_indus_to_head(rawdataframe,hhcat):
	#sort by head to be able to have the spouse first when the head does not have an industry
	rawdataframe=rawdataframe.sort(columns=['idh','head'],ascending=False)
	#solve pbs with duplicate head of household or missing head
	checkheads=rawdataframe.ix[:,['idh','ishead']].groupby('idh',sort=False).apply(lambda x:x['ishead'].sum())
	#if no head, replaced by spouse first and then adult
	subset=rawdataframe.ix[(rawdataframe['idh'].isin(checkheads.ix[checkheads==0].index))&(1-rawdataframe['isachild']),['head','idh','ishead']]
	hop=subset.groupby('idh')
	rawdataframe.loc[rawdataframe['idh'].isin(hop['idh'].head(1).values),'ishead']=1
	#if more than one head, drop duplicates in the hh dataframe later.
	#associate a category to each person based on hhcat
	for group in rawdataframe.groupby(list(categories.values)):
		therow=(hhcat[list(categories.values)].values==np.array(group[0])).all(1)
		cat=hhcat.ix[therow,'hhcat'].values
		rawdataframe.loc[group[1].index,'headcat']=cat
	return rawdataframe
	
def indus_to_bool(rawdataframe,industringlist,useskill=False):
	for industring in industringlist:
		newstring='is'+industring
		rawdataframe[newstring]=(rawdataframe['isanadult']&(rawdataframe['newindus']==industring))+0
	rawdataframe['noindustry']=(rawdataframe['isanadult']&isnull(rawdataframe['newindus']))+0
	return rawdataframe
	
def associate_cat2people(rawdataframe,hhcat):
	categories=hhcat.columns[1::]
	for group in rawdataframe.groupby(list(categories.values)):
		therow=(hhcat[list(categories.values)].values==np.array(group[0])).all(1)
		cat=hhcat.ix[therow,'hhcat'].values
		rawdataframe.loc[group[1].index,'cat']=int(cat)
	rawdataframe['cat']=rawdataframe['cat'].astype('int64')
	for thecat in hhcat['hhcat'].unique():
		catstring='iscat{}'.format(thecat)
		rawdataframe[catstring]=(rawdataframe['cat']==thecat)&(rawdataframe['isanadult'])
	return rawdataframe
	
def deal_with_head_issues(hhdataframe):
	#look for households where someone else than the head has an industry (but not the head)
	otherworkers=(hhdataframe['noindustry']==1)&((hhdataframe['agworkers']>0)|(hhdataframe['manuworkers']>0)|(hhdataframe['servworkers']>0))
	sub=rawdataframe.ix[(rawdataframe['idh'].isin(hhdataframe.ix[otherworkers,:].index))&(rawdataframe['isanadult'])&(rawdataframe['headcat']<7),['head','idh','headcat']].copy()
	#takes the category of the spouse or if not, the first member of the hh with an industry
	hop=sub.groupby('idh')
	newcats=hop['headcat'].head(1)
	theindexes=hop['idh'].head(1).values
	hhdataframe.loc[theindexes,'headcat']=newcats.values
	return hhdataframe

def correct_zeroY(rawdataframe):
	min_Y=min(rawdataframe.ix[rawdataframe['Y']>0,'Y'])
	rawdataframe.ix[rawdataframe['Y']==0,'Y']=min_Y
	return rawdataframe
	
def sumoverhh(hhdataframe,rawdataframe,hhcolstring,rawstring):
	hhdataframe.loc[hhdataframe.index,hhcolstring]=rawdataframe.ix[:,['idh',rawstring]].groupby('idh',sort=False).apply(lambda x:x[rawstring].sum())
	return hhdataframe
	
def intensify_cat_columns(hhdataframe,rawdataframe,hhcat):
	for thecat in hhcat['hhcat'].unique():
		catstring='iscat{}'.format(thecat)
		intstring='cat{}workers'.format(thecat)
		hhdataframe=sumoverhh(hhdataframe,rawdataframe,intstring,catstring)
	return hhdataframe
	
def match_deciles(hhdataframe,deciles):
	hhdataframe.loc[hhdataframe['Y']<=deciles[0],'decile']=1
	for j in np.arange(1,len(deciles)):
		hhdataframe.loc[(hhdataframe['Y']<=deciles[j])&(hhdataframe['Y']>deciles[j-1]),'decile']=j+1
	return hhdataframe
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data
	
def get_pop_description(rawdataframe,hhcat,industry_list,listofdeciles,issplit=False):
	"Extracts the three main components of our pb from the country dataframe: characteristics is a matrix that has all important household characteristics in columns and hh heads in lines. weights is the weight of each hh head and pop_description is a summary of total population: number of children, skilled people etc"
	rawdataframe=rawdataframe.drop(rawdataframe.ix[isnull(rawdataframe["wgthh2007"]),:].index)
	rawdataframe=convert_int_to_float(rawdataframe)
	rawdataframe=del_missing_Y(rawdataframe)
	rawdataframe=correct_zeroY(rawdataframe)
	rawdataframe['ishead']=rawdataframe['head']=="Head of household"
	rawdataframe=find_indus(rawdataframe,industry_list)
	rawdataframe=create_age_col(rawdataframe)
	rawdataframe=indus_to_bool(rawdataframe,['serv','manu','ag'])
	rawdataframe=associate_cat2people(rawdataframe,hhcat)
	# rawdataframe['isskillworker']=rawdataframe['skill']&(rawdataframe['isanadult'])&(~rawdataframe['noindustry'])
	# rawdataframe=associate_indus_to_head(rawdataframe,hhcat)
	#keep households instead of people.
	hhdataframe=rawdataframe.groupby('idh').apply(lambda x:x.head(1))
	hhdataframe.index=hhdataframe['idh'].values
	hhdataframe.loc[hhdataframe.index,'totY']=rawdataframe.loc[:,['idh','Y']].groupby('idh',sort=False).apply(lambda x:x['Y'].sum())
	# hhdataframe['meanY']=hhdataframe['Y']
	#deletes rows corresponding to the same household
	hhdataframe=hhdataframe.drop_duplicates(['idh'])
	#calculate the number of children, adults and old people in each hh
	hhdataframe=sumoverhh(hhdataframe,rawdataframe,'totwgt','wgthh2007')
	hhdataframe=sumoverhh(hhdataframe,rawdataframe,'children','isachild')
	hhdataframe=sumoverhh(hhdataframe,rawdataframe,'adults','isanadult')
	hhdataframe=sumoverhh(hhdataframe,rawdataframe,'old','isold')
	#calculate the number of workers in each category defined in hhcat
	hhdataframe=intensify_cat_columns(hhdataframe,rawdataframe,hhcat)	
	# hhdataframe=deal_with_head_issues(hhdataframe)
	#calculates the decile of each hh and adds a column
	deciles=wp(reshape_data(hhdataframe['Y']),reshape_data(hhdataframe['totwgt']),listofdeciles,cum=False)
	hhdataframe=match_deciles(hhdataframe,deciles)
	#columns to keep for description of population
	if not issplit:
		int_columns=['children','old','decile']+['cat{}workers'.format(thecat) for thecat in hhcat['hhcat'].unique()]
		finalhhframe=merges_rows(int_columns,hhdataframe)
	else:
		int_columns=['idh','Y','totY','reg02','children','old','decile']+['cat{}workers'.format(thecat) for thecat in hhcat['hhcat'].unique()]
		finalhhframe=hhdataframe[int_columns]
		finalhhframe['totweight']=hhdataframe['totwgt']
		finalhhframe['weight']=hhdataframe['wgthh2007']
		finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
		finalhhframe['nbpeople'].fillna(0, inplace=True)
	finalhhframe = finalhhframe.drop('totweight')
	return finalhhframe
	
def merges_rows(int_columns,hhdataframe):
	"merges the rows that have the same characteristics for the int_columns variables, to reduce the number of households. Weights are summed and I take the mean income between similar households"
	inter_wh=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totwgt'].sum())
	inter_w=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['wgthh2007'].sum())
	indexes=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['idh'].head(1))
	inter_c=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x[int_columns].head(1))
	inter_it=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totY'].mean())
	inter_i=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['Y'].mean())
	finalhhframe=DataFrame(inter_c.values,columns=int_columns,index=indexes.values)
	# finalhhframe.drop('decile', axis=1, inplace=True)
	finalhhframe['totweight']=inter_wh.values
	finalhhframe['weight']=inter_w.values
	finalhhframe['totY']=inter_it.values
	finalhhframe['Y']=inter_i.values
	finalhhframe['idh']=indexes.values
	finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
	return finalhhframe
	
def merges_rows_bis(int_columns,hhdataframe):
	"merges the rows that have the same characteristics for the int_columns variables, to reduce the number of households. Weights are summed and I take the mean income between similar households"
	inter_wh=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totweight'].sum())
	inter_w=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['weight'].sum())
	indexes=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['idh'].head(1))
	inter_c=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x[int_columns].head(1))
	inter_it=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['totY'].mean())
	inter_i=hhdataframe.groupby(int_columns,sort=False).apply(lambda x: x['Y'].mean())
	finalhhframe=DataFrame(inter_c.values,columns=int_columns,index=indexes.values)
	# finalhhframe.drop('decile', axis=1, inplace=True)
	finalhhframe['totweight']=inter_wh.values
	finalhhframe['weight']=inter_w.values
	finalhhframe['totY']=inter_it.values
	finalhhframe['Y']=inter_i.values
	finalhhframe['idh']=indexes.values
	finalhhframe['nbpeople']=finalhhframe['totweight']/finalhhframe['weight']
	return finalhhframe

def get_pop_data_from_UN(UNpop,countrycode,theyear):
	"get the description of the country's population in the projected year from UN/WB data"
	year='YR'+str(theyear)
	select=(UNpop['Country_Code']==countrycode)&((UNpop['Time_ValueCode']==year))
	country_pop=UNpop.ix[select,:].pivot(index='Time_ValueCode',columns='Indicator_Code',values='Value')
	pop_tot=country_pop.ix[year,'SP.POP.TOTL']
	pop_0014=country_pop.ix[year,'SP.POP.0014.TO']
	pop_1564=country_pop.ix[year,'SP.POP.1564.TO']
	pop_65up=country_pop.ix[year,'SP.POP.65UP.TO']
	return pop_tot,pop_0014,pop_1564,pop_65up
    
def get_pop_data_from_ssp(ssp_data,ssp,year,countrycode):
	model='IIASA-WiC POP'
	if ssp==4: ssp='4d'
	scenario="SSP{}_v9_130115".format(ssp)
	selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)&(ssp_data['REGION']==countrycode)
	pop_tot=ssp_data.ix[selection&(ssp_data['VARIABLE']=="Population"),str(year)]
	pop_0014=0
	pop_1564=0
	pop_65up=0
	for gender in ['Male','Female']:
		for age in ['0-4','5-9','10-14']:
			var="Population|{}|Aged{}".format(gender,age)
			pop_0014+=ssp_data.ix[selection&(ssp_data['VARIABLE']==var),str(year)].values
		for age in ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64']:
			var="Population|{}|Aged{}".format(gender,age)
			pop_1564+=ssp_data.ix[selection&(ssp_data['VARIABLE']==var),str(year)].values
		for age in ['65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+']:
			var="Population|{}|Aged{}".format(gender,age)
			pop_65up+=ssp_data.ix[selection&(ssp_data['VARIABLE']==var),str(year)].values
	skilled_adults=0
	for gender in ['Male','Female']:
		for age in ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64']:
			for edu in ['Secondary Education','Tertiary Education']:
				var="Population|{}|Aged{}|{}".format(gender,age,edu)
				skilled_adults+=ssp_data.ix[selection&(ssp_data['VARIABLE']==var),str(year)].values
	pop_tot=pop_tot*10**6
	pop_0014=pop_0014*10**6
	pop_1564=pop_1564*10**6
	pop_65up=pop_65up*10**6
	skilled_adults=skilled_adults*10**6
	return pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults
	
def rescale_to_2007(UNpop,countrycode,year,characteristics,weight,income):
	"rescales the population to match 2007 description but keep aggregate income constant"
	pop_description=calc_pop_desc(characteristics,weight)
	pop_tot,pop_0014,pop_1564,pop_65up=get_pop_data_from_UN(UNpop,countrycode,year)
	ini_pop_desc=pop_description[['children','adults','old']]
	ini_pop_desc['income']=GDP(income,weight)
	new_pop_desc=ini_pop_desc.copy()
	new_pop_desc[['children','adults','old']]=[pop_0014,pop_1564,pop_65up]
	charac=characteristics[['children','adults','old']]
	charac['income']=income
	new_weights,result=build_new_weights(ini_pop_desc,new_pop_desc,charac,weight,ismosek=True)
	return new_weights
	
def country2r32(codes_tables,countrycode):
	r32='R32{}'.format(codes_tables.loc[codes_tables['ISO']==countrycode,'R32'].values[0])
	return r32

def get_gdp_growth(ssp_data,year,ssp,r32,ini_year):
	model='OECD Env-Growth'
	scenario="SSP{}_v9_130325".format(ssp)
	selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)&(ssp_data['REGION']==r32)&(ssp_data['VARIABLE']=='GDP|PPP')
	if ini_year<2010:
		y1=ssp_data.ix[selection,'2005'].values[0]
		y2=ssp_data.ix[selection,'2010'].values[0]
		f=interpolate.interp1d([2005,2010], [y1,y2],kind='slinear')
	else:
		y1=ssp_data.ix[selection,'2010'].values[0]
		y2=ssp_data.ix[selection,'2015'].values[0]
		f=interpolate.interp1d([2010,2015], [y1,y2],kind='slinear')
	gdp_ini=f(ini_year)
	gdp_growth=ssp_data.ix[selection,str(year)].values[0]/gdp_ini
	return gdp_growth


def create_correct_data(countrycode,data_gidd_csv,hhcat,industry_list,issplit=False):
	rawdataframe=read_csv(data_gidd_csv+countrycode+"_GIDD.csv")
	listofdeciles=np.sort(np.append(np.arange(0.1, 1.1, 0.1),[0.99]))
	if issplit:
		finalhhframe=get_pop_description(rawdataframe,hhcat,industry_list,listofdeciles,issplit=True)
	else:
		finalhhframe=get_pop_description(rawdataframe,hhcat,industry_list,listofdeciles)
	if finalhhframe["idh"].dtype=='O':
		finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
		finalhhframe["idh"]=finalhhframe["idh"].astype(float)
		finalhhframe.index=finalhhframe['idh']
	return finalhhframe
	
def filter_country(countrycode,all_surveys,codes):
	if len(codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'])>0:
		wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
	else:
		return None
	if wbreg=='YHI':
		return None
	if countrycode in ["IRQ","COL","BGR","TUR","IND2","IND1","COL1","COL2"]:
		return None
	else:
		if (countrycode=='COL')|(countrycode=='IND'):
			toobig=True
		else:
			toobig=False
		finalhhframe=load_correct_data(all_surveys[countrycode])
		countrycode=correct_countrycode(countrycode)
		return finalhhframe,countrycode,toobig,wbreg
		
def load_correct_data(finalhhframe):
	if finalhhframe["idh"].dtype=='O':
		finalhhframe["idh"]=[re.sub('[^0-9a]+', '3', x) for x in finalhhframe["idh"]]
		finalhhframe["idh"]=finalhhframe["idh"].astype(float)
		finalhhframe.index=finalhhframe['idh']
	if 'hhweights' in finalhhframe.columns:
		finalhhframe.rename(columns={'hhweights':'weight'},inplace=True)
	return finalhhframe

		
def get_scenario_dataframe(outputs,countrycode,year,scenarname):
	finalhhframe = read_csv(outputs+"futurehhframe{}_{}_{}.csv".format(countrycode,year,scenarname))
	if (countrycode=='COL')|(countrycode=='IND'):
		toobig=True
	else:
		toobig=False
	return finalhhframe,countrycode,toobig
	
def correct_countrycode(countrycode):
	'''
	Corrects countrycodes in the database that don't correspond to official 3 letters codes.
	'''
	if countrycode=='TMP':
		countrycode='TLS'
	if countrycode=='ZAR':
		countrycode='COD'
	if countrycode=='ROM':
		countrycode='ROU'
	return countrycode

def reverse_correct_countrycode(countrycode):
	'''
	Corrects countrycodes in the database that don't correspond to official 3 letters codes.
	'''
	if countrycode=='TLS':
		countrycode='TMP'
	if countrycode=='COD':
		countrycode='ZAR'
	if countrycode=='ROU':
		countrycode='ROM'
	return countrycode
