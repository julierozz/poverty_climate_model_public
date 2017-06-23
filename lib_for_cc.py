import numpy as np
from pandas import read_csv, Series
from lib_for_growth_model import *

def get_pik_data(pik_data,ssp,cc,year):
	'''PIK data on food prices -- not used at the moment. We use IIASA prices'''
	baseline = read_csv("{}cof_no_climate_change_SSP{}.csv".format(pik_data,ssp))
	rcp8p5   = read_csv("{}cof_rcp8p5_SSP{}.csv".format(pik_data,ssp))
	column   = "y{}.1".format(year)
	price_increase = rcp8p5.ix[rcp8p5['country']==cc,column]/baseline.ix[baseline['country']==cc,column]-1
	if len(price_increase.values)>0:
		return price_increase.values[0]
	else:
		return 0.4
		
def get_shares_outside(futurehhframe):
	'''Ad hoc. Share of people working outside for each category. Need better data here'''
	shares_outside = Series(index=['ag','manu','serv'])
	shares_outside['ag']   = 0.8
	shares_outside['manu'] = 0.5
	shares_outside['serv'] = 0.3
	if np.average(futurehhframe['Y'],weights=futurehhframe['weight']*futurehhframe['nbpeople'])>10000:
		shares_outside['manu'] = 0.1
		shares_outside['serv'] = 0.05
	return shares_outside
	
def price_increase(wbreg,ssp,price_increase_reg,food_cursor):
	'''applies IIASA price increases'''
	select = (price_increase_reg.Macro=="SSP{}".format(ssp))&(price_increase_reg.Reg==wbreg)
	price_change = (1-food_cursor)*price_increase_reg.ix[select&(price_increase_reg.min_max=="min"),"priced"].values[0]+food_cursor*price_increase_reg.ix[select&(price_increase_reg.min_max=="max"),"priced"].values[0]
	return price_change
	
def food_impact_on_farmers(wbreg,ssp,price_increase_reg,food_cursor):
	'''calculates the revenue changes for farmers. Based on IIASA model and consistent with food price increases.'''
	select = (price_increase_reg.Macro=="SSP{}".format(ssp))&(price_increase_reg.Reg==wbreg)
	rev_change = (1-food_cursor)*price_increase_reg.ix[select&(price_increase_reg.min_max=="min"),"pqd"].values[0]+food_cursor*price_increase_reg.ix[select&(price_increase_reg.min_max=="max"),"pqd"].values[0]
	return rev_change


def food_price_impact(futurehhframe,price_change,wbreg,food_share_data,day2data):
	'''food price impact depends on income category (below, based on data from WB) and WB region'''
	list_segments = [(0,2.97),(2.97,8.44),(8.44,23.03),(23.03,float("inf"))]
	for (i,j) in list_segments:
		select=(futurehhframe['Y']>=i*day2data)&(futurehhframe['Y']<j*day2data)
		if j<24:
			food_share = food_share_data.ix[(food_share_data.wbregion==wbreg)&(food_share_data.consSeg=="{}-{}".format(i,j)),"food_share"]
		else:
			food_share = food_share_data.ix[(food_share_data.wbregion==wbreg)&(food_share_data.consSeg==">{}".format(i)),"food_share"]
		new_price_index = float(food_share)*(1+price_change)+(1-float(food_share))
		futurehhframe.ix[select,'Y'] = futurehhframe.ix[select,'Y']/new_price_index
	return futurehhframe
	
def shock(finalhhframe,sh_people_affected,losses_poor,losses_rich):
	'''
	This function models any type of shock on households' income. It duplicates all households to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	Then it reduces the income of affected people, with a difference between the poor and the rich. Here the poor is defined as the bottom 20% but this can be changed depending on the context. An absolute poverty line can also be used for country specific work.
	'''
	if sh_people_affected<=0:
		return finalhhframe
	else:
		sh_people_affected = min(sh_people_affected,0.99) #this prevents bugs if 100% of people are affected
		#finalhhframetemp is the survey with non affected people
		finalhhframetemp           = finalhhframe.copy()
		finalhhframetemp['weight'] = finalhhframetemp['weight']*(1-sh_people_affected)
		#finalhhframe21 is the survey with affected people
		finalhhframe21             = finalhhframe.copy()
		finalhhframe21['weight']   = finalhhframe21['weight']*(sh_people_affected)
		
		#select poor households in the affected survey
		th = perc_with_spline(finalhhframe21['Y'],finalhhframe21['nbpeople']*finalhhframe21['weight'],0.2)
		isbelowline                            = (finalhhframe21['Y']<=float(th))
		#apply income shocks
		finalhhframe21.ix[isbelowline,'Y']     = finalhhframe21.ix[isbelowline,'Y']*(1-losses_poor)
		finalhhframe21.ix[isbelowline,'totY']  = finalhhframe21.ix[isbelowline,'totY']*(1-losses_poor)
		finalhhframe21.ix[~isbelowline,'Y']    = finalhhframe21.ix[~isbelowline,'Y']*(1-losses_rich)
		finalhhframe21.ix[~isbelowline,'totY'] = finalhhframe21.ix[~isbelowline,'totY']*(1-losses_rich)
		#re-index affected households
		finalhhframe21.index                   = finalhhframe21.index.astype(object).astype(str)+"s"
		#merge the two surveys
		finalhhframetemp                       = finalhhframetemp.append(finalhhframe21)
	return finalhhframetemp
	
def shock_flood(new_hhframe,hazard_share_p,hazard_share_r,losses_poor,losses_rich):
	'''
	This function models any type of shock on households' income, with a different exposure for the poor and non poor. It duplicates all households (separating poor and non poor) to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	Then it reduces the income of affected people, with a difference between the poor and the rich. Here the poor is defined as the bottom 20% but this can be changed depending on the context. An absolute poverty line can also be used for country specific work.
	'''
	if hazard_share_p<=0:
		return new_hhframe
	else:
		hazard_share_p = min(0.999,hazard_share_p)
		hazard_share_r = min(0.999,hazard_share_r)
		
		th = perc_with_spline(new_hhframe['Y'],new_hhframe['nbpeople']*new_hhframe['weight'],0.2)
		isbelowline    = new_hhframe.Y<float(th)
		finalhhframe_r = new_hhframe.ix[~isbelowline,:].copy()
		finalhhframe_p = new_hhframe.ix[isbelowline,:].copy()
		
		finalhhframe_r_affected           = finalhhframe_r.copy()
		finalhhframe_r_affected['weight'] = finalhhframe_r_affected['weight']*(hazard_share_r)
		finalhhframe_r_affected['Y']      = finalhhframe_r_affected['Y']*(1-losses_rich)
		finalhhframe_r_affected.index     = finalhhframe_r_affected.index.astype(object).astype(str)+"ra"

		finalhhframe_r['weight']          = finalhhframe_r['weight']*(1-hazard_share_r)
		
		finalhhframe_p_affected           = finalhhframe_p.copy()
		finalhhframe_p_affected['weight'] = finalhhframe_p_affected['weight']*(hazard_share_p)
		finalhhframe_p_affected['Y']      = finalhhframe_p_affected['Y']*(1-losses_poor)
		finalhhframe_p_affected.index     = finalhhframe_p_affected.index.astype(object).astype(str)+"pa"

		finalhhframe_p['weight']          = finalhhframe_p['weight']*(1-hazard_share_p)

	return finalhhframe_r.append(finalhhframe_r_affected).append(finalhhframe_p).append(finalhhframe_p_affected)
	
def shock_drought(finalhhframe,sh_people_affected,losses_poor):
	'''
	This function models drought shocks on households' income. It only affects farmers and makes no difference between the poor and non poor.
	It duplicates all households to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	'''
	if sh_people_affected<=0:
		return finalhhframe
	else:
		finalhhframetemp                    = finalhhframe.copy()
		finalhhframe21                      = finalhhframe.copy()
		finalhhframetemp['weight']          = finalhhframetemp['weight']*(1-sh_people_affected)
		finalhhframe21['weight']            = finalhhframe21['weight']*(sh_people_affected)
		select_farmers                      = (finalhhframe21.cat3workers>0)|(finalhhframe21.cat4workers>0)
		finalhhframe21.ix[select_farmers,'Y'] = finalhhframe21.ix[select_farmers,'Y']*(1-losses_poor)
		finalhhframe21.ix[select_farmers,'totY'] = finalhhframe21.ix[select_farmers,'totY']*(1-losses_poor)
		finalhhframe21.index                   = finalhhframe21.index.astype(object).astype(str)+"s"
		finalhhframetemp                       = finalhhframetemp.append(finalhhframe21)
	return finalhhframetemp

	
def get_shares_stunting(wbreg,paramstunt,shares_stunting,ssp):
	'''imports the share of children being stunt'''
	select = (shares_stunting.wbreg==wbreg)&(shares_stunting.ssp==ssp)
	sh_people_stunt = (1-paramstunt)*shares_stunting.ix[select&(shares_stunting.min_max=="min"),"sh_stunt"].values[0]/100+paramstunt*shares_stunting.ix[select&(shares_stunting.min_max=="max"),"sh_stunt"].values[0]/100
	return sh_people_stunt
	
	
def stunting(finalhhframe,threshold,wbreg,shockstunting,sh_people_stunt):
	'''This function models stunting shocks on households' income.
	It duplicates households who live below a given income threshold to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	Then it reduces the income of affected people.
	'''
	finalhhframetemp = finalhhframe.copy()
	select         = (finalhhframetemp['Y']<=threshold)
	#this survey contains only household who live below the income threshold.
	finalhhframe22 = finalhhframetemp.ix[select,:].copy()
	#recalculates the share of people affected within this survey of only poor people
	new_share      = min(0.99,sh_people_stunt*sum(finalhhframetemp['weight']*finalhhframetemp['nbpeople'])/sum(finalhhframe22['weight']*finalhhframe22['nbpeople']))
	#applies the shock, reindexes households and merges the two surveys
	if new_share>0:
		finalhhframetemp.ix[select,'weight'] = finalhhframetemp.ix[select,'weight']*(1-new_share)
		finalhhframe22['weight']         = finalhhframe22['weight']*(new_share)
		finalhhframe22['Y']              = finalhhframe22['Y']*(1-shockstunting)
		finalhhframe22['totY']           = finalhhframe22['totY']*(1-shockstunting)
		finalhhframe22.index             = finalhhframe22.index.astype(object).astype(str)+"st"
		finalhhframetemp = finalhhframetemp.append(finalhhframe22)
	return finalhhframetemp

def temperature_impact(futureinc,shares_outside,temp_impact):
	'''reduces people's income due to temperature impact, based on their sector'''
	futureinc['cat3workers']   = futureinc['cat3workers']*(1-shares_outside['ag'])+futureinc['cat3workers']*(1-temp_impact)*shares_outside['ag']
	futureinc['cat4workers']   = futureinc['cat4workers']*(1-shares_outside['ag'])+futureinc['cat4workers']*(1-temp_impact)*shares_outside['ag']
	futureinc['cat1workers']   = futureinc['cat1workers']*(1-shares_outside['serv'])+futureinc['cat1workers']*(1-temp_impact)*shares_outside['serv']
	futureinc['cat2workers']   = futureinc['cat2workers']*(1-shares_outside['serv'])+futureinc['cat2workers']*(1-temp_impact)*shares_outside['serv']
	futureinc['cat5workers']   = futureinc['cat5workers']*(1-shares_outside['manu'])+futureinc['cat5workers']*(1-temp_impact)*shares_outside['manu']
	futureinc['cat6workers']   = futureinc['cat6workers']*(1-shares_outside['manu'])+futureinc['cat6workers']*(1-temp_impact)*shares_outside['manu']
	return futureinc
	
def get_malaria_share(cc,wbreg,malaria_diff):
	'''imports share of people affected by malaria'''
	share = malaria_diff.ix[malaria_diff.country==cc,"diff 2020s"].values
	if len(share)==0:
		share = malaria_diff.ix[malaria_diff.wbregion==wbreg,"diff 2020s"].mean()
	return float(share)/100
	
def get_disease_impact(malaria_cursor,malaria_bounds):
	'''imports the impact of the disease on number of lost working days and fixed cost'''
	lostdays = (1-malaria_cursor)*malaria_bounds.loc["lostdays","min"]+(malaria_cursor)*malaria_bounds.loc["lostdays","max"]
	eventcost = (1-malaria_cursor)*malaria_bounds.loc["eventcost","min"]+(malaria_cursor)*malaria_bounds.loc["eventcost","max"]
	return lostdays,eventcost
	
def malaria_impact(finalhhframe,malaria_yr_occ,lostdays,eventcost,sh_people_affected):
	'''This function models malaria shocks on households' income.
	It duplicates all households to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	Then it reduces the income of affected people.
	There is a possibility for reduced malaria incidence because of climate change, so people's income may increase.
	'''
	if sh_people_affected==0:
		return finalhhframe
	else:
		finalhhframetemp         = finalhhframe.copy()
		finalhhframe21           = finalhhframe.copy()
		if sh_people_affected>0:
			finalhhframetemp['weight']   = finalhhframetemp['weight']*(1-sh_people_affected)
			finalhhframe21['weight']     = finalhhframe21['weight']*(sh_people_affected)
			finalhhframe21.totY          = finalhhframe21.totY*(1-malaria_yr_occ*lostdays/30/12)-malaria_yr_occ*eventcost
			finalhhframe21.Y             = finalhhframe21.totY/finalhhframe21.nbpeople
			finalhhframe21.ix[finalhhframe21.Y<0,"Y"]=0
		elif sh_people_affected<0:
			finalhhframetemp['weight']   = finalhhframetemp['weight']*(1+sh_people_affected)
			finalhhframe21['weight']     = finalhhframe21['weight']*(-sh_people_affected)
			finalhhframe21.totY          = finalhhframe21.totY*(1+malaria_yr_occ*lostdays/30/12)+malaria_yr_occ*eventcost
			finalhhframe21.Y             = finalhhframe21.totY/finalhhframe21.nbpeople

		finalhhframe21.index          = finalhhframe21.index.astype(object).astype(str)+"m"
		finalhhframetemp              = finalhhframetemp.append(finalhhframe21)
		return finalhhframetemp
	
	
def diarrhea_impact(futurehhframe,diarrhea_yr_occ,lostdays_dia,eventcost_dia,diarrhea_share,th_diarrhea):
	'''This function models diarrhea shocks on households' income.
	It duplicates households who live below a given income threshold to create affected and non-affected households, and adjusts the weights so that the total number of people remains the same.
	Then it reduces the income of affected people.
	There is a possibility for reduced diarrhea incidence because of climate change, so people's income may increase.
	'''
	if diarrhea_share==0:
		return futurehhframe
	else:
		finalhhframetemp = futurehhframe.copy()
		select           = (finalhhframetemp['Y']<=th_diarrhea)&(finalhhframetemp['children']>0)
		finalhhframe22   = finalhhframetemp.ix[select,:].copy()
		new_share        = min(0.99,diarrhea_share*sum(finalhhframetemp['weight']*finalhhframetemp['nbpeople'])/sum(finalhhframe22['weight']*finalhhframe22['nbpeople']))
		if diarrhea_share>0:
			finalhhframetemp.ix[select,'weight'] = finalhhframetemp.ix[select,'weight']*(1-new_share)
			finalhhframe22['weight']             = finalhhframe22['weight']*(new_share)
			finalhhframe22['totY']               = finalhhframe22['totY']*(1-diarrhea_yr_occ*lostdays_dia/30/12)-diarrhea_yr_occ*eventcost_dia
			finalhhframe22.Y                     = finalhhframe22.totY/finalhhframe22.nbpeople
			finalhhframe22.ix[finalhhframe22.Y<0,"Y"]=0
		elif diarrhea_share<0:
			finalhhframetemp.ix[select,'weight'] = finalhhframetemp.ix[select,'weight']*(1+new_share)
			finalhhframe22['weight']             = finalhhframe22['weight']*(-new_share)
			finalhhframe22['totY']               = finalhhframe22['totY']*(1+diarrhea_yr_occ*lostdays_dia/30/12)+diarrhea_yr_occ*eventcost_dia
			finalhhframe22.Y                     = finalhhframe22.totY/finalhhframe22.nbpeople
		finalhhframe22.index          = finalhhframe22.index.astype(object).astype(str)+"di"
		finalhhframetemp              = finalhhframetemp.append(finalhhframe22)
		return finalhhframetemp
		
def valuefromcursor(boundsrow,cursor):
	'''finds a value between two bounds. cursor is between 0 and 1.'''
	return boundsrow['min']*(1-cursor)+boundsrow['max']*(cursor)
	
def get_disasters_vulnerabilities(countrycode,vulnerabilities,disasters_cursor,climate_param_bounds):
	if countrycode in vulnerabilities.index:
		# the basic_cursor is a 0/+20% variation around the value found in the resilience indicator
		cursor = valuefromcursor(climate_param_bounds.loc['basic_cursor',:],disasters_cursor)
		vp     = (1+cursor)*vulnerabilities.loc[countrycode,'vp']
		vr     = (1+cursor)*vulnerabilities.loc[countrycode,'vr']
	else:
		vp = valuefromcursor(climate_param_bounds.loc['losses_poor',:],disasters_cursor)
		vr = valuefromcursor(climate_param_bounds.loc['losses_rich',:],disasters_cursor)
	return vp,vr
	
def get_disasters_exposure(countrycode,fa):
	fa_poor       = DataFrame(index=['floodglofris','surge','drought','wind'],columns=['fa'])
	fa_nonpoor    = DataFrame(index=['floodglofris','surge','drought','wind'],columns=['fa'])
	if countrycode in fa.index:
		fa_all        = fa.loc[countrycode,:]
		for hazard in ['floodglofris','surge','drought','wind']:
			if hazard in list(fa_all.hazard_name):
				select = fa_all.hazard_name==hazard
				fa_poor.loc[hazard,'fa']    = fa_all.ix[select,'fa'].values[0]
				fa_nonpoor.loc[hazard,'fa'] = fa_all.ix[select,'fa'].values[0]
			else:
				fa_poor.loc[hazard,'fa']    = 0
				fa_nonpoor.loc[hazard,'fa'] = 0

	else:
		for hazard in ['floodglofris','surge','drought','wind']:
			fa_poor.loc[hazard,'fa']    = 0
			fa_nonpoor.loc[hazard,'fa'] = 0
	return fa_poor,fa_nonpoor

		
def get_model_cc_parameters(food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor, ssp, wbreg, price_increase_reg, malaria_bounds, diarrhea_bounds, finalhhframe, shares_stunting, malaria_diff, countrycode, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds,fa,vulnerabilities,voice):
	'''
	Loads all climate change parameters from data sources.
	'''
		
	#until we find something better we don't change the frequency of the desease
	malaria_yr_occ             = (climate_param_bounds.loc['malaria_yr_occ','min']+climate_param_bounds.loc['malaria_yr_occ','max'])/2
	diarrhea_yr_occ            = (climate_param_bounds.loc['diarrhea_yr_occ','min']+climate_param_bounds.loc['diarrhea_yr_occ','max'])/2
	
	losses_poor,losses_rich    = get_disasters_vulnerabilities(countrycode,vulnerabilities,disasters_cursor,climate_param_bounds)
	fa_poor,fa_nonpoor         = get_disasters_exposure(countrycode,fa)
	
	shockstunting              = valuefromcursor(climate_param_bounds.loc['shockstunting',:],paramshstunt)   
	cyclones_increase          = valuefromcursor(climate_param_bounds.loc['cyclones_increase',:],disasters_cursor)
	flood_increase             = valuefromcursor(climate_param_bounds.loc['flood_increase',:],disasters_cursor)
	fprice_increase            = price_increase(wbreg,ssp,price_increase_reg,food_cursor)		
	farmer_rev_change          = food_impact_on_farmers(wbreg,ssp,price_increase_reg,food_cursor)	
	th_nostunt                 = climate_param_thresholds.loc['th_nostunt','threshold']
	th_diarrhea                = climate_param_thresholds.loc['th_diarrhea','threshold']
	lostdays_mal,eventcost_mal = get_disease_impact(malaria_cursor,malaria_bounds)
	lostdays_dia,eventcost_dia = get_disease_impact(diarrhea_cursor,diarrhea_bounds)
	shares_outside             = get_shares_outside(finalhhframe)
	sh_people_stunt            = get_shares_stunting(wbreg,paramshstunt,shares_stunting,ssp)
	malaria_share              = get_malaria_share(countrycode,wbreg,malaria_diff)
	diarrhea_increase		   = valuefromcursor(climate_param_bounds.loc['diarrhea_increase',:],voice)   
	transmission      		   = valuefromcursor(climate_param_bounds.loc['transmission',:],voice)   

	
	diarrhea_share             = diarrhea_increase*diarrhea_reg.who.replace(diarrhea_shares.ratio).fillna(0)[reverse_correct_countrycode(countrycode)]/diarrhea_yr_occ
	
	flood_share_poor    = (flood_increase)*fa_poor.loc['floodglofris','fa']
	flood_share_nonpoor = (flood_increase)*fa_nonpoor.loc['floodglofris','fa']
	
	drought_share       = (flood_increase)*fa_poor.loc['drought','fa']
	wind_share          = (cyclones_increase)*fa_poor.loc['wind','fa']
	surge_share         = (cyclones_increase)*fa_poor.loc['surge','fa']
	
	ccparam      = fprice_increase,farmer_rev_change,cyclones_increase,losses_poor,losses_rich,shares_outside,temp_impact,shockstunting,sh_people_stunt,th_nostunt,transmission,malaria_yr_occ,lostdays_mal,eventcost_mal,lostdays_dia,eventcost_dia,th_diarrhea,malaria_share,diarrhea_share,diarrhea_yr_occ,flood_share_poor,flood_share_nonpoor,drought_share,wind_share,surge_share
	
	ccparam2keep = fprice_increase,farmer_rev_change,transmission,food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor,malaria_share,diarrhea_share,flood_share_poor,flood_share_nonpoor,drought_share,wind_share,surge_share
	ccparam2keeptitles = ['fprice_increase','farmer_rev_change','transmission','food_cursor','temp_impact','paramshstunt','malaria_cursor','diarrhea_cursor','disasters_cursor','malaria_share','diarrhea_share','flood_share_poor','flood_share_nonpoor','drought_share','wind_share','surge_share']
	
	return ccparam,ccparam2keep,ccparam2keeptitles

	
	