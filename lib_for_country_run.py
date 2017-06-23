import sys
from pandas import Series,DataFrame,read_csv
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

def estimate_income_and_all(hhcat,hhdataframe):
	inc             = estime_income(hhcat,hhdataframe)
	characteristics = keep_characteristics_to_reweight(hhdataframe)
	ini_pop_desc    = calc_pop_desc(characteristics,hhdataframe['weight'])
	inimin          = hhdataframe['Y'].min()
	return inc,characteristics,ini_pop_desc,inimin
	
def run_one_baseline(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,wbreg,data2day,food_share_data,istoobig=False):
	shareag,sharemanu = correct_shares(inputs['shareag'],inputs['sharemanu'])
	shareemp          = inputs['shareemp']
	
	future_pop_desc,pop_0014=build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp)
	
	if istoobig:
		weights_proj = Series()
		finalhhframe1,finalhhframe2 = finalhhframe

		ini_weights_sum   = (finalhhframe1['weight']*finalhhframe1['nbpeople']).sum()+(finalhhframe2['weight']*finalhhframe2['nbpeople']).sum()
		for finalhhframehalf in [finalhhframe1,finalhhframe2]:
			characteristicshalf = keep_characteristics_to_reweight(finalhhframehalf)
			ini_weightshalf     = finalhhframehalf['weight']*finalhhframehalf['nbpeople']
			ratio               = sum(ini_weightshalf)/ini_weights_sum
			ini_pop_deschalf    = calc_pop_desc(characteristicshalf,ini_weightshalf)
			weights_projh       = find_new_weights(characteristicshalf,ini_weightshalf,future_pop_desc*ratio)
			weights_proj        = weights_proj.append(weights_projh)
		finalhhframe  = concat([finalhhframe1,finalhhframe2],axis=0)
		weights_proj = DataFrame(weights_proj,index=finalhhframe.index,columns=["weight"])
	else:
		ini_weights   = finalhhframe['weight']
		weights_proj  = find_new_weights(characteristics,ini_weights,future_pop_desc)
		weights_proj = DataFrame(weights_proj,index=finalhhframe.index,columns=["weight"])
		
	futurehhframe              = futurehh(finalhhframe,pop_0014)
	futurehhframe['weight']    = weights_proj["weight"]

	income_proj,futureinc  = future_income_simple_no_cc(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year)
	income_proj.fillna(0, inplace=True)
	futurehhframe['Y']         = income_proj
		
	return futurehhframe,futureinc

	
def run_one_cc_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health):
	print(switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)
	shareag,sharemanu = correct_shares(inputs['shareag'],inputs['sharemanu'])
	shareemp          = inputs['shareemp']
	
	fprice_increase,farmer_rev_change,cyclones_increase,losses_poor,losses_rich,shares_outside,temp_impact,shockstunting,sh_people_stunt,th_nostunt,transmission,malaria_yr_occ,lostdays_mal,eventcost_mal,lostdays_dia,eventcost_dia,th_diarrhea,malaria_share,diarrhea_share,diarrhea_yr_occ,flood_share_poor,flood_share_nonpoor,drought_share,wind_share,surge_share = ccparam
	
	future_pop_desc,pop_0014=build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp)
	
	futurehhframe          = futurehhframe_bau
	if not switch_ag_rev:
		farmer_rev_change = 0
		
	if not switch_temp:
		temp_impact = 0
	
	income_proj,futureinc  = future_income_simple(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,transmission*farmer_rev_change)
	income_proj.fillna(0, inplace=True)
	futurehhframe['Y']     = income_proj
	
	if switch_ag_prices:
		futurehhframe = food_price_impact(futurehhframe,fprice_increase,wbreg,food_share_data,data2day)
	
	if switch_disasters:
		futurehhframe = shock(futurehhframe,wind_share,losses_poor,losses_rich)
		futurehhframe = shock(futurehhframe,surge_share,losses_poor,losses_rich)
		futurehhframe = shock_flood(futurehhframe,flood_share_poor,flood_share_nonpoor,losses_poor,losses_rich)
		futurehhframe = shock_drought(futurehhframe,drought_share,losses_poor)
	if switch_health:
		futurehhframe = stunting(futurehhframe,th_nostunt,wbreg,shockstunting,sh_people_stunt)
		futurehhframe = malaria_impact(futurehhframe,malaria_yr_occ,lostdays_mal,eventcost_mal,malaria_share)
		futurehhframe = diarrhea_impact(futurehhframe,diarrhea_yr_occ,lostdays_dia,eventcost_dia,diarrhea_share,th_diarrhea)
	
	return futurehhframe,futureinc