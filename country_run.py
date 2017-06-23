from pandas import concat,Series,DataFrame,read_csv,HDFStore,set_option
from lib_for_growth_model import *
from lib_for_cc import *
from lib_for_country_run import *
import numpy as np

def country_run(countrycode,scenar,datalist,paramvar,all_surveys,switches):
	'''
	Runs one scenario for one country (with 2 ssps and baseline/climate) and returns a dataframe with aggregated results per scenario.
	
	countrycode: an ISO3 code
	scenar: number of the scenario to run
	datalist: all data required to run the scenario
	paramvar: all parameters defined in run_model.py
	all_surveys: dict of dataframes, indexed by countrycode, with household surveys for all countries
	'''
	
	(year,ini_year,data2day,ssp_to_test,povline) = paramvar
	(hhcat,industry_list,codes_tables,lhssample,ranges,ssp_pop,ssp_gdp,food_share_data,impact_scenars,price_increase_reg, malaria_bounds, diarrhea_bounds, shares_stunting, malaria_diff, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds,codes,fa,vulnerabilities) = datalist
	
	#get the country data from list of surveys
	out = filter_country(countrycode,all_surveys,codes)
	if out is None:
		return DataFrame([[countrycode,year,scenar]],columns=['country','year','scenar']),DataFrame([[countrycode,year,scenar]],columns=['country','year','scenar'])
	else:
		finalhhframe,countrycode,istoobig,wbreg = out

	#estimates income
	inc,characteristics,ini_pop_desc,inimin = estimate_income_and_all(hhcat,finalhhframe)
	
	#initializes outputs
	forprim_bau = DataFrame()
	forprim_cc  = DataFrame()
	
	#gets the ranges of values to generate the scenarios from a lhs table
	ranges    = scenar_ranges(ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year)
	scenarios = lhssample.values*np.diff(ranges[['min','max']].values).T+ranges['min'].values
	scenarios = DataFrame(scenarios,columns=ranges.index)
	
	#in the version designed for the cluster, we get run only one scenario (called scenar) per call of this function
	inputs = scenarios.ix[scenar,:]

	if istoobig:
		#sometimes the survey is too big so we have to split the survey in two. We split it here because it takes a lot of time so we don't repeat it in each ssp scenario
		#the big dataframe is split by regions and each subdataframe is reduced by merging households
		finalhhframes = split_big_dframe(finalhhframe,hhcat)
		finalhhframe1,finalhhframe2 = finalhhframes
		finalhhframe  = concat([finalhhframe1,finalhhframe2],axis=0)
	else:
		finalhhframes = finalhhframe
	
	for ssp in ssp_to_test:
		
		#run the baseline scenario
		print("I am running baseline {} with ssp {} of {}".format(scenar,ssp,countrycode))
		futurehhframe_bau,futureinc_bau = run_one_baseline(ssp,inputs,finalhhframes,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,wbreg,data2day,food_share_data,istoobig)
		
		if sum(futurehhframe_bau['weight'])==0:
			futurehhframe_bau = futurehhframe_bau.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),DataFrame([inputs.values],columns=inputs.index)],ignore_index=True))
			continue
		
		#calculate indicators
		indicators_bau  = calc_indic(futurehhframe_bau['Y'],futurehhframe_bau['weight']*futurehhframe_bau['nbpeople'],futurehhframe_bau['weight'],futurehhframe_bau,data2day,futureinc_bau,povline)
		
		#additional indicators: average productivity growth for all workers (skilled and unskilled) by sector
		prod_gr_serv_bau,prod_gr_ag_bau,prod_gr_manu_bau = actual_productivity_growth(futurehhframe_bau,inc,futurehhframe_bau,futureinc_bau,year,ini_year)
		
		#store results (this was useful for a previous version that ran all scenarios within this function)
		forprim_bau = forprim_bau.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),indicators_bau,DataFrame([inputs.values],columns=inputs.index),DataFrame([[prod_gr_serv_bau,prod_gr_ag_bau,prod_gr_manu_bau]],columns=['prod_gr_ag','prod_gr_serv','prod_gr_manu'])],axis=1),ignore_index=True)
		
		for ccint in impact_scenars.index:
			for switch in switches:
				if switch == 'all':
					switch_ag_rev    = True
					switch_temp      = True
					switch_ag_prices = True
					switch_disasters = True
					switch_health    = True
				else:
					switch_ag_rev    = switch == 'switch_ag_rev'
					switch_temp      = switch == 'switch_temp'
					switch_ag_prices = switch == 'switch_ag_prices'
					switch_disasters = switch == 'switch_disasters'
					switch_health    = switch == 'switch_health'
								
				#runs the climate impacts scenarios. Here no need to recalculate the weights
				#extracts cc parameters
				food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor = impact_scenars.loc[ccint,:]
				ccparam,ccparam2keep,ccparam2keeptitles = get_model_cc_parameters(food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor, ssp, wbreg, price_increase_reg, malaria_bounds, diarrhea_bounds, finalhhframe, shares_stunting, malaria_diff, countrycode, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds, fa,vulnerabilities,inputs.voice)
				#run the cc scenario
				print("I am running climate scenario {} of baseline {} with ssp {} of {}".format(ccint,scenar,ssp,countrycode))
				futurehhframe_cc,futureinc_cc = run_one_cc_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)
				
				if sum(futurehhframe_cc['weight'])==0:
					futurehhframe_cc = futurehhframe_cc.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),DataFrame([inputs.values],columns=inputs.index)],ignore_index=True))
					continue

				#calculate indicators and store results
				indicators_cc   = calc_indic(futurehhframe_cc['Y'],futurehhframe_cc['weight']*futurehhframe_cc['nbpeople'],futurehhframe_cc['weight'],futurehhframe_cc,data2day,futureinc_cc,povline)
				prod_gr_serv_cc, prod_gr_ag_cc, prod_gr_manu_cc  = actual_productivity_growth(futurehhframe_cc, inc,futurehhframe_cc, futureinc_cc, year,ini_year)
				
				switches_values = DataFrame([[switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health]],columns=['switch_ag_rev','switch_temp','switch_ag_prices','switch_disasters','switch_health'])
							
				forprim_cc  = forprim_cc.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp),ccint]],columns=['country','year','scenar','ssp','ccint']),switches_values,indicators_cc,DataFrame([inputs.values],columns=inputs.index),DataFrame([ccparam2keep],columns=ccparam2keeptitles),DataFrame([[prod_gr_serv_cc,prod_gr_ag_cc,prod_gr_manu_cc]],columns=['prod_gr_ag','prod_gr_serv','prod_gr_manu'])],axis=1),ignore_index=True)

	return forprim_bau,forprim_cc
	