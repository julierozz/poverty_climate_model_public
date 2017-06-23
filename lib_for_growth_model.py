from pandas import Series,DataFrame
import numpy as np
import cvxopt
from cvxopt.solvers import qp
from perc import *
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from scipy import integrate, optimize
from kde import gaussian_kde
from lib_for_data_reading import *
from lib_for_cc import *
import scipy

from statsmodels.nonparametric.kde import kdensity

################# Functions used to define the scenarios (ranges are specific to each country) #########################################

def scenar_ranges(ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year):
	'''
	This is a messy function at the moment. It sets the ranges of uncertainties. For redistribution (p and b) it is just a fixed range. For structural change, the ranges depend on the initial shares and are calculated in find_range_struct. For growth rates,xxx
	'''
	characteristics = keep_characteristics_to_reweight(finalhhframe)
	ini_pop_desc    = calc_pop_desc(characteristics,finalhhframe['weight'])
	shareemp_ini,shareag_ini,sharemanu_ini,share_skilled = indicators_from_pop_desc(ini_pop_desc)
	
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])

	gr4=(get_gdp_growth(ssp_gdp,year,4,country2r32(codes_tables,countrycode),ini_year))**(1/(year-ini_year))-1
	gr5=(get_gdp_growth(ssp_gdp,year,5,country2r32(codes_tables,countrycode),ini_year))**(1/(year-ini_year))-1
	ssp_growth = np.mean([gr4,gr5])

	pop_tot,pop_0014,pop_1564_4,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,4,year,countrycode)
	pop_tot,pop_0014,pop_1564_5,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,5,year,countrycode)
	pop_growth = np.mean([(pop_1564_5/adults)**(1/(year-ini_year))-1,(pop_1564_5/adults)**(1/(year-ini_year))-1])
	
	ranges.ix['shareag',['min','max']]   = find_range_struct(shareag_ini,'ag')
	ranges.ix['sharemanu',['min','max']] = find_range_struct(sharemanu_ini,'ind')
	ranges.ix['shareemp',['min','max']]  = find_range_struct(shareemp_ini,'emp')
		
	select_gr=['grag','grmanu','grserv']
	ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.05,ssp_growth-pop_growth+0.01]
	
	if countrycode in ['BTN']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.08,ssp_growth-pop_growth-0.01]
	if countrycode in ['AFG','ALB','BIH','CHN','DOM','ECU','EGY','FSM','GEO','GIN','JAM','KGZ','MAR','MDA','MKD','MNG','MOZ','NPL','PHL']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.06,ssp_growth-pop_growth]
	if countrycode in ['BDI']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.04,ssp_growth-pop_growth+0.03]
	if countrycode in ['TCD','ZMB','SWZ']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.02,ssp_growth-pop_growth+0.04]
		
	ranges.ix[['skillpag','skillpserv','skillpmanu'],['min','max']] = [1,5]
	ranges.ix['p',['min','max']]=[0.001,0.2]
	ranges.ix['b',['min','max']]=[0.001,0.2]
	
	ranges.ix['voice',['min','max']]=[0,1]
	
	return ranges
	
	
def find_range_struct(ini_share,sector):
	x    = [0,0.01,0.1,0.3,0.5,0.7,0.9,1]
	if sector=='ag':
		ymin = [0,0.001,0.01,0.1,0.2,0.3,0.4,0.6]
		ymax = [0,0.2,0.15,0.4,0.5,0.6,0.8,0.8]
	elif sector=='ind':
		ymin = [0,0.1,0.15,0.1,0.2,0.3,0.35,0.4]
		ymax = [0,0.25,0.3,0.35,0.4,0.5,0.7,0.8]
	elif sector=='emp':
		ymin = [0,0.007,0.07,0.25,0.4,0.6,0.75,0.8]
		ymax = [0,0.4,0.5,0.6,0.7,0.9,0.99,1]
	w         = [1,2,2,2,1,1,1,1]
	smin      = UnivariateSpline(x, ymin, w)
	smax      = UnivariateSpline(x, ymax, w)
	range_out = [max(float(smin(ini_share)),0),min(float(smax(ini_share)),1)]
	return range_out
	
def correct_shares(shareag,sharemanu):
	if shareag+sharemanu>1:
		tot=shareag+sharemanu
		shareag=shareag/tot-0.001
		sharemanu=sharemanu/tot-0.001
	return shareag,sharemanu
	


####### Functions used for the reweighting process #######################################

def calc_pop_desc(characteristics,weights):
	pop_description             = DataFrame(columns=characteristics.columns)
	pop_description.ix['pop',:] = np.dot(characteristics.T,weights)
	return pop_description

	
def keep_characteristics_to_reweight(finalhhframe):
	characteristics                 = DataFrame()
	characteristics['old']          = finalhhframe['old']
	characteristics['children']     = finalhhframe['children']
	characteristics['unemployed']   = finalhhframe['cat7workers']
	characteristics['skillworkers'] = finalhhframe['cat2workers']+finalhhframe['cat4workers']+finalhhframe['cat6workers']
	characteristics['agworkers']    = finalhhframe['cat3workers']+finalhhframe['cat4workers']
	characteristics['manuworkers']  = finalhhframe['cat5workers']+finalhhframe['cat6workers']
	characteristics['servworkers']  = finalhhframe['cat1workers']+finalhhframe['cat2workers']
	return characteristics
	
def build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,ischildren=False):
	'''builds a new description vector for the projected year, from ssp data and exogenous share for skilled people and agri people'''
	pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,ssp,year,countrycode)
	new_pop_desc                 = ini_pop_desc.copy()
	new_pop_desc['children']     = ini_pop_desc['children']
	if ischildren:
		new_pop_desc['children'] = pop_0014
	new_pop_desc['old']          = pop_65up
	new_pop_desc['skillworkers'] = skilled_adults*shareemp
	new_pop_desc['agworkers']    = pop_1564*shareag*shareemp
	new_pop_desc['manuworkers']  = pop_1564*sharemanu*shareemp
	new_pop_desc['servworkers']  = pop_1564*(1-shareag-sharemanu)*shareemp
	new_pop_desc['unemployed']   = pop_1564*(1-shareemp)
	return new_pop_desc,pop_0014
	
def futurehh(finalhhframe,pop_0014,ischildren=False):
	'''
	ischildren is True if the number of children is taken into account in the re-weighting process. Otherwise, we re-scale the number of people based on the new number of children.
	'''
	futurehhframe=finalhhframe.copy()
	if not ischildren:
		futurehhframe['children']=finalhhframe['children']*pop_0014/sum(finalhhframe['children']*finalhhframe['weight'])
	futurehhframe['nbpeople']=finalhhframe['nbpeople']+futurehhframe['children']-finalhhframe['children']
	futurehhframe.drop(['Y','weight'], axis=1, inplace=True)
	return futurehhframe
	
def build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True):
	'''optimize new weights to match current households and new population description'''
	t_tilde = cvxopt.matrix((future_pop_desc.values-ini_pop_desc.values).astype(np.float,copy=False))
	aa      = cvxopt.matrix(characteristics.values.astype(np.float,copy=False))
	w1      = 1/(ini_weights.values)**2
	n       = len(w1)
	P       = cvxopt.spdiag(cvxopt.matrix(w1))
	G       = -cvxopt.matrix(np.identity(n))
	h       = cvxopt.matrix(ini_weights.values.astype(np.float,copy=False))
	q       = cvxopt.matrix(0.0,(n,1))
	if ismosek:
		result = qp(P,q,G,h,aa.T,t_tilde.T,solver='mosek')['x']
	else:
		result = qp(P,q,G,h,aa.T,t_tilde.T)['x']
	if result is None:
		new_weights = 0*ini_weights
	else:
		new_weights = ini_weights+list(result)
	return new_weights
	
def find_new_weights(characteristics,ini_weights,future_pop_desc):
	ini_pop_desc = calc_pop_desc(characteristics,ini_weights)
	weights_proj = build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True)
	weights_proj = Series(np.array(weights_proj),index=ini_weights.index.values,dtype='float64')
	return weights_proj
	
########### Functions used before changing future income #####################

def estime_income(hhcat,finalhhframe):
	'''
	Estimates income brought by each category of adults/elderly. The objective is not to have a good model of income but rather to find a starting point for making the income grow based on the household's composition. If the income of the unemployed or elderly is found negative, it is put equal to zero and we re-estimate.
	If the coefficients are non significant, we try different categories (by grouping existing categories) and keep the new coefficients only if they become significant.
	We ignore the richest 5%.
	'''
	select     = finalhhframe.Y<float(perc_with_spline(finalhhframe.Y,finalhhframe.weight*finalhhframe.nbpeople,0.95))
	X          = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
	w          = finalhhframe.ix[select,'weight'].copy()
	w[w==0]    = 10**(-10)
	Y          = (finalhhframe.ix[select,'Y']*finalhhframe.ix[select,'nbpeople'])
	result     = sm.WLS(Y, X, weights=1/w).fit()
	inc        = result.params
	nonworkers = inc[['cat7workers','old']].copy()
	negs       = nonworkers[nonworkers<0].index
	if len(negs)>0:
		X.drop(negs.values,axis=1,inplace=True)
		result = sm.WLS(Y, X, weights=1/w).fit()
		inc    = result.params
		for ii in negs:
			inc[ii] = 0
	a        = result.pvalues
	nonsign1 = a[a>0.05].index
	nonsign2 = []
	nonsign3 = []
	if len(nonsign1)>0:
		X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
		X['serv'] = X['cat1workers']+X['cat2workers']
		X['ag']   = X['cat3workers']+X['cat4workers']
		X['manu'] = X['cat5workers']+X['cat6workers']
		X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
		result3   = sm.WLS(Y, X, weights=1/w).fit()
		a3        = result3.pvalues
		nonsign3  = a3[a3>0.05].index
		if (len(nonsign3)==0):
			inctemp            = result3.params
			inc['cat2workers'] = inctemp['serv']
			inc['cat4workers'] = inctemp['ag']
			inc['cat6workers'] = inctemp['manu']
			inc['cat1workers'] = inctemp['serv']
			inc['cat3workers'] = inctemp['ag']
			inc['cat5workers'] = inctemp['manu']
		else:
			X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
			X['skilled']   = X['cat2workers']+X['cat4workers']+X['cat6workers']
			X['unskilled'] = X['cat1workers']+X['cat3workers']+X['cat5workers']
			X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
			result2        = sm.WLS(Y, X, weights=1/w).fit()
			a2             = result2.pvalues
			nonsign2       = a2[a2>0.05].index
			if len(nonsign2)==0|((len(nonsign2)<len(nonsign1))&(len(nonsign2)<len(nonsign3))):
				inctemp           = result2.params
				inc['cat2workers']= inctemp['skilled']
				inc['cat4workers']= inctemp['skilled']
				inc['cat6workers']= inctemp['skilled']
				inc['cat1workers']= inctemp['unskilled']
				inc['cat3workers']= inctemp['unskilled']
				inc['cat5workers']= inctemp['unskilled']
			else:
				if (len(nonsign3)<len(nonsign1))&(len(nonsign3)<len(nonsign2)):
					inctemp            = result3.params
					inc['cat2workers'] = inctemp['serv']
					inc['cat4workers'] = inctemp['ag']
					inc['cat6workers'] = inctemp['manu']
					inc['cat1workers'] = inctemp['serv']
					inc['cat3workers'] = inctemp['ag']
					inc['cat5workers'] = inctemp['manu']
	return inc
	
def estimate_income_and_all(hhcat,hhdataframe):
	inc             = estime_income(hhcat,hhdataframe)
	characteristics = keep_characteristics_to_reweight(hhdataframe)
	ini_pop_desc    = calc_pop_desc(characteristics,hhdataframe['weight'])
	inimin          = hhdataframe['Y'].min()
	return inc,characteristics,ini_pop_desc,inimin
	
	
############ Functions used for changing household income ##############################
	
def Y_from_inc(futurehhframe,inc):
	'''
	Calculates total hh income as the sum of each people's income in the household (estimated income)
	'''
	listofvariables = list(inc.index)
	out             = 0*futurehhframe['nbpeople']
	for var in listofvariables:
		out += inc[var]*futurehhframe[var]
	Ycalc           = Series(out,index=out.index)
	return Ycalc
	
def before_tax(inc,finalhhframe):
	'''Calculates the pre-tax revenues, assuming that elderly and unemployed incomes come from redistribution only. The error term (difference btw calculated and actual income) is included in the taxed revenue. We therefore calculate a pre-tax error term.
	Note: obsolete in the latest version of the model.
	'''
	inc_bf=inc.copy()
	errorterm=finalhhframe['totY']-Y_from_inc(finalhhframe,inc)
	gdpobserved=GDP(finalhhframe['totY'],finalhhframe['weight'])
	pensions=GDP(Y_from_inc(finalhhframe,inc[['old']]),finalhhframe['weight'])
	benefits=GDP(Y_from_inc(finalhhframe,inc[['cat7workers']]),finalhhframe['weight'])
	p=pensions/gdpobserved
	b=benefits/gdpobserved
	for thecat in range(1,7):
		string='cat{}workers'.format(int(thecat))
		inc_bf[string]=inc[string]*1/(1-p-b)
	inc_bf['old']=0
	inc_bf['cat7workers']=0
	errorterm=errorterm*1/(1-p-b)
	return b,p,errorterm,inc_bf
	
def keep_workers(inputs):
	thebool=(inputs.index!='old')&(inputs.index!='cat7workers')
	return thebool

	
def after_pensions(inc,errorterm,p,finalhhframe):
	'''
	Transfers income from workers to retirees. The error term is taxed also, only for households that are not only composed of unemployed or elderlies.
	'''
	inigdp = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	select = ~((finalhhframe['cat7workers']+finalhhframe['old'])==(finalhhframe['nbpeople']-finalhhframe['children']))
	inc_af = inc.copy()
	for thecat in range(1,7):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-p)
	errorterm[select] = errorterm[select]*(1-p)
	totrev            = inigdp-GDP(Y_from_inc(finalhhframe,inc_af)+errorterm,finalhhframe['weight'])
	pensions          = totrev/sum(finalhhframe['old']*finalhhframe['weight'])
	inc_af['old']     = inc['old']+pensions
	return inc_af,errorterm
	
def after_bi(inc,errorterm,b,finalhhframe):
	'''Recalculates the after-basic-income incomes. All categories are taxed (including unemployed and retirees) and all adults receive the basic income. The error term is taxed also.'''
	inc_af  = inc.copy()
	gdpcalc = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	bI      = b*gdpcalc/sum((finalhhframe['nbpeople']-finalhhframe['children'])*finalhhframe['weight'])
	for thecat in range(1,8):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-b)+bI
	inc_af['old'] = inc['old']*(1-b)+bI
	errorterm     = errorterm*(1-b)
	return inc_af,errorterm
	
def future_income_simple(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,price_increase):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''
	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']

	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)*(1+price_increase)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	futureinc = temperature_impact(futureinc,shares_outside,temp_impact)
	
	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(futureinc,futurerrorterm,b,futurehhframe)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc
	
def future_income_simple_no_cc(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''
	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']
	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(futureinc,futurerrorterm,b,futurehhframe)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc
	
###################### In progress: functions to split households with a very high weight into several types of households ###########

def weighted_std(values, weights):
	'''Weighted standard deviation'''
	average = np.average(values, weights=weights)
	variance = np.average((values-average)**2, weights=weights)
	return np.sqrt(variance)

def create_new_hh(a_serie,chosen_std):
	'''
	Creates new households based on those that have very high weights after the re-weighting process (currently not used but in progress)
	'''

	#h=med/m = exp(-s**2/2)   log(h) = -s**2/2
	#h=med/m   med = m*h = exp(mu)   mu = log(m*h)
	h = np.exp(-chosen_std**2/2)
	norm=scipy.stats.lognorm(s=chosen_std,loc=np.log(h*a_serie.Y))
	x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 50)
	oo = DataFrame(columns=a_serie.index)
	oo = oo.append([a_serie]*50,ignore_index=True)
	oo['Y'] = x
	oo.ix[oo['Y']<0,'Y']=0
	oo['weight'] = oo['weight']*norm.pdf(x)/sum(norm.pdf(x))
	oo.index = [str(a_serie.name)+"_"+str(i) for i in oo.index]
	return oo
	
def add_errors_to_distrib(futurehhframe_old,e,w_th):
	'''
	Creates normal income distributions around households that have very high weights after the re-weighting process (currently not used but in progress)
	'''
	futurehhframe = futurehhframe_old.copy()
	cat_df = futurehhframe.ix[futurehhframe.Y<float(perc_with_spline(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,0.9)),:].groupby(['cat1workers', 'cat2workers','cat3workers', 'cat4workers', 'cat5workers', 'cat6workers','cat7workers']).apply(lambda x:weighted_std(x.Y,x.weight*x.nbpeople))
	chosen_std = np.median(cat_df[cat_df>0])/e
	select = futurehhframe.weight*futurehhframe.nbpeople>w_th*sum(futurehhframe.weight*futurehhframe.nbpeople)
	t = futurehhframe.ix[select,:].copy()
	futurehhframe = futurehhframe.drop(futurehhframe.ix[select,:].index)
	new_households = DataFrame()
	for index, row in t.iterrows():
		new_households = new_households.append(create_new_hh(row,chosen_std))
	futurehhframe = futurehhframe.append(new_households)
	return futurehhframe


###################### Functions used to calculate indicators at the end of the run ############################################

def GDP(income,weights):
	GDP = np.nansum(income*weights)
	return GDP
	
def actual_productivity_growth(finalhhframe,inc,futurehhframe,futureinc,year,ini_year):
	
	out=list()
	
	for cat1,cat2 in (['cat1workers','cat2workers'],['cat3workers','cat4workers'],['cat5workers','cat6workers']):
		prod_ini  = (sum(finalhhframe[cat1]*finalhhframe['weight'])*inc[cat1]+sum(finalhhframe[cat2]*finalhhframe['weight'])*inc[cat2])/sum((finalhhframe[cat1]+finalhhframe[cat2])*finalhhframe['weight'])
		prod_last = (sum(futurehhframe[cat1]*futurehhframe['weight'])*futureinc[cat1]+sum(futurehhframe[cat2]*futurehhframe['weight'])*futureinc[cat2])/sum((futurehhframe[cat1]+futurehhframe[cat2])*futurehhframe['weight'])
		
		prod_gr   = (prod_last/prod_ini)**(1/(year-ini_year))-1
		out.append(prod_gr)
	return tuple(out)
	
def poor_people(income,weights,povline):
	isbelowline = (income<povline)
	thepoor     = weights.values*isbelowline.values
	nbpoor      = thepoor.sum()
	return nbpoor
	
def find_perc(y,w,theperc,density):
	'''
	The very sophisticated way of finding percentiles
	'''
	normalization = integrate.quad(density,0,np.inf)
	estime = wp(y,w,[theperc],cum=False)
	def find_root(x,normalization,density,theperc):
		integrale = integrate.quad(density,0,x)
		return integrale[0]/normalization[0]-theperc
	out = optimize.fsolve(find_root, estime, args=(normalization,density,theperc))
	return out
	
def poverty_indic_kde(income,weights,threshold,density):
	'''
	The very sophisticated way of finding percentiles and the average income of people between percentiles
	'''
	if type(threshold)==float:
		inclim20        = find_perc(income,weights,threshold,density)[0]
		isbelowline     = (income<=inclim20)
	elif type(threshold)==list:
		minlim = find_perc(income,weights,threshold[0],density)[0]
		maxlim = find_perc(income,weights,threshold[1],density)[0]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic_spec(income,weights,threshold):
	'''
	For special cases
	'''
	if type(threshold)==list:
		minlim = threshold[0]
		maxlim = threshold[1]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	else:
		isbelowline     = (income<=threshold)
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic(percentiles,limit1,limit2):
	out = percentiles[limit1:limit2].sum()/(limit2-limit1)
	return out
	
def gini(income,weights):
	inc = np.asarray(reshape_data(income))
	wt  = np.asarray(reshape_data(weights))
	i   = np.argsort(inc) 
	inc = inc[i]
	wt  = wt[i]
	y   = np.cumsum(np.multiply(inc,wt))
	y   = y/y[-1]
	x   = np.cumsum(wt)
	x   = x/x[-1]
	G   = 1-sum((y[1::]+y[0:-1])*(x[1::]-x[0:-1]))
	return G
	
def poverty_gap(income,weights,povline):
	isbelowline = (income<povline)
	gap         = sum((1-income[isbelowline]/povline)*weights[isbelowline]/sum(weights))
	return gap
		
def distrib2store(income,weights,nbdots,tot_pop):
	categories         = np.arange(0, 1+1/nbdots, 1/nbdots)
	y                  = np.asarray(wp(reshape_data(income),reshape_data(weights),categories,cum=False))
	inc_o              = (y[1::]+y[0:-1])/2
	o2store            = DataFrame(columns=['income','weights'])
	o2store['income']  = list(inc_o)
	o2store['weights'] = list([tot_pop/nbdots]*len(inc_o))
	return o2store
	
def indicators_from_pop_desc(ini_pop_desc):
	children      = ini_pop_desc['children']
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])
	earn_income   = adults+float(ini_pop_desc['old'])
	tot_pop       = earn_income+children
	skilled       = float(ini_pop_desc['skillworkers'])
	
	shareemp_ini  = float(1-ini_pop_desc['unemployed']/adults)
	shareag_ini   = ag/work
	sharemanu_ini = manu/work
	
	share_skilled = skilled/adults
	
	return shareemp_ini,shareag_ini,sharemanu_ini,share_skilled
	

def calc_indic(income_proj,weights_proj_tot,weights_proj,futurehhframe,data2day,futureinc,povline):
	'''
	Indicators that are calculated at the end and stored. Since we cannot store the entire survey for each scenario we summarize the information with these indicators.
	'''
	# density = gaussian_kde(income_proj,weights=weights_proj_tot)
	# density._compute_covariance()
	
	indicators                 = DataFrame()
	percentiles                = perc_with_spline(income_proj,weights_proj_tot,np.arange(0,1,0.01))
	quintilescum               = wp(income_proj,weights_proj_tot,[0.2,0.4,1])
	quintilespc                = quintilescum/quintilescum[-1]
	indicators['GDP']          = [GDP(income_proj,weights_proj_tot)]
	indicators['avincome']     = [np.average(income_proj,weights=weights_proj_tot)]
	indicators['incbott10']    = [poverty_indic(percentiles,0,10)]
	indicators['incbott20']    = [poverty_indic(percentiles,0,20)]
	indicators['inc2040']      = [poverty_indic(percentiles,20,40)]
	indicators['incbott40']    = [poverty_indic(percentiles,0,40)]
	indicators['quintilecum1'] = [quintilescum[0]]
	indicators['quintilecum2'] = [quintilescum[1]]
	indicators['quintilepc1']  = [quintilespc[0]]
	indicators['quintilepc2']  = [quintilespc[1]]
	indicators['belowpovline']     = [poor_people(income_proj,weights_proj_tot,povline*data2day)]
	indicators['below2']       = [poor_people(income_proj,weights_proj_tot,2*data2day)]
	indicators['below4']       = [poor_people(income_proj,weights_proj_tot,4*data2day)]
	indicators['below6']       = [poor_people(income_proj,weights_proj_tot,6*data2day)]
	indicators['below8']       = [poor_people(income_proj,weights_proj_tot,8*data2day)]
	indicators['below10']      = [poor_people(income_proj,weights_proj_tot,10*data2day)]
	indicators['gini']         = [gini(income_proj,weights_proj_tot)]
	indicators['tot_pop']      = [sum(weights_proj*futurehhframe['nbpeople'])]
	indicators['gappovline']       = [poverty_gap(income_proj,weights_proj_tot,povline*data2day)]
	indicators['gap2']         = [poverty_gap(income_proj,weights_proj_tot,2*data2day)]
	indicators['gap4']         = [poverty_gap(income_proj,weights_proj_tot,4*data2day)]
	indicators['gap6']         = [poverty_gap(income_proj,weights_proj_tot,6*data2day)]
	indicators['gap8']         = [poverty_gap(income_proj,weights_proj_tot,8*data2day)]
	indicators['gap10']        = [poverty_gap(income_proj,weights_proj_tot,10*data2day)]
		
	ag = (futurehhframe['cat3workers']>0)|(futurehhframe['cat4workers']>0)
	percentiles_ag = perc_with_spline(income_proj[ag],weights_proj_tot[ag],np.arange(0,1,0.01))
	percentiles_nonag = perc_with_spline(income_proj[~ag],weights_proj_tot[~ag],np.arange(0,1,0.01))
	
	for pp in np.arange(0,100):
		indicators['percentiles_ag_{}'.format(pp)] = percentiles_ag[pp]
		indicators['percentiles_nonag_{}'.format(pp)] = percentiles_nonag[pp]
	
	indicators['childrenag']  = sum(futurehhframe.ix[ag,'children']*futurehhframe.ix[ag,'weight'])
	indicators['childrenonag']= sum(futurehhframe.ix[~ag,'children']*futurehhframe.ix[~ag,'weight'])
	indicators['peopleag']    = sum(futurehhframe.ix[ag,'nbpeople']*futurehhframe.ix[ag,'weight'])
	indicators['peoplenonag'] = sum(futurehhframe.ix[~ag,'nbpeople']*futurehhframe.ix[~ag,'weight'])
	
	indicators['avincomeag'] = [np.average(income_proj[ag],weights=weights_proj_tot[ag])]
	indicators['avincomenonag']   = [np.average(income_proj[~ag],weights=weights_proj_tot[~ag])]
	
	return indicators
	
	