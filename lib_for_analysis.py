from pandas import read_excel,concat,Series,DataFrame,read_csv,isnull,notnull,HDFStore,set_option
import numpy as np
import matplotlib.pyplot as plt
from perc import wp
from lib_for_growth_model import get_gdp_growth
from sklearn import tree
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols



def find_median(data,variable,thlist):
    var=data[variable]
    out=np.percentile(var,thlist)
    return out
	
def optandpess(data,varpoors,varineq,thlist):
	th1=find_median(data,varpoors,thlist)
	th2=find_median(data,varineq,thlist)
	if varpoors=="incbott20":
		opt=(data.ix[:,varineq]>th2[1])&(data.ix[:,varpoors]>th1[1])
		pess=(data.ix[:,varineq]<th2[0])&(data.ix[:,varpoors]<th1[0])
	else:
		opt=(data.ix[:,varineq]>th2[1])&(data.ix[:,varpoors]<th1[0])
		pess=(data.ix[:,varineq]<th2[0])&(data.ix[:,varpoors]>th1[1])
	return opt,pess

def best_scenar(alist):
    alls=[item for sublist in alist for item in sublist]
    l_counts = [ (alls.count(x), x) for x in set(alls)]
    l_counts.sort(reverse=True)
    l_result = [ y for x,y in l_counts ]
    return l_counts[0:10]

def intercept_scenar(alistoflists):
    result = set(alistoflists[0])
    for s in alistoflists[1:]:
        result.intersection_update(s)
    return result

def countries_missing(thescenars,alistoflists):
    finallistoflist=list()
    for ascenar in thescenars:
        listofcountries=list()
        for s in alistoflists:
            if ascenar not in s.values:
                listofcountries.append(s.name)
        finallistoflist.append(listofcountries)
    return finallistoflist

def other_scenar(listofcountries,alistoflists):
    firsttime=True
    for s in alistoflists:
        if (s.name in listofcountries) and firsttime:
            result = set(s)
            firsttime=False
        elif (s.name in listofcountries)and(not firsttime):
            result.intersection_update(s)
    return result
	
def choose_ssp(varpoor,varineq,outcomes,experiments,cc,isplot=True):
	filtering=experiments['isssp4']
	
	qt_poor5=outcomes.ix[~filtering,varpoor].describe()
	qt_ineq5=outcomes.ix[~filtering,varineq].describe()
	# thres5poor=(qt_poor5['max']+qt_poor5['min'])/2
	thres5poor=qt_poor5['50%']
	# thres5ineq=(qt_ineq5['max']+qt_ineq5['min'])/2
	thres5ineq=qt_ineq5['50%']
	
	qt_poor4=outcomes.ix[filtering,varpoor].describe()
	qt_ineq4=outcomes.ix[filtering,varineq].describe()
	# thres4poor=(qt_poor4['max']+qt_poor4['min'])/2
	thres4poor=qt_poor4['50%']
	# thres4ineq=(qt_ineq4['max']+qt_ineq4['min'])/2
	thres4ineq=qt_ineq4['50%']
	
	selectssp5=(~filtering)&(outcomes[varpoor]<thres5poor)&(outcomes[varineq]>thres5ineq)
	selectssp4=(filtering)&(outcomes[varpoor]>thres4poor)&(outcomes[varineq]<thres4ineq)
	
	if thres5poor==qt_poor5['min']:
		selectssp5=(~filtering)&(outcomes[varineq]>qt_ineq5['50%'])
		
	
	if isplot:
		plt.figure(figsize=(12,4))
		plt.subplot(121)
		plt.plot(outcomes.ix[~filtering,varpoor],[qt_ineq5['50%']]*len(outcomes.ix[~filtering,varineq]),'k')
		plt.plot([qt_poor5['50%']]*len(outcomes.ix[~filtering,varineq]),outcomes.ix[~filtering,varineq],'k')
		plt.plot(outcomes.ix[~filtering,varpoor],outcomes.ix[~filtering,varineq],'ko')
		plt.plot(outcomes.ix[selectssp5,varpoor],outcomes.ix[selectssp5,varineq],'bo')
		plt.axis([outcomes[varpoor].min()*0.8,outcomes[varpoor].max()*1.1, outcomes[varineq].min()*1.1,outcomes[varineq].max()*1.1])
		plt.xlabel(varpoor)
		plt.ylabel(varineq)
		plt.title('SSP5')
		plt.subplot(122)
		plt.plot(outcomes.ix[filtering,varpoor],[qt_ineq4['50%']]*len(outcomes.ix[filtering,varineq]),'k')
		plt.plot([qt_poor4['50%']]*len(outcomes.ix[filtering,varineq]),outcomes.ix[filtering,varineq],'k')
		plt.plot(outcomes.ix[filtering,varpoor],outcomes.ix[filtering,varineq],'ko')
		plt.plot(outcomes.ix[selectssp4,varpoor],outcomes.ix[selectssp4,varineq],'ro')
		plt.axis([outcomes[varpoor].min()*0.8,outcomes[varpoor].max()*1.1, outcomes[varineq].min()*1.1,outcomes[varineq].max()*1.1])
		plt.xlabel(varpoor)
		plt.ylabel(varineq)
		plt.title('SSP4')
		plt.savefig('png/'+cc+'.png')
		plt.close()
	return selectssp5,selectssp4
	
def compare2GDP(var1,var2,outcomes,filtering,gr5,gr4,inigdp):
	y=outcomes.ix[:,'GDP']/inigdp.values[0]
	plt.figure(figsize=(12,4))
	plt.subplot(121)
	plt.plot(outcomes.ix[~filtering,var1],y[~filtering],'bo')
	plt.plot(outcomes.ix[filtering,var1],y[filtering],'ro')
	plt.plot(outcomes[var1],[gr5]*len(outcomes),'b')
	plt.plot(outcomes[var1],[gr4]*len(outcomes),'r')
	plt.xlabel(var1)
	plt.ylabel('GDP growth')
	plt.subplot(122)
	plt.plot(outcomes.ix[~filtering,var2],y[~filtering],'bo')
	plt.plot(outcomes.ix[filtering,var2],y[filtering],'ro')
	plt.plot(outcomes[var2],[gr5]*len(outcomes),'b')
	plt.plot(outcomes[var2],[gr4]*len(outcomes),'r')
	plt.xlabel(var2)
	plt.ylabel('GDP growth')
	
def test_GDP(future_gdp_ssp5,future_gdp_ssp4,results,countrycode):
	out=False
	if (results['GDP'].max()<future_gdp_ssp5.values[0]):
		print(countrycode+": all GDP lower than SSP5")
		print(results['GDP'].describe()[['50%','max']],future_gdp_ssp5)
	elif (results['GDP'].min()>future_gdp_ssp4.values[0]):
		print(countrycode+": all GDP higher than SSP4")
		print(results['GDP'].describe()[['50%','min']],future_gdp_ssp4)
	elif sum((results['GDP']>0.9*future_gdp_ssp4.values[0])&(results['GDP']<1.1*future_gdp_ssp5.values[0])&(results['ssp']=='ssp4'))<len(results)/10:
		print(countrycode+" not enough scenarios for ssp4")
	elif sum((results['GDP']>0.9*future_gdp_ssp4.values[0])&(results['GDP']<1.1*future_gdp_ssp5.values[0])&(results['ssp']=='ssp5'))<len(results)/10:
		print(countrycode+" not enough scenarios for ssp5")
	else:
		out=True
	return out
	
def find_closest2today_scenar(outcomes,experiments,scenarios2keep,ini_data,cc,year,ini_year,isopt=True):
	
	if isopt:
		inichoice=(outcomes['opt']==1)&(experiments['isssp5'])
		if sum(inichoice)==0:
			inichoice=(outcomes['opt']==1)
			print(cc+" opt not ssp5 pop")
	else:
		inichoice=(outcomes['pess']==1)&(1-experiments['isssp5'])
		if sum(inichoice)==0:
			inichoice=(outcomes['pess']==1)
			print(cc+" pess not ssp4 pop")
	
	sous_scenar_opt=scenarios2keep.ix[inichoice,:]
	
	# if (isopt)&(outcomes.ix[inichoice,'poors'].max()>0.035):
		# w=0.5
		# opt=w*(outcomes.ix[inichoice,'poors'])**2
	# else:
		# opt=0
		# w=0
	opt=0
	w=0
	
	for gr in ['gr1','gr2','gr3','gr4','gr5','gr6']:
		opt+=(1-w)*(experiments.ix[inichoice,gr])**2
	for share in ['agchange','manuchange','empchange']:
		opt+=(1-w)*((experiments.ix[inichoice,share])**(1/(year-ini_year))-1)**2
	
	outopt=sous_scenar_opt.ix[opt==opt.min(),'scenar']
	thebool=inichoice&(scenarios2keep['scenar']==outopt.values[0])
	return outopt.values[0],thebool
	
def find_the_scenario(future_gdp_ssp5,future_gdp_ssp4,outcomes,experiments,scenarios2keep,ini_data,cc,isopt=True):
	if isopt:
		inichoice=(outcomes['opt']==1)&(experiments['isssp5'])
		if sum(inichoice)==0:
			inichoice=(outcomes['opt']==1)
			print(cc+" opt not ssp5 pop")
		gdp2compare=future_gdp_ssp5.values
		opt = experiments.ix[inichoice,'agchange']
		opt += (1-outcomes.ix[inichoice,'shprosp'])
		# for skillp in ['skillpag','skillpmanu','skillpserv']:
			# opt+=(experiments.ix[inichoice,skillp])**2
		# opt=0
	else:
		inichoice=(outcomes['pess']==1)&(1-experiments['isssp5'])
		if sum(inichoice)==0:
			inichoice=(outcomes['pess']==1)
			print(cc+" pess not ssp4 pop")
		gdp2compare=future_gdp_ssp4.values
		for share in ['agchange','manuchange','empchange']:
			opt=(experiments.ix[inichoice,share]-1)**2
		opt+=(outcomes.ix[inichoice,'incbott20']/ini_data.ix[ini_data['countrycode']==cc,'incbott20'].values[0]-1)**2
		# opt+=(outcomes.ix[inichoice,'below125']/(1+ini_data.ix[ini_data['countrycode']==cc,'below125'].values[0])-1)**2
		# for skillp in ['skillpag','skillpmanu','skillpserv']:
			# opt+=(experiments.ix[inichoice,skillp])**2
			
	opt+=(outcomes['GDP']/gdp2compare-1)**2
	
	sous_scenar_opt=scenarios2keep.ix[inichoice,:]
		
	outopt=sous_scenar_opt.ix[opt==opt.min(),'scenar']
	thebool=inichoice&(scenarios2keep['scenar']==outopt.values[0])
	return outopt.values[0],thebool

	
def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     
    def recurse(left, right, child, lineage=None):          
        if lineage is None:
           lineage = [child]
        if child in left:
           parent = np.where(left == child)[0].item()
           split = '<='
        else:
           parent = np.where(right == child)[0].item()
           split = '>='

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
           lineage.reverse()
           return lineage
        else:
           return recurse(left, right, parent, lineage)
    out = list() 
    threevar = list()
    threesign = list()
    threevalues = list()
    for child in idx: 
        thestring=str()
        vartuple=list()
        signtuple=list()
        valuetuple=list()
        for thetuple in recurse(left, right, child):
            if type(thetuple)==tuple:
                thestring+="(experiments["+"\'"+str(thetuple[3])+"\'"+"]"+thetuple[1]+str(thetuple[2])+")&"
                vartuple.append(thetuple[3])
                signtuple.append(thetuple[1])
                valuetuple.append(thetuple[2])
        thestring = thestring[:-1]
        out.append(thestring)
        threevar.append(vartuple)
        threesign.append(signtuple)
        threevalues.append(valuetuple)
    return out,threevar,threesign,threevalues
	
def find_scenars_from_tree(var1,var2,outcomes,experiments,w,isopt=True):
	x=experiments.values
	if var2 is None:
		y=outcomes[var1].values
	else:
		y=outcomes[[var1,var2]].values
	header=experiments.columns
	mass_min = 0.05
	clf = tree.DecisionTreeRegressor(max_depth=3,min_samples_leaf=int(mass_min*x.shape[0]))
	clf = clf.fit(x,y)
	list_boxes,threevar,threesign,threevalues=get_lineage(clf, header)
	poorsandineq=list()
	for ii in list_boxes:
		thebool=eval(ii)
		if var2 is None:
			poorsandineq.append(outcomes.ix[thebool,var1].mean())
		else:
			poorsandineq.append([outcomes.ix[thebool,var1].mean(),outcomes.ix[thebool,var2].mean()])
	poorsandineq=np.asarray(poorsandineq)
	if var2 is None:
		obj=-poorsandineq
	else:
		obj=w*poorsandineq[:,0]-(1-w)*poorsandineq[:,1]
	if isopt:
		outstr=list_boxes[list(obj).index(min(obj))]
		threevarout=threevar[list(obj).index(min(obj))]
		threesignout=threesign[list(obj).index(min(obj))]
		threevaluesout=threevalues[list(obj).index(min(obj))]
	else:
		outstr=list_boxes[list(obj).index(max(obj))]
		threevarout=threevar[list(obj).index(max(obj))]
		threesignout=threesign[list(obj).index(max(obj))]
		threevaluesout=threevalues[list(obj).index(max(obj))]
	return outstr,threevarout,threesignout,threevaluesout

	
def eliminate_absurd(results,wbreg,future_gdp_ssp4,future_gdp_ssp5,cc):
	if results['quintile2'].dtype!='float64':
		results=results.ix[results['quintile2']!='Infeasible',:]
		results['quintile2']=results['quintile2'].astype(float)
	if (wbreg=="EAP")|(wbreg=="SAS"):
		absurd=(results['GDP']<0.9*future_gdp_ssp4.values[0])|(results['GDP']>1.1*future_gdp_ssp5.values[0])|(results['quintile2']<0)|(results['quintilepc2']>0.4)
	elif (wbreg=="LAC")|(wbreg=="ECA")|(wbreg=="MNA"):
		absurd=(results['GDP']<0.9*future_gdp_ssp4.values[0])|(results['GDP']>1.1*future_gdp_ssp5.values[0])|(results['quintile2']<0)|(results['quintilepc2']>0.4)
	elif (wbreg=="SSA"):
		absurd=(results['GDP']<0.9*future_gdp_ssp4.values[0])|(results['GDP']>1.2*future_gdp_ssp5.values[0])|(results['quintile2']<0)|(results['quintilepc2']>0.4)
	if cc=='BGR':
		absurd=(results['GDP']<0.7*future_gdp_ssp4.values[0])|(results['GDP']>1.3*future_gdp_ssp5.values[0])|(results['quintile2']<0)|(results['quintilepc2']>0.4)
	new_results=results[~absurd]
	return new_results
	
def append_outputs(df,cc,threevarout,outcomes,scenarios2keep,experiments,thebool,correlation):
	thescenar=scenarios2keep.ix[thebool,'scenar'].values[0]
	ind=outcomes.ix[thebool,:].index
	if len(threevarout)==3:
		df=df.append(concat([DataFrame([{'correlation':correlation,'countrycode':cc,'scenar':thescenar,'1st determinant':threevarout[0],'2nd determinant':threevarout[1],'3rd determinant':threevarout[2]}],index=ind),outcomes.ix[thebool,:],experiments.ix[thebool,:]],axis=1),ignore_index=True)
	elif len(threevarout)==2:
		df=df.append(concat([DataFrame([{'correlation':correlation,'countrycode':cc,'scenar':thescenar,'1st determinant':threevarout[0],'2nd determinant':threevarout[1]}],index=ind),outcomes.ix[thebool,:],experiments.ix[thebool,:]],axis=1),ignore_index=True)
	elif len(threevarout)==2:
		df=df.append(concat([DataFrame([{'correlation':correlation,'countrycode':cc,'scenar':thescenar,'1st determinant':threevarout[0]}],index=ind),outcomes.ix[thebool,:],experiments.ix[thebool,:]],axis=1),ignore_index=True)
	return df
	
def append_drivers(df,cc,threevarout,threesignout,threevaluesout):
	for i in range(0,len(threevarout)):
		var=threevarout[i]
		df.ix[df['countrycode']==cc,'{} sign'.format(var)]=threesignout[i]
		df.ix[df['countrycode']==cc,var]=threevaluesout[i]
	return df
	
def change_name(symbol):
	if symbol=='b':
		out='Redistribution'
	elif (symbol=='gr3'):
		out='Agriculture growth (unskilled)'
	elif (symbol=='gr4'):
		out='Agriculture growth (skilled)'
	elif (symbol=='agchange'):
		out='Agriculture share change'
	elif (symbol=='shareag'):
		out="Agriculture share"
	elif symbol=='manuchange':
		out='Manufacture share change'
	elif (symbol=='sharemanu'):
		out="Manufacture share"
	elif (symbol=='empchange'):
		out='Participation change'
	elif (symbol=='shareemp'):
		out='Participation'
	elif symbol=='isssp5':
		out='Population'
	elif (symbol=='gr1'):
		out='Services growth (unskilled)'
	elif (symbol=='gr2'):
		out='Services growth (skilled)'
	elif (symbol=='gr5'):
		out='Manufacture growth (unskilled)'
	elif (symbol=='gr6'):
		out='Manufacture growth (skilled)'
	elif (symbol=='p'):
		out='Tax for pensions'
	else:
		out=symbol
	return out
	
def leg_from_strings(threevaroutopt,threesignoutopt,threevaluesoutopt,i):
	firstpart=change_name(threevaroutopt[i])
	if firstpart=="Population":
		if threesignoutopt[i]==">=":
			firstpart="Low Population"
			secondpart=""
			thirdpart=""
		else:
			firstpart="High Population"
			secondpart=" "
			thirdpart=""
	elif firstpart=="Participation change":
		secondpart=threesignoutopt[i]
		thirdpart=str(round(100*(float(threevaluesoutopt[i])-1),1))+" %"
	else:
		secondpart=threesignoutopt[i]
		thirdpart=str(round(100*(float(threevaluesoutopt[i])),1))+" %"
		
	leg=firstpart+secondpart+thirdpart
	return leg
	
def drivers_from_anova(varin,data,experiments_cols):
	formula = varin+" ~ " + "+".join(experiments_cols)
	olsmodel=ols(formula,data=data).fit()
	table=anova_lm(olsmodel)
	table['sum_sq_pc']=table['sum_sq']/table['sum_sq'].sum()
	table=table.sort(['sum_sq'],ascending=False)
	sumvar=0
	drivers=list()
	for var in table.index:
		if var!='Residual':
			drivers.append(var)
			sumvar+=table.loc[var,'sum_sq_pc']
		if len(drivers)==3:
			break
	return drivers,sumvar
	
		
	