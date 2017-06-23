"""This computes order statistics on data with weights. 
"""
 
import numpy 
from scipy.interpolate import UnivariateSpline,interp1d
 
 
def wp(data, wt, percentiles,cum=False): 
	"""Compute weighted percentiles. 
	If the weights are equal, this is the same as normal percentiles. 
	Elements of the C{data} and C{wt} arrays correspond to 
	each other and must have equal length (unless C{wt} is C{None}). 

	@param data: The data. 
	@type data: A L{numpy.ndarray} array or a C{list} of numbers. 
	@param wt: How important is a given piece of data. 
	@type wt: C{None} or a L{numpy.ndarray} array or a C{list} of numbers. 
		 All the weights must be non-negative and the sum must be 
		 greater than zero. 
	@param percentiles: what percentiles to use.  (Not really percentiles, 
		 as the range is 0-1 rather than 0-100.) 
	@type percentiles: a C{list} of numbers between 0 and 1. 
	@rtype: [ C{float}, ... ] 
	@return: the weighted percentiles of the data. 
	"""
	assert numpy.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero" 
	assert numpy.less_equal(percentiles, 1.0).all(), "Percentiles greater than one" 
	data = numpy.asarray(data) 
	# data = numpy.reshape(data,(len(data)))
	assert len(data.shape) == 1 
	if wt is None: 
		 wt = numpy.ones(data.shape, numpy.float) 
	else: 
		 wt = numpy.asarray(wt, numpy.float) 
		 # wt = numpy.reshape(wt,(len(wt)))
		 assert wt.shape == data.shape 
		 assert numpy.greater_equal(wt, 0.0).all(), "Not all weights are non-negative." 
	i = numpy.argsort(data) 
	sd = numpy.take(data, i, axis=0)
	sw = numpy.take(wt, i, axis=0) 
	aw = numpy.add.accumulate(sw) 
	if not aw[-1] > 0: 
		 raise ValueError("Nonpositive weight sum" )
	w = (aw)/aw[-1] 
	spots = numpy.searchsorted(w, percentiles) 
	if cum:
		sd = numpy.add.accumulate(numpy.multiply(sd,sw))
	f = interp1d(w,sd)
	return f(percentiles)
	

def perc_with_spline(data, wt, percentiles):
	assert numpy.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero" 
	assert numpy.less_equal(percentiles, 1.0).all(), "Percentiles greater than one" 
	data = numpy.asarray(data) 
	assert len(data.shape) == 1 
	if wt is None: 
		wt = numpy.ones(data.shape, numpy.float) 
	else: 
		wt = numpy.asarray(wt, numpy.float) 
		assert wt.shape == data.shape 
		assert numpy.greater_equal(wt, 0.0).all(), "Not all weights are non-negative." 
	assert len(wt.shape) == 1 
	i = numpy.argsort(data) 
	sd = numpy.take(data, i, axis=0)
	sw = numpy.take(wt, i, axis=0) 
	aw = numpy.add.accumulate(sw) 
	if not aw[-1] > 0: 
	 raise ValueError("Nonpositive weight sum" )
	w = (aw)/aw[-1] 
	# f = UnivariateSpline(w,sd,k=1)
	f = interp1d(numpy.append([0],w),numpy.append([0],sd))
	return f(percentiles)	 
	


