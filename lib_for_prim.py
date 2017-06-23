import sys
sys.path.insert(0, 'C:\\Users\\julierozenberg\\Documents\\GitHub\\EMAworkbench\\src')
import numpy.lib.recfunctions as recfunctions
from analysis import prim
from expWorkbench import ema_logging
from pandas import DataFrame
import numpy as np

class fPrim(prim.Prim):
    '''
    This is a small extension to the normal prim. This 
    extension adds functionality for automatically
    selecting a specific box on the peeling_trajectory
    found by normal prim. In the literature, this is
    known as fPrim. 
    
    The automatic selection is based on making a 
    tradeoff between coverage and density. More 
    specifically, the user specifies an f_value (between 0 and 1)
    that determines the weight of coverage, the weight
    of density then becomes 1-f_value. 
    
    The box on the peeling trajectory that is automatically chosen
    is the box that has the maximum score on the objective function.
    
    Outside of the automatic selection of a box, this extension has
    all the functionality of normal prim. 
    
    
    '''
    
    
    def __init__(self, 
                 results,
                 classify, 
                 f_value,
                 obj_function=prim.DEFAULT, 
                 peel_alpha=0.05, 
                 paste_alpha=0.05,
                 mass_min=0.05, 
                 threshold=None, 
                 threshold_type=prim.ABOVE,
                 incl_unc=[]):
        self.f_value = f_value
        
        super(fPrim, self).__init__(results, 
                                    classify,
                                    obj_function=obj_function, 
                                    peel_alpha=peel_alpha, 
                                    paste_alpha=paste_alpha,
                                    mass_min=mass_min, 
                                    threshold=threshold, 
                                    threshold_type=threshold_type,
                                    incl_unc=incl_unc)
        
        
    
    def find_box(self):
        box = super(fPrim, self).find_box()
        
        # here the f prim part should go
        obj = self.f_value *box.peeling_trajectory['coverage'] + (1-self.f_value)*box.peeling_trajectory['density']
        i = np.where(obj==np.max(obj))[0][0]
        
        box.select(i)
        box._cur_box = i
        
        return box
		
def format_data(outcomes,experiments,var):
	x = experiments.astype(float)
	y = outcomes.ix[:,var].values
	x = x.to_records()
	x = recfunctions.drop_fields(x, 'index')
	results = (x,{'y':y})
	return results
	
def classify(outcomes):
	outcome = outcomes['y']
	classes = np.zeros(outcome.shape)
	classes[(outcome==1)] =1
	return classes
	
def perform_prim(results):
	x,y     = results
	prim    = fPrim(results, classify, f_value=0.2, threshold=0.5, threshold_type=1)
	box     = prim.find_box()
	indices = box.yi
	logical = np.zeros(x.shape[0], dtype=np.bool)
	logical[indices] = 1
	
	index_last_box = box._cur_box
	box_lim = box.box_lims[index_last_box]
	res=DataFrame.from_records(box_lim)
	return logical,res