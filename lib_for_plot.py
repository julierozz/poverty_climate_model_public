from itertools import cycle, islice
from matplotlib.ticker import FuncFormatter as funfor
import matplotlib.pyplot as plt

def y_thousands_sep(ax=None):
    if ax is None:
        ax=plt.gca()
    ax.get_yaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
	
def savefig(path, **kwargs):
    #Saves in both png and pdf
    
    plt.tight_layout()
    
    path = path.replace(".png","")
    path = path.replace(".pdf","")

    plt.savefig(path+".png", )
    plt.savefig(path+".pdf", )
	
def spine_and_ticks(ax,reverted=False, thousands=False):
    
    if reverted:
    
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color("none")

        #removes ticks 
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        
    else:
        
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color("none")

            #removes ticks 
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        
    if thousands:
        ax.get_xaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))