import numpy as np
import matplotlib.pyplot as plt

def summary(data):
    print(f'Title: Petrophysical Summary of the Parameters Evaluated')
    return data.describe()


def log_plot(logs, depth=False, *argv):
    
    '''
    Plot log signatures of petrophysical parameters.
    Function accepts a dataframe and a depth argument.
    Pass True for the depth value if dataframe has a depth column, default is fault (uses index as depth)
    '''
    
    try:
        if depth == False:
            logs['DEPTH'] = logs.index
            logs = logs.reset_index(drop=True)
        else:
            logs['DEPTH'] = logs.depth
        
        
        logs = logs.sort_values(by='DEPTH')
        
        top = logs.DEPTH.min()
        bot = logs.DEPTH.max()
        
        f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,8))
        ax[0].plot(logs.GR, logs.DEPTH, color='black')
        ax[1].plot(logs.NPHI, logs.DEPTH, color='c')
        ax[2].plot(logs.RHOB, logs.DEPTH, color='blue')
        ax[3].plot(np.log(logs.RT), logs.DEPTH, color='red')
        
        for i in range(len(ax)):
            ax[i].set_ylim(top,bot)
            ax[i].invert_yaxis()
            ax[i].grid()
            
        ax[0].set_xlabel("GR (API)")
        ax[0].set_xlim(logs.GR.min(), 200)
        ax[0].set_ylabel("Depth(ft)")
        ax[1].set_xlabel("NPHI (v/v)")
        ax[1].set_xlim(logs.NPHI.min(),logs.NPHI.max())
        ax[2].set_xlabel("RHOB (g/cm3)")
        ax[2].set_xlim(logs.RHOB.min(),logs.RHOB.max())
        ax[3].set_xlabel("RT (ohm.m)")
        ax[3].set_xlim(-2,np.log(logs.RT.max()))
        
        ax[0].set_yticklabels([]); ax[1].set_yticklabels([]);
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([]); #ax[4].set_yticklabels([]) 
        
        f.suptitle('Log Plots, fontsize=14,y=0.94')
    
    except ModuleNotFoundError as err:
        print(f'Install module. {err}')

