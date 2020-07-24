import matplotlib.pyplot as plt
import numpy as np

def four_plots(logs, depth=False):
    
    '''
    Function to automatically plot well logs
    Returns a plot of four logs(Gamma ray, Porosity, Density and Resistivity)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    '''

    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth
            
    try:

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
            ax[0].set_title(f"Plot of Depth Against GR")
            ax[1].set_xlabel("NPHI (v/v)")
            ax[1].set_xlim(logs.NPHI.min(),logs.NPHI.max())
            ax[1].set_title(f"Plot of Depth Against Neutron Porosity")
            ax[2].set_xlabel("RHOB (g/cm3)")
            ax[2].set_xlim(logs.RHOB.min(),logs.RHOB.max())
            ax[2].set_title(f"Plot of Depth Against Density")
            ax[3].set_xlabel("RT (ohm.m)")
            ax[3].set_xlim(-2,np.log(logs.RT.max()))
            ax[3].set_title(f"Plot of Depth Against Resistivity")
                    
            f.suptitle('Log Plots, fontsize=14,y=0.94')
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

def three_plots(logs, x1, x2, x3, depth=False):

    '''
    Function to automatically plot well logs
    Returns a plot of three logs(x1, x2, x3)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    '''

    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth
            
    try:

        logs = logs.sort_values(by='DEPTH')
                    
        top = logs.DEPTH.min()
        bot = logs.DEPTH.max()
                    
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,8))
        
        ax[0].plot(logs[x1], logs.DEPTH, color='black')
        ax[1].plot(logs[x2], logs.DEPTH, color='c')
        ax[2].plot(logs[x3], logs.DEPTH, color='blue')
                    
        for i in range(len(ax)):
            ax[i].set_ylim(top,bot)
            ax[i].invert_yaxis()
            ax[i].grid()
                        
            ax[0].set_xlabel(f"{x1}  ")
            ax[0].set_xlim(logs[x1].min(), logs[x1].max())
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2} ")
            ax[1].set_xlim(logs[x2].min(),logs[x2].max())
            ax[1].set_title(f"Plot of Depth Against {x2}")
            ax[2].set_xlabel(f"{x3}")
            ax[2].set_xlim(logs[x3].min(),logs[x3].max())
            ax[2].set_title(f"Plot of Depth Against {x3}")
                    
            #f.suptitle('Log Plots, fontsize=14,y=0.94')
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

def two_plots(logs, x1, x2, depth=False):

    '''
    Function to automatically plot well logs
    Returns a plot of two logs(x1, x2)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    '''

    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth
            
    try:

        logs = logs.sort_values(by='DEPTH')
                    
        top = logs.DEPTH.min()
        bot = logs.DEPTH.max()
                    
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,10))
        
        ax[0].plot(logs[x1], logs.DEPTH, color='black')
        ax[1].plot(logs[x2], logs.DEPTH, color='c')
                    
        for i in range(len(ax)):
            ax[i].set_ylim(top,bot)
            ax[i].invert_yaxis()
            ax[i].grid()
                        
            ax[0].set_xlabel(f"{x1}  ")
            ax[0].set_xlim(logs[x1].min(), logs[x1].max())
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2}  ")
            ax[1].set_xlim(logs[x2].min(),logs[x2].max())
            ax[1].set_title(f"Plot of Depth Against {x2}")
                    
            #f.suptitle('Log Plots, fontsize=14,y=0.94')
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

def one_plot(logs, x1, depth=False):

    '''
    Function to automatically plot a single well log
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    '''

    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth
            
    try:

        logs = logs.sort_values(by='DEPTH')
                    
        top = logs.DEPTH.min()
        bot = logs.DEPTH.max()
                    
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,10))
        ax.plot(logs[x1], logs.DEPTH, color='black')
        
        ax.set_ylim(top,bot)
        ax.invert_yaxis()
        ax.grid()
                        
        ax.set_xlabel(f"{x1}")
        ax.set_xlim(logs[x1].min(), logs[x1].max())
        ax.set_ylabel("Depth(ft)")
        ax.set_title(f"Plot of Depth Against {x1}")
                    
        #ax.set_yticklabels([])
                    
        #f.suptitle('Log Plots, fontsize=14,y=0.94')
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')
