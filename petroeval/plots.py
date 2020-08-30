import matplotlib.pyplot as plt
import numpy as np



def four_plot(logs, top, base, depth=False):
    
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

    logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]
            
    try:

        logs = logs.sort_values(by='DEPTH')

        f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,10))

        for i in range(len(ax)):
            ax[i].set_ylim(top, base)
            ax[i].invert_yaxis()
            ax[i].grid()
            ax[i].locator_params(axis='x', nbins=4)

        
        ax[0].plot(logs.GR, logs.DEPTH, color='black')
        ax[1].plot(logs.NPHI, logs.DEPTH, color='c')
        ax[2].plot(logs.RHOB, logs.DEPTH, color='blue')
        ax[3].plot(logs.RT, logs.DEPTH, color='red')
                    
        ax[0].set_xlabel("GR (API)")
        ax[0].set_xlim(logs.GR.min(), logs.GR.max())
        ax[0].set_ylabel("Depth(ft)")
        ax[0].set_title(f"Plot of Depth Against GR")
        ax[1].set_xlabel("NPHI (v/v)")
        ax[1].set_xlim(logs.NPHI.min(),logs.NPHI.max())
        ax[1].set_title(f"Plot of Depth Against Neutron Porosity")
        ax[2].set_xlabel("RHOB (g/cm3)")
        ax[2].set_xlim(logs.RHOB.min(),logs.RHOB.max())
        ax[2].set_title(f"Plot of Depth Against Density")
        ax[3].set_xlabel("RT (ohm.m)")
        ax[3].set_xscale("log")
        ax[3].set_xlim(logs.RT.min(), logs.RT.max())
        ax[3].set_title(f"Plot of Depth Against Resistivity")
        
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

def four_plots(logs, x1, x2, x3, x4, top, base, depth=False, scale=False):

    '''
    Function to automatically plot well logs
    Returns a plot of three logs(x1, x2, x3)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used

    '''

    #Setting the value of the y axis. Using index or property specified
    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth

    logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]

    if scale == True:  

        try:

            logs = logs.sort_values(by='DEPTH')
                        
            #top = logs.DEPTH.min()
            #bot = logs.DEPTH.max()
                        
            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)

            if (logs[x1].min() < logs[x2].min()) or (logs[x1].min() < logs[x3].min()) or (logs[x1].min() < logs[x4].min()):
                x_min=logs[x1].min()
            elif (logs[x2].min() < logs[x1].min()) or (logs[x2].min() < logs[x3].min() or (logs[x2].min() < logs[x4].min())):
                x_min=logs[x2].min()
            elif (logs[x3].min() < logs[x1].min()) or (logs[x3].min() < logs[x2].min() or (logs[x3].min() < logs[x4].min())):
                x_min=logs[x3].min()
            else:
                x_min=logs[x4].min()
                    
            if (logs[x1].max() < logs[x2].max()) or (logs[x1].max() < logs[x3].max()) or (logs[x1].max() < logs[x4].max()):
                x_max=logs[x1].max()
            elif (logs[x2].max() < logs[x1].max()) or (logs[x2].max() < logs[x3].max() or (logs[x2].max() < logs[x4].max())):
                x_max=logs[x2].max()
            elif (logs[x3].max() < logs[x1].max()) or (logs[x3].max() < logs[x2].max() or (logs[x3].max() < logs[x4].max())):
                x_max=logs[x3].max()
            else:
                x_max=logs[x4].max()
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
            ax[2].plot(logs[x3], logs.DEPTH, color='blue')
            ax[3].plot(logs[x4], logs.DEPTH, color='red')
                        
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(x_min, x_max)
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2} ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(x_min,x_max)
            ax[1].set_title(f"Plot of Depth Against {x2}")
            ax[2].set_xlabel(f"{x3}")
            if x3 == 'RT':
                ax[2].set_xscale("log")
            ax[2].set_xlim(x_min,x_max)
            ax[2].set_title(f"Plot of Depth Against {x3}")
            if x4 == 'RT':
                ax[3].set_xscale("log")
            ax[3].set_xlim(x_min,x_max)
            ax[3].set_title(f"Plot of Depth Against {x4}")
            ax[3].set_xlabel(f"{x4}")
                        
                    
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    elif scale == False:

        try:

            logs = logs.sort_values(by='DEPTH')
                        
            #top = logs.DEPTH.min()
            #bot = logs.DEPTH.max()
                        
            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
            ax[2].plot(logs[x3], logs.DEPTH, color='blue')
            ax[3].plot(logs[x4], logs.DEPTH, color='red')
                        
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(logs[x1].min(), logs[x1].max())
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2} ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(logs[x2].min(),logs[x2].max())
            ax[1].set_title(f"Plot of Depth Against {x2}")
            ax[2].set_xlabel(f"{x3}")
            if x3 == 'RT':
                ax[2].set_xscale("log")
            ax[2].set_xlim(logs[x3].min(),logs[x3].max())
            ax[2].set_title(f"Plot of Depth Against {x3}")
            if x4 == 'RT':
                ax[3].set_xscale("log")
            ax[3].set_xlim(logs[x3].min(),logs[x3].max())
            ax[3].set_title(f"Plot of Depth Against {x4}")
            ax[3].set_xlabel(f"{x3}")
                    
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    else:
        print(f'scale attribute takes in only True or False')

def three_plots(logs, x1, x2, x3, top, base, depth=False, scale=False):

    '''
    Function to automatically plot well logs
    Returns a plot of three logs(x1, x2, x3)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used

    '''

    #Setting the value of the y axis. Using index or property specified
    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)

    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth

    logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]

    if scale==True:
            
        try:

            logs = logs.sort_values(by='DEPTH')
                        
            #top = logs.DEPTH.min()
            #bot = logs.DEPTH.max()
                        
            f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)

            if (logs[x1].min() < logs[x2].min()) or (logs[x1].min() < logs[x3].min()):
                x_min=logs[x1].min()
            elif (logs[x2].min() < logs[x1].min()) or (logs[x2].min() < logs[x3].min()):
                x_min=logs[x2].min()
            else:
                x_min=logs[x3].min()
                    
            if (logs[x1].max() < logs[x2].max()) or (logs[x1].max() < logs[x3].max()):
                x_max=logs[x1].max()
            elif (logs[x2].max() < logs[x1].max()) or (logs[x2].max() < logs[x3].max()):
                x_max=logs[x2].max()
            else:
                x_max=logs[x3].max()
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
            ax[2].plot(logs[x3], logs.DEPTH, color='blue')
                        
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(x_min, x_max)
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2} ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(x_min, x_max)
            ax[1].set_title(f"Plot of Depth Against {x2}")
            ax[2].set_xlabel(f"{x3}")
            if x3 == 'RT':
                ax[2].set_xscale("log")
            ax[2].set_xlim(x_min, x_max)
            ax[2].set_title(f"Plot of Depth Against {x3}")
                        
                    
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    elif scale == False:

        try:

            logs = logs.sort_values(by='DEPTH')
                        
            #top = logs.DEPTH.min()
            #bot = logs.DEPTH.max()
                        
            f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
            ax[2].plot(logs[x3], logs.DEPTH, color='blue')
                        
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(logs[x1].min(), logs[x1].max())
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2} ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(logs[x2].min(),logs[x2].max())
            ax[1].set_title(f"Plot of Depth Against {x2}")
            ax[2].set_xlabel(f"{x3}")
            if x3 == 'RT':
                ax[2].set_xscale("log")
            ax[2].set_xlim(logs[x3].min(),logs[x3].max())
            ax[2].set_title(f"Plot of Depth Against {x3}")
                    
                
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    else:
        print(f'scale attribute takes in only True or False')

       
        
def two_plots(logs, x1, x2, top, base, depth=False, scale=False):

    '''
    Function to automatically plot well logs
    Returns a plot of two logs(x1, x2)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    

    #Converting the values of the resistivity logs to log scale
    if x1 == 'RT':
        logs[x1] = np.log(logs[x1])
        #logs[x1] = logs[x1].replace({np.Inf:0, np.nan:0}, inplace=False)

    if x2 == 'RT':
        logs[x2] = np.log(logs[x2])
        #logs[x2] = logs[x2].replace({np.Inf:0, np.nan:0}, inplace=False)

    '''

    #Setting the value of the y axis. Using index or property specified
    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth

    logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]

    if scale == True:        

        try:

            logs = logs.sort_values(by='DEPTH')
                        
            f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)
                

            if logs[x1].min() < logs[x2].min():
                x_min=logs[x1].min()
            else:
                x_min=logs[x2].min() 
                
            if logs[x1].max() < logs[x2].max():
                x_max=logs[x1].max()
            else:
                x_max=logs[x2].max() 
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
                            
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(x_min, x_max)
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2}  ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(x_min, x_max)
            ax[1].set_title(f"Plot of Depth Against {x2}")
                        
                    
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    elif scale == False:

        try:

            logs = logs.sort_values(by='DEPTH')
                        
            f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,10))

            for i in range(len(ax)):
                ax[i].set_ylim(top, base)
                ax[i].invert_yaxis()
                ax[i].grid()
                ax[i].locator_params(axis='x', nbins=4)
            
            ax[0].plot(logs[x1], logs.DEPTH, color='black')
            ax[1].plot(logs[x2], logs.DEPTH, color='c')
                            
            ax[0].set_xlabel(f"{x1}  ")
            if x1 == 'RT':
                ax[0].set_xscale("log")
            ax[0].set_xlim(logs[x1].min(), logs[x1].max())
            ax[0].set_ylabel("Depth(ft)")
            ax[0].set_title(f"Plot of Depth Against {x1}")
            ax[1].set_xlabel(f"{x2}  ")
            if x2 == 'RT':
                ax[1].set_xscale("log")
            ax[1].set_xlim(logs[x2].min(),logs[x2].max())
            ax[1].set_title(f"Plot of Depth Against {x2}")
                
        except NameError as err:
            print(f'Depth column could not be located. {err}')

    else:
        print(f'Attributes takes in True or False')



def one_plot(logs, x1, top, base, depth=False):

    '''
    Function to automatically plot a single well log
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    

    #Converting the values of the resistivity logs to log scale
    if x1 == 'RT':
        logs[x1] = np.log(logs[x1])
        #logs[x1] = logs[x1].replace({np.Inf:0, np.nan:0}, inplace=False)

    '''

    #Setting the value of the y axis. Using index or property specified   
    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth

    logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]
            
    try:

        logs = logs.sort_values(by='DEPTH')
                    
                    
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,15))
        ax.plot(logs[x1], logs.DEPTH, color='black')
        
        ax.set_ylim(top, base)
        ax.plot(logs[x1], logs.DEPTH, color='black')
        ax.invert_yaxis()
        ax.grid()
        ax.locator_params(axis='x', nbins=4)
                        
        ax.set_xlabel(f"{x1}")
        if x1 == 'RT':
            ax.set_xscale("log")
        ax.set_xlim(logs[x1].min(), logs[x1].max())
        ax.set_ylabel("Depth(ft)")
        ax.set_title(f"Plot of Depth Against {x1}")
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

    logs[x1] = np.log10(logs[x1])