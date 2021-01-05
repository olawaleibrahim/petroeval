from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from utils import process
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def four_plot(logs, top, base, depth=False):
    '''
    Function to automatically plot well logs
    Returns a plot of four logs(Gamma ray, Porosity, Density and Resistivity)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used
    '''

    logs = process(logs)

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

        if logs.NPHI.max() == np.Inf or logs.NPHI.max() == np.nan:
            nphi_max = 0.9
        ax[0].plot(logs.GR, logs.DEPTH, color='black')
        ax[1].plot(logs.NPHI, logs.DEPTH, color='c')
        ax[2].plot(logs.RHOB, logs.DEPTH, color='blue')
        ax[3].plot(logs.RT, logs.DEPTH, color='red')
                    
        ax[0].set_xlabel("GR (API)")
        ax[0].set_xlim(logs.GR.min(), nphi_max)
        ax[0].set_ylabel("Depth(ft)")
        ax[0].set_title(f"Plot of Depth Against GR")
        ax[1].set_xlabel("NPHI (v/v)")
        ax[1].set_xlim(0, logs.NPHI.max())
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

def four_plots(logs, x1, x2, x3, x4, top, base, depth=False):
    '''
    Function to automatically plot well logs
    Returns a plot of three logs(x1, x2, x3)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used

    '''

    logs = process(logs)

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
        ax[3].set_xlabel(f"{x4}")
                    
                
    except NameError as err:
        print(f'Depth column could not be located. {err}')

def three_plots(logs, x1, x2, x3, top, base, depth=False):
    '''
    Function to automatically plot well logs
    Returns a plot of three logs(x1, x2, x3)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used

    '''

    logs = process(logs)

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


def two_plots(logs, x1, x2, top, base, depth=False):
    '''
    Function to automatically plot well logs
    Returns a plot of two logs(x1, x2)
    args::
          logs: Dataframe object of well logs
          depth: Set to false or leave as default to use dataframe index
                 Set to column title if column depth should be used

    '''

    logs = process(logs)

    #Setting the value of the y axis. Using index or property specified
    if depth == False:
        logs['DEPTH'] = logs.index
        logs = logs.reset_index(drop=True)
    else:
        depth = np.array(logs[depth])
        logs = logs.reset_index(drop=True)
        logs['DEPTH'] = depth

    #logs = logs.loc[(logs.DEPTH >= float(top)) & (logs.DEPTH <= float(base))]
            
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
        
        
def two_plot(logs, x1, x2, top, base, depth=False, scale=False):
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

    logs = process(logs)

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

    '''

    logs = process(logs)

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

'''
The functions below are adapted and modified from the SEG 2015 tutorials on SEG's
github page "The Leading Edge column";
https://github.com/seg/tutorials-2016/blob/master/1610_Facies_classification/
'''

def make_facies_log_plot(logs, x1, x2, x3, x4, x5, Depth=False):
    '''
    Plots well logs against depth and corresponding predicted lithofacies
        in a labelled color plot

        Arguments
        ---------

        df: dataframe of well data
        predictions: predicted values in integers (0, 1, 2....)
        log1: str -> well log 1
        ''''''''''''''''''''
        log5: str -> well log 2
        depth_col: depth column
    '''

    logs = logs.fillna(0)
    if Depth == False:
        logs['Depth'] = logs.index
        Depth = 'Depth'
        ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    else:
        logs['Depth'] = logs[Depth]
        Depth = 'Depth'
        ztop=logs.Depth.min(); zbot=logs.Depth.max()
        
    logs = logs.sort_values(by='Depth', ascending=True)

    facies_colors = [
        '#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', 
        '#AED6F1', '#A569BD', '#196F3D', '#10003D', '#A56222', '#000000'
    ]

    facies_labels = [
        'Sandstone', 'SS/SH', 'Shale', 'Marl', 'Dolomite',
        'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]

    facies_colormap = {}
    for ind, label in enumerate(facies_labels):
        facies_colormap[label] = facies_colors[ind]

    no = 12
    #no = len(list(dict(logs[target].value_counts())))
    cmap_facies = colors.ListedColormap(
            facies_colors[0 : no], 'indexed'
            )

    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 12))
    ax[0].plot(logs[x1], logs.Depth, '-g')
    ax[1].plot(logs[x2], logs.Depth, '-')
    ax[2].plot(logs[x3], logs.Depth, '-', color='0.5')
    ax[3].plot(logs[x4], logs.Depth, '-', color='r')
    ax[4].plot(logs[x5], logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=12)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((7*' ').join([
        'Sandstone', 'SS/SH', 'Shale', 'Marl', 'Dolomite',
        'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel(x1)
    ax[0].set_xlim(logs[x1].min(), logs[x1].max())
    ax[1].set_xlabel(x2)
    ax[1].set_xlim(logs[x2].min(), logs[x2].max())
    ax[2].set_xlabel(x3)
    ax[2].set_xlim(logs[x3].min(), logs[x3].max())
    ax[3].set_xlabel(x4)
    ax[3].set_xlim(logs[x4].min(), logs[x4].max())
    ax[4].set_xlabel(x5)
    ax[4].set_xlim(logs[x5].min(), logs[x5].max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['WELL'], fontsize=14,y=0.94)


def compare_plots(logs, x1, x2, x3, x4, x5, Depth=False):
    '''
    Plots well logs against depth and corresponding predicted and actual 
    lithofacies in a labelled color plot for comparism

    Arguments
    ---------

    df: dataframe of well data
    label: actual lithofacies values
    predictions: predicted values in integers (0, 1, 2....)
    log1: str -> well log 1
    ''''''''''''''''''''
    log5: str -> well log 2
    depth_col: depth column
    '''

    logs = logs.fillna(0)
    if Depth == False:
        logs['Depth'] = logs.index
        Depth = 'Depth'
        ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    else:
        logs['Depth'] = logs[Depth]
        Depth = 'Depth'
        ztop=logs.Depth.min(); zbot=logs.Depth.max()
        
    logs = logs.sort_values(by='Depth', ascending=True)

    facies_colors = [
        '#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', 
        '#AED6F1', '#A569BD', '#196F3D', '#10003D', '#A56222', '#000000'
    ]

    facies_labels = [
        'Sandstone', 'SS/SH', 'Shale', 'Marl', 'Dolomite',
        'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]

    facies_colormap = {}
    for ind, label in enumerate(facies_labels):
        facies_colormap[label] = facies_colors[ind]

    no = 12
    #no = len(list(dict(logs[target].value_counts())))
    cmap_facies = colors.ListedColormap(
            facies_colors[0 : no], 'indexed'
            )

    cluster1=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2=np.repeat(np.expand_dims(logs['Actual'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(12, 12))
    ax[0].plot(logs[x1], logs.Depth, '-g')
    ax[1].plot(logs[x2], logs.Depth, '-')
    ax[2].plot(logs[x3], logs.Depth, '-', color='0.5')
    ax[3].plot(logs[x4], logs.Depth, '-', color='r')
    ax[4].plot(logs[x5], logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=12)
    im=ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=12)
    
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((7*' ').join([
        'Sandstone', 'SS/SH', 'Shale', 'Marl', 'Dolomite',
        'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel(x1)
    ax[0].set_xlim(logs[x1].min(), logs[x1].max())
    ax[1].set_xlabel(x2)
    ax[1].set_xlim(logs[x2].min(), logs[x2].max())
    ax[2].set_xlabel(x3)
    ax[2].set_xlim(logs[x3].min(), logs[x3].max())
    ax[3].set_xlabel(x4)
    ax[3].set_xlim(logs[x4].min(), logs[x4].max())
    ax[4].set_xlabel(x5)
    ax[4].set_xlim(logs[x5].min(), logs[x5].max())
    ax[5].set_xlabel('Predictions')
    ax[6].set_xlabel('Actual')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[6].set_yticklabels([])
    ax[5].set_xticklabels([]); ax[6].set_xticklabels([]); 
    f.suptitle('Well: %s'%logs.iloc[0]['WELL'], fontsize=14,y=0.94)