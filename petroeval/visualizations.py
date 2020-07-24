from .plots import four_plots, three_plots, two_plots, one_plot

def summary(data):
    
    '''
    Function to display summaryof the dataframe
    Returns: Summary object

    Args::
            Takes in the dataframe object
    '''

    print(f'Title: Petrophysical Summary of the Parameters Evaluated')
    return data.describe()


def log_plot(logs, GR=True, NPHI=True, RHOB=True, RT=True, no_plots=4):
    
    '''
    Plot log signatures of petrophysical parameters.
    Args::
            Function accepts a dataframe and a depth argument.
            Plots the GR, Porosity, Density and Resistivity logs respectively

            Pass True for the depth value if dataframe has a depth column, 
            default is fault (uses index as depth)

            no_plots: No of plots to display depending on petrophysical parameters to be visualized
                      Default and max value is 4
    '''
    
    try:

        '''
        Setting up all possible combinations of required plots
        '''

        if GR and NPHI and RHOB and RT:
            no_plots = 4

        elif GR and NPHI and RHOB or GR and RHOB and RT or GR and NPHI and RT or NPHI and RHOB and RT:
            no_plots = 3
        
        elif GR and NPHI or GR and RHOB or GR and RT or NPHI and RHOB or NPHI and RT or RHOB and GR or RHOB and RT:
            no_plots = 2

        elif GR or NPHI or RHOB or RT:
            no_plots = 1
        
        else:
            no_plots = 0
            raise InvalidEntryError(f'Enter an integer in the range (1-4). Set one or more petrophysical arguments to True')

        #if number of plots is equal to four
        if no_plots == 4:

            four_plots(logs)

        #if number of plots is equal to four
        elif no_plots == 3:
            if GR and NPHI and RHOB:
                three_plots(logs, 'GR', 'NPHI', 'RHOB')
            elif GR and NPHI and RT:
                three_plots(logs, 'GR', 'NPHI', 'RT')
            elif GR and RHOB and RT:
                three_plots(logs, 'GR', 'RHOB', 'RT')
            elif NPHI and RHOB and RT:
                three_plots(logs, 'NPHI', 'RHOB', 'RT')

        #if number of plots is equal to two (possible combinations)
        elif no_plots == 2:

            if GR and NPHI:
                two_plots(logs, 'GR', 'NPHI')
            elif GR and RHOB:
                two_plots(logs, 'GR', 'RHOB')
            elif GR and RT:
                two_plots(logs, 'GR', 'RT')
            elif NPHI and RHOB:
                two_plots(logs, 'NPHI', 'RHOB')
            elif NPHI and RT:
                two_plots(logs, 'NPHI', 'RT')
            elif RHOB and RT:
                two_plots(logs, 'RHOB', 'RT')

        #if number of plots is equal to 1 (possible combinations)

        elif no_plots == 1:

            if GR:
                one_plot(logs, 'GR')
            elif NPHI:
                one_plot(logs, 'NPHI')
            elif RHOB:
                one_plot(logs, 'RHOB')
            else:
                one_plot(logs, 'RT')

    
    except ModuleNotFoundError as err:
        print(f'Install module. {err}')

    except AttributeError as err:
        print(f'NameError: Attritubute not found. Specify attribute. {err}')



