'''
Python module for formation reservoir evaluation

Classes
-------

FormationEvaluation
'''

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class FormationEvaluation:
    '''
    Class to evaluate a reservoir based on four main petrophysical parameters.
    Creates an instance of the reservoir to be evaluated
    Arguments
    --------
        data: dataframe  or csv format of data
        GR: gamma ray column of table
        NPHI: neutron porosity column title
        RHOB: density column title
        RT: resistivity column title
        top: top of reservoir (in, metres, feets)
        base: base of reservoir (in, metres, feets)
        cutoff: Shale baseline value in API
    '''

    def __init__(self, data, GR, NPHI, RHOB, RT, top, base, cutoff):
        try:
            self.data = data
            self.GR = str(GR)
            self.NPHI = str(NPHI)
            self.RHOB = str(RHOB)
            self.RT = str(RT)
            self.top = top
            self.base = base
            self.cutoff = cutoff
            
        except ValueError as err:
            print(f'Input right format {err}')
        
        except TypeError as err:
            print(f'Input right format {err}')
            
        
    def fill_missing(self, use_mean, value):

        '''
        Method to fill in missing values in the dataset and return dataframe

        Returns
        -------
        Dataframe with missing values filled

        Arguments
        ---------
        use_mean: Specify True or False
                  Use the mean value of a column to fill the column if specified as True
                  Value should be specified if use_mean is set to False
        value: Specified value is used to fill the whole columns
        '''

        if isinstance(value, str):
            raise NameError(f'{value} is nota valid data type for filling missing values.\n Use integer or float')

        try:
            print('Filling missing values...')
            data = self.data
            cols = list(data.columns)

            if use_mean == True:
                for col in cols:

                    data[col] = data[col].fillna(data[col].mean())
                    '''for i in range(len(data[col])):
                        if np.isnan(data[col].iloc[i]):
                            
                            data[col].iloc[i] = data[col].mean()
                        else:
                            data[col].iloc[i] = data[col].iloc[i]
                    '''
            
            elif use_mean == False:
                cols = list(data.columns)
                data = data.fillna(value)
                #'''
                #    for i in range(len(data[col])):
                #        if np.isnan(data[col].iloc[i]):
                #            
                #            data[col].iloc[i] = value
                #        else:
                #            data[col].iloc[i] = data[col].iloc[i]
               

            return data

        except ModuleNotFoundError as err:
            print (f'Install required module. {err}')

        
        except TypeError as err:
            print(f'Unsupported Format: Process inputs as data input type is incompatible with method')
            

    def show_table(self, baseline_default):
        
        '''
        Method to carry out formation evaluation of a reservoir
        Returns
        -------
        Dataframe object with the petrophysical parameters evaluated

        Arguments
        ---------
        baseeine_default: Default cutoff is used to determine shalebaseline for evaluation
               Set to True to use default baseline. If set to False, cutoff value specified during class instation is used
        
        Default baseline is calculated by finding the average of the minimum and maximum shale values
        '''

        top = self.top
        base = self.base
        base = base + 1
        data = self.data.iloc[top:base]
        GR = self.GR
        NPHI = self.NPHI
        RHOB = self.RHOB
        RT = self.RT
        cutoff = self.cutoff

        df3 = data.copy()

        if baseline_default == True:
            cutoff_test = (data[GR].min() + data[GR].max() / 2)
            if cutoff_test > 80:
                cutoff_test = 80
                print(f'Default baseline {cutoff_test} is used for evaluation')
            else:
                print(f'Default baseline {cutoff_test} is used for evaluation')

        else:
            cutoff_test = self.cutoff
            print(f'{cutoff_test} will be used for evaluation')

        try:
            #converting resistivity values of 0 to 0.01
            
            #data[self.RT] = np.where(data[self.RT]==0, 0.01, data[self.RT])
            data.loc[data[self.RT] == 0, 'RT'] = 0.01
            #for i in range(data.shape[0]):
                #if data[self.RT].iloc[i] == 0:
                    #data[self.RT].iloc[i] = 0.1
                #else:
                    #data[self.RT].iloc[i] == data[self.RT].iloc[i]
        
        
            #to classify each point as shale or sandstone based on the Gamma Ray readings
            #df3['litho'] = 0
            
            df3.loc[df3.GR < cutoff_test, 'litho'] = 0
            df3.loc[df3.GR > cutoff_test, 'litho'] = 1

            min_GR = data[GR].min()
            max_GR = data[GR].max()

            #data['IGR'] = 0
            data['IGR'] = (data[GR] - min_GR) / (max_GR - min_GR)
            #data['reading'] = 0
            data['reading'] = (0.083 * (2 ** (3.7 * data['IGR']) - 1))
            #df3['vsh'] = 0
            #data['vsh'] = 0
            data['vsh'] = np.where(data['reading'] < 0, 0, data['reading'])
            data['vsh'] = np.where(data['reading'] > 1, 1, data['reading'])


            df3['vsh'] = data['vsh']

            #df3['ntg'] = 0
            df3['ntg'] = 1 - df3['vsh']
                    
            #Calculating Net Pay using GR and Porosity readings
            
            df3['Net_Pay'] = np.nan
            df3.loc[df3[GR] > cutoff_test, 'Net_Pay'] = 0
            df3.loc[(df3[GR] < cutoff_test) & (df3[NPHI] > 0.25), 'Net_Pay'] = 1
            
            
            FL = 1   #Formation liquid (assummed to be water, 0.8 for oil)

            #df3['phid'] = 0
            df3['phid'] = (2.65 - df3[RHOB]) / (2.65 - FL)
                
            #Setting cutoff conditions for the porosity values

            df3['phidf'] = np.where(df3['phid'] <= 0.1, 0.1, df3['phid'])
            df3['phidf'] = np.where(df3['phid'] >= 0.85, 0.85, df3['phid'])

                    
            #Calculating water saturation
            
            df3['sw'] = np.nan
            df3['sw'] = (np.sqrt(0.1/(df3[RT].mean() * (df3['phidf'] ** 1.74))))
            df3['sw'] = np.where(df3['sw'] < 0, 0, df3['sw'])
            df3['sw'] = np.where(df3['sw'] > 1, 1, df3['sw'])
            
            #j = np.sqrt(0.10/(df3[RT].mean() * (df3['phidf'].iloc[i] ** 1.74)))
                    
            #Calculating oil saturation
            
            df3['oil_sat'] = 1-df3['sw']

            #effective porosity calculation
            df3['phie'] = df3['phidf'] * df3['ntg']
            df3['phie'] = np.where(df3['phie'] < 0, 0, df3['phie'])
            
            #return evaluated parameters
            
            df4 = pd.DataFrame()
            #df4['Depth'] = df3.index
            df4['GR'] = df3[GR]
            df4['LITHO'] = df3['litho']
            df4['VSH'] = df3['vsh']
            df4['NET_PAY'] = df3['Net_Pay']
            df4['PHIDF'] = df3['phidf']
            df4['PHIE'] = df3['phie']
            df4['SW'] = df3['sw']
            df4['OIL_SAT'] = df3['oil_sat']
            
            print('ESTIMATED PETROPHYSICAL PARAMETERS')
            return df4

        except ModuleNotFoundError as err:
            print(f'Install required module. {err}')

        except TypeError as err:
            print(f'Unsupported Format: Process inputs as data input type is incompatible with method')


    def parameters(self, baseline_default):

        '''
        Method to return a dictionary of paraeters evaluated
        Returns
        -------
        Dictionary of the parameters values;
        Values::
                Net to Gross ratio
                Total porosity evaluated
                Water saturation and
                Oil saturation

        Arguments
        -------
                baseeine_default: Default cutoff is used to determine shalebaseline for evaluation
               Set to True to use default baseline. If set to False, cutoff value specified during class instation is used
        '''

        try:
            table = self.show_table(baseline_default)

            grk = table.LITHO.sum()
            ntg = table.NET_PAY.sum()/table.LITHO.sum()
            net_pay = table.NET_PAY.sum()
            phidf = table.PHIDF.mean()
            phie = table.PHIE.mean()
            sw = table.SW.mean()
            oil_sat = 1-sw

            #values = {'net': net, 'phid': phid, 'sw': sw, 'oil_sat': oil_sat}

            gross_rock = 'Gross rock'
            net_to_gross = 'The Net to Gross is:'
            reservoir_net_pay = 'Net Pay of reservoir:'
            total_porosity = 'Total Porosity:'
            effective_porosity = 'Effective Porosity:'
            water_saturation = 'Water Saturation:'
            oil_saturation = 'Oil Saturation:'

            values = {gross_rock: grk, net_to_gross: ntg, reservoir_net_pay: net_pay, total_porosity: phidf, effective_porosity: phie, water_saturation: sw, oil_saturation: oil_sat}

            return values

        except ModuleNotFoundError as err:
            print (f'Install required module. {err}')
        


