'''
Python module for formation reservoir evaluation
Author: Ibrahim Olawale
'''

import numpy as np
import pandas as pd


class FormationEvaluation:
    '''
    Class to evaluate a reservoir based on four main petrophysical parameters.
    Creates an instance of the reservoir to be evaluated
    args::
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
        Returns: Dataframe object with the petrophysical parameters evaluated
        Args:: baseeline_default: Default cutoff is used to determine shalebaseline for evaluation
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
            
            data[self.RT] = np.where(data[self.RT]==0, 0.01, data[self.RT])
            for i in range(data.shape[0]):
                if data[self.RT].iloc[i] == 0:
                    data[self.RT].iloc[i] = 0.1
                else:
                    data[self.RT].iloc[i] == data[self.RT].iloc[i]
        
        
            #to classify each point as shale or sandstone based on the Gamma Ray readings
            df3['litho'] = 0
            
            for i in range(data.shape[0]):
                
                if (data[GR].iloc[i] < cutoff_test):
                    df3['litho'].iloc[i] = 1
                else:
                    df3['litho'].iloc[i] = 0

            vsh = []

            min_GR = data[GR].min()
            max_GR = data[GR].max()

            for i in range(data.shape[0]):

                IGR = (data[GR].iloc[i] - min_GR) / (max_GR - min_GR)
                reading = 0.083 * ((2** (3.7 * IGR)) - 1)
                #reading = (((3.7 * (data[GR].iloc[i] - 25)/(130-25)) ** 2) - 1) * 0.083
                
                #To correct for negative volumes of shale as this is practically not correct

                if reading < 0:
                    vsh.append(0)

                elif reading > 1:
                    vsh.append(1)
                else:
                    vsh.append(reading)

            df3['vsh'] = vsh

            ntg = []
            for vsh_ in df3['vsh']:
                amount = 1 - vsh_
                ntg.append(amount)

            df3['ntg'] = ntg
                    
            #Calculating Net Pay using GR and Porosity readings
            
            net = []

            for i in range(data.shape[0]):
                if (data[GR].iloc[i] < cutoff_test) and (data[NPHI].iloc[i] > 0.25):
                    i = 1
                    net.append(i)
                else:
                    i = 0
                    net.append(i)
                    
            
            df3['Net_Pay'] = net

            #df3['litho'] = 0.5 * df3['litho']
            #df3['Net_Pay'] = 0.5 * df3['Net_Pay']
            
            #Calculating total formation porosity (phid)
            
            FL = 1

            df3['phid'] = 0
            for i in range(df3.shape[0]):
                df3['phid'].iloc[i] = (2.65 - df3[RHOB].iloc[i]) / (2.65 - FL)
                
            #Setting cutoff conditions for the porosity values

            df3['phidf'] = np.where(df3['phid'] <= 0.1, 0.1, df3['phid'])
            df3['phidf'] = np.where(df3['phid'] >= 0.85, 0.85, df3['phid'])

                    
            #Calculating water saturation
            
            sw = []
            
            for i in range(df3.shape[0]):
                #i = np.sqrt(0.10 / (df3[RT].iloc[i]) * (phidf[i] ** 1.74))
                #if i == float('inf'):
                j = np.sqrt(0.10/(df3[RT].mean() * (df3['phidf'].iloc[i] ** 1.74)))
                if j < 0:
                    sw.append(j)
                elif j > 1:
                    sw.append(1)
                else:
                    sw.append(j)
                    
            #Calculating oil saturation
            
            oil_sat = []
            
            for i in range(0, df3.shape[0]):
                i = 1 - sw[i]
                oil_sat.append(i)
            
            df3['vsh'] = vsh

            
            df3['sw'] = sw
            df3['oil_sat'] = oil_sat

            #effective porosity calculation
            df3['phie'] = 0
            for i in range(df3.shape[0]):
                df3['phie'].iloc[i] = df3['phidf'].iloc[i] * df3['ntg'].iloc[i]
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
        Returns a dictionary of the parameters values;
        Values::
                Net to Gross ratio
                Total porosity evaluated
                Water saturation and
                Oil saturation
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
        


