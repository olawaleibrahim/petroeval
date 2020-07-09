'''
Python module for formation reservoir evaluation
First release v1.0.1 by Ibrahim Olawale
'''

import numpy as np
import pandas as pd
from visualizations import summary

class FormationEvaluation:
    '''
    Class to evaluate a reservoir based on four main petrophysical parameters.
    data: dataframe  or csv format of data
    gr: gamma ray column of table
    nphi: neutron porosity column title
    dens: density column title
    res: resistivity column title
    top: top of reservoir (in, metres, feets)
    bottom: bottom of reservoir (in, metres, feets)
    cutoff: Shale baseline value in API
    '''

    def __init__(self, data, gr, nphi, dens, res, top, bottom, cutoff):
        try:
            self.data = data
            self.gr = str(gr)
            self.nphi = str(nphi)
            self.dens = str(dens)
            self.res = str(res)
            self.top = top
            self.bottom = bottom
            self.cutoff = cutoff
            
        except ValueError as err:
            print(f'Input right format {err}')
        
        except TypeError as err:
            print(f'Input right format {err}')
            
        
    def fill_missing(self, use_mean, *value):

        '''
        Method to fill in missing values in the dataset and return dataframe
        use_mean: Use the mean value of a column to fill the column
        '''
        try:
            print('Time computation is proportional to size of dataset and amount of missing values present')
            data = self.data
            cols = list(data.columns)

            if use_mean == True:
                for col in cols:
                    for i in range(len(data[col])):
                        if np.isnan(data[col].iloc[i]):
                            
                            data[col].iloc[i] = data[col].mean()
                        else:
                            data[col].iloc[i] = data[col].iloc[i]
            
            else:
                cols = list(data.columns)
                for col in cols:
                    for i in range(len(data[col])):
                        if np.isnan(data[col].iloc[i]):
                            
                            data[col].iloc[i] = value
                        else:
                            data[col].iloc[i] = data[col].iloc[i]

            return data

        except ModuleNotFoundError as err:
            print (f'Install required module. {err}')
        
        except TypeError as err:
            print(f'Unsupported Format: Process inputs as data input type is incompatible with method')
            

    def show_table(self):
        
        '''
        Method to carry out formation evaluation of a reservoir
        and return the table
        '''

        top = self.top
        bottom = self.bottom
        bottom = bottom + 1
        data = self.data.iloc[top:bottom]
        #data = self.data
        #data = data.iloc[top:bottom, :]
        gr = self.gr
        nphi = self.nphi
        dens = self.dens
        res = self.res
        cutoff = self.cutoff

        try:
            for i in range(data.shape[0]):
                if data[self.res].iloc[i] == 0:
                    data[self.res].iloc[i] = 0.1
                else:
                    data[self.res].iloc[i] == data[self.res].iloc[i]
        
        
            #to classify each point as shale or sandstone based on the Gamma Ray readings
            gr_cutoff = []
            
            for i in range(data.shape[0]):
                if (data[gr].iloc[i] < cutoff):
                    i = 1
                    gr_cutoff.append(i)
                else:
                    i = 0
                    gr_cutoff.append(i)

            #To calculate volume of shale
            vsh = []

            for i in range(data.shape[0]):
                reading = (((3.7 * (data[gr].iloc[i] - 25)/(130-25)) ** 2) - 1) * 0.83
                
                #To correct for negative volumes of shale as this is practically not correct

                if reading < 0:
                    reading = 0
                    vsh.append(reading)
                else:
                    reading1 = reading
                    vsh.append(reading1)

            #ntg = []



                    
            #Calculating Net Pay using GR and Porosity readings
            
            net = []
            

            for i in range(data.shape[0]):
                if (data[gr].iloc[i] < cutoff) and (data[nphi].iloc[i] > 0.25):
                    i = 1
                    net.append(i)
                else:
                    i = 0
                    net.append(i)
                    
            df3 = data.copy()
                    
            df3['litho'] = gr_cutoff
            df3['Net_Pay'] = net

            #df3['litho'] = 0.5 * df3['litho']
            #df3['Net_Pay'] = 0.5 * df3['Net_Pay']
            
            #Calculating total formation porosity (phid)
            
            FL = 1
            phid = [] 

            for i in range(df3.shape[0]):
                i = (2.65 - df3[dens].iloc[i]) / (2.65 - FL)
                phid.append(i)
                
            phidf = []

            for i in range(0, df3.shape[0]):
                if (phid[i] <= 0.15):
                    i = 0.15
                    phidf.append(i)
                elif (phid[i]) >= 0.8:
                    i = 0.8
                    phidf.append(i)
                else:
                    i = phid[i]
                    phidf.append(i)

                    
            #Calculating water saturation
            
            sw = []
            
            for i in range(df3.shape[0]):
                i = np.sqrt(0.10 / (df3[res].iloc[i]) * (phidf[i] ** 1.74))
                if i == float('inf'):
                    i == 0.1
                    sw.append(i)
                else:
                    sw.append(i)
                    
            #Calculating oil saturation
            
            oil_sat = []
            
            for i in range(0, df3.shape[0]):
                i = 1 - sw[i]
                oil_sat.append(i)
            
            df3['vsh'] = vsh
            #df3['Net_Pay'] = net_pay
            
            #Calculate Net to Gross
            ntg = []
            for vsh in df3['vsh']:
                amount = 1 - vsh
                ntg.append(amount)

            df3['ntg'] = ntg
            df3['phid'] = phid
            df3['phidf'] = phidf
            df3['sw'] = sw
            df3['oil_sat'] = oil_sat

            #effective porosity calculation

            phie = []
            for i in range(df3.shape[0]):
                reading = df3['phidf'].iloc[i] * (1 - df3['vsh'].iloc[i])
                if reading < 0:
                    reading = 0
                    phie.append(reading)
                else:
                    reading1 = reading
                    phie.append(reading1)
            
            #return evaluated parameters
            
            df4 = pd.DataFrame()
            #df4['Depth'] = df3.index
            df4['GR'] = df3[gr]
            df4['LITHO'] = df3['litho']
            df4['VSH'] = df3['vsh']
            #df4['NTG'] = df3['ntg']
            df4['NET_PAY'] = df3['Net_Pay']
            #df4['PHID'] = df3['phid']
            df4['PHIDF'] = df3['phidf']
            df4['PHIE'] = phie
            df4['SW'] = df3['sw']
            df4['OIL_SAT'] = df3['oil_sat']
            
            return df4

        except ModuleNotFoundError as err:
            print (f'Install required module. {err}')

        except TypeError as err:
            print(f'Unsupported Format: Process inputs as data input type is incompatible with method')


    def parameters(self):

        '''
        Method to return a dictionary of paraeters evaluated;
        Net to Gross ratio
        Total porosity evaluated
        Water saturation and
        Oil saturation
        '''

        try:
            table = self.show_table()

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
        
        #except TypeError as err:
            #print(f'Unsupported Format: Process inputs as data input type is incompatible with method')