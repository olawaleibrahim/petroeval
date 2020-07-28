'''
Module for preprocessing las file before formation evaluation
For a better user experience, this module is advised to be used before formation evaluation
to avoid errors and glitches
'''
import pandas as pd

def set_mnemonics(data, GR, NPHI, RHOB, RT):
    
    '''
    Function to set logs mnemonics to a standard title for later use
    args::
            data: dataframe object of well logs
            GR: gamma Ray mnemonics; set equal to the gamma ray title of the well log/dataframe
            NPHI: neutron porosity mnemonic; set equal to the neutron porosity title of the well log/dataframe
            RHOB: bulk density mnemonic; set equal to the bulk density title of the well log/dataframe
            RT: resistivity mnemonic; set equal to the resistivity title of the well log/dataframe

    returns: Dataframe ofbject with adjusted mnemonics (To be used for formation evaluation to avoid errors)
    '''
    
    assert isinstance(data, pd.DataFrame), 'Data should be a DataFrame object'
    mnemonics = [GR, NPHI, RHOB, RT]
    for _ in mnemonics:
        assert isinstance(_, str), f'Mnemonic ({_}) should be a string.'

    data = data.rename(columns={GR:'GR', NPHI:'NPHI', RHOB: 'RHOB', RT: 'RT'})

    return data
    