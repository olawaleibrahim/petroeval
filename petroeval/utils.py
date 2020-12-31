import pandas as pd
import numpy as np

def drop_columns(data, *args):

    '''
    function used to drop columns.
    args:: 
      data:  dataframe to be operated on
      *args: a list of columns to be dropped from the dataframe

    return: returns a dataframe with the columns dropped
    '''
    
    columns = []
    for _ in args:
        columns.append(_)
        
    data = data.drop(columns, axis=1)
        
    return data

def process(df):

    '''
    Function to process log and replace missing or infinity values with zero
    for easier plotting
    '''

    cols = list(df.columns)
    for _ in cols:

        #df[_] = np.where(df[_] == np.inf, 0, df[_])
        df.loc[df[_] == np.inf] = 0
        #df[_] = np.where(df[_] == np.nan, 0, df[_])
        df.loc[df[_] == np.NaN] = 0
        #df[_] = np.where(df[_] == -np.inf, 0, df[_])
        df.loc[df[_] == -np.inf] = 0
        
    return df

def check_cardinality(df, column):

    no_rows = df.shape[0]
    value_count = df[column].value_counts()
    unique_count = len(value_count)
    
    value = None
    
    if (unique_count/no_rows) >= 0.9 and unique_count > 10:
        value = 'Unique'

    elif (unique_count/no_rows) < 0.9 and unique_count > 10:
        value = 'High'

    elif unique_count < 5:
        value = 'Low'

    return value

def label_encode(df, column):

    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes

    return df

def one_hot_encode(df, column):

    df = pd.get_dummies(df, prefix=column, columns=[column])

    return df
