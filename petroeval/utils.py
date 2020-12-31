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