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
        value = 'Distinct'

    if (unique_count/no_rows) <= 0.1 and unique_count <= 3:
        value = 'Unique'

    elif (unique_count/no_rows) < 0.9 and unique_count > 9:
        value = 'High'

    elif unique_count < 5:
        value = 'Low'

    return value

def label_encode(df, column):

    df[column + '_enc'] = df[column].astype('category')
    df[column + '_enc'] = df[column + '_enc'].cat.codes
    df = df.drop(column, axis=1, inplace=False)

    return df

def one_hot_encode(df, column):

    df = pd.get_dummies(df, prefix=column, columns=[column])

    return df


#Paolo Bestagini's feature augmentation technique from SEG 2016 ML competition
#Link : https://github.com/seg/2016-ml-contest/tree/master/ispl

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]
 
    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
 
    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row
 
    return X_aug
 
# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad
 
# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows
