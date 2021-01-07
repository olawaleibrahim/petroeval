'''
Utilities modules for other modules
Contains reusable functions

Functions
---------

prepare_datasets(df, start, end, target)
scale_train_test(train_df, test_df)
drop_columns(data, *args)
process(df)
check_cardinality(df, column: str)
label_encode(df, column)
one_hot_encode(df, column)
sample_evaluation(y_test, y_pred)
'''

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def prepare_datasets(df, start, end, target):
    '''
    Function to prepare the dataframe into train and test features

    returns: train features, test features, train target

    Arguments
    ---------

    target: target column string name
            start: specify start point for test features from train data if test features
                    dataframe does not exist i.e if desired prediction section is a missing
                    section from the supplied train data
            end: where test features should stop from train data provided

    The general idea is to use the depth column and the range passed by the parameters.
    The range specified represent the range needed for prediction. Every other part 
    is used as the training data set
    '''

    top_df = df.iloc[:df[df['depth'] == start].index[0]]
    bottom_df = df.iloc[df[df['depth'] == end].index[0]: ]
    test_features = df.iloc[df[df['depth'] == start].index[0] : df[df['depth'] == end].index[0]]

    top_target = df.iloc[:df[df['depth'] == start].index[0]]
    bottom_target = df.iloc[df[df['depth'] == end].index[0]: ]

    train_features = pd.concat((top_df, bottom_df), axis=0)
    train_features = train_features.drop(target, axis=1, inplace=False)
    test_features = test_features.drop(target, axis=1, inplace=False)

    train_target_df = pd.concat((top_target, bottom_target), axis=0)
    train_target = train_target_df[target]

    return train_features, test_features, train_target


def scale_train_test(train_df, test_df):
    '''
    Function to scale train and test data sets

    returns: scaled train and test data

    Arguments
    ---------
    train_df: train dataframe or first dataframe or data
    test_df: test dataframe or secind dataframe or data
    '''

    scaler = StandardScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)

    return train_df, test_df


def drop_columns(data, *args):

    '''
    function used to drop columns

    Returns
    -------
    dataframe wkth dropped column(s)

    Arguments
    ---------
      data:  dataframe to be operated on
      *args: a list of columns to be dropped from the dataframe
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

    returns: dataframe with substituted and processed values

    Arguments
    ---------
    df = dataframe to be processed
    '''

    cols = list(df.columns)
    for _ in cols:

        df = df.fillna(0)
        df.loc[df[_] == np.inf] = 0
        df.loc[df[_] == np.nan] = 0
        df.loc[df[_] == -np.inf] = 0
        
    return df


def check_cardinality(df, column: str):
    '''
    Function to check the cardinality of a column
    
    returns a value (string) based on deduced column cardinality

    Arguments
    ---------

    df: dataframe
    column: column in dataframe to check cardinality
    '''

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
    '''
    Function to label encode a categorical column

    returns: Dataframe with encoded column is returned while original column is dropped

    Arguments
    ---------

    df: dataframe
    column: column to be encoded
    '''
            

    df[column + '_enc'] = df[column].astype('category')
    df[column + '_enc'] = df[column + '_enc'].cat.codes
    df = df.drop(column, axis=1, inplace=False)

    return df


def one_hot_encode(df, column):
    '''
    Function to one hot encode a categorical column

    returns: Dataframe with encoded column is returned while original column is dropped

    Arguments
    ---------

    df: dataframe
    column: column to be encoded
    '''

    df = pd.get_dummies(df, prefix=column, columns=[column])

    return df


def sample_evaluation(y_test, y_pred):
    '''
    Function to print the RMSE and R2 of the predicted and actual values

    Arguments
    ---------
    y_test: actual values
    y_pred: predictions
    '''
    
    print(f'The test RMSE is : {mean_squared_error(y_test, y_pred) ** 0.5}')
    print(f'The test R2 score is : {r2_score(y_test, y_pred)}')


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
