"""
Machine learning module for predicting lithology and lithofacies labels
and other ML functionalities
"""

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from utils import drop_columns, check_cardinality
from sklearn.preprocessing import StandardScaler
import preprocessing
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import xgboost as XGBRegressor
from plots import four_plots
import pandas as pd
import numpy as np
import joblib


class PredictLitho():

    '''
    Class for predicting lithology
    '''

    def __init__(self, df, depth_col, plot=True):

        self.df = df
        self.depth_col = depth_col
        self.plot = plot

    def __call__(self, plot=True):
        return self.train(plot)

    
    def _preprocess(self, df, target, start, end):

        self.df = df
        self.target = target
        self.start = start
        self.end = end

        df.fillna(-9999, inplace=True)

        #new_df = (df.drop(target, axis=1))
        new_df = df.copy()

        columns_ = list(new_df.columns)

        # dropping columns with categorical contents for easier processing

        for column in columns_:
            if new_df[column].dtype == 'object':
                new_df.drop(column, axis=1, inplace=True)

        new_df['depth'] = range(0, new_df.shape[0])
        #columns = new_df.columns


        # divide dataframe into train part and part needed for prediction

        '''
        The idea is to use the depth column and the range passed by the parameters.
        The range specified represent the range needed for prediction. Every other part 
        is used as the training data set
        '''

        top_df = new_df.iloc[:new_df[new_df['depth'] == start].index[0]]
        bottom_df = new_df.iloc[new_df[new_df['depth'] == end].index[0]: ]
        test_features = new_df.iloc[new_df[new_df['depth'] == start].index[0] : new_df[new_df['depth'] == end].index[0]]

        top_target = new_df.iloc[:new_df[new_df['depth'] == start].index[0]]
        bottom_target = new_df.iloc[new_df[new_df['depth'] == end].index[0]: ]

        train_features = pd.concat((top_df, bottom_df), axis=0)
        train_features.drop(target, axis=1, inplace=True)
        test_features.drop(target, axis=1, inplace=True)

        columns = list(train_features.columns)

        new_df = pd.concat((top_target, bottom_target), axis=0)

        train_target = new_df[target]

        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        train_features = pd.DataFrame(train_features, columns=columns)
        test_features = pd.DataFrame(test_features, columns=columns)
        
        # dropping added depth column which was used to aid preprocessing

        train_features.drop('depth', axis=1, inplace=True)
        test_features.drop('depth', axis=1, inplace=True)

        return train_features, train_target, test_features


    def train(self, target, start, end, plot, model='RF', CV=3):

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV
        df = self.df

        try:
            if CV < 3:
                raise ValueError(f'Number of cross validation folds should be greaterb than 2; {CV} specified')
            
        except ValueError as err:
            print(err)

        train_features, train_target, test_features = self._preprocess(self.df, self.target, start, end)

        #print(f'Train features: {train_features.head(3)}')
        #print(f'Test features: {test_features.head(3)}')

        # divide dataframe into train part and part needed for prediction

        '''
        The idea is to use the depth column and the range passed by the parameters.
        The range specified represent the range needed for prediction. Every other part 
        is used as the training data set
        '''

        if model == 'RF':

            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=2)
            model.fit(train_features, train_target)

        elif model == 'XGB':
            model = xgb.XGBRegressor(n_estimators=3000, max_depth=8, reg_lambda=500,
            random_state=20)

        X_train, X_test, y_train, y_test = ms.train_test_split(train_features, train_target,
                                                                test_size=0.2, random_state=20)

        print(f'(X_train, y_train): {X_train.shape, y_train.shape}, (X_test, y_test): {X_test.shape, y_test.shape}')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'The test RMSE is : {mean_squared_error(y_test, y_pred) ** 0.5}')
        print(f'The test R2 score is : {r2_score(y_test, y_pred)}')

        if plot:
            self.plot_feat_imp(model, list(train_features.columns))

        return model, test_features


    def predict(self, target, start, end, model='RF', CV=3):

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV

        trained_model, test_features = self.train(target, start, end, self.plot, CV=CV)
        prediction = trained_model.predict(test_features)

        return prediction
    
    def plot_feat_imp(self, model, columns):

        self.columns = columns
        self.model = model

        x = len(columns)
        x = np.arange(0, x)
        feat_imp = pd.Series(model.feature_importances_).sort_values(ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.figure(figsize=(12,8))
        ax.bar(x, feat_imp)
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation='vertical', fontsize=18)
        ax.set_ylabel('Feature Importance Score')

        plt.show()

    
class PredictLabels():

    '''
    Class for predicting lithofacies
    '''

    def __init__(self, df, depth_col):

        self.df = df
        self.depth_col = depth_col


    def __call__(self, plot=True):
        return self.train(plot)


    def _preprocess(self, df, target):
        
        self.df = df
        self.target = target

        df.fillna(-9999, inplace=True)

        lithology_numbers = {30000: 0,
                        65030: 1,
                        65000: 2,
                        80000: 3,
                        74000: 4,
                        70000: 5,
                        70032: 6,
                        88000: 7,
                        86000: 8,
                        99000: 9,
                        90000: 10,
                        93000: 11,
                        -9999: 'Unknown'}

        lithology = df[target]
        lithology = lithology.map(lithology_numbers)

        df_wells = df.WELLS.values
        df_depth = df.DEPTH.values

        cols = ['FORCE_2020_LITHOFACIES_CONFIDENCE', 'SGR', 'DTS', 'RXO', 'ROPA'] #columns to be dropped
        df = drop_columns(df, *cols)

        df['GROUP_encoded'] = df['GROUP'].astype('category')
        df['GROUP_encoded'] = df['GROUP_encoded'].cat.codes 
        df['FORMATION_encoded'] = df['FORMATION'].astype('category')
        df['FORMATION_encoded'] = df['FORMATION_encoded'].cat.codes
        df['WELL_encoded'] = df['WELL'].astype('category')
        df['WELL_encoded'] = df['WELL_encoded'].cat.codes

    def train(self, pretrained=True):

        self.pretrained = pretrained

        if pretrained:

            model = joblib.load('data/lithofacies_model')

        return model, test_features


    def plot_feat_imp(self, model, columns):

        self.columns = columns
        self.model = model

        x = len(columns)
        x = np.arange(0, x)
        feat_imp = pd.Series(model.feature_importances_).sort_values(ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.figure(figsize=(12,8))
        ax.bar(x, feat_imp)
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation='vertical', fontsize=18)
        ax.set_ylabel('Feature Importance Score')    

    
    def predict(self, target, start, end, model='RF', CV=3):

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV

        trained_model, test_features = self.train(target, start, end, self.plot, CV=CV)
        prediction = trained_model.predict(test_features)

        return prediction


class DataHandlers():

    '''
    Will handle the preprocessing of the dataframe for categorical and numerical variables
    as well as handling different mnemonics issues
    '''

    def __init__(self, df):

        self.df = df

    
    def __call__(self):
        return self.


    def encode_categorical(self):

        df = self.df

        columns = list(df.columns)

        cat_df = pd.DataFrame()

        for column in columns:
            if df[column].dtype == object:
                cat_df[column] = df[column]

        # check cardinality of categorical variables then encode based on cardinality


        return cat_df


    def set_mnemonics(self, GR, RHOB, RT, NPHI, depth=False):

        self.GR = GR
        self.RHOB = RHOB
        self.RT = RT
        self.NPHI = NPHI
        self.depth = depth
        df = self.df

        if depth == False:
            df['depth'] = df.index

        df = preprocessing.set_mnemonics(self.df, GR, NPHI, RHOB, RT)

        return df
