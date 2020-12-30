"""
Machine learning module for predicting lithology and lithofacies labels
and other ML functionalities
"""

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
        print(df.head())
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

        print(f'The shape is {new_df.shape}')
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

        print(f'Train features: {train_features.head(3)}')
        print(f'Test features: {test_features.head(3)}')
        # divide dataframe into train part and part needed for prediction

        '''
        The idea is to use the depth column and the range passed by the parameters.
        The range specified represent the range needed for prediction. Every other part 
        is used as the training data set
        '''

        if model == 'RF':

            model = RandomForestRegressor(n_estimators=5, max_depth=6, random_state=42, verbose=2)
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
            self.plot_feat_imp(model)

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
    
    def plot_feat_imp(self, model):

        feat_imp = pd.Series(model.feature_importances_).sort_values(ascending=False)
        plt.figure(figsize=(12,8))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')    

    
class PredictLabels():

    '''
    Class for predicting lithofacies
    '''

    def __init__(self, df, depth_col):

        self.df = df
        self.depth_col = depth_col


    def train(self, pretrained=True):

        self.pretrained = pretrained

        if pretrained:

            model = joblib.load('data/lithofacies_model')

        return model, test_features


    #def plot_feat_imp(self):



    
    def predict(self, pretrained, plot_pred=True):

        self.pretrained = pretrained

        model, test_features = self.train(pretrained)
        predictions = model.predict(test_features)

        if plot_pred:
            df = self.df.copy()
            df['Lithofacies'] = predictions
            four_plots(self.df, 'GR', 'NPHI', 'RHOB', 'Lithofacies')

        return predictions