"""
Machine learning module for predicting lithology and lithofacies labels
and other ML functionalities
"""

from utils import drop_columns, label_encode, one_hot_encode, augment_features, check_cardinality
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import preprocessing
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import xgboost as XGBRegressor
from plots import four_plots
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle, os


class PredictLitho():

    '''
    Class for predicting lithology
    '''

    def __init__(self, df, depth_col, plot=True):

        '''
        args::
            df: dataframe
            depth_col: depth column, specify False if absent
        '''

        self.df = df
        self.depth_col = depth_col
        self.plot = plot

    def __call__(self, plot=True):
        return self.train(plot)

    
    def _preprocess(self, df, target, start, end):

        '''
        Method for preprocessing data by generating train features and labels
        using the specified start and end points and also to generate the 
        portion of the data needed for predictions (test features). This is done in
        a way to prevent data leakage while still using the maximum data points
        for a better accuracy
        returns: train features, train target, test features

        args::
            df: dataframe to be preprocessed
            target: column to be predicted
            start: where prediction should start from
            end: where prediction should stop
        '''

        self.df = df
        self.target = target
        self.start = start
        self.end = end

        df = df.fillna(-9999, inplace=False)

        new_df = df.copy()

        '''
        # dropping columns with categorical contents for easier processing
        columns_ = list(new_df.columns)
        
        for column in columns_:
            if new_df[column].dtype == 'object':
                new_df.drop(column, axis=1, inplace=True)
        '''

        encoding = DataHandlers(new_df)
        new_df = encoding.encode_categorical()

        new_df['depth'] = range(0, new_df.shape[0])


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
        train_features = train_features.drop(target, axis=1, inplace=False)
        test_features = test_features.drop(target, axis=1, inplace=False)

        columns = list(train_features.columns)

        new_df = pd.concat((top_target, bottom_target), axis=0)

        train_target = new_df[target]

        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        train_features = pd.DataFrame(train_features, columns=columns)
        test_features = pd.DataFrame(test_features, columns=columns)
        
        # dropping added depth column which was used to aid preprocessing

        train_features = train_features.drop('depth', axis=1, inplace=False)
        test_features = test_features.drop('depth', axis=1, inplace=False)

        return train_features, train_target, test_features


    def train(self, target, start, end, plot, model='RF', CV=3):

        '''
        Method used in making prediction
        returns: trained model, test features needed for predictions

        args::
            target: Column to be predicted
            start: where prediction should start from
            end: where prediction should stop
            model: model to be used; default value is 'RF' for random forest
                                     other options are 'XG' for XGBoost
                                                       'CAT' for CatBoost
        '''

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

        train_features, train_target, test_features = self._preprocess(df, target, start, end)

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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'The test RMSE is : {mean_squared_error(y_test, y_pred) ** 0.5}')
        print(f'The test R2 score is : {r2_score(y_test, y_pred)}')

        if plot:
            self.plot_feat_imp(model, list(train_features.columns))

        return model, test_features


    def predict(self, target, start, end, model='RF', CV=3):

        '''
        Method used in making prediction
        returns: prediction values

        args::
            target: Column to be predicted
            start: where prediction should start from
            end: where prediction should stop
            model: model to be used; default value is 'RF' for random forest
                                     other options are 'XG' for XGBoost
                                                       'CAT' for CatBoost
                                                       
        '''

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV

        trained_model, test_features = self.train(target, start, end, self.plot, CV=CV)
        prediction = trained_model.predict(test_features)

        return prediction
    
    def plot_feat_imp(self, model, columns):

        '''
        Method to plot the feature importance of the model in a bar chart
        according to rank (importance)

        returns: plot of the features importance

        args::
            model: trained model object
            columns: features names used for training (dataframe.columns)
        '''

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

    def __init__(self, df, depth_col, plot=True):

        '''
        args::
            df: dataframe for predicting lithofacies
            depth_col: depth column if available, specify False if not
        '''

        self.df = df
        self.depth_col = depth_col
        self.plot = plot


    def __call__(self, plot=True):
        return self.train(plot)


    def _preprocess(self, df):

        '''
        Preprocessing method: Takes care of missing values, encoding categorical features
                              augmenting features
        returns: preprocessed dataframe

        args::
            df: dataframe to be preprocessed
        '''
        
        self.df = df

        df = df.fillna(-9999, inplace=False)
        
        # beginning augmentation procedure

        df_wells = df.WELL.values
        df_depth = df.DEPTH_MD.values

        # augmentation procedure terminated to resume below (reasons)
        df = df.drop('WELL', axis=1, inplace=False)

        group_encoded = pickle.load(open('model/group_encoded', 'rb'))
        formation_encoded = pickle.load(open('model/formation_encoded', 'rb'))

        df['GROUP_enc'] = (df.GROUP).map(group_encoded)
        df['FORMATION_enc'] = (df.FORMATION).map(formation_encoded)

        df = df.drop(['GROUP', 'FORMATION'], axis=1, inplace=False)

        print('Augmenting features...')
        print(f'Shape of dataframe before augmentation: {df.shape}')
        # augmentation procedure continues...

        df, padded_rows = augment_features(df.values, df_wells, df_depth)
        del(padded_rows)
        print(f'Shape of dataframe after augmentation: {df.shape}')

        df = pd.DataFrame(df)

        return df


    def train(self, start, end, pretrained=True):

        '''
        Training method
        returns: a list of the pretrained models, test features needed for prediction

        args::
            start: where prediction should start from
            end: where prediction should stop
            pretrained: takes in boolean values; 
                        True if pretrained models should be used
                        False if model should be trained from scratch

        '''

        self.start = start
        self.end = end
        self.pretrained = pretrained

        if pretrained:

            models = []
            i = 0
            for i in range(1, (len(os.listdir('model')) - 1)):
                model = xgb.Booster()
                model.load_model(f'model/lithofacies_model{i}.model')
                models.append(model)

        else:
            print('Only pretrained model support is currently available')

        test_features = self._preprocess(self.df)

        return models, test_features   

    
    def predict(self, start, end):

        '''
        Method used in making prediction
        returns: prediction values

        args::
            start: where prediction should start from
            end: where prediction should stop
        '''

        self.start = start
        self.end = end

        trained_models, test_features1 = self.train(start, end)
        test_features = xgb.DMatrix(test_features1.values)

        predictions = np.zeros((test_features1.shape[0], 12))
        i = 1
        for model in trained_models:
            predictions += model.predict(test_features)
            print(f'Model {i}, predicting...')
            i += 1

        predictions = predictions/(len(os.listdir('model')) - 2)
        predictions = pd.DataFrame(predictions).idxmax(axis=1)
        print('Predictions complete!')

        return predictions

    def plot_feat_imp(self, model, columns):

        '''
        Method to plot the feature importance of the model in a bar chart
        according to rank (importance)

        returns: plot of the features importance

        args::
            model: trained model object
            columns: features names used for training (dataframe.columns)
        '''

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


class DataHandlers():

    '''
    Handle the preprocessing of the dataframe for categorical and numerical variables
    as well as handling different mnemonics issues.
    args::
        df: dataframe
    '''

    def __init__(self, df):

        self.df = df

    
    def __call__(self):

        df = self.encode_categorical()

        return df


    def encode_categorical(self):

        '''
        Method for encoding categorical variables in a dataframe

        returns: dataframe with the categorical variables encoded
        based on their cardinality
        
        '''

        df = self.df

        columns = list(df.columns)

        cat_df = pd.DataFrame()   #categorical dataframe

        for column in columns:
            if df[column].dtype == object:
                cat_df[column] = df[column]

        # check cardinality of categorical variables then encode based on cardinality

        for column in cat_df.columns:

            # if cardinality is too high/feature is distinct (e.g. a unique ID column), the column will be dropped
            if check_cardinality(cat_df, column) == 'Unique' or check_cardinality(cat_df, column) == 'Distinct':
                cat_df.drop(column, axis=1, inplace=True)

            elif check_cardinality(cat_df, column) == 'High':
                cat_df = label_encode(cat_df, column)

            elif check_cardinality(cat_df, column) == 'Low':
                cat_df = one_hot_encode(cat_df, column)

        df = df.drop(cat_df.columns, axis=1, inplace=False)
        df = pd.concat((df, cat_df), axis=1)

        return df


    def set_mnemonics(self, GR='GR', RHOB='RHOB', NPHI='NPHI', CALI='CALI', BS='BS', RDEP='RDEP',
                      RMED='RMED', RSHA='RSHA', PEF='PEF', DTC='DTC', SP='SP', ROP='ROP', DTS='DTS', 
                      DCAL='DCAL', DRHO='DRHO', MUDWEIGHT='MUDWEIGHT', RMIC='RMIC', ROPA='ROPA', 
                      RXO='RXO', GROUP='GROUP', FORMATION='FORMATION', X_LOC='X_LOC', Y_LOC='Y_LOC', 
                      Z_LOC='Z_LOC', DEPTH_MD='DEPTH_MD', WELL='WELL', target=None, depth=False):

        '''
        Method to set well mnemonics
        This preprocessing is necessary to have the data in the right format for prediction
        by the pretrained model

        returns: dataframe with 24 log types (the arguments passed)
        args::
            Well curves. Default mnemonics are used. Specify the curve value in 
            cases of different value. Specify False if curve does not appear in
            the data/well.
        Set arguments/logs to False if they are not present in well data
        '''

        self.GR, self.RHOB, self.NPHI, self.CALI, self.BS, self.RDEP = GR, RHOB, NPHI, CALI, BS, RDEP
        self.RMED, self.RSHA, self.PEF, self.DTC, self.SP, self.ROP, self.DTS = RMED, RSHA, PEF, DTC, SP, ROP, DTS
        self.DCAL, self.DRHO, self.MUDWEIGHT, self.RMIC, self.ROPA = DCAL, DRHO, MUDWEIGHT, RMIC, ROPA
        self.RXO, self.GROUP, self.FORMATION, self.X_LOC, self.Y_LOC, self.Z_LOC = RXO, GROUP, FORMATION, X_LOC, Y_LOC, Z_LOC
        self.depth, self.DEPTH_MD, self.target, self.WELL, df = depth, DEPTH_MD, target, WELL, self.df

        if depth == False:
            df['depth'] = df.index

        if WELL == False:
            df['WELL'] = 'Same Well'
            
        
        new_df = pd.DataFrame()
        new_df['WELL'], new_df['DEPTH_MD'], new_df['X_LOC'] = df[WELL], df[DEPTH_MD], df[X_LOC]
        new_df['Y_LOC'], new_df['Z_LOC'], new_df['GROUP'] = df[Y_LOC], df[Z_LOC], df[GROUP]
        new_df['FORMATION'], new_df['CALI'], new_df['RSHA'] = df[FORMATION], df[CALI], df[RSHA]
        new_df['RMED'], new_df['RDEP'], new_df['RHOB'] = df[RMED], df[RDEP], df[RHOB]
        new_df['GR'], new_df['NPHI'], new_df['PEF'] = df[GR], df[NPHI], df[PEF]
        new_df['DTC'], new_df['SP'], new_df['BS'] = df[DTC], df[SP], df[BS]
        new_df['BS'], new_df['ROP'] = df[BS], df[ROP]
        new_df['DCAL'], new_df['DRHO'], new_df['MUDWEIGHT'], new_df['RMIC'] = df[DCAL], df[DRHO], df[MUDWEIGHT], df[RMIC]

        return new_df