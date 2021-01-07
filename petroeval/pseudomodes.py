"""
Machine learning module for predicting lithology and lithofacies labels
and other ML functionalities

Classes
-------

PredictLitho()
    class for predicting well lithology values by estimating values
    from same well but different log data

PredictLabels()
    class for predicting lithofacies from well logs

DataHandlers()
    class for handling differences in log headers,
    encoding data, and setting log mnemonics to a common 
    standard for easier well data use by the ML classes
    defined above
"""

from .utils import prepare_datasets, label_encode, sample_evaluation, scale_train_test
from .utils import augment_features, check_cardinality, drop_columns, one_hot_encode
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .plots import four_plots, make_facies_log_plot, compare_plots
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle, os


class PredictLitho():
    '''
    A Class for predicting well logs lithology

    Methods
    -------
    _preprocess(df, target, start, end)
        returns the processed train and test features and train 
        targets needed for training the model

    train(target, start, end, plot, model='RF', CV=2)
        returns the trained lithology model and test features 
        which predictions will be made on
    
    predict(target, start, end, model='RF', CV=2)
        returns the predicted lithology values (continous values)
    '''

    def __init__(self, df, depth_col, plot=True):

        '''
        args::
            df: dataframe
            depth_col: depth column, specify False if absent,
            plot: to return the feature importance plot after model training
        '''

        self.df = df
        self.depth_col = depth_col
        self.plot = plot

    def __call__(self, plot=True):

        '''
        returns the train method when the class is called
        '''
        return self.train(plot)

    
    def _preprocess(self, df, target, start, end):

        '''
        Method for preprocessing data by generating train features and labels
        using the specified start and end points and also to generate the 
        portion of the data needed for predictions (test features). This is done in
        a way to prevent data leakage while still using the maximum data points
        for a better accuracy

        Returns
        -------
        Train features, train target, test features

        Arguments
        ---------
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

        encoding = DataHandlers(df=new_df, target=target)
        new_df = encoding.encode_categorical()

        new_df['depth'] = range(0, new_df.shape[0])

        # divide dataframe into train part and part needed for prediction

        '''
        The idea is to use the depth column and the range passed by the parameters.
        The range specified represent the range needed for prediction. Every other part 
        is used as the training data set
        '''

        train_features, test_features, train_target = prepare_datasets(new_df, start, end, target)
        columns = list(test_features.columns)

        # scaling train and test features

        train_features, test_features = scale_train_test(train_features, test_features)

        train_features = pd.DataFrame(train_features, columns=columns)
        test_features = pd.DataFrame(test_features, columns=columns)
        
        # dropping added depth column which was used to aid preprocessing

        train_features = train_features.drop('depth', axis=1, inplace=False)
        test_features = test_features.drop('depth', axis=1, inplace=False)

        return train_features, train_target, test_features


    def train(self, target, start, end, plot, model='RF', CV=2):

        '''
        Method used in making prediction
        Returns
        -------
        Trained model, test features needed for predictions

        Arguments
        ---------
            target: Column to be predicted
            start: where prediction should start from
            end: where prediction should stop
            model: model to be used; default value is 'RF' for random forest
                                     other options are 'XG' for XGBoost
                                                       'LGB' for LightGBM
            CV: number of cross validation folds to run (currently not implemented)
        '''

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV
        df = self.df

        try:
            if CV < 2:
                raise ValueError(f'Number of cross validation folds should be greater than 2; {CV} specified')
            
        except ValueError as err:
            print(err)

        train_features, train_target, test_features = self._preprocess(df, target, start, end)

        '''
        Divide dataframe into train part and part needed for prediction:

        The idea is to use the depth column and the range passed by the parameters.
        The range specified represent the range needed for prediction. Every other part 
        is used as the training data set
        '''

        X_train, X_test, y_train, y_test = ms.train_test_split(train_features, train_target,
                                                                test_size=0.2, random_state=20)

        if model == 'RF':

            model1 = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=20, verbose=2)

        elif model == 'XGB':
            model1 = xgb.XGBRegressor(n_estimators=3000, max_depth=6, reg_lambda=300, random_state=20)

        elif model == 'LGB':
            model1 = lgb.LGBMRegressor(n_estimators=3000, max_depth=6, reg_lambda=300, random_state=20)
        
        if model == 'RF':
            model1.fit(X_train, y_train)
            y_pred = model1.predict(X_test)
            print(sample_evaluation(y_test, y_pred))

        elif model == 'XGB':
            model1.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
                       early_stopping_rounds=100, verbose=50)

            y_pred = model1.predict(X_test)
            print(sample_evaluation(y_test, y_pred))

        elif model == 'LGB':
            model1.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                       early_stopping_rounds=100, verbose=200)

            y_pred = model1.predict(X_test)
            print(sample_evaluation(y_test, y_pred))

        if plot:
            self.plot_feat_imp(model1, list(train_features.columns))

        return model1, test_features


    def predict(self, target, start, end, model='RF', CV=2):

        '''
        Method used in making prediction
        Returns
        -------
        Prediction values

        Arguments
        ---------
            target: Column to be predicted
            start: where prediction should start from
            end: where prediction should stop
            model: model to be used; default value is 'RF' for random forest
                                     other options are 'XG' for XGBoost
                                                       'CAT' for CatBoost
            CV: number of cross validation folds to run (currently not implemented)
                                                       
        '''

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV

        trained_model, test_features = self.train(target, start, end, self.plot, model=model, CV=CV)
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

    Methods
    -------
    _preprocess(df)
        returns the preprocessed dataframe (
            takes care of encoding, missing values, 
            well data augmentation
        )
    pretrain()
        returns a list of the pretrained lithofacies models,
        and preprocessed test dataframe to be predicted on

    prepare(train, target, test=None, start=None, end=None)
        returns the train features, test features and train targets
        need for training the lithofacies model and making predictions

    _train(train_df, target, start=None, end=None, test_df=None, model='RF')
        returns the trained lithofacies model and test features needed 
        for predictions

    predict(test_df=None, target=None, model=False)
        returns the test data predicted lithofacies

    plot_feat_imp(model, columns)
        plots of a bar chart the features importance after model training

    plot_lithofacies(df, predictions, log1, log2, log3, log4, log5, depth_col)
        makes plots and colorplot of the well logs and predicted
        lithofacies respectively

    compare_lithofacies(df, label, predictions, log1, log2, log3, log4, log5, depth_col)
        makes plots and colorplots of the well logs and predicted and 
        actual lithofacies respectively
    '''

    def __init__(self, df, depth_col=None, plot=True):

        '''
        Arguments
        ----------
            df: dataframe for predicting lithofacies
            depth_col: specify column name if prediction should be based on that depth,
                        leave as default (None), if dataframe index should be used or
                        if depth column is not available
            plot: to return the feature importance plot after model training
        '''

        self.df = df
        self.depth_col = depth_col
        self.plot = plot


    def __call__(self, plot=True):

        '''
        returns the train method when the class is called
        '''

        return self.pretrain(plot)


    def _preprocess(self, df):

        '''
        Preprocessing method: Takes care of missing values, encoding categorical features
                              augmenting features

        Returns
        -------
        Preprocessed dataframe

        Arguments
        ---------
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

        print(f'Shape of dataframe before augmentation: {df.shape}')
        print('Augmenting features...')
        # augmentation procedure continues...

        df, padded_rows = augment_features(df.values, df_wells, df_depth)
        del(padded_rows)
        print(f'Shape of dataframe after augmentation: {df.shape}')

        df = pd.DataFrame(df)

        return df


    def pretrain(self):

        '''
        Training method
        
        Returns
        -------
        A list of the pretrained models, test features needed for prediction

        '''
            
        models = []
        i = 0
        for i in range(1, (len(os.listdir('model')) - 1)):
            model = xgb.Booster()
            model.load_model(f'model/lithofacies_model{i}.model')
            models.append(model)

        test_features = self._preprocess(self.df)

        return models, test_features


    def prepare(self, train, target, test=None, start=None, end=None):

        '''
        Method to prepare dataset(s) for training

        Returns
        -------
        Train data features, test data features, train target

        Arguments
        ---------
            train: train data
            target: target column string name
            test: test dataframe if test features is in a different dataframe
            start: specify start point for test features from train data if test features
                    dataframe does not exist i.e if desired prediction section is a missing
                    section from the supplied train data
            end: where test features should stop from train data provided
        '''

        self.train, self.test = train, test
        self.start, self.end = start, end
        self.target = target

        if type(test) == type(None):

            '''
            that is: if test data frame is not specified but will be created
            from train data provided
            '''

            try:
                
                start/2
            except TypeError as err:
                raise err

            except TypeError as err:
                raise err

            if type(self.depth_col) == type(None):
                train['depth'] = range(0, train.shape[0])
            else:
                # converting the depths to integers, for easier slicing operation when preparing data
                depths = list(train[self.depth_col])
                depths_ = []
                for depth in depths:
                    depths_.append(int(depth))
                train['depth'] = depths_

            encode_cat_var = DataHandlers(df=train, target=target)
            train = encode_cat_var.encode_categorical()
                    
            train_features, test_features, train_target = prepare_datasets(
                train, start, end, target
                )

            # dropping added depth column which was used to aid preprocessing
            train_features = train_features.drop('depth', axis=1, inplace=False)
            test_features = test_features.drop('depth', axis=1, inplace=False)  
            train_features = train_features.fillna(-9999, inplace=False)
            test_features = test_features.fillna(-9999, inplace=False)

        else:

            # if test data frame is specified
            
            df = pd.concat((train, test))
            
            label = df[target]
            ntrain = train.shape[0]
            
            encode_cat_var = DataHandlers(df=df, target=target)
            df = encode_cat_var.encode_categorical()

            new_train = df[:ntrain]
            new_test = df[ntrain:]

            '''
            Here, the method takes care of the target encoding.
            If the target is a categorical variable (the lithofacies), it returns the 
            label encoded form, if it is not, it returns the original form (in numbers) 
            for easier processing. The encode_categorical function adds _enc to the target
            encoded column from utils.label_encode function. Hence, reason for the addition
            of '_enc' below
            '''

            train_target = label[:ntrain]

            if train[target].dtype == object:
                train_target = (df[target + '_enc'])[:ntrain]

            columns = new_train.columns
            train_features, test_features = scale_train_test(new_train, new_test)

            train_features = pd.DataFrame(train_features, columns=columns)
            test_features = pd.DataFrame(test_features, columns=columns)
            train_features = train_features.fillna(-9999, inplace=False)
            test_features = test_features.fillna(-9999, inplace=False)

            train_features = train_features.drop(target, axis=1, inplace=False)
            test_features = test_features.drop(target, axis=1, inplace=False)

        return train_features, test_features, train_target


    def _train(self, train_df, target, start=None, end=None, test_df=None, model='RF'):

        '''
        Training method
        Returns
        -------
        Trained model, test features needed for prediction

        Arguments
        ---------
            train_df: train dataframe
            start: where prediction should start from
            end: where prediction should stop
            target: target column to be used for training (string/column name)
            model: model to be used; default value is 'RF' for random forest
                                     other options are 'XG' for XGBoost
        '''

        self.train_df = train_df
        self.test_df = test_df
        self.start = start
        self.end, self.model = end, model
        self.target = target

        train_features, test_features, train_target = self.prepare(
            train=train_df, test=test_df, target=target, start=start, end=end
            )

        if model == 'RF':
            model1 = RandomForestClassifier(
                n_estimators=100, class_weight='balanced', verbose=2, random_state=20
                                            )
            model1.fit(train_features, train_target)

            if self.plot:
                self.plot_feat_imp(model1, train_features.columns)
            print('Model training completed...')
        
        elif model == 'XGB':
            X_train, X_test, y_train, y_test = ms.train_test_split(
                train_features, train_target, test_size=0.2, stratify=train_target
            )

            model1 = xgb.XGBClassifier(
                n_estimators=100, max_depth=10, learning_rate=0.1,
                reg_lambda=300, random_state=20, tree_method='gpu_hist'
                )

            model1.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=10)
            print('Model training completed...')
            if self.plot:
                self.plot_feat_imp(model1, train_features.columns)

        return model1, test_features

    
    def predict(self, test_df=None, target=None, model=False):

        '''
        Method used in making prediction
        Returns
        -------
        Prediction values

        Arguments
        ---------
            test_df: test dataframe if test features is in a different dataframe
            model: default value is false (pretraioned model is used for prediction), 
                    trained model object should be specified if available;
                    if not, the model is trained based on other arguments passed
        '''

        self.test_df = test_df
        self.model = model

        if model == False:
            trained_models, test_features1 = self.pretrain()
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

        else:

            # that is if nodel is specified, predictions should be made on the test data

            predictions = model.predict(test_df)
            print('Predictions complete!')
            
        return predictions


    def plot_feat_imp(self, model, columns):
        '''
        Method to plot the feature importance of the model in a bar chart
        according to rank (importance)

        Returns
        -------
        Plot of the features importance

        Arguments
        ---------
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
        ax.set_title('The Model Features Importance')


    def plot_lithofacies(
        self, df, predictions, log1: str, log2: str, log3: str, log4: str, log5: str, depth_col: str
    ):

        '''
        Plots well logs against depth and corresponding predicted lithofacies
        in a labelled color plot

        Arguments
        ---------

        df: dataframe of well data
        predictions: predicted values in integers (0, 1, 2....)
        log1: str -> well log 1
        ''''''''''''''''''''
        log5: str -> well log 2
        depth_col: depth column
        '''

        self.df = df
        self.predictions = predictions
        self.log1, self.log2, self.log3, self.log4, self.log5 = log1, log2, log3, log4, log5
        self.depth_col = depth_col

        df['predictions'] = predictions
        df['Facies'] = predictions

        for WELL in df['WELL'].unique():
            
            make_facies_log_plot(
                (df[df['WELL'] == WELL]), log1, log2, 
                log3, log4, log5, Depth=depth_col
            )

    
    def compare_lithofacies(
        self, df, label, predictions, log1, log2, log3, log4, log5, depth_col
    ):

        '''
        Plots well logs against depth and corresponding predicted and actual 
        lithofacies in a labelled color plot for comparism

        Arguments
        ---------

        df: dataframe of well data
        label: actual lithofacies values
        predictions: predicted values in integers (0, 1, 2....)
        log1: str -> well log 1
        ''''''''''''''''''''
        log5: str -> well log 2
        depth_col: depth column
        '''

        self.df = df
        self.label, self.predictions = label, predictions
        self.log1, self.log2, self.log3, self.log4, self.log5 = log1, log2, log3, log4, log5
        self.depth_col = depth_col
        
        df['Facies'] = predictions
        df['Actual'] = label

        for WELL in df['WELL'].unique():
            
            compare_plots(
                df[df['WELL'] == WELL], log1, log2, log3, log4, log5, Depth=depth_col
                )


class DataHandlers():
    '''
    Handle the preprocessing of the dataframe for categorical and numerical variables
    as well as handling different mnemonics issues.
    
    Methods
    --------
    encode_categorical()
        Returns
        -------
        Dataframe with categorical variables encoded or 
        dropped based on the degree of cardinality of each 
        categorical column. 
        
        Encoding could be (label encoding
        or one hot encoding). Categorical columns with low cardinality
        are one hot encoded while those with medium to high cardinality
        are label encoded while columns with extremely high cardinalities 
        (tending to be unique columns), are dropped

    set_mnemonics(GR='GR', RHOB='RHOB', NPHI='NPHI', CALI='CALI', BS='BS', RDEP='RDEP',
                      RMED='RMED', RSHA='RSHA', PEF='PEF', DTC='DTC', SP='SP', ROP='ROP', DTS='DTS', 
                      DCAL='DCAL', DRHO='DRHO', MUDWEIGHT='MUDWEIGHT', RMIC='RMIC', ROPA='ROPA', 
                      RXO='RXO', GROUP='GROUP', FORMATION='FORMATION', X_LOC='X_LOC', Y_LOC='Y_LOC', 
                      Z_LOC='Z_LOC', DEPTH_MD='DEPTH_MD', WELL='WELL', depth=False)

        returns dataframe with log headers corrected/adjusted in arguments
    '''

    def __init__(self, df, target=None):

        '''
        Arguments
        ------------
        df: dataframe
        target: target column name (string)
                target column is label encoded if present as string
                returns the target as received if in int or float
        '''

        self.df = df
        self.target = target

    
    def __call__(self):

        df = self.encode_categorical()

        return df


    def encode_categorical(self):

        '''
        Method for encoding categorical variables in a dataframe

        Returns
        -------
        Dataframe with the categorical variables encoded
        based on their cardinality
        
        '''
        
        target = self.target
        df = self.df

        if type(target) == type(None):
            pass
            
        elif df[target].dtype == object:
            df = label_encode(df, target)

        columns = list(df.columns)

        cat_df = pd.DataFrame()   #categorical dataframe

        for column in columns:
            if df[column].dtype == object:
                cat_df[column] = df[column]

        # check cardinality of categorical variables then encode based on cardinality

        previous_cat_columns = []
        for column in cat_df.columns:
            previous_cat_columns.append(column)

            # if cardinality is too high/feature is distinct (e.g. a unique ID column), the column will be dropped
            if check_cardinality(cat_df, column) == 'Unique' or check_cardinality(cat_df, column) == 'Distinct':
                cat_df.drop(column, axis=1, inplace=True)

            elif check_cardinality(cat_df, column) == 'High':
                cat_df = label_encode(cat_df, column)

            elif check_cardinality(cat_df, column) == 'Low':
                cat_df = one_hot_encode(cat_df, column)

        df = df.drop(previous_cat_columns, axis=1, inplace=False)
        df = pd.concat((df, cat_df), axis=1)

        return df


    def set_mnemonics(self, GR='GR', RHOB='RHOB', NPHI='NPHI', CALI='CALI', BS='BS', RDEP='RDEP',
                      RMED='RMED', RSHA='RSHA', PEF='PEF', DTC='DTC', SP='SP', ROP='ROP', DTS='DTS', 
                      DCAL='DCAL', DRHO='DRHO', MUDWEIGHT='MUDWEIGHT', RMIC='RMIC', ROPA='ROPA', 
                      RXO='RXO', GROUP='GROUP', FORMATION='FORMATION', X_LOC='X_LOC', Y_LOC='Y_LOC', 
                      Z_LOC='Z_LOC', DEPTH_MD='DEPTH_MD', WELL='WELL', depth=False):

        '''
        Method to set well mnemonics
        This preprocessing is necessary to have the data in the right format for prediction
        by the pretrained model

        Returns
        -------
        Dataframe with 24 log types (the arguments passed)
        
        Arguments
        ---------
            Well curves. Default mnemonics are used. Specify the curve value in 
            cases of different value. Specify False if curve does not appear in
            the data/well.
        Set arguments/logs to False if they are not present in well data
        '''

        self.GR, self.RHOB, self.NPHI, self.CALI, self.BS, self.RDEP = GR, RHOB, NPHI, CALI, BS, RDEP
        self.RMED, self.RSHA, self.PEF, self.DTC, self.SP, self.ROP, self.DTS = RMED, RSHA, PEF, DTC, SP, ROP, DTS
        self.DCAL, self.DRHO, self.MUDWEIGHT, self.RMIC, self.ROPA = DCAL, DRHO, MUDWEIGHT, RMIC, ROPA
        self.RXO, self.GROUP, self.FORMATION, self.X_LOC, self.Y_LOC, self.Z_LOC = RXO, GROUP, FORMATION, X_LOC, Y_LOC, Z_LOC
        self.depth, self.DEPTH_MD, self.WELL, df = depth, DEPTH_MD, WELL, self.df
        target = self.target

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
        
        if target:
            new_df[target] = df[target]

        return new_df