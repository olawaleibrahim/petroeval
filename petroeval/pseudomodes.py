"""
Machine learning module for predicting lithology and lithofacies labels
and other ML functionalities
"""

from plots import four_plots
import joblib
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as XGBRegressor

class PredictLitho():

    '''
    Class for predicting lithology
    '''

    def __init__(self, df, depth_col):

        self.df = df
        self.depth_col = depth_col

    def __call__(self, plot=True):
        self.train(plot)

    
    def _preprocess(self, df, target):

        self.df = df
        self.target = target

        df.fillna(-9999, inplace=True)
        target = df[target]
        new_df = (df.drop(target, axis=1)).copy()


        scaled_df = StandardScaler().fit_transform(new_df)

        return scaled_df, target


    def train(self, target, start, end, model='RF', CV=3):

        self.model = model
        self.target = target
        self.start = start
        self.end = end
        self.CV = CV
        df = self.df

        try:
            if CV < 3:
                print(f'Number of cross validation folds should be greaterb than 2; {CV} specified')
            raise ('Invalid Entry Error')

        except ValueError as err:
            print(err)

        train_features, train_target = self._preprocess(self.df, self.target)

        # divide dataframe into train part and part needed for prediction

        top_df = train_features.iloc[0:train_features[train_features.depth == start].index[0]]
        bottom_df = train_features.iloc[train_features[train_features.depth == end].index[0]: train_features.shape[0] + 1]

        train_features = pd.concat((top_df, bottom_df), axis=0)



        #test_features = df.
        #train_features = df.iloc[0:df.depth.iloc[start]]
        #train_target = 

        features = df.drop([target, self.depth_col], axis=1)
        feature_columns = list(features.columns)

        if model == 'RF':

            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            model.fit(train_features, train_target)

        elif model == 'XGB':
            model = xgb.XGBRegressor(n_estimators=3000, max_depth=8, reg_lambda=500,
            random_state=20)

        X_train, y_train, X_test, y_test = ms.train_test_split(train_features, train_target,
                                                                test_size=0.2, random_state=20)

        

        if plot:
            self.plot_feat_imp(model)

        return model, test_features


    def predict(self, target, start, end, model='RF', CV=3):

        self.model = model
        self.target = target
        self.start = start
        self.end = end

        trained_model, test_features = self.train(model, target, start, end)
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