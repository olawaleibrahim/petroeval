Modules
=======
All modules containd in the package

files_reader
^^^^^^^^^^^^
Reading/Importing a single lasio file.::

    las = pet.read_lasio('BOGI 01.las')

Reading/Importing multiple lasio files.::

    las1 = 'ATAGA 5.LAS'  #file path
    las2 = 'ATAGA 10.las'  #file path

    dual_las = pet.read_lasios(las1, las2)
    print(dual_las)

    >>> [<lasio.las.LASFile at 0x7fd19c63db50>, <lasio.las.LASFile at 0x7fd19c63ddd0>]

Cconverting las file to dataframe.::

    df = las.df()

evaluate_reservoir
^^^^^^^^^^^^^^^^^^^

This module is used for evaluating and estimating reservoir properties.

FormationEvaluation(data, GR, NPHI, RHOB, RT, top, base, cutoff)
*****************************************************************

Class to evaluate a reservoir based on four main petrophysical parameters.
Creates an instance of the reservoir to be evaluated

Arguments
---------
**data**: dataframe  or csv format of data, **GR**: gamma ray column of table, **NPHI**: neutron porosity column title, **RHOB**: density column title

**RT**: resistivity column title, **top**: top of reservoir (in, metres, feets), **base**: base of reservoir (in, metres, feets), cutoff: Shale baseline value in API

Example.::

    reservoir1 = FormationEvaluation(data=data, GR=GR, NPHI=NPHI, RHOB=RHOB, RT=RT, top=0, base=1000, cutoff=75)
    print(reservoir1)

    >>> <petroeval.evaluate_reservoir.FormationEvaluation object at 0x7fd171c18450>
    
show_table method
********************

Example.::

    table = reservoir1.show_table(baseline_default=False)

    >>> 75 will be used for evaluation
    ESTIMATED PETROPHYSICAL PARAMETERS

Fill missing values using mean values of the columns, specify value if mean shouldn't be applied.::

    df1 = reservoir1.fill_missing(use_mean=False, value=55)

    pet.visualizations.summary(table)

    >>> Title: Petrophysical Summary of the Parameters Evaluated
            GR	                LITHO	        VSH	    NET_PAY	    PHIDF	        PHIE	        SW	       OIL_SAT
    count16641.000000	16641.000000	16641.000000	8448.00000	16300.000000	16238.000000	16300.000000	16300.000000
    mean55.551470	    0.441981	    0.221180	    0.12938	    0.312697	    0.251374	    0.324355	    0.675645
    std	    24.589393	    0.496637	    0.154419	    0.33564	    0.084605	    0.103454	    0.078342	    0.078342
    min	    0.079000	    0.000000	    0.000000	    0.00000	    0.093091	    0.001913	    0.196181	    0.124013
    25%	    33.010000	    0.000000	    0.082945	    0.00000	    0.240667	    0.147560	    0.264379	    0.616633
    50%	    53.100000	    0.000000	    0.170235	    0.00000	    0.307152	    0.259540	    0.310063	    0.689937
    75%	    80.784000	    1.000000	    0.370387	    0.00000	    0.368909	    0.321667	    0.383367	    0.735621
    max	    121.982000	    1.000000	    0.995671	    1.00000	    0.519818	    0.496655	    0.875987	    0.803819

Print out a summary of the petrophysical estimates.::

    #baseline_default argument is set to False, so specified shale baseline cutoff is used

    print(reservoir1.parameters(baseline_default=False))

pseudomodes
^^^^^^^^^^^
Machine learning module for predicting lithology and lithofacies labels

Quick Start.::

    import petroeval.pseudomodes as pds

PredictLitho()
--------------
Class used for predicting lithology values from well logs. Well log data is provided
and missing section range to be predicted is passed. Takes in two arguments; the data, 
and the depth column. Specify False if dataframe index should be used/if depth column 
is not available.

Example.::

    litho = pds.PredictLitho(df=df)

PredictLitho Methods
--------------------

_preprocess(df, target, start, end)
***********************************


Method for preprocessing data by generating train features and labels
using the specified start and end points and also to generate the 
portion of the data needed for predictions (test features). This is done in
a way to prevent data leakage while still using the maximum data points
for a better accuracy

Returns
*******
Train features, train target, test features

Arguments
**********

**df**: dataframe to be preprocessed, **target**: column to be predicted

**start**: where prediction should start from, **end**: where prediction should stop.::

    train_features, test_features, train_target = litho._preprocess(df=df, target='target_column', start=0, end=1000)


train(target, start, end, plot, model='RF', CV=2)
*************************************************

Method used in making prediction

Returns
*******

Trained model, test features needed for predictions

Arguments
*********
**target**: Column to be predicted, **start**: where prediction should start from
**end**: where prediction should stop, **model**: model to be used; default value is 'RF' for random forest
other options are 'XG' for XGBoost,'LGB' for LightGBM

**CV**: number of cross validation folds to run (currently not implemented).::
    
    trained_model = litho.train(target='target_column', start=0, end=1000, plot=True)


predict(self, target, start, end, model='RF', CV=2)
***************************************************

Method used in making prediction

Returns
*******

Prediction values

Arguments
**********

**target**: Column to be predicted, **start**: where prediction should start from

**end**: where prediction should stop, **model**: model to be used; default value is 'RF' for random forest, 
other options are 'XG' for XGBoost, 'CAT' for CatBoost,

**CV**: number of cross validation folds to run (currently not implemented).::

    predictions = lithos.predict('GR', 0, 500, model='LGB')

PredictLabels()
---------------

Class for predicting lithofacies from well logs.

Arguments
*********

**df**: dataframe for predicting lithofacies, **depth_col**: specify column name if prediction should be based on that depth,
leave as default (None), if dataframe index should be used or if depth column is not available
**plot**: to return the feature importance plot after model training

Example.::

    facies = pds.PredictLabels(df=df, depth_col='DEPTH_MD', plot=True)

PredictLitho Methods
--------------------

_preprocess(df)
***************

Preprocessing method: Takes care of missing values, encoding categorical features
augmenting features

Returns
*******

Preprocessed dataframe

Arguments
*********

df: dataframe to be preprocessed

Example.::

    processed_data = facies._preprocess(df)

pretrain()
**********

Training method
        
Returns
*******

A list of the pretrained models, test features needed for prediction

Example.::

    pretrained_models = facies.pretrain()

prepare(train, target, test=None, start=None, end=None)
*******************************************************

Method to prepare dataset(s) for training

Returns
********

Train data features, test data features, train target

Arguments
*********

**train**: train data, **target**: target column string name
**test**: test dataframe if test features is in a different dataframe, default is none (if test is part of train dataset). start and end should be specified.

**start**: specify start point for test features from train data if test features, dataframe does not exist i.e if desired prediction section is a missing section from the supplied train data, **end**: where test features should stop from train data provided

Sample.::

    train_features, test_features, train_target = facies.prepare(train=traindata, 
                                                                 target='target_column', 
                                                                 test=None, start=None, end=None)

_train(train_df, target, start=None, end=None, test_df=None, model='RF')
*************************************************************************

Returns
*******

Trained model, test features needed for prediction

Arguments
**********

**train_df**: train dataframe, start: where prediction should start from, **end**: where prediction should stop, target: target column to be used for training (string/column name)

**model**: model to be used; default value is 'RF' for random forest, other options are 'XG' for XGBoost

Example.::

    trained_model, test_features = facies._train(train_df=traindata, target='target_column', 
                                                 start=None, end=None, test_df=None, model='RF')

predict(test_df=None, target=None, model=False)
***********************************************

Returns
*******

Prediction values

Arguments
*********

**test_df**: test dataframe if test features is in a different dataframe, **model**: default value is false (pretraioned model is used for prediction), 

Trained model object should be specified if available, if not, the model is trained based on other arguments passed

Example.::

    predictions = facies.predict(model=model, test_df=test_features, 
                                 start=0, end=1000)

visualizations
^^^^^^^^^^^^^^

plots
^^^^^

utils
^^^^^

preprocessing
^^^^^^^^^^^^^

