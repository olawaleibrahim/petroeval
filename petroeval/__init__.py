'''
This package helps to aid the formation evaluation and interpretation 
process of a reservoir by estimating reservoir properties

Modules
-------

evaluate_reservoir.py
    used for evaluating reservoir properties

pseudomodes.py
    for predicting lithology and lithofacies from well logs

files_reader.py
    module for reading las files

plots.py, visualiazations.py
    helps to make dynamic visualizations of well logs

utils.py
    utilities modules used by other modules; contains
    all relevant functions which are reusable by end user
'''

__all__=[
    'read_lasio','read_lasios','plots','visualizations', 'files_reader',
    'evaluate_reservoir', 'preprocessing', 'pseudomodes','utils'
    ]

from petroeval import visualizations, preprocessing, pseudomodes
from petroeval import evaluate_reservoir, plots, utils
from .evaluate_reservoir import FormationEvaluation
from .pseudomodes import PredictLabels, PredictLitho, DataHandlers
from .visualizations import log_plot
from petroeval import files_reader
from petroeval.files_reader import  read_lasio,read_lasios

from .plots import *
from .pseudomodes import *
from .utils import *