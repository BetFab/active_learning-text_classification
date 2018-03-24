import os

import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_dir+"/../..")
sys.path.insert(0, curr_dir+"/../../utils")

from saving import *

import numpy as np
import pandas as pd

import pickle

# ----------- IMPORT MODELS -------------

#from modAL.models import Committee
from modAL.models import ActiveLearner

#from modAL.disagreement import vote_entropy_sampling
#from modAL.disagreement import max_disagreement_sampling
from modAL.uncertainty import uncertainty_sampling

from models import *
from score import *
from initialization import *

#----------- IMPORT CLASSIFIERS -----------

#from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


name_exp = os.path.dirname(os.path.realpath(__file__)).split('/')[-1]

print('Initializing the experiment {} : '.format(name_exp))

curr_path_snapshot_init = curr_dir+'/snapshot/init/'

if not os.path.exists(curr_path_snapshot_init):
	os.makedirs(curr_path_snapshot_init)

#---------------------------------------------------------------------------------
#                                   DATA
#---------------------------------------------------------------------------------


print('- DATASET : ')
print(' ... loading')

data_path = curr_dir+'/../../data/blobs_5clusters_5features/blobs_5centers_5features.csv'
# ex: data_path = '../../data/blobs_5clusters_5features/blobs_5centers_5features.csv'


X, y = dataframe_to_Xy_array(data_path)

print('     > the dataset contains {} samples of {} features.'.format(X.shape[0], X.shape[1]))

print(' ... splitting train/test')

X_train, y_train, X_pool, y_pool = initialize_n_per_class(X, y, n=5)
#ex: X_train, y_train, X_pool, y_pool = initialize_n_per_class(X, y, n=5)

print('     > the train set contains {} samples, it remains {} samples in the pool.'.format(len(y_train), len(y_pool)))
print(' ... saving\n')

Xy_array_to_dataframe(X_train, y_train, curr_path_snapshot_init + 'train.csv')
Xy_array_to_dataframe(X_pool, y_pool, curr_path_snapshot_init + 'pool.csv')

#---------------------------------------------------------------------------------
#                                  LEARNER
#---------------------------------------------------------------------------------

print(' - LEARNER :')




learner = SemiActiveLearner(estimator = KNeighborsClassifier(n_neighbors= 10, metric='cosine', weights='distance'), 
				  query_strategy = uncertainty_sampling,  automatic_learning_strategy = max_certainty_sampling, 
				  X_training = X_train, y_training = y_train)
"""
INITIALIZE THE LEARNER : 
	-> ActiveLearner :
	  >> classifier = KNeighborsClassifier(n_neighbors= 10, metric='cosine', weights='distance')
	  >> query_strategy = uncertainty_sampling
	  >> learner = ActiveLearner(estimator = classifier, query_strategy = query_strategy,
								X_training = X_train, y_training = y_train)

	-> SemiActiveLearner : 
	  >> classifier = KNeighborsClassifier(n_neighbors= 10, metric='cosine', weights='distance')
	  >> query_strategy = uncertainty_sampling
	  >> learner =  SemiActiveLearner(estimator = classifier, query_strategy = query_strategy,
							  automatic_learning_strategy = max_certainty_sampling, 
							  X_training = X_train, y_training = y_train)

	-> Committee : 
	  >> learner = Committee([learner1, learner2], max_disagreement_sampling)
"""

print('     {}'.format(learner))

print("... saving\n")

serialize(learner, curr_path_snapshot_init + "learner.pickle")


#---------------------------------------------------------------------------------
#                               EXPERIMENT PARAMETERS
#---------------------------------------------------------------------------------

print('- EXPERIMENT PARAMETERS :')
# Experiment parameter
exp_parameter = { 
				# ITERATION MAX
				  'iter_max' : 10,
				# INTERVAL BETWEEN SNAPSHOT 		
				  'iter_snapshot' : 2, 
				# INTERVAL BETWEEN TESTING 
				  'iter_test' : 3, 
				# LIST OF TUPLES (names, path) OF DB ON WHICH THE LEARNER IS TESTED	
				  'test_db' : [('pool_init', curr_dir+'/snapshot/init/pool.csv')],
				# LIST OF TUPLES (names, measure) THAT SCORE THE LEARNER ON THE TEST SETS
				  'test_measure' : [('accuracy', score_accuracy)],

				# NUMBER OF QUERIES TO ADD BY ITERATION
				  'nb_query_by_iter' : 2,

				# NUMBER OF SAMPLES ADDED AUTOMATICLY (SEMIACTIVE)
				  'nb_max_auto_samples' : 10,  

				# AUTOMATIC PARAMETERS  
				  'iter_curr' : 0,
				  'exp_path' : curr_dir
}

for keys, value in exp_parameter.items():
	print('     > {} : {}'.format(keys, value))

print(' ... saving')

serialize(exp_parameter, curr_dir+"/param.pickle")


#---------------------------------------------------------------------------------
#                           TESTING DATA INITIALIZATION
#---------------------------------------------------------------------------------
if (len(exp_parameter['test_db']) >0) & (len(exp_parameter['test_measure'])>0) :
	score_df = pd.DataFrame(columns=['database', 'measure', 'score', 'iter'])
	score_df.to_csv(curr_dir+'/scores.csv')