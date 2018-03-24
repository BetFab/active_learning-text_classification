from modAL.models import Committee
from modAL.models import ActiveLearner

from modAL.disagreement import vote_entropy_sampling
from modAL.uncertainty import uncertainty_sampling

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

import pickle
import os

import argparse

import sys
sys.path.insert(0, "./utils")
from saving import *
from models import *
from score import *

def snapshot(X_train, y_train, X_pool, y_pool, learner, exp_path, iter):

	state_path = 'iter'+str(iter)+'/'
	new_state_dir = exp_path+'snapshot/'+state_path

	# Create a new folder at exp_path/snapshot/iter
	if not os.path.exists(new_state_dir):
		os.makedirs(new_state_dir)

	# Save train and pool 
	Xy_array_to_dataframe(X_train, y_train, new_state_dir+'train.csv')
	Xy_array_to_dataframe(X_pool,  y_pool,  new_state_dir+'pool.csv')

	# Serialize the learner
	serialize(learner, new_state_dir+'learner.pickle')

	return  state_path 



#----------- ARGUMENT PARSING ----------------------

parser = argparse.ArgumentParser(description='Launch an experiment.')
parser.add_argument('name', metavar='name', type=str,
					help='name of the experiment to launch, ./experiments/name/')

parser.add_argument('--iter', metavar='iter_init', type=int,
					help='update iteration N from when resume the experiment, exp/snapshot/iterN')

parser.add_argument('--iter_max', metavar='iter_max', type=int,
					help='update iteration when the experiment stop')

parser.add_argument('--iter_snapshot', metavar='iter_snapshot', type=int,
					help='update iteration interval between snapshots')

parser.add_argument('--iter_test', metavar='iter_test', type=int,
					help='update iteration interval between tests')

parser.add_argument('--verbose', dest='verbose',action='store_const', const=True, default=False,
					help='print more detailed informations on the terminal')

args = parser.parse_args()
#print(args)
#print(args.name)

verbose = args.verbose

name_exp = args.name

exp_path = './experiments/'+name_exp+'/'

assert(os.path.exists(exp_path))

print('- Loading experiment {}'.format(exp_path))
param_dict = pickle.load( open( exp_path+"param.pickle", "rb" ) )

for keys, value in param_dict.items():
	print('     > {} : {}'.format(keys, value))

print('- Update from command line:')
if args.iter != None :
	assert(os.path.exists(exp_path+'snapshot/iter'+str(args.iter)+'/'))
	param_dict['iter_curr'] = args.iter
	print('     > {} : {}'.format('iter_curr', args.iter))

if args.iter_max != None :
	param_dict['iter_max'] = args.iter_max
	print('     > {} : {}'.format('iter_max', args.iter_max))

if args.iter_snapshot !=  None:
	param_dict['iter_snapshot'] = args.iter_snapshot
	print('     > {} : {}'.format('iter_snapshot', args.iter_snapshot))

if args.iter_test !=  None :
	param_dict['iter_test'] = args.iter_test
	print('     > {} : {}'.format('iter_test', args.iter_test))

serialize(param_dict, exp_path+"param.pickle")

iter_init = param_dict['iter_curr']
iter_max = param_dict['iter_max']
iter_snapshot = param_dict['iter_snapshot']
iter_test = param_dict['iter_test']
state_path = 'init/'

nb_query = param_dict['nb_query_by_iter']
nb_auto_max = param_dict['nb_max_auto_samples']

if iter_init > 0 :
	state_path = 'iter'+str(iter_init)+'/'

assert(os.path.exists(exp_path+'snapshot/'+state_path))

print('- Loading snapshot {}'.format(iter_init))

data_path = exp_path + 'snapshot/' + state_path

print('  > training data ');

X_train, y_train = dataframe_to_Xy_array(data_path+'train.csv')
print("  ... data shape : {}".format(X_train.shape))

print('  > pool data ')

X_pool, y_pool = dataframe_to_Xy_array(data_path+'pool.csv')
print('  ... data shape : {}'.format(X_pool.shape))

print('  > Learner')
learner = pickle.load( open( data_path+"learner.pickle", "rb" ) )

print('- Loading test datasets')

dict_test = {}
for test_sets in param_dict['test_db']:
	print(  '> {}'.format(test_sets[0]))
	X_test, y_test = dataframe_to_Xy_array(test_sets[1])
	dict_test[test_sets[0]] = (X_test, y_test)

measures_list = param_dict['test_measure']

print('- Loading scores')

scores = pd.read_csv(exp_path+'scores.csv', index_col=0)

print(" - EXPERIMENT")

bool_auto = (type(learner) == SemiActiveLearner)

if(bool_auto):
	stats_auto_learn = pd.read_csv(exp_path+'stats_auto_learning.csv', index_col=0)

iter_curr = iter_init

while (iter_curr < iter_max) and (len(y_pool) >= nb_query): 

	if(verbose) :print("ITER {}".format(iter_curr))

	if(verbose) :print("-- querying")
	query_idx, query_sample = learner.query(X_pool, n_instances=nb_query)
	if(verbose) :print("     ... query {}".format(query_idx))

	if(verbose) :print("-- teaching")
	learner.teach( X = query_sample, y = y_pool[query_idx].flatten())

	if(verbose) :print("-- updating the pool")
	X_train = np.vstack([X_train, query_sample])

	y_train = np.concatenate((y_train, y_pool[query_idx]))
	if(verbose) :print('    ... train shape : {}'.format(X_pool.shape))


	X_pool = np.delete(X_pool, query_idx, 0)
	y_pool = np.delete(y_pool, query_idx, 0)
	if(verbose) :print('    ... pool shape : {}'.format(X_pool.shape))

	if (bool_auto):
		if(verbose) :print("-- auto learning")
		indx_samples_added, samples_added = learner.auto_learn(X_pool,y_pool, n_instances_max=nb_auto_max)
		if(verbose) :print("-- computing stats") #	col = ['NB_ADD_SAMPLES', 'NB_ALL_AUTO_SAMPLES', 'FRAC_ALL_AUTO_SAMPLES_IN_TRAINING',
									#			'SCORE_ADD_SAMPLES', 'AVG_CONF_ADD_SAMPLES',
									#			'SCORE_ALL_AUTO_SAMPLES', 'AVG_CONF_ALL_AUTO_SAMPLES']
		stats = [len(indx_samples_added), len(learner.y_auto_samples), len(learner.y_auto_samples)/len(learner.y_training),
				learner.score(samples_added, y_pool[indx_samples_added]), np.mean(classifier_certainty(learner.estimator, samples_added)),
				learner.score_auto_samples(), np.mean(classifier_certainty(learner.estimator, learner.X_training[learner.indx_auto_samples])),iter_curr]
		if(verbose) :print(stats)	
		stats_auto_learn.loc[stats_auto_learn.shape[0]] = stats						

		if(verbose) :print("-- updating the pool")
		X_pool = np.delete(X_pool, indx_samples_added, 0)
		y_pool = np.delete(y_pool, indx_samples_added, 0)
		if(verbose) :print('    ... pool shape : {}'.format(X_pool.shape))

	#----------- SNAPSHOT ---------------------	
	if ((iter_curr - iter_init)>0 )& ((iter_curr - iter_init) % iter_snapshot == 0):
		print("ITER {} SNAHPSHOT".format(iter_curr))
		state_path = snapshot(X_train, y_train, X_pool, y_pool, learner, exp_path, iter_curr)
		if(verbose) :print("... state_path: {}".format(state_path))

		param_dict['iter_curr'] = iter_curr
		serialize(param_dict, exp_path+"param.pickle")
		if(bool_auto):
			stats_auto_learn.to_csv(exp_path+'stats_auto_learning.csv')

	#----------- TESTTING ----------------------
	if ((iter_curr - iter_init) % iter_test == 0):
		print("ITER {} TEST".format(iter_curr))
		for m in measures_list: 
			measure_name = m[0]
			measure_fct = m[1]
			for test_name, set_test in dict_test.items():

				score = measure_fct(learner, set_test[0] ,set_test[1])
				scores.loc[scores.shape[0]] = [test_name, measure_name, score, iter_curr]
				print('.. testing on {} : {} = {}'.format(test_name, measure_name, score))
		scores.to_csv(exp_path+'scores.csv')

	#--------------------------------------------
	iter_curr += 1

print("END SNAHPSHOT")
state_path = snapshot(X_train, y_train, X_pool, y_pool, learner, exp_path, iter_curr)
if(verbose) :print("... state_path: {}".format(state_path))

param_dict['iter_curr'] = iter_curr
serialize(param_dict, exp_path+"param.pickle")

