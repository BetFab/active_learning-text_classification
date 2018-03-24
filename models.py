"""

"""

import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.utils import check_array

import pandas as pd

def nargmax(array):
	return np.argwhere(array == np.amax(array)).flatten()

def classifier_certainty(classifier, X, **predict_proba_kwargs):

	classwise_certainty = classifier.predict_proba(X, **predict_proba_kwargs)
	certainty = np.max(classwise_certainty, axis=1)

	return certainty

def max_certainty_sampling(classifier, X, n_instances_max = -1, **certainty_measure_kwargs):
	certainty = classifier_certainty(classifier, X, **certainty_measure_kwargs)
	indx_certainty_samples = nargmax(certainty)

	if (n_instances_max != -1) & (n_instances_max < len(indx_certainty_samples)):
		indx_certainty_samples = np.random.choice(indx_certainty_samples, size=n_instances_max)

	return indx_certainty_samples, X[indx_certainty_samples]

def sure_certainty_sampling(classifier, X, n_instances_max = -1, **certainty_measure_kwargs):
	certainty = classifier_certainty(classifier, X, **certainty_measure_kwargs)
	indx_sure_samples = np.argwhere(certainty == 1).flatten()

	if (n_instances_max != -1) & (n_instances_max < len(indx_sure_samples)):
		indx_sure_samples = np.random.choice(indx_sure_samples, size=n_instances_max)

	return indx_sure_samples, X[indx_sure_samples]


class SemiActiveLearner(ActiveLearner):
	"""
	This class is ...

	Parameters
	----------

	Attributes
	----------

	Examples
	--------
	>>>
	...
	>>>

	"""

	def __init__(
			self,
			estimator,
			query_strategy = uncertainty_sampling,
			automatic_learning_strategy = sure_certainty_sampling,
			X_training=None, y_training=None,
			indx_auto_samples=None, y_auto_samples=None,
			bootstrap_init=False,
			**fit_kwargs
	):
		# ATTENTION : initialization with indx_auto non empty supposes that there exist a label for 
		# those samples (ex previous training) in y_training.


		ActiveLearner.__init__(self, estimator, query_strategy, X_training, y_training, bootstrap_init, **fit_kwargs)

		assert callable(automatic_learning_strategy), 'automatic_learning_strategy must be callable'

		self.automatic_learning_strategy = automatic_learning_strategy


		if (type(indx_auto_samples) == type(None)) and (type(y_auto_samples) == type(None)):
			self.indx_auto_samples = np.array([], dtype=np.int)
			self.y_auto_samples = np.array([])
		elif (type(indx_auto_samples) != type(None)) and (type(y_auto_samples) != type(None)):
			
			self.indx_auto_samples = check_array(indx_auto_samples, ensure_2d = False)
			self.y_auto_samples = check_array(y_auto_samples, ensure_2d=False)

			assert len(self.indx_auto_samples) == len(np.unique(self.indx_auto_samples))
			assert len(self.indx_auto_samples) == len(self.y_auto_samples)


	def auto_learn(self, X, y=None, **automatic_learning_kwargs):
		check_array(X, ensure_2d=True)

		indx_samples_added, samples_to_add = self.automatic_learning_strategy(self.estimator, X, **automatic_learning_kwargs)
		y_samples_to_add = self.predict(samples_to_add)

		true_y_samples_to_add = np.empty(len(y_samples_to_add))
		true_y_samples_to_add.fill(np.nan)
		# check if y done -> same size as X
		if type(y) != type(None):
			y = check_array(y, ensure_2d=False)
			assert len(X) == len(y)
			true_y_samples_to_add = y[indx_samples_added]

		# Add to y_auto_samples
		self.y_auto_samples = np.concatenate((self.y_auto_samples, true_y_samples_to_add))
				
		# Add to X_train
		self.X_training = np.vstack([self.X_training, samples_to_add])

		# Add to y_train
		self.y_training = np.concatenate((self.y_training, y_samples_to_add))

		# update indx_auto_samples
		indx_in_training = np.arange(len(self.y_training) - len(y_samples_to_add), len(self.y_training))
		self.indx_auto_samples = np.concatenate((self.indx_auto_samples, indx_in_training))

		# train
		self._fit_to_known()

		return indx_samples_added, samples_to_add



	def predict_auto_samples(self, **predict_kwargs):
		print('predict auto samples')

		return self.estimator.predict(X_training[indx_auto_samples], **predict_kwargs)

	def predict_proba_auto_samples(self, X, **predict_proba_kwargs):
		print('predict proba auto samples')

		return self.estimator.predict_proba(X_training[indx_auto_samples], **predict_proba_kwargs)

	def score_auto_samples(self, **score_kwargs):
		#select X in automatic samples where y is defined
		X = self.X_training[self.indx_auto_samples[np.isnan(self.y_auto_samples)==False]]
		#select only defined y
		y = self.y_auto_samples[np.isnan(self.y_auto_samples)==False]

		score = self.score(X, y, **score_kwargs)
		return score

