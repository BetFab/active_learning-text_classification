import numpy as np

from modAL.disagreement import vote_entropy, consensus_entropy, KL_max_disagreement
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy

def score_accuracy(learner, X, y, **score_kwargs):
	return learner.score(X, y, **score_kwargs)

def score_mean_uncertainty(learner, X, y=None, **uncertainty_kwargs):
	uncertainty = classifier_uncertainty(learner, X, **uncertainty_kwargs)
	return np.mean(uncertainty)

def score_mean_margin(learner, X, y=None, **uncertainty_kwargs):
	margin = classifier_margin(learner, X, **uncertainty_kwargs)
	return np.mean(margin)

def score_mean_entropy(learner, X, y=None, **uncertainty_kwargs):
	entropy = classifier_entropy(learner, X, **uncertainty_kwargs)
	return np.mean(entropy)

def score_mean_vote_entropy(committee, X, y=None, **disagreement_kwargs):
	entr = vote_entropy(committee, X, **disagreement_kwargs)
	return np.mean(entr)

def score_mean_consensus_entropy(committee, X, y=None, **disagreement_kwargs):
	entr = consensus_entropy(committee, X, **disagreement_kwargs)
	return np.mean(entr)

def score_mean_KL_max_disagreement(committee, X, y=None, **disagreement_kwargs):
	disagreement = KL_max_disagreement(committee, X, **disagreement_kwargs)
	return np.mean(disagreement)

