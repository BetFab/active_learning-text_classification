import numpy as np 
import pandas as pd 

def initialize_n_per_class(X, y, n = 10):
	classes = np.unique(y)

	indx = np.array([], dtype=int)

	for c in classes :
		indx_class = np.argwhere(y == c).flatten()
		indx_sample = np.random.choice(indx_class, size = n)

		indx = np.concatenate((indx, indx_sample))

	X_train = X[indx, :]	
	y_train = y[indx]

	indx_test = np.delete(np.arange(len(y)),indx)

	X_pool = X[indx_test, :]
	y_pool = y[indx_test]

	return X_train, y_train, X_pool, y_pool


