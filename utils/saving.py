import pandas
import numpy 
import pickle

def dataframe_to_Xy_array(path):
	df = pandas.read_csv(path, index_col=0)
	X = numpy.asarray(df.drop('Cat', axis=1))
	y = numpy.asarray(df['Cat'])
	return X, y

def Xy_array_to_dataframe(X, y, path):
	df = pandas.DataFrame(data = X)
	df['Cat'] = y

	df.to_csv(path)
	return 1

def serialize(obj, path):
	pickle_object = open(path, "wb")
	pickle.dump(obj, pickle_object)
	pickle_object.close()

	return 1
