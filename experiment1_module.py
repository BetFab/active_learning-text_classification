"""
Only used in the first scripts 
"""

import numpy 
import pandas 

# Dataset
from sklearn.datasets import fetch_20newsgroups

# Feature extraction from text
from sklearn.feature_extraction.text import TfidfVectorizer

# Classification
from sklearn.neighbors import KNeighborsClassifier

# Classification results
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def toDataFrame(news_group):
    """
    Transform the data structure of a the news_group scikit learn dataset to a pandas DataFrame
    The input format should contain .data and .target and .target_names element.
    
    The output DataFrame as a RangeIndex and the following columns ['Text', 'Target', 'Category']
    The information on Target and Category is redundant.
    
    Examples
    --------
    >> news_group  = fetch_20newsgroups() 
    >> df = toDataFrame(news_group)
    
    """

    df = pandas.DataFrame(data = news_group.data, columns=['Text'])
    df['Target'] = news_group.target
    df['Category'] = df['Target'].apply(lambda x: news_group.target_names[x])
    
    return df

def init_train_test(df, N=20, verbose=True):
    """
    Initialize the train and test sets from a DataFrame set of data according to the following procedure:
     - select N documents from each categories
     - append it to the train set
     - drop it from the input set
     - what remains from the input set becomes the test set
     
     At the end, are returned a training set containing N samples of each labels and the testing set containing
     the other available samples. The sets are DataFrame of the same format as the initial DataFrame. 
     The index of the train and test sets are dijoints and corresponds to the index in the initial DataFrame.
     
     Parameters
     ----------
     N : int
         Number of documents to take from each categories in order to create the initial train set
     verbose : boolean
         Print information of the process
     
     Examples
     --------
     >> train, test = init_train_test(df, N=100)
    
    """
    
    categories = df["Category"].unique()
    
    if(verbose) : print('Number of categories : {}\n'.format(len(categories)))
    
    test = df.copy()
    train = pandas.DataFrame(columns = df.columns)
    
    if(verbose) : 
        print("Number of samples : {}".format(df.shape[0]))
        print("Number of columns : {}\n".format(df.shape[1]))
        print("Splitting the data...")
    
    
    for i in range(len(categories)):
        
        #if(verbose) : print("Adding the category {}".format(news_group.target_names[i]))
        
        idx = df[df['Target']==i].sample(N).index
                
        train = train.append(test.loc[idx])
        test.drop(index=idx, inplace=True)
         
    # Shuffle the train
    if(verbose) : print("Shuffling the train set\n")
        
    train = train.sample(frac=1)
            
    if(verbose) :
        print('Number of samples in the train set : {}'.format(train.shape[0]))
        print("Number of samples remining in the test set : {}\n".format(test.shape[0]))
    
    return train,test


def update_train_test(train, test, indx_in_test, verbose=False):
    """
    Update the train and test set by extracting some samples of the test set and adding them to the train set.
    
    Parameters
    ----------
    
    train : DataFrame
        train set to be updated
    test : DataFrame
        test set to be updated
    indx_in_test : list of int
        positions (/!\ may be different from the DataFrame Index) of the testing samples to extract and
        add to the train set
    verbose : boolean
        print some informations on the update
        
    Output
    ------
    
    train : DataFrame
        updated train set
    test : DataFrame
        updated test set
        
    Example 
    -------
    >> _, pred_proba,_ = classify(train, [test])
    >> pred_proba = pred_proba[0]
    >> indx_max = np.argwhere(pred_proba == np.max(pred_proba))
    >> train, test = update_train_test(train, test, indx_max[:,0])    # Update the train set with the predicted samples
                                                                        of the test set with the highest confidence.

    """
    if(verbose) :
        print('Number of samples in the train set : {}'.format(train.shape[0]))
        print("Number of samples in the test set : {}".format(test.shape[0]))
        print("Number of samples to move : {}\n".format(len(indx_in_test)))
    
    indx = test.iloc[indx_in_test].index
    train = train.append(test.loc[indx])
    test.drop(index=indx, inplace=True)
    
    train = train.sample(frac=1) # Probably useless ? depends on the classifier
            
    if(verbose) :
        print('Number of samples in the updated train set : {}'.format(train.shape[0]))
        print("Number of samples in the updated test set : {}".format(test.shape[0]))
    
    
    
    return train,test

def high_confidence_indx( pred_proba ):
    """
    return the indices of the sample predicted with the highest confidence.
    the indices correspond to the position of the sample on the DataFrame used for the classification
    
    Parameter
    ---------
    pred_proba : numpy.ndarray
        contains all the probabilities of classification of the n samples (row) for the m labels (columns)        
    
    Example
    -------
    >> indx_max = high_confidence_indx(pred_proba[0])
    >> test.iloc[indx_max].head(3)
    """
    
    indx_max = numpy.argwhere(pred_proba == numpy.max(pred_proba))
    return indx_max[:,0]
    
def low_confidence_indx( pred_proba ):
    """
    return the indices of the sample predicted with the lowest confidence.
    the indices correspond to the position of the sample on the DataFrame used for the classification
    
    Parameter
    ---------
    pred_proba : numpy.ndarray
        contains all the probabilities of classification of the n samples (row) for the m labels (columns)        
    
    Example
    -------
    >> indx_min = low_confidence_indx(pred_proba[0])
    >> test.iloc[indx_min].head(3)
    """
    
    proba_on_predicted = pred_proba.max(axis=1)
    indx_min = numpy.where(proba_on_predicted == numpy.min(proba_on_predicted))
    
    return indx_min[0]