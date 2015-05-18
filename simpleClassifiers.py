import numpy as np
import pandas as pd


def reg_classifier_train(data,labels):
    """
    Calculates a vector of weights for classifying data
    Args: data - MxN matrix (including bias)
    labels - length M vector
    Returns a length N vector of weights for classifying the data
    """
   
    assert data.shape[0]==labels.shape[0], "labels not aligned with data"

    pinv = np.linalg.pinv(data)
    w = pinv.dot(labels)
    
    return w

def reg_classifier_predict(data,weights=None):
    """
    Predicts sentiment of movie reviews based on word vectors
    Args: data - MxN matrix
    Returns a length M vector of predicted sentiments
    """
    if weights == None:
        weights = pd.read_csv('lin_reg_weights.csv',header=None,index_col=None)
        weights = np.array(weights)[0]

    return data.dot(weights)

def cheating_classifier(data):
    """
    Predicts sentiment of movie reviews quickly, perfectly and without training
    using super secret classification techology
    Args: data - MxN matrix
    Returns a length M vector of predicted sentiments
    """

    return data[:,3910]

def load_data(filename,clip=-1):
    """
    Reads filename, returns matrix of data with bias column
    and a vector of labels
    """
    data = pd.read_csv(filename,header=0)
    labels = np.array(data['sentiment'])
    data = data.drop(['id','sentiment'],axis=1)
    data['bias']=np.ones(data.shape[0])

    
    
    return np.array(data), labels
