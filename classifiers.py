import numpy as np
import pandas as pd


def reg_classifier(data,labels):
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

def load_data(filename,clip=-1):
    """
    Reads filename, returns matrix of data with bias column
    and a vector of labels
    """
    data = pd.read_csv(filename,header=0)
    labels = np.array(data['sentiment'])
    data = data.drop(['id','sentiment'],axis=1)
    data['bias']=np.ones(data.shape[0])

    
    
    return np.array(data[:clip]), labels[:clip]
