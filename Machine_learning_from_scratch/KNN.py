import numpy as np
from collections import Counter

def euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def accuracy(y_pred,y_test):
    return np.sum(y_pred == y_test)/len(y_test)

class KNN:
    def __init__(self,k=5):
        self.k = k
        
    def fit(self,x,y):
        
        self.x_train = x
        self.y_train = y
        
    def predict_x(self,x):
        distances = [euclidean(x,x2) for x2 in self.x_train]
        #k nearest point indices
        k_indices =np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        max_vote = Counter(k_nearest_labels).most_common()[0][0]
        return max_vote 
    
    def predict(self,x):
        predictions = [self.predict_x(x) for x in self.x_train]
        return predictions 
        