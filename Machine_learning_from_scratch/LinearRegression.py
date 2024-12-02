import numpy as np

def mse(y_test,y_pred):
    return np.mean((y_pred-y_test)**2)


class LinearRegression:
    def __init__(self,epochs = 500, lr = 0.01):
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.lr = lr
                
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
        n_sample, n_features = x.shape
        self.weights = np.zeros((n_features)) 
        self.bias = 0 
        for _ in range(self.epochs):
            y_pred = np.dot(x,self.weights) + self.bias 
            print(mse(y_pred,self.y_train))
            #calculate the derivatives 
            dw = (1/x.shape[0])*(np.dot(x.T,(y_pred-y))) 
            db = (1/x.shape[0])*(np.sum(y_pred-y)) 
            #update weights and biases
            self.weights = self.weights - self.lr*dw 
            self.bias = self.bias - self.lr*db 
    def predict(self,x_test):
        predictions = [(np.dot(x,self.weights) + self.bias) for x in x_test]
        return predictions 
            
