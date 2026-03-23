import numpy as np
from .base import Loss

class Binary_cross_entropy(Loss):
    def forward(self,y_pred,y):
        self.y_pred=y_pred
        self.y=y
        eps=1e-8
        return -np.mean(y*np.log(y_pred+eps)+(1-y)*np.log(1-y_pred+eps))
    
    def backward(self):
        return (self.y_pred-self.y)/self.y.shape[1]