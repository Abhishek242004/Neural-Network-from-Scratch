import numpy as np
from .base import Loss

class Cross_entropy(Loss):
    def forward(self,y_pred,y):
        self.y_pred=y_pred
        self.y=y
        eps=1e-8
        return -np.log(np.sum(y*np.log(y_pred+eps),axis=0))
    def backward(self):
        return (self.y_pred-self.y)/self.y.shape[1]