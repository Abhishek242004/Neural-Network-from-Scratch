import numpy as np
from .base import Loss

class MSE(Loss):
    expects_logits=False
    def forward(self,y_pred,y):
        self.y_pred=y_pred
        self.y=y
        return np.mean((y_pred-y)**2)
    
    def backward(self):
        return 2*(self.y_pred-self.y)/self.y.shape[0]