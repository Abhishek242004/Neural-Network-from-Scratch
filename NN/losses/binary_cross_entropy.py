import numpy as np
from .base import Loss

class Binary_cross_entropy(Loss):
    expects_logits=False
    def forward(self,y_pred,y):
        self.y_pred=y_pred
        self.y=y
        eps=1e-8
        return -np.mean(y*np.log(y_pred+eps)+(1-y)*np.log(1-y_pred+eps))
    
    def backward(self):
        eps = 1e-8
        p = np.clip(self.y_pred, eps, 1 - eps)
        batch_size = p.shape[0]
        return (p - self.y) / (p * (1 - p) * batch_size)
        