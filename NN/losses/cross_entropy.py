import numpy as np
from .base import Loss

class Cross_entropy(Loss):
    expects_logits=False
    def forward(self,y_pred,y):
        if y.ndim == 1:
            raise ValueError("y must be one-hot encoded")
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            raise ValueError("y_pred must be probabilities (apply softmax first)")
        self.y_pred=y_pred
        self.y=y
        eps=1e-8
        return -np.sum(y*np.log(y_pred+eps))/y.shape[0]
    def backward(self):
        eps=1e-8
        return -(self.y/(self.y_pred+eps))/self.y.shape[0]