import numpy as np
from .base import Activation

class Softmax(Activation):
    def forward(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        self.output=exp_x/np.sum(exp_x,axis=1,keepdims=True)
        return self.output
    
    def backward(self,grad_out):
        return grad_out
