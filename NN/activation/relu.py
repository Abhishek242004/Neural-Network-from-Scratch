import numpy as np
from .base import Activation

class Relu(Activation):
    def forward(self,x):
        self.input=x
        return np.maximum(x,0)
    def backward(self, grad_out):
        return grad_out*(self.input>0)
