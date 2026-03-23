import numpy as np
from .base import Activation

class Tanh(Activation):
    def forward(self,x):
        self.output=np.tanh(x)
        return self.output
    def backward(self,grad_out):
        return grad_out*(1-self.output**2)