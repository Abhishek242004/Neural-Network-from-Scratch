import numpy as np
from .base import Activation

class Sigmoid(Activation):
    def forward(self,x,clip=True):
        if(clip==True):
            x=np.clip(x,-500,500)
        self.output=1/(1+np.exp(-x))
        return self.output
    
    def backward(self,grad_out):
        return grad_out*self.output*(1-self.output)
