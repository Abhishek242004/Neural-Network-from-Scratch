import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self,lr=0.01):
        self.lr=lr

    def update(self,layer):
        if hasattr(layer,'weights'):
            layer.weights=layer.weights-self.lr*layer.grad_weight
        if hasattr(self,"bias"):
            layer.bias=layer.bias-self.lr*layer.grad_bias   


