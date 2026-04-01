import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self,lr=0.01):
        self.lr=lr
        self.tracker=None

    def set_tracker(self, tracker):
        self.tracker = tracker

    def update(self,layer):
        if hasattr(layer,'weights'):
            update_w = self.lr*layer.grad_weight
            layer.weights = layer.weights - update_w
            
            if self.tracker:
                weight_norm = np.linalg.norm(layer.weights)
                self.tracker.log_update(self.lr*layer.grad_bias, update_w, weight_norm)
        
        if hasattr(layer,"bias"):
            layer.bias=layer.bias-self.lr*layer.grad_bias



