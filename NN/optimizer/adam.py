import numpy as np
from .base import Optimizer
class Adam(Optimizer):
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.t=0
        self.tracker=None
    
    def set_tracker(self, tracker):
        self.tracker = tracker
    
    def update(self,layer):
        t = getattr(self, "t", 1)
        if t < 1:
            t = 1

        if hasattr(layer, "weights") and hasattr(layer, "grad_weight"):
            if not hasattr(layer, "mw"):
                layer.mw = np.zeros_like(layer.weights)
                layer.vw = np.zeros_like(layer.weights)

            dw = layer.grad_weight
            layer.mw = self.beta1 * layer.mw + (1 - self.beta1) * dw
            layer.vw = self.beta2 * layer.vw + (1 - self.beta2) * (dw ** 2)

            mw_hat = layer.mw / (1 - self.beta1 ** t)
            vw_hat = layer.vw / (1 - self.beta2 ** t)

            update_w = self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            layer.weights -= update_w
            
            if self.tracker:
                weight_norm = np.linalg.norm(layer.weights)
                update_b = np.zeros_like(layer.bias) if hasattr(layer, "bias") else np.zeros(1)
                self.tracker.log_update(update_b, update_w, weight_norm)

        if hasattr(layer, "bias") and hasattr(layer, "grad_bias"):
            if not hasattr(layer, "mb"):
                layer.mb = np.zeros_like(layer.bias)
                layer.vb = np.zeros_like(layer.bias)

            db = layer.grad_bias
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * db
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (db ** 2)

            mb_hat = layer.mb / (1 - self.beta1 ** t)
            vb_hat = layer.vb / (1 - self.beta2 ** t)

            layer.bias -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
