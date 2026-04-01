import numpy as np
class Tracker:
    def __init__(self):
        self.history={
            "loss" : [],
            "grad_norm" : [],
            "weight_norm" : [],
            "update_norm" : [],
            "update_weight_ratio" : []
        }

    def log_epoch_start(self):
        self.history["grad_norm"].append([])
        self.history["weight_norm"].append([])
        self.history["update_norm"].append([])
        self.history["update_weight_ratio"].append([])
    
    def log_loss(self,loss):
        self.history["loss"].append(loss)

    def log_layer(self,layer):
        grad_norm=np.linalg.norm(layer.grad_weight)+np.linalg.norm(layer.grad_bias)
        weight_norm=np.linalg.norm(layer.weights)+np.linalg.norm(layer.bias)

        self.history["grad_norm"][-1].append(grad_norm)
        self.history["weight_norm"][-1].append(weight_norm)

    def log_update(self,update_b,update_w,weight_norm):
        update_norm=np.linalg.norm(update_w)+np.linalg.norm(update_b)

        ratio=update_norm/(weight_norm+1e-8)

        self.history["update_norm"][-1].append(update_norm)
        self.history["update_weight_ratio"][-1].append(ratio)