import numpy as np
from .base import Loss

class OptimizedCE(Loss):
    expects_logits=True
    def forward(self,logits,y):
        if np.all((logits >= 0) & (logits <= 1)):
            print("Warning: Inputs look like probabilities. Did you apply softmax before OptimizedCE?")        
        self.y=y
        shifted=logits-np.max(logits,axis=1,keepdims=True)
        log_prob=shifted-np.log(np.sum(np.exp(shifted),axis=1,keepdims=True))
        self.prob=np.exp(log_prob)

        loss=-np.sum(y*log_prob)/logits.shape[0]
        return loss
    
    def backward(self):
        return (self.prob-self.y)/self.y.shape[0]