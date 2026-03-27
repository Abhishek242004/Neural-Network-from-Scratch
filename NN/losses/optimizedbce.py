import numpy as np
from .base import Loss

class BCEWithLogits(Loss):
    expects_logits=True
    def forward(self, z, y):
        self.z = z
        self.y = y
        
        # numerically stable version
        loss = np.maximum(z, 0) - z*y + np.log(1 + np.exp(-np.abs(z)))
        return np.mean(loss)

    def backward(self):
        batch_size = self.z.shape[0]
        # sigmoid(z)
        sigmoid = 1 / (1 + np.exp(-self.z))
        
        return (sigmoid - self.y) / batch_size
        