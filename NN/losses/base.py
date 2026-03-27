class Loss:
    def forward(self,y_pred,y):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError