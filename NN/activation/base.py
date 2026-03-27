class Activation:
    def forward(self,x):
        raise NotImplementedError
    def backward(self,grad_out):
        raise NotImplementedError
    