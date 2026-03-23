import numpy as np

class Dense:
    def __init__(self,in_feat,out_feat):
        self.in_features=in_feat
        self.out_feature=out_feat
        
        self.weights=np.random.randn(in_feat,out_feat)*np.sqrt(2. / in_feat)
        self.bias=np.zeroes(1,out_feat)

        def forward(self,x):    
            self.input=x
            return np.dot(self.input,self.weights)+self.bias
        
        def backward(self,grad_out):
            self.grad_weight=np.dot(self.input.T,grad_out)
            self.grad_bias=np.sum(grad_out,axis=0,keepdims=True)

            self.grad_inp=np.dot(grad_out,self.weights.T)
            return self.grad_inp
        

