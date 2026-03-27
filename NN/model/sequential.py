import numpy as np
class Sequential:
    def __init__(self,layers):
        self.layers=layers
    
    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x

    def backward(self,grad):
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
        return grad

    def update(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def compile(self,loss_fn,optimizer):
        self.loss_fn=loss_fn
        self.optimizer=optimizer

    
    def train(self,x,y,epochs=5,batch_size=32):
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError("Call the compile() function before training")
        n=x.shape[0]
        num_batch=int(np.ceil(n/batch_size))
        for epoch in range(epochs):
            
            indices=np.random.permutation(n)
            x_shuffled=x[indices]
            y_shuffled=y[indices]
            epoch_loss=0

            for i in range(0,n,batch_size):
                x_batch=x_shuffled[i:i+batch_size]
                y_batch=y_shuffled[i:i+batch_size]

                #output
                output=self.forward(x_batch)
                
                #Loss Computation
                loss=self.loss_fn.forward(output,y_batch)
                epoch_loss+=loss


                #Backward prop
                grad=self.loss_fn.backward()
                self.backward(grad)


                #Update weights and bias
                self.update()
            
            #Average loss
            avg_epoch=epoch_loss/num_batch
            print(f"Epoch {epoch+1}, Loss: {avg_epoch}")

    def predict(self, x, mode="raw", threshold=0.5):
        y_pred = self.forward(x)

        if mode == "raw":
            return y_pred

        elif mode == "binary":
            if getattr(self.loss_fn, "expects_logits", False):
                probs = 1 / (1 + np.exp(-y_pred))  # sigmoid
            else:
                probs = y_pred

            return (probs > threshold).astype(int)

        elif mode == "multiclass":
            if getattr(self.loss_fn, "expects_logits", False):
                shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
                exp = np.exp(shifted)
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                return np.argmax(probs, axis=1)
            else:
                return np.argmax(y_pred, axis=1)

        else:
            raise ValueError("Invalid mode. Choose from ['raw', 'binary', 'multiclass']")
        
    def evaluate(self, x, y, metric=None):
        y_pred = self.forward(x)

        loss = self.loss_fn.forward(y_pred, y)

        if metric is not None:
            score = metric(y_pred, y)
            return loss, score

        return loss