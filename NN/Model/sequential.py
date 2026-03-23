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

    def update(self,optimizer):
        for layer in self.layers:
            if hasattr(layer,"weights"):
                optimizer.update(layer)
    
    def train(self,x,y,loss_fn,optimizer,epochs=5):
        for epoch in range(epochs):
            # Forward pass
            output=self.forward(x)

            #Loss Computation
            loss=loss_fn.forward(output,y)

            #Backward Pass
            grad=loss_fn.backward()
            self.backward(grad)

            #Update Backwards
            self.update(optimizer)

            print(f"Epoch {epoch+1}, Loss: {loss}")