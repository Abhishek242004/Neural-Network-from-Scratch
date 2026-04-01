import numpy as np

class Analyze:
    def __init__(self,tracker):
        self.tracker=tracker
    
    def analyze_epoch(self):
        self.check_gradients()
        self.check_updates()
        self.check_learning()

    def check_gradients(self):
        history=self.tracker.history["grad_norm"]

        if len(history) < 5 :
            return

        current=history[-1]
        prev=np.mean(history[-5:],axis=0)    


        for i,(g,avg) in enumerate(zip(current,prev)):
            if g > avg * 10 : 
                print (f"Layer {i} : Gradient Spike")
            
            if g < 0.01 * avg : 
                print(f"Layer {i} : Gradient Vanished")


    def check_updates(self):
        updates = self.tracker.history["update_norm"][-1]
        ratios = self.tracker.history["update_weight_ratio"][-1]

        for i, (u, r) in enumerate(zip(updates, ratios)):

            if u < 1e-8:
                print(f"Layer {i}: No effective updates")

            if r > 1e-1:
                print(f"Layer {i}: Updates too aggressive")

            elif r < 1e-6:
                print(f"Layer {i}: Updates too small")

    def check_learning(self):
        loss = self.tracker.history["loss"]

        if len(loss) < 5:
            return

        recent = loss[-5:]

        if abs(recent[-1] - recent[0]) < 1e-4:
            print("Model not learning (loss stagnant)")

        elif recent[-1] > recent[0]:
            print("Loss increasing → unstable training")
