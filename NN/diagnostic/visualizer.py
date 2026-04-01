import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.tracker.history["loss"])
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
    
    def plot_gradients(self):
        grads = np.array(self.tracker.history["grad_norm"])
        if len(grads) == 0:
            print("No gradient data available")
            return

        plt.figure()
        for i in range(len(grads[0]) if len(grads) > 0 else 0):
            plt.plot([g[i] if i < len(g) else None for g in grads], label=f"Layer {i}")

        plt.title("Gradient Norms per Layer")
        plt.xlabel("Epoch")
        plt.ylabel("Grad Norm")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_update_ratio(self):
        ratios = np.array(self.tracker.history["update_weight_ratio"])
        if len(ratios) == 0:
            print("No update ratio data available")
            return

        plt.figure()
        for i in range(len(ratios[0]) if len(ratios) > 0 else 0):
            plt.plot([r[i] if i < len(r) else None for r in ratios], label=f"Layer {i}")

        plt.title("Update / Weight Ratio")
        plt.xlabel("Epoch")
        plt.ylabel("Ratio")
        plt.legend()
        plt.grid(True)
        plt.show()