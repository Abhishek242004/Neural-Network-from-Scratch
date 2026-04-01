import numpy as np
from sklearn.datasets import make_moons
from NN.model import Sequential
from NN.layers import Dense
from NN.activation import Relu, Sigmoid
from NN.losses import Binary_cross_entropy   # or BCEWithLogits
from NN.optimizer import SGD
from NN.diagnostic import Tracker
from NN.diagnostic import Visualizer
from NN.diagnostic import Analyze

# 1. Create dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

# 2. Define model
model = Sequential([
    Dense(2, 8),
    Relu(),
    Dense(8, 4),
    Relu(),
    Dense(4, 1),
    Sigmoid()   # REMOVE this if using BCEWithLogits
])

# 3. Compile
model.compile(
    loss_fn=Binary_cross_entropy(),   # or BCEWithLogits()
    optimizer=SGD(lr=0.1)
)

# 4. Train
tracker = Tracker()
model.train(X, y, epochs=2000, batch_size=16, tracker=tracker)

# 5. Predict
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int)

# 6. Accuracy
accuracy = np.mean(y_pred_labels == y)
print("Accuracy:", accuracy)

# 7. Analyze
analyzer = Analyze(tracker)
try:
    analyzer.analyze_epoch()
except:
    print("Note: Full diagnostic analysis requires layer-level gradient tracking")

# 8. Visualize
visualizer = Visualizer(tracker)
visualizer.plot_loss()