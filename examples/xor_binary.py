import numpy as np

# Your repo imports
from NN.model import Sequential
from NN.layers import Dense
from NN.activation import Relu   # assuming you have this
from NN.losses import BCEWithLogits
from NN.optimizer import SGD
from NN.diagnostic import Tracker
from NN.diagnostic import Visualizer
from NN.diagnostic import Analyze


# -----------------------
# XOR Dataset
# -----------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# -----------------------
# Build Model
# -----------------------
model = Sequential([
    Dense(2, 4),   # hidden layer
    Relu(),
    Dense(4, 1)    # output logits
])

model.compile(
    loss_fn=BCEWithLogits(),
    optimizer=SGD(lr=0.1)
)


# -----------------------
# Train
# -----------------------
tracker = Tracker()
model.train(X, y, epochs=2000, batch_size=4, tracker=tracker)


# -----------------------
# Predict
# -----------------------
preds = model.predict(X, mode="binary")

print("Predictions:")
print(preds)

print("Ground Truth:")
print(y)

accuracy = np.mean(preds == y)
print("Accuracy:", accuracy)

# -----------------------
# Analyze
# -----------------------
analyzer = Analyze(tracker)
try:
    analyzer.analyze_epoch()
except:
    print("Note: Full diagnostic analysis requires layer-level gradient tracking")

# -----------------------
# Visualize
# -----------------------
visualizer = Visualizer(tracker)
visualizer.plot_loss()