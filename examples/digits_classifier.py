import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from NN.model import Sequential
from NN.layers import Dense
from NN.activation import Relu, Softmax
from NN.losses import Cross_entropy
from NN.optimizer import SGD
from NN.diagnostic import Tracker
from NN.diagnostic import Visualizer
from NN.diagnostic import Analyze


# 1. Load dataset
data = load_digits()
X = data.data          # (1797, 64)
y = data.target        # (1797,)

# 2. Normalize
X = X / 16.0

# 3. One-hot encode
num_classes = 10
y_onehot = np.zeros((y.size, num_classes))
y_onehot[np.arange(y.size), y] = 1

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# 5. Model (matching your style)
model = Sequential([
    Dense(64, 32),
    Relu(),
    Dense(32, 16),
    Relu(),
    Dense(16, 10),
    Softmax()
])

# 6. Compile (your naming)
model.compile(
    loss_fn=Cross_entropy(),
    optimizer=SGD(lr=0.1)
)

# 7. Train (your function name)
tracker = Tracker()
model.train(X_train, y_train, epochs=200, batch_size=32, tracker=tracker)

# 8. Predict
y_pred = model.predict(X_test)

# Convert to labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# 9. Accuracy
accuracy = np.mean(y_pred_labels == y_true_labels)
print("Test Accuracy:", accuracy)

# 10. Inspect predictions
print("\nSample Predictions (probabilities):\n", y_pred[:5])
print("\nPredicted Labels:", y_pred_labels[:10])
print("True Labels     :", y_true_labels[:10])

# 11. Analyze
analyzer = Analyze(tracker)
try:
    analyzer.analyze_epoch()
except:
    print("Note: Full diagnostic analysis requires layer-level gradient tracking")

# 12. Visualize
visualizer = Visualizer(tracker)
visualizer.plot_loss()