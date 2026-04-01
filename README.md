# Neural Network from Scratch

A modular, educational neural network framework built with NumPy. This framework provides a PyTorch-like API for building, training, and diagnosing deep learning models from scratch.

## Features

- **Pure NumPy Implementation**: No external deep learning dependencies, everything built from first principles
- **Modular Architecture**: Clean separation of concerns with dedicated modules for layers, activations, losses, and optimizers
- **Sequential API**: PyTorch-style `Sequential` model building for intuitive model construction
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: Binary Cross-Entropy (standard & optimized), Cross-Entropy (standard & optimized), Mean Squared Error
- **Optimizers**: SGD and Adam with configurable learning rates
- **Diagnostic Tools**: Built-in tracking, analysis, and visualization of training dynamics
- **Educational Focus**: Code designed to be readable and understandable for learning purposes

## Installation

### Requirements
- Python 3.7+
- NumPy
- Scikit-learn (for example datasets)
- Matplotlib (for visualization)
- tkinter (for interactive plots)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd FS

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn matplotlib

# Install tkinter (system-specific)
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk

# Windows: Usually included with Python
```

## Quick Start

### Basic Model Training

```python
import numpy as np
from NN.model import Sequential
from NN.layers import Dense
from NN.activation import Relu, Sigmoid
from NN.losses import Binary_cross_entropy
from NN.optimizer import SGD

# Create dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Build model
model = Sequential([
    Dense(2, 4),      # Input: 2, Hidden: 4
    Relu(),
    Dense(4, 1),      # Hidden: 4, Output: 1
    Sigmoid()
])

# Compile
model.compile(
    loss_fn=Binary_cross_entropy(),
    optimizer=SGD(lr=0.1)
)

# Train
model.train(X, y, epochs=1000, batch_size=4)

# Predict
predictions = model.predict(X, mode="binary")
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
```

### With Diagnostics

```python
from NN.diagnostic import Tracker, Visualizer, Analyze

# Create tracker
tracker = Tracker()

# Train with tracking
model.train(X, y, epochs=1000, batch_size=4, tracker=tracker)

# Analyze training
analyzer = Analyze(tracker)
analyzer.analyze_epoch()

# Visualize results
visualizer = Visualizer(tracker)
visualizer.plot_loss()
visualizer.plot_gradients()
visualizer.plot_update_ratio()
```

## Architecture Overview

```
NN/
├── activation/          # Activation functions
│   ├── base.py
│   ├── relu.py         # Rectified Linear Unit
│   ├── sigmoid.py      # Sigmoid activation
│   ├── softmax.py      # Softmax for multi-class
│   └── tanh.py         # Hyperbolic tangent
├── layers/             # Neural network layers
│   └── dense.py        # Fully connected layer
├── losses/             # Loss functions
│   ├── base.py
│   ├── binary_cross_entropy.py
│   ├── cross_entropy.py
│   ├── mse.py          # Mean squared error
│   ├── optimizedbce.py # Optimized BCE
│   └── optimizedCE.py  # Optimized cross-entropy
├── optimizer/          # Optimization algorithms
│   ├── base.py
│   ├── sgd.py          # Stochastic gradient descent
│   └── adam.py         # Adaptive moment estimation
├── metrics/            # Evaluation metrics
│   └── classification.py
├── model/              # Model classes
│   └── sequential.py   # Sequential model builder
└── diagnostic/         # Training diagnostics
    ├── tracker.py      # Track metrics during training
    ├── analyzer.py     # Analyze training issues
    └── visualizer.py   # Visualize training dynamics
```

## Module Documentation

### Core Modules

#### Dense Layer
```python
from NN.layers import Dense

layer = Dense(in_features=784, out_features=128)
output = layer.forward(input_data)
grad = layer.backward(output_gradient)
```

**Features:**
- Xavier initialization for stable training
- Efficient matrix operations using NumPy
- Gradient computation for backpropagation

#### Activation Functions

```python
from NN.activation import Relu, Sigmoid, Tanh, Softmax

relu = Relu()           # Hide negative values
sigmoid = Sigmoid()     # Probability output (0-1)
tanh = Tanh()          # Symmetric activation (-1 to 1)
softmax = Softmax()    # Multi-class probabilities
```

#### Loss Functions

```python
from NN.losses import Binary_cross_entropy, Cross_entropy, MSE

bce = Binary_cross_entropy()    # Binary classification
ce = Cross_entropy()            # Multi-class classification
mse = MSE()                     # Regression
```

#### Optimizers

```python
from NN.optimizer import SGD, Adam

sgd = SGD(lr=0.01)                           # Stochastic gradient descent
adam = Adam(lr=0.001, beta1=0.9, beta2=0.999)  # Adaptive optimizer
```

### Diagnostic Tools

#### Tracker
Automatically logs training metrics:
- Loss per epoch
- Gradient norms per layer
- Weight norms per layer
- Update norms and update/weight ratios

```python
from NN.diagnostic import Tracker

tracker = Tracker()
model.train(X, y, epochs=100, tracker=tracker)

# Access history
loss_history = tracker.history["loss"]
grad_history = tracker.history["grad_norm"]
```

#### Analyzer
Detects common training issues:

```python
from NN.diagnostic import Analyze

analyzer = Analyze(tracker)
analyzer.analyze_epoch()  # Prints warnings for:
# - Gradient spikes or vanishing gradients
# - Weight updates too aggressive/small
# - Loss stagnation or divergence
```

#### Visualizer
Interactive plots of training dynamics:

```python
from NN.diagnostic import Visualizer

visualizer = Visualizer(tracker)
visualizer.plot_loss()           # Loss curve over epochs
visualizer.plot_gradients()      # Gradient norms per layer
visualizer.plot_update_ratio()   # Update magnitude ratios
```

## Examples

### 1. XOR Problem (Binary Classification)
```bash
python -m examples.xor_binary
```
Solves the classic XOR problem with a 2-4-1 network.
- **Accuracy**: ~100% after 2000 epochs
- **Key Concepts**: Hidden layer necessity, non-linearity

### 2. Two Moons (Binary Classification)
```bash
python -m examples.two_moons
```
Classifies the Two Moons dataset using scikit-learn.
- **Accuracy**: ~97%
- **Dataset**: 200 samples of 2-class problem

### 3. Digits Recognition (Multi-class Classification)
```bash
python -m examples.digits_classifier
```
Recognizes handwritten digits (0-9) from sklearn's digits dataset.
- **Accuracy**: High (varies with initialization)
- **Architecture**: 64 → 32 → 16 → 10
- **Approach**: Multi-class classification with softmax

## Usage Examples

### Building Custom Models

```python
from NN.model import Sequential
from NN.layers import Dense
from NN.activation import Relu, Softmax
from NN.losses import Cross_entropy
from NN.optimizer import Adam

# Multi-layer network
model = Sequential([
    Dense(20, 64),
    Relu(),
    Dense(64, 32),
    Relu(),
    Dense(32, 16),
    Relu(),
    Dense(16, 10),
    Softmax()  # 10 classes
])

model.compile(
    loss_fn=Cross_entropy(),
    optimizer=Adam(lr=0.001)
)

model.train(X_train, y_train, epochs=50, batch_size=32, tracker=tracker)
```

### Making Predictions

```python
# Raw output
raw_preds = model.predict(X_test, mode="raw")

# Binary classification
binary_preds = model.predict(X_test, mode="binary", threshold=0.5)

# Multi-class
class_labels = model.predict(X_test, mode="multiclass")
```

### Evaluating Models

```python
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# With metrics
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## Training Tips

1. **Normalize Data**: Scale inputs to [-1, 1] or [0, 1]
   ```python
   X = X / X.max()  # Simple normalization
   ```

2. **Use Appropriate Learning Rates**: Start with 0.01-0.1 for SGD, 0.001 for Adam
   ```python
   optimizer = SGD(lr=0.01)
   ```

3. **Monitor with Diagnostics**: Track gradients and loss
   ```python
   analyzer.analyze_epoch()  # Detects issues early
   ```

4. **Batch Size**: Typically 16-32, use 4 for XOR
   ```python
   model.train(X, y, batch_size=32)
   ```

5. **Epochs**: More epochs for complex tasks (100-2000)
   ```python
   model.train(X, y, epochs=1000)
   ```

## Diagnostic Features

### Gradient Tracking
Monitors gradient health to detect:
- **Gradient Spikes**: Loss jumps suddenly
- **Vanishing Gradients**: Gradients approach zero
- **Solutions**: Reduce learning rate, use different activation

### Update Analysis
Checks weight update magnitudes:
- **Too Aggressive**: Large updates cause instability
- **Too Small**: Model learns too slowly
- **Ratio**: Should be 1e-3 to 1e-4

### Loss Analysis
Detects training problems:
- **Stagnation**: Loss stops improving
- **Divergence**: Loss keeps increasing
- **Solutions**: Different learning rate, add regularization

## Mathematical Background

### Forward Pass
```
z = x @ W + b         # Linear transformation
a = activation(z)     # Non-linearity
```

### Backward Pass
```
dL/dW = x.T @ dL/dz   # Weight gradient
dL/db = sum(dL/dz)    # Bias gradient
dL/dx = dL/dz @ W.T   # Input gradient for previous layer
```

### Optimization
**SGD**: `W -= lr * gradient`

**Adam**: Adaptive learning rate with momentum
```
m = β₁*m + (1-β₁)*g           # First moment
v = β₂*v + (1-β₂)*g²          # Second moment
W -= (lr * m) / (√v + ε)      # Update
```

## Performance Characteristics

- **Forward Pass**: O(n*m) where n=batch size, m=layer size
- **Backward Pass**: Same complexity as forward
- **Memory**: O(layer_sizes) for weights + activations cache

## Limitations & Future Work

### Current Limitations
- No GPU acceleration (NumPy only)
- No convolutional layers yet
- No recurrent layers (RNN/LSTM)
- Limited regularization (no dropout/batch norm)

### Future Enhancements
- Conv2D and MaxPooling layers
- Recurrent networks (LSTM, GRU)
- Batch normalization
- Dropout regularization
- Data augmentation utilities
- Mixed precision training

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Areas for contribution:
- New layer types
- Additional activation functions
- Regularization techniques
- Performance optimizations
- Documentation improvements

## License

MIT License - See LICENSE file for details

## References

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_LFPM5VHV)
- [Understanding Backpropagation](http://neuralnetworksanddeeplearning.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Goodfellow, Bengio, Courville - Deep Learning (MIT Press)

## Citation

To cite this framework in academic or professional work, use the following format:

```bibtex
@software{neural_network_from_scratch_2026,
  title={Neural Network Framework: A NumPy-Based Implementation},
  author={Abhishek},
  year={2026},
  url={https://github.com/yourusername/FS}
}
```

## Acknowledgments

Built as an educational tool to understand deep learning fundamentals. Inspired by PyTorch's clean API and TensorFlow's documentation style.

---

**Start learning neural networks from scratch!** 
