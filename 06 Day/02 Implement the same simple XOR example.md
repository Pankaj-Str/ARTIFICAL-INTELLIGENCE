# Implement the same simple XOR example with an Artificial Neural Network

### Option 1: Using scikit-learn (very simple, no manual training loop)

scikit-learn has a built-in multi-layer perceptron (`MLPClassifier`) — perfect when you want something quick without writing forward/backward passes yourself.

```python
# Option 1: scikit-learn MLP (very beginner-friendly)

from sklearn.neural_network import MLPClassifier
import numpy as np

# Data (same as before)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])   # 1D array — scikit-learn expects this for classification

# Create the model
# hidden_layer_sizes=(2,) → one hidden layer with 2 neurons
# max_iter=10000 → like epochs
# random_state=42 → for reproducible results
model = MLPClassifier(hidden_layer_sizes=(2,),
                      activation='logistic',    # sigmoid
                      solver='adam',
                      learning_rate_init=0.1,
                      max_iter=10000,
                      random_state=42)

# Train (super simple — one line!)
model.fit(X, y)

# Check training progress (optional)
print("Final loss:", model.loss_)

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("\nResults:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {predictions[i]}, Probability: {probabilities[i][1]:.4f}, Actual: {y[i]}")

# Check if it learned XOR correctly
print("\nAccuracy:", model.score(X, y))
```

**Typical output** (after training):
```
Final loss: 0.0032   # usually gets very low

Results:
Input: [0 0], Predicted: 0, Probability: 0.0031, Actual: 0
Input: [0 1], Predicted: 1, Probability: 0.9968, Actual: 1
Input: [1 0], Predicted: 1, Probability: 0.9967, Actual: 1
Input: [1 1], Predicted: 0, Probability: 0.0034, Actual: 0

Accuracy: 1.0
```

→ Easiest version — great for first experiments.

### Option 2: Using TensorFlow / Keras (industry standard, clean high-level API)

Keras (now inside TensorFlow) is probably the most popular way to build neural networks quickly.

```python
# Option 2: TensorFlow / Keras (very clean and modern style)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Data (same)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0], [1], [1], [0]], dtype=np.float32)   # 2D for Keras

# Build model
model = Sequential([
    Dense(2, activation='sigmoid', input_shape=(2,)),   # hidden layer: 2 neurons
    Dense(1, activation='sigmoid')                      # output layer
])

# Compile (like choosing loss + optimizer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X, y,
                    epochs=5000,
                    verbose=0)   # set verbose=1 if you want to see progress

# Show final loss & accuracy
print("Final loss:", history.history['loss'][-1])
print("Final accuracy:", history.history['accuracy'][-1])

# Predict
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(int)

print("\nResults:")
for i in range(len(X)):
    print(f"Input: {X[i].tolist()}, Predicted prob: {predictions[i][0]:.4f}, Class: {predicted_classes[i][0]}, Actual: {y[i][0]}")
```

**Typical output**:
```
Final loss: 0.0041
Final accuracy: 1.0000

Results:
Input: [0.0, 0.0], Predicted prob: 0.0038, Class: 0, Actual: 0.0
Input: [0.0, 1.0], Predicted prob: 0.9962, Class: 1, Actual: 1.0
Input: [1.0, 0.0], Predicted prob: 0.9961, Class: 1, Actual: 1.0
Input: [1.0, 1.0], Predicted prob: 0.0039, Class: 0, Actual: 0.0
```

### Quick Comparison Table – Which to Choose?

| Library          | Code Length | Training Control | Best For                          | Install Command             |
|------------------|-------------|------------------|------------------------------------|-----------------------------|
| PyTorch          | Medium      | Very high        | Research, custom models            | `pip install torch`         |
| scikit-learn     | Very short  | Low              | Quick experiments, small problems  | `pip install scikit-learn`  |
| TensorFlow/Keras | Short       | Medium-High      | Industry, production, fast prototyping | `pip install tensorflow` |

All three can solve XOR perfectly with this tiny network.

