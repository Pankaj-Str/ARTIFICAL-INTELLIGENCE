# Dense neural network example 
### for classifying MNIST handwritten digits, but now using 
- TensorFlow  Keras

Keras provides a simpler, higher-level API — many people find it easier for the first few models.

### Step-by-Step Example with Keras

#### Step 1: Install (if needed) and Import Libraries
In most environments (Colab, Jupyter, etc.) you already have TensorFlow. If not:

```bash
pip install tensorflow
```

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
```

#### Step 2: Load and Prepare the Data
Keras has built-in MNIST loader — super convenient!

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

print(f"Training samples: {x_train.shape[0]}, shape: {x_train.shape}")
print(f"Test samples: {x_test.shape[0]}, shape: {x_test.shape}")
# → Training samples: 60000, shape: (60000, 28, 28)
```

No need for manual DataLoaders — Keras handles batching internally.

#### Step 3: Build the Dense Model
We use the **Sequential** API (easiest for beginners).

```python
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),     # Flatten 28×28 → 784
    layers.Dense(128, activation='relu'),     # Hidden layer 1
    layers.Dense(64,  activation='relu'),     # Hidden layer 2
    layers.Dense(10)                          # Output (logits for 10 classes)
])

# See what we created
model.summary()
```

**Output example:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
 dense (Dense)               (None, 128)               100480    
 dense_1 (Dense)             (None, 64)                8256      
 dense_2 (Dense)             (None, 10)                650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
_________________________________________________________________
```

- No manual `forward` method — Keras handles it.
- Last layer has no activation (we'll use it with `from_logits=True` in the loss).

#### Step 4: Compile the Model
Choose optimizer, loss, and metrics.

```python
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

- `SparseCategoricalCrossentropy`: Use this when labels are integers (0–9), not one-hot.
- `from_logits=True`: Because our last layer has raw logits (no softmax).

#### Step 5: Train the Model

```python
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.1,          # Use 10% of training data for validation
    verbose=1
)
```

Typical output:
```
Epoch 1/5
844/844 [==============================] - 4s 4ms/step - loss: 0.3124 - accuracy: 0.9102 - val_loss: 0.1418 - val_accuracy: 0.9600
Epoch 2/5
844/844 [==============================] - 3s 4ms/step - loss: 0.1326 - accuracy: 0.9603 - val_loss: 0.1054 - val_accuracy: 0.9693
...
Epoch 5/5
844/844 [==============================] - 3s 4ms/step - loss: 0.0612 - accuracy: 0.9815 - val_loss: 0.0821 - val_accuracy: 0.9760
```

Training is usually faster and the code is shorter than the PyTorch version.

#### Step 6: Evaluate on Test Set

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
```

You should get ~96.5–97.5% accuracy after just 5 epochs (often a bit higher than the small PyTorch example because of slight differences in initialization / defaults).

#### Optional: Make Predictions (Inference)

```python
# Predict on first 5 test images
predictions = model(x_test[:5])                # raw logits
probs = tf.nn.softmax(predictions, axis=1)    # probabilities
predicted_classes = np.argmax(probs, axis=1)

print("Predicted:", predicted_classes)
print("True labels:", y_test[:5])
```

### Complete One-File Script (Keras Version)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Load & prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# 2. Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,  activation='relu'),
    layers.Dense(10)
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 4. Train
model.fit(x_train, y_train,
          batch_size=64,
          epochs=5,
          validation_split=0.1)

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
```

### Why Choose Keras/TensorFlow Over PyTorch (for Beginners)?
- Less code for the same result
- Built-in dataset loaders
- Easier high-level API (`Sequential`, `model.fit()` handles training loop)
- Great integration with Google Colab (free GPU/TPU)
- Production-ready (TensorFlow Serving, TFLite, etc.)

### Quick Variations to Try
- Add dropout: `layers.Dropout(0.2)` after each Dense
- Change optimizer: `optimizer='sgd'` or `keras.optimizers.Adam(learning_rate=0.0005)`
- More epochs → better accuracy (try 10–15)
- Use one-hot labels + `CategoricalCrossentropy` (instead of sparse)
