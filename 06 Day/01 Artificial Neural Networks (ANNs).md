# Artificial Neural Network (ANN) 

### What is Artificial Neural Network (ANN)?

An Artificial Neural Network (ANN) is a computer model that works like the human brain. It learns patterns from data, just like how a child learns by seeing examples.

ANN is the basic building block of Deep Learning.

### Simple Structure of ANN

A simple ANN has 3 main parts:

1. **Input Layer**  
   This layer takes the data inside the network.  
   Example: If you give a 28x28 pixel image, the input layer has 784 neurons (28 × 28 = 784).

2. **Hidden Layers**  
   These are the middle layers where the real learning happens.  
   You can have one or many hidden layers. More layers = deeper network.

3. **Output Layer**  
   This layer gives the final answer.  
   Example: For recognizing digits 0 to 9, there are 10 neurons in the output layer.

### How Does One Neuron Work?

Each neuron does a simple calculation:

- It takes inputs
- Multiplies them by **weights** (importance of each input)
- Adds a **bias** (a small extra number)
- Then applies an **Activation Function** (like ReLU) to add non-linearity

Simple Formula:  
`z = (w1×x1 + w2×x2 + ...) + b`  
`Output = Activation(z)`

Popular Activation Functions:
- **ReLU** → Most commonly used in hidden layers
- **Softmax** → Used in output layer for classification

### How Does ANN Learn?

ANN learns in two main steps:

1. **Forward Propagation**  
   Data goes from input layer → hidden layers → output layer and makes a prediction.

2. **Back Propagation**  
   If the prediction is wrong, it calculates the error.  
   Then it adjusts the **weights** and **bias** using an **Optimizer** (like Adam) to reduce the error.

This process repeats many times (called **epochs**) until the network becomes good at its task.

### Practical Example with TensorFlow

We will use the **MNIST dataset** — a very famous beginner example.  
It contains handwritten digits (0 to 9) as 28x28 gray images. The goal is to teach the ANN to recognize which digit is written.

#### Complete Code

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Step 2: Normalize data (make values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 3: Create the Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # Convert 28x28 image to 784 numbers
    keras.layers.Dense(128, activation='relu'),   # Hidden Layer 1
    keras.layers.Dense(64, activation='relu'),    # Hidden Layer 2
    keras.layers.Dense(10, activation='softmax')  # Output Layer (10 digits)
])

# Step 4: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Step 6: Check accuracy on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Step 7: Predict on first 5 test images
predictions = model.predict(x_test[:5])

for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.show()
```

### Simple Explanation of the Code:

- **Flatten**: Changes the 2D image (28x28) into a 1D list of 784 numbers.
- **Dense(128, relu)**: A hidden layer with 128 neurons using ReLU activation.
- **Dense(10, softmax)**: Output layer with 10 neurons (one for each digit 0-9).
- **adam**: A smart optimizer that adjusts weights automatically.
- **epochs=10**: The model will see the full training data 10 times.

### Advantages and Disadvantages

**Advantages:**
- Can learn very complex patterns
- Works well with images, text, and sound
- Automatically finds important features from data

**Disadvantages:**
- Needs a lot of data
- Takes time and powerful computer (GPU is better)
- Hard to understand why it makes a particular decision (Black Box)

---

# Backpropagation

---

### What is Backpropagation?

**Backpropagation** is the most important algorithm in training Artificial Neural Networks (ANNs).  

It answers this question:  
**“How should we change the weights and biases so that the network’s error becomes smaller?”**

It is called **Backpropagation** because it propagates (sends) the error **backwards** through the network — from the output layer to the input layer.

---

### Why Do We Need Backpropagation?

After **Forward Propagation**, the network gives a prediction.  
Most of the time, this prediction is **wrong** in the beginning.  

Example:  
- Actual digit = **7**  
- Network predicted = **3**  

We calculate the **error** (loss).  
Now we need to fix the network so that next time it predicts better.  

Backpropagation tells us **how much to change each weight** in the entire network to reduce this error.

---

### Step-by-Step Explanation of Backpropagation

Let’s understand it with a simple 3-layer network:

- Input Layer → Hidden Layer → Output Layer

#### Step 1: Forward Propagation (Already Done)
- Input goes through the network.
- We get the final prediction.
- We calculate **Loss** (Error).  
  Common loss for classification = **Cross-Entropy Loss**

#### Step 2: Calculate Error at Output Layer
We measure how wrong the prediction is.

#### Step 3: Backward Pass (This is Backpropagation)

Now the error travels **backwards**:

1. **Output Layer**  
   - Find out how much each output neuron contributed to the total error.  
   - This is called **Gradient** (∂Loss / ∂Output).

2. **Hidden Layer(s)**  
   - The error from output layer is distributed to the hidden layer neurons.  
   - Each hidden neuron gets to know: “How much did I contribute to the final mistake?”

3. **Input Layer**  
   - Finally, the error reaches the weights connected to the input.

At every step, we calculate **how sensitive the loss is to small changes in weights and biases**. This is done using **Calculus** (Chain Rule).

#### Step 4: Update Weights and Biases
Using the gradients, we update every weight using this formula:

```python
New_Weight = Old_Weight - (Learning_Rate × Gradient)
```

- **Learning Rate**: A small number (like 0.001) that controls how big a step we take while updating.
- If gradient is positive → decrease the weight.
- If gradient is negative → increase the weight.

This is done for **every single weight and bias** in the network.

---

### Simple Analogy

Imagine you are playing a game with 5 friends in a line:

You → Friend1 → Friend2 → Friend3 → Final Score

- You give input.
- Final score is very bad.
- Backpropagation is like asking from the end:
  - “Friend3, how much did you affect the bad score?”
  - “Friend2, how much did you affect Friend3’s mistake?”
  - “Friend1, how much did you affect Friend2?”
  - And finally, you also adjust your input contribution.

Everyone adjusts their behavior a little bit so that next time the final score becomes better.

This is exactly what backpropagation does — it tells every neuron and every weight:  
**“You were responsible for X% of the error, so change yourself by this much.”**

---

### Mathematical Idea (Simple Version)

For every weight **W**, backpropagation calculates:

**Gradient = ∂Loss / ∂W**

This tells us:  
“If I change this weight a tiny bit, how much will the loss change?”

Then we move the weight in the direction that **reduces the loss**.

The **Chain Rule** of calculus makes this possible — it allows us to break down the total error into small parts and calculate gradients layer by layer efficiently.

---

### Role of Optimizer (Adam, SGD, etc.)

Backpropagation only gives the **gradients** (direction to move).  
The **Optimizer** decides **how much and in what way** to update the weights.

- **SGD** (Stochastic Gradient Descent) → Simple
- **Adam** → Most popular (smart + fast)

That’s why in the code we write:
```python
optimizer='adam'
```

---

### Summary of Backpropagation Process

1. Do **Forward Pass** → Get prediction
2. Calculate **Loss** (how wrong we are)
3. Do **Backward Pass** (Backpropagation):
   - Start from output layer
   - Move backwards to hidden layers
   - Calculate gradient for every weight and bias
4. Update all weights using gradients and learning rate
5. Repeat for many epochs

---

### One Important Thing to Remember

- In the beginning, weights are random → predictions are bad.
- After many epochs of backpropagation, weights become “smart”.
- The network slowly starts recognizing patterns (edges → shapes → digits).

---

