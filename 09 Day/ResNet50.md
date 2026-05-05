## 1. Introduction to ResNet

### What is a Neural Network?

A **neural network** is a computational model inspired by how the human brain processes information. It is made up of layers of interconnected units called neurons.

* **Input layer**: receives raw data (for example, an image).
* **Hidden layers**: process the data through weights and activation functions.
* **Output layer**: produces the final prediction (for example, “cat” or “dog”).

Each connection has a weight, and during training the network learns the best weights to minimize error.

In simple terms:
A neural network learns patterns from data by adjusting numbers (weights) so it can make accurate predictions.

---

### Problem with Deep Neural Networks

As neural networks became deeper (more layers), researchers expected better performance. In theory, deeper networks can learn more complex patterns.

However, in practice, two major problems appeared:

1. **Vanishing Gradient Problem**
2. **Degradation Problem**

These issues made very deep networks difficult to train.

---

### Vanishing Gradient Problem

To understand this, we need to look at how neural networks learn.

* Training uses **backpropagation**, where errors are passed backward from output to input.
* Gradients (small numerical values) tell the model how much to update weights.

In deep networks:

* As gradients move backward through many layers, they become very small.
* Eventually, they become almost zero.

This causes:

* Early layers stop learning.
* Training becomes extremely slow or completely stuck.

Example idea:
If you multiply many small numbers:
0.9 × 0.9 × 0.9 × ... → becomes very close to 0

That is exactly what happens with gradients in deep networks.

---

### Degradation Problem

This is different from vanishing gradients.

Observation:

* A deeper network should perform better than a shallow one.
* But in reality, after a certain depth, accuracy starts getting worse.

This is called the **degradation problem**.

Important point:

* This is not due to overfitting.
* Even training accuracy becomes worse, not just validation accuracy.

Why it happens:

* It becomes harder for deeper networks to learn the correct mapping.
* Ideally, deeper layers should at least copy the previous layers (identity mapping), but they fail to do so.

---

### Why ResNet Was Introduced

To solve these problems, researchers from Microsoft Research introduced **ResNet (Residual Network)**.

Main goal:

* Make very deep networks easier to train.
* Solve vanishing gradient and degradation problems.

Key idea:
Instead of forcing layers to learn a direct mapping:

H(x)

ResNet lets layers learn a **residual function**:

F(x) = H(x) − x

So the final output becomes:

H(x) = F(x) + x

This is implemented using **skip connections (shortcut connections)**.

---

### Overview of ResNet

ResNet is a deep neural network architecture that introduces **skip connections**.

What makes ResNet different:

* It allows information to bypass some layers.
* It makes gradient flow easier during backpropagation.
* It enables training of very deep networks (50, 101, 152 layers).

Basic idea of a residual block:

* Input goes through some layers.
* The original input is added back to the output.

So instead of learning everything from scratch, the network learns only the **difference (residual)**.

Benefits:

* Faster training
* Better accuracy
* Stable performance even with very deep networks

---

### Simple Intuition

Think of it like this:

Instead of learning a completely new function:
“Learn full transformation”

ResNet says:
“Just learn what is different from the input”

This makes learning much easier and more efficient.


