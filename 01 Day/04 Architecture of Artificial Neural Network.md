## Artificial Neural Network (ANN) Architecture

An **Artificial Neural Network (ANN)** is a computing system inspired by the **human brain**.
Its **architecture** defines **how neurons are arranged and connected** to process data.

---

## What is ANN Architecture?

ANN architecture is the **structure of a neural network**, consisting of:

* Layers
* Neurons
* Connections (weights)
* Activation functions

👉 It explains **how data flows from input to output**.

---

## Main Components of ANN Architecture

### 1. Input Layer

* First layer of the network
* Receives raw data (features)
* No computation is done here

**Example:**
If a dataset has 3 features:

```
x1, x2, x3
```

Then the input layer has **3 neurons**.

---

### 2. Hidden Layer(s)

* Located between input and output layers
* Perform **actual learning and computation**
* Can be **one or multiple hidden layers**

Each hidden neuron:

```
Input → Weight → Sum → Activation → Output
```

🔹 More hidden layers = **Deep Neural Network**

---

### 3. Output Layer

* Final layer
* Produces the prediction or result

**Examples:**

* Binary classification → 1 neuron (0 or 1)
* Multi-class classification → multiple neurons
* Regression → 1 neuron (continuous value)

---

## Simple ANN Architecture Diagram (Text Format)

```
Input Layer     Hidden Layer      Output Layer
 x1  ───▶  ○
 x2  ───▶  ○  ───▶  ○  ───▶  Output
 x3  ───▶  ○
```

---

## Working of ANN Architecture (Step-by-Step)

### Step 1: Forward Propagation

1. Inputs are passed to hidden layer
2. Weights are applied
3. Bias is added
4. Activation function is used
5. Output is generated

Formula:

```
z = (x1w1 + x2w2 + ... + xnwn) + bias
output = activation(z)
```

---

### Step 2: Loss Calculation

* Difference between actual output and predicted output
* Common loss functions:

  * Mean Squared Error (MSE)
  * Cross-Entropy Loss

---

### Step 3: Backpropagation

* Errors are sent backward
* Weights are adjusted
* Learning happens here

---

### Step 4: Weight Update

* Uses **Gradient Descent**
* Improves accuracy step by step

---

## Types of ANN Architectures

### 1. Single Layer Neural Network

* Only input and output layer
* No hidden layer
* Example: Perceptron

---

### 2. Multi-Layer Perceptron (MLP)

* One or more hidden layers
* Most commonly used ANN

---

### 3. Feedforward Neural Network

* Data flows in one direction
* No loops

---

### 4. Recurrent Neural Network (RNN)

* Has feedback loops
* Used for sequential data (text, time series)

---

### 5. Convolutional Neural Network (CNN)

* Specialized for image data
* Uses convolution layers

---

## Activation Functions Used in ANN

* **ReLU** – most common in hidden layers
* **Sigmoid** – binary classification
* **Softmax** – multi-class classification
* **Tanh**

---

## Example Use Case of ANN

| Task                   | ANN Usage            |
| ---------------------- | -------------------- |
| Email Spam Detection   | Classification       |
| House Price Prediction | Regression           |
| Face Recognition       | Image classification |
| Stock Prediction       | Time series          |

---

## Advantages of ANN

* Learns complex patterns
* Handles large data
* Works well with non-linear data

---

## Limitations of ANN

* Needs large data
* High computational cost
* Hard to interpret (black box)

---

## Summary (Quick View)

| Component           | Role                 |
| ------------------- | -------------------- |
| Input Layer         | Takes data           |
| Hidden Layer        | Learns patterns      |
| Output Layer        | Gives result         |
| Weights             | Importance of inputs |
| Bias                | Shifts activation    |
| Activation Function | Adds non-linearity   |

---

### One-Line Definition:

> **ANN architecture defines how neurons are organized in layers to learn patterns and make predictions from data.**


