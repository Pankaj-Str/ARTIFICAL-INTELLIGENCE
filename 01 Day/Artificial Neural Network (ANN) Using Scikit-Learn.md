
# Artificial Neural Network (ANN) Using Scikit-Learn

Scikit-Learn provides an easy way to create ANN using
👉 **`MLPClassifier`** (Multi-Layer Perceptron)

---

## What is MLP?

**MLP (Multi-Layer Perceptron)** is a type of **Artificial Neural Network** with:

* Input layer
* One or more hidden layers
* Output layer

---

## Problem Statement

We will build an ANN model to **classify the Iris dataset** (simple & popular dataset).

---

## Step 1: Import Required Libraries

```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## Step 2: Load Dataset

```python
iris = load_iris()

X = iris.data      # Input features
y = iris.target    # Output labels
```

---

## Step 3: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 4: Feature Scaling (Very Important for ANN)

ANN works best when data is **scaled**.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Step 5: Create ANN Model (MLPClassifier)

```python
ann = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # two hidden layers
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
```

### Explanation:

| Parameter            | Meaning                  |
| -------------------- | ------------------------ |
| `hidden_layer_sizes` | Neurons in hidden layers |
| `activation`         | Activation function      |
| `solver`             | Optimizer                |
| `max_iter`           | Training iterations      |

---

## Step 6: Train the ANN Model

```python
ann.fit(X_train, y_train)
```

---

## Step 7: Make Predictions

```python
y_pred = ann.predict(X_test)
```

---

## Step 8: Model Evaluation

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## Sample Output

```
Accuracy: 0.97
```

---

## ANN Architecture Used

```
Input Layer : 4 neurons (features)
Hidden Layer 1 : 10 neurons
Hidden Layer 2 : 10 neurons
Output Layer : 3 neurons (classes)
```

---

## ANN Modules Mapping (Scikit-Learn)

| ANN Module     | Scikit-Learn          |
| -------------- | --------------------- |
| Input Layer    | X (features)          |
| Hidden Layers  | hidden_layer_sizes    |
| Weights & Bias | Automatically handled |
| Activation     | activation            |
| Learning       | solver                |
| Output         | predict()             |

---

## Advantages of ANN in Scikit-Learn

* Easy to implement
* No manual backpropagation
* Good for beginners
* Built-in optimization

---

## Limitations

* Less flexible than TensorFlow/Keras
* Not ideal for very deep networks

---

## When to Use Scikit-Learn ANN?

* Learning ANN concepts
* Small to medium datasets
* Fast experimentation
* Academic projects

---

## Summary

* ANN in Scikit-Learn uses **MLPClassifier**
* Scaling data is mandatory
* Architecture is controlled via parameters
* Great starting point for Deep Learning

---


