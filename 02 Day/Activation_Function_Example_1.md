# Activation Functions in Neural Networks

**(With Real Dataset & Step-by-Step Example)**

## 1. What is an Activation Function?

An **activation function** decides whether a neuron should be activated or not.
It adds **non-linearity**, allowing neural networks to learn complex patterns.

Without activation functions, a neural network would behave like **linear regression**, no matter how many layers it has.

---

## 2. Why Activation Functions are Important?

Activation functions help to:

* Learn **non-linear relationships**
* Control **output range**
* Speed up **training**
* Solve **classification & regression problems**

---

## 3. Common Activation Functions

| Activation | Formula                  | Used For                   |
| ---------- | ------------------------ | -------------------------- |
| Step       | 0 or 1                   | Perceptron                 |
| Sigmoid    | 1 / (1 + e⁻ˣ)            | Binary Classification      |
| Tanh       | (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)  | Hidden Layers              |
| ReLU       | max(0, x)                | Most Popular               |
| Softmax    | Probability Distribution | Multi-class Classification |

---

## 4. Real Dataset Used (Breast Cancer Dataset)

We will use **Breast Cancer Wisconsin Dataset** from `sklearn`.

**Problem Type:** Binary Classification
**Target:**

* `0` → Malignant
* `1` → Benign

---

## 5. Load Dataset

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load data
data = load_breast_cancer()

X = data.data
y = data.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

df.head()
```

---

## 6. Train-Test Split & Scaling

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 7. Activation Function Example – **Sigmoid**

### Why Sigmoid?

* Output range: **0 to 1**
* Best for **binary classification**

---

### Neural Network with Sigmoid

```python
from sklearn.neural_network import MLPClassifier

model_sigmoid = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='logistic',   # Sigmoid
    max_iter=1000,
    random_state=42
)

model_sigmoid.fit(X_train, y_train)
```

---

### Model Evaluation

```python
from sklearn.metrics import accuracy_score

y_pred = model_sigmoid.predict(X_test)
accuracy_score(y_test, y_pred)
```

---

## 8. Activation Function Example – **ReLU**

### Why ReLU?

* Faster training
* Solves vanishing gradient problem
* Most commonly used

---

### Neural Network with ReLU

```python
model_relu = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    max_iter=1000,
    random_state=42
)

model_relu.fit(X_train, y_train)

y_pred_relu = model_relu.predict(X_test)
accuracy_score(y_test, y_pred_relu)
```

---

## 9. Activation Function Example – **Tanh**

```python
model_tanh = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='tanh',
    max_iter=1000,
    random_state=42
)

model_tanh.fit(X_train, y_train)

y_pred_tanh = model_tanh.predict(X_test)
accuracy_score(y_test, y_pred_tanh)
```

---

## 10. Comparison of Activation Functions

| Activation | Accuracy | Best Use      |
| ---------- | -------- | ------------- |
| Sigmoid    | Medium   | Binary output |
| Tanh       | Better   | Hidden layers |
| ReLU       | Best     | Deep networks |

👉 **ReLU usually performs best**

---

## 11. Softmax Activation (Multi-Class Example)

Softmax is used in **output layer** for **multi-class classification**.

```python
MLPClassifier(
    hidden_layer_sizes=(20,20),
    activation='relu',
    solver='adam'
)
```

*(Softmax is applied internally for multi-class problems)*

---

## 12. Key Takeaways

* Activation functions introduce **non-linearity**
* **Sigmoid** → Binary classification
* **ReLU** → Most widely used
* **Tanh** → Zero-centered
* **Softmax** → Multi-class output

---

## 13. Interview Questions

1. Why is activation function required?
2. Difference between Sigmoid and ReLU?
3. Why ReLU is better than Sigmoid?
4. What happens if no activation function is used?

---




