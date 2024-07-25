# Cost Functions

## Understanding Cost Functions in Machine Learning: A Step-by-Step Guide

Cost functions, also known as loss functions, play a critical role in the training of machine learning models. They measure how well or poorly a model's predictions match the actual data. The objective of most machine learning algorithms is to minimize this cost function. Below is a step-by-step guide to understanding and implementing cost functions in Python.

### Step 1: Understanding the Theory

Before diving into the code, let's understand some basic concepts:

1. **Cost Function**: A mathematical function that measures the error between predicted and actual values.
2. **Objective**: Minimize the cost function to improve the model's predictions.

Common cost functions include:
- **Mean Squared Error (MSE)**: Used for regression problems.
- **Cross-Entropy Loss**: Used for classification problems.

### Step 2: Importing Necessary Libraries

First, we need to import the necessary libraries.

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Step 3: Implementing Mean Squared Error (MSE)

Let's implement the Mean Squared Error (MSE) cost function for a simple linear regression problem.

1. **Generating Data**: We'll start by generating some synthetic data for a linear regression problem.

```python
# Generating synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

2. **Initializing Parameters**: We'll initialize the parameters (weights) for our linear model.

```python
theta = np.random.randn(2, 1)  # Random initialization
```

3. **Adding Bias Term to Input**: We'll add a column of ones to the input matrix \(X\) to account for the bias term.

```python
X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
```

4. **Computing the Cost Function**: We'll define the MSE cost function and compute the cost.

```python
def compute_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    mse = (1 / (2 * m)) * np.sum(error ** 2)
    return mse

# Compute initial cost
initial_cost = compute_mse(X_b, y, theta)
print(f"Initial MSE: {initial_cost}")
```

### Step 4: Implementing Gradient Descent

Next, we'll implement gradient descent to minimize the cost function.

1. **Setting Hyperparameters**: We'll define the learning rate and the number of iterations.

```python
learning_rate = 0.1
n_iterations = 1000
m = len(y)
```

2. **Gradient Descent Function**: We'll define the gradient descent function to update the parameters.

```python
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    for iteration in range(n_iterations):
        gradients = 1/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

# Perform gradient descent
theta_optimized = gradient_descent(X_b, y, theta, learning_rate, n_iterations)
print(f"Optimized parameters: {theta_optimized}")
```

3. **Plotting the Regression Line**: Finally, we'll plot the regression line to visualize the results.

```python
# Plotting the results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_optimized), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.show()
```

### Step 5: Implementing Cross-Entropy Loss

For classification problems, we'll implement the cross-entropy loss.

1. **Generating Data**: We'll generate synthetic data for a binary classification problem.

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
y = y.reshape(-1, 1)
```

2. **Sigmoid Function**: We'll define the sigmoid function for our logistic regression model.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

3. **Cross-Entropy Loss Function**: We'll define the cross-entropy loss function.

```python
def compute_cross_entropy_loss(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    loss = -1/m * (y.T.dot(np.log(predictions)) + (1 - y).T.dot(np.log(1 - predictions)))
    return loss[0][0]

# Compute initial loss
initial_loss = compute_cross_entropy_loss(X, y, theta)
print(f"Initial Cross-Entropy Loss: {initial_loss}")
```

4. **Gradient Descent for Logistic Regression**: We'll modify our gradient descent function for logistic regression.

```python
def gradient_descent_logistic(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    for iteration in range(n_iterations):
        predictions = sigmoid(X.dot(theta))
        gradients = 1/m * X.T.dot(predictions - y)
        theta = theta - learning_rate * gradients
    return theta

# Perform gradient descent
theta_optimized_logistic = gradient_descent_logistic(X, y, theta, learning_rate, n_iterations)
print(f"Optimized parameters: {theta_optimized_logistic}")
```

5. **Plotting Decision Boundary**: Finally, we'll plot the decision boundary to visualize the results.

```python
# Plotting the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

x_values = [np.min(X[:, 0] - 1), np.max(X[:, 1] + 1)]
y_values = -(theta_optimized_logistic[0] + np.dot(theta_optimized_logistic[1], x_values)) / theta_optimized_logistic[2]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
```

### Conclusion

Understanding and implementing cost functions are essential steps in building effective machine learning models. By following the steps above, you can grasp the fundamentals of cost functions and how to minimize them using gradient descent. This foundational knowledge is crucial for further exploring and mastering more complex machine learning algorithms.
