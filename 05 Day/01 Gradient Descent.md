# Gradient Descent 

### Understanding Gradient Descent in Neural Networks with Python

#### Introduction
Gradient Descent is a fundamental optimization algorithm used in training machine learning models, particularly neural networks. It is used to minimize the loss function by iteratively moving towards the minimum of the function. This tutorial will explain gradient descent, its variations, and provide a Python example to illustrate how it works.

#### What is Gradient Descent?
Gradient descent is an iterative optimization algorithm for finding the local minimum of a differentiable function. In the context of machine learning, this function is typically the loss function, and the goal is to find the model parameters (weights) that minimize the loss.

#### Key Concepts:
- **Loss Function**: A function that measures how well the model performs. The goal of training is to minimize this function.
- **Gradient**: The gradient of the loss function with respect to the model parameters. It points in the direction of steepest ascent.
- **Learning Rate**: A hyperparameter that determines the step size during each iteration of the gradient descent.

#### Types of Gradient Descent
1. **Batch Gradient Descent**: Computes the gradient using the whole dataset. This is precise but can be slow and computationally expensive for large datasets.
2. **Stochastic Gradient Descent (SGD)**: Computes the gradient using a single sample at each iteration. This is faster and can help escape local minima, but is noisier than batch gradient descent.
3. **Mini-batch Gradient Descent**: A compromise between batch and stochastic gradient descent. It computes the gradient for a small subset of the data at each step, providing a balance between speed and stability.

#### Example: Implementing Gradient Descent for Linear Regression
We will use a simple linear regression example where we try to fit a line to some data points. This will help in understanding how gradient descent optimizes the parameters.

##### Step 1: Import Necessary Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
```

##### Step 2: Generate Some Data
```python
# Generate random data around a line
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
```

##### Step 3: Implementing Batch Gradient Descent
```python
# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = 100  # number of data points

# Initial parameters
theta = np.random.randn(2,1)  # random initialization

# Add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), x]

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print(f"The model parameters after Gradient Descent: \n{theta}")
```

##### Step 4: Plotting the Results
```python
# Plot the results
plt.plot(x, y, "b.")
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]  # add x0 = 1 to each instance
y_predict = x_new_b.dot(theta)
plt.plot(x_new, y_predict, "r-")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.title("Linear Regression with Gradient Descent")
plt.axis([0, 2, 0, 15])
plt.show()
```

#### Conclusion
This tutorial demonstrated how to implement and understand gradient descent in the context of a simple linear regression model. By adjusting the model parameters iteratively in the direction that minimally reduces the loss function, gradient descent helps in optimizing the model. It's a powerful tool that forms the backbone of many machine learning algorithms, especially in neural networks. Understanding gradient descent is essential for any aspiring data scientist or machine learning practitioner.
