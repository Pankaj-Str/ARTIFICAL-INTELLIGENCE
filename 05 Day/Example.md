
### Additional Examples of Gradient Descent

#### Example 1: Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent updates the parameters using just one training example at a time. It is generally noisier than batch gradient descent but can converge faster on large datasets.

##### Python Implementation:
```python
import numpy as np
import matplotlib.pyplot as plt

# Create some data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# SGD parameters
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

# Function to determine the learning rate at each step
def learning_schedule(t):
    return t0 / (t + t1)

# Initial parameters
theta = np.random.randn(2,1)  # random initialization

# Add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), x]

# Perform Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(100):
        random_index = np.random.randint(100)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * 100 + i)
        theta -= eta * gradients

print(f"SGD Model Parameters: \n{theta}")
```

#### Example 2: Mini-batch Gradient Descent
Mini-batch Gradient Descent is a compromise between batch and stochastic versions. It calculates the gradient on small random sets of instances called mini-batches.

##### Python Implementation:
```python
import numpy as np
import matplotlib.pyplot as plt

# Create some data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Mini-batch Gradient Descent parameters
n_iterations = 50
minibatch_size = 20

t0, t1 = 10, 1000  # learning schedule hyperparameters

# Function to determine the learning rate at each step
def learning_schedule(t):
    return t0 / (t + t1)

# Initial parameters
theta = np.random.randn(2,1)  # random initialization

# Add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), x]

# Perform Mini-batch Gradient Descent
t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(100)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, 100, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta -= eta * gradients

print(f"Mini-batch Gradient Descent Model Parameters: \n{theta}")
```

#### Conclusion
These examples illustrate different forms of gradient descent, showing how they can be implemented in Python to optimize a simple linear regression model. Stochastic and mini-batch gradient descent are particularly useful for large datasets or when the training set does not fit into memory. By experimenting with these methods, you can better understand their dynamics and applications, which is crucial for applying gradient descent in more complex machine learning scenarios.
