# Build a Neural Network from Scratch in Python

This guide walks you through building a simple neural network from scratch using Python. It covers the essential concepts such as defining layers, initializing weights and biases, and performing forward passes.

## Prerequisites

Ensure you have the following libraries installed in your Python environment:

```python
import sys
import numpy as np
import matplotlib 

print(sys.version)
print(np.__version__)
print(matplotlib.__version__)
```

Expected Output:
```
3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
1.19.5
3.4.3
```

## Step 1: Create a Dense Layer Class

Start by creating a class for a dense (fully connected) layer:

```python
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        pass # using pass statement as a placeholder
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        pass # using pass statement as a placeholder
```

## Step 2: Generate Random Weights

For our example, we use a random number generator to initialize the weights:

```python
import numpy as np 
np.random.seed(0)

# Define our dataset 
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
```

## Step 3: Implement the Dense Layer

Next, we define the dense layer with its forward pass:

```python
class Dense_layer:
    def __init__(self, n_inputs, n_neurons): 
        # Generate weights randomly and multiply with 0.1 to make the numbers smaller
        self.weight = 0.10 * np.random.randn(n_inputs, n_neurons) 
        # Generate bias 
        self.bias = np.zeros((1, n_neurons)) 
    
    # Define the forward function, it takes the dataset as input
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.bias
```

## Step 4: Test the Dense Layer

Create two layers and perform a forward pass with the sample dataset:

```python
# Create layers
layer1 = Dense_layer(4, 5)
layer2 = Dense_layer(5, 2)

# Forward pass
layer1.forward(X)
layer2.forward(layer1.output)

# Print the results
print(layer1.output, '\n')
print(layer2.output)
```

Expected Output:
```
[[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
 [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
 [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]] 

[[ 0.148296   -0.08397602]
 [ 0.14100315 -0.01340469]
 [ 0.20124979 -0.07290616]]
```

