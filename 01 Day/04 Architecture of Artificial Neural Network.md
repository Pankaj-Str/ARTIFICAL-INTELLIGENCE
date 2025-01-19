
# Codes with Pankaj: Understanding Neural Network Architecture

## Introduction
Neural networks are like digital brains - they're made up of layers of connected neurons that work together to solve complex problems. Let's understand their architecture from the ground up!

## Basic Structure of a Neural Network

Let's start with a simple implementation to visualize the concept:

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: list of integers representing neurons in each layer
        Example: [3, 4, 2] means:
        - 3 input neurons
        - 4 hidden layer neurons
        - 2 output neurons
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases between layers
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1])
            b = np.random.randn(1, layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        current_input = X
        
        # Store activations for visualization
        activations = [current_input]
        
        for w, b in zip(self.weights, self.biases):
            # Calculate net input
            net_input = np.dot(current_input, w) + b
            # Apply activation function (ReLU)
            current_input = np.maximum(0, net_input)
            activations.append(current_input)
            
        return activations
```

## Neural Network Components

### 1. Input Layer
- First layer of the network
- Receives raw data
- Each neuron represents a feature

Example of preparing input data:
```python
def prepare_input_layer(data):
    """
    Normalize input data between 0 and 1
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Example usage
input_data = np.array([
    [5.1, 3.5, 1.4],  # First sample
    [4.9, 3.0, 1.4],  # Second sample
])
normalized_input = prepare_input_layer(input_data)
```

### 2. Hidden Layers
- Layers between input and output
- Perform complex feature extraction
- Can be multiple hidden layers (deep learning)

```python
class HiddenLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.bias
        self.activated_output = self.relu(self.output)
        return self.activated_output
    
    def relu(self, x):
        return np.maximum(0, x)
```

### 3. Output Layer
- Final layer of the network
- Produces the network's prediction
- Architecture depends on the problem type:
  - Binary classification: 1 neuron
  - Multi-class classification: Multiple neurons
  - Regression: 1 or more neurons

```python
class OutputLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(self.output)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

## Common Neural Network Architectures

### 1. Feedforward Neural Network (FNN)
The most basic architecture where information flows in one direction.

```python
class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layers = []
        
        # Create hidden layers
        for i in range(len(layer_sizes)-2):
            self.layers.append(
                HiddenLayer(layer_sizes[i], layer_sizes[i+1])
            )
        
        # Create output layer
        self.layers.append(
            OutputLayer(layer_sizes[-2], layer_sizes[-1])
        )
    
    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
```

### 2. Convolutional Neural Network (CNN)
Specialized for processing grid-like data (images).

```python
class SimpleConvLayer:
    def __init__(self, kernel_size):
        self.kernel = np.random.randn(kernel_size, kernel_size)
    
    def convolve2d(self, image):
        i_height, i_width = image.shape
        k_height, k_width = self.kernel.shape
        
        output_height = i_height - k_height + 1
        output_width = i_width - k_width + 1
        
        output = np.zeros((output_height, output_width))
        
        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(
                    image[i:i+k_height, j:j+k_width] * self.kernel
                )
        
        return output
```

### 3. Recurrent Neural Network (RNN)
For processing sequential data like text or time series.

```python
class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        
    def forward(self, x, h_prev):
        # Combine input and previous hidden state
        h_next = np.tanh(
            np.dot(x, self.Wxh) + 
            np.dot(h_prev, self.Whh) + 
            self.bh
        )
        return h_next
```

## Choosing the Right Architecture

Here's a decision guide:

```python
def recommend_architecture(problem_type, data_type, data_size):
    if data_type == 'image':
        return 'CNN'
    elif data_type == 'sequence':
        return 'RNN'
    elif data_size < 10000:
        return 'Simple Feedforward NN'
    else:
        return 'Deep Feedforward NN'

# Example usage
problem = {
    'type': 'classification',
    'data_type': 'image',
    'data_size': 50000
}

recommended = recommend_architecture(
    problem['type'],
    problem['data_type'],
    problem['data_size']
)
print(f"Recommended architecture: {recommended}")
```

## Common Patterns in Neural Architecture

1. **Increasing Layer Width**:
```python
layer_sizes = [64, 128, 256, 512]  # Common pattern
```

2. **Decreasing Layer Width**:
```python
layer_sizes = [512, 256, 128, 64]  # Encoder pattern
```

3. **Bottleneck Architecture**:
```python
layer_sizes = [256, 128, 64, 128, 256]  # Autoencoder pattern
```

## Best Practices for Architecture Design

1. Start Simple:
```python
def create_initial_architecture(input_size, output_size):
    return {
        'input_layer': input_size,
        'hidden_layers': [input_size * 2],
        'output_layer': output_size
    }
```

2. Gradually Add Complexity:
```python
def add_layer(architecture, size):
    architecture['hidden_layers'].append(size)
    return architecture
```

3. Monitor Performance:
```python
def evaluate_architecture(architecture, X_train, y_train):
    model = create_model(architecture)
    history = train_model(model, X_train, y_train)
    return history.metrics
```

## Practice Exercise
Create a neural network for a specific problem:
1. Define input and output sizes
2. Choose appropriate layer sizes
3. Implement forward propagation
4. Test with sample data

## Next Steps
- Learn about advanced architectures (Transformers, GANs)
- Study optimization techniques
- Explore regularization methods
- Practice implementing different architecture ?​​