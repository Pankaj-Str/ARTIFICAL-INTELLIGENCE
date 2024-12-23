import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class StepByStepNN:
    """
    A step-by-step implementation of a neural network to understand each component.
    This example uses a simple binary classification problem.
    """
    
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize the neural network with specified layer sizes.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
                        [input_size, hidden_layer1_size, ..., output_size]
        """
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.gradients = {}
        self.cache = {}
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """
        Step 1: Initialize weights and biases for all layers
        Using He initialization for weights
        """
        print("\nStep 1: Initializing Parameters")
        
        for l in range(1, len(self.layer_sizes)):
            # He initialization for weights
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], 
                self.layer_sizes[l-1]
            ) * np.sqrt(2 / self.layer_sizes[l-1])
            
            # Initialize biases to zeros
            self.parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
            
            print(f"Layer {l} - Weight shape: {self.parameters[f'W{l}'].shape}, "
                  f"Bias shape: {self.parameters[f'b{l}'].shape}")
    
    def relu(self, Z: np.ndarray) -> np.ndarray:
        """
        Step 2a: ReLU Activation Function
        A(x) = max(0, x)
        """
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU function
        A'(x) = 1 if x > 0 else 0
        """
        return np.where(Z > 0, 1, 0)
    
    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Step 2b: Sigmoid Activation Function
        A(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    def sigmoid_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Derivative of Sigmoid function
        A'(x) = A(x) * (1 - A(x))
        """
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Step 3: Forward Propagation
        Compute activations for each layer
        """
        print("\nStep 3: Forward Propagation")
        
        # Input layer
        self.cache['A0'] = X
        
        # Hidden layers with ReLU
        for l in range(1, len(self.layer_sizes) - 1):
            # Linear transformation
            Z = np.dot(self.parameters[f'W{l}'], self.cache[f'A{l-1}']) + self.parameters[f'b{l}']
            self.cache[f'Z{l}'] = Z
            
            # Apply ReLU activation
            self.cache[f'A{l}'] = self.relu(Z)
            print(f"Layer {l} (Hidden) - Activation shape: {self.cache[f'A{l}'].shape}")
        
        # Output layer with Sigmoid
        L = len(self.layer_sizes) - 1
        Z = np.dot(self.parameters[f'W{L}'], self.cache[f'A{L-1}']) + self.parameters[f'b{L}']
        self.cache[f'Z{L}'] = Z
        self.cache[f'A{L}'] = self.sigmoid(Z)
        
        print(f"Layer {L} (Output) - Activation shape: {self.cache[f'A{L}'].shape}")
        return self.cache[f'A{L}']
    
    def compute_cost(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Step 4: Compute Binary Cross-Entropy Cost
        J = -1/m * Σ(y*log(a) + (1-y)*log(1-a))
        """
        m = y_true.shape[1]
        epsilon = 1e-15  # Prevent log(0)
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -1/m * np.sum(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
        
        return float(cost)
    
    def backward_propagation(self, y_true: np.ndarray):
        """
        Step 5: Backward Propagation
        Compute gradients for all parameters
        """
        print("\nStep 5: Backward Propagation")
        
        m = y_true.shape[1]
        L = len(self.layer_sizes) - 1
        
        # Output layer
        dZ = self.cache[f'A{L}'] - y_true
        self.gradients[f'dW{L}'] = 1/m * np.dot(dZ, self.cache[f'A{L-1}'].T)
        self.gradients[f'db{L}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA * self.relu_derivative(self.cache[f'Z{l}'])
            
            self.gradients[f'dW{l}'] = 1/m * np.dot(dZ, self.cache[f'A{l-1}'].T)
            self.gradients[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
            print(f"Layer {l} - Gradient shapes: dW: {self.gradients[f'dW{l}'].shape}, "
                  f"db: {self.gradients[f'db{l}'].shape}")
    
    def update_parameters(self, learning_rate: float):
        """
        Step 6: Update Parameters
        Update weights and biases using computed gradients
        """
        print("\nStep 6: Updating Parameters")
        
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f'W{l}'] -= learning_rate * self.gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * self.gradients[f'db{l}']
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             learning_rate: float = 0.01, 
             epochs: int = 1000,
             print_cost: bool = True) -> List[float]:
        """
        Step 7: Training Loop
        Combine all steps to train the neural network
        """
        print("\nStep 7: Training Neural Network")
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(y_pred, y)
            
            # Backward propagation
            self.backward_propagation(y)
            
            # Update parameters
            self.update_parameters(learning_rate)
            
            # Print progress
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
            costs.append(cost)
        
        return costs

def generate_sample_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample binary classification data
    """
    np.random.seed(42)
    
    # Generate circular data
    r = np.random.normal(1, 0.3, (n_samples, 1))
    theta = np.random.uniform(0, 2*np.pi, (n_samples, 1))
    
    X = np.hstack([r*np.cos(theta), r*np.sin(theta)])
    y = (r < 1).astype(int)
    
    return X.T, y.T

def plot_decision_boundary(model: StepByStepNN, X: np.ndarray, y: np.ndarray):
    """
    Plot the decision boundary of the trained model
    """
    h = 0.01
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.forward_propagation(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.3)
    plt.scatter(X[0], X[1], c=y.ravel(), cmap='RdYlBu', alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    print("Step 0: Generating Sample Data")
    X, y = generate_sample_data(1000)
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    
    # Create and train neural network
    layer_sizes = [2, 4, 3, 1]  # [input_size, hidden1, hidden2, output_size]
    model = StepByStepNN(layer_sizes)
    
    # Train the model
    costs = model.train(X, y, learning_rate=0.1, epochs=1000)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y)