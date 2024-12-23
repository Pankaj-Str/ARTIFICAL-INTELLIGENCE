import numpy as np
import matplotlib.pyplot as plt

def simple_activation_demo():
    """
    Simple demonstration of activation functions with clear visualizations
    and examples.
    """
    # Create input values for our demonstration
    x = np.linspace(-5, 5, 100)
    
    # 1. ReLU (Rectified Linear Unit)
    def relu(x):
        return np.maximum(0, x)
    
    # 2. Sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # 3. Tanh
    def tanh(x):
        return np.tanh(x)
    
    # Create a figure with three subplots
    plt.figure(figsize=(15, 5))
    
    # Plot ReLU
    plt.subplot(1, 3, 1)
    plt.plot(x, relu(x))
    plt.title('ReLU Function')
    plt.grid(True)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    
    # Plot Sigmoid
    plt.subplot(1, 3, 2)
    plt.plot(x, sigmoid(x))
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    
    # Plot Tanh
    plt.subplot(1, 3, 3)
    plt.plot(x, tanh(x))
    plt.title('Tanh Function')
    plt.grid(True)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate practical examples
    test_values = [-2, -1, 0, 1, 2]
    print("\nPractical Examples with inputs:", test_values)
    
    print("\n1. ReLU Examples:")
    print("ReLU is like a gate that only lets positive values through.")
    for val in test_values:
        result = relu(val)
        print(f"Input: {val:>4} → Output: {result:>4}")
    
    print("\n2. Sigmoid Examples:")
    print("Sigmoid squishes any input into a value between 0 and 1.")
    for val in test_values:
        result = sigmoid(val)
        print(f"Input: {val:>4} → Output: {result:.4f}")
    
    print("\n3. Tanh Examples:")
    print("Tanh squishes any input into a value between -1 and 1.")
    for val in test_values:
        result = tanh(val)
        print(f"Input: {val:>4} → Output: {result:.4f}")

if __name__ == "__main__":
    simple_activation_demo()