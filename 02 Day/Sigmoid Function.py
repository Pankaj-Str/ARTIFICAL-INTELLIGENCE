import numpy as np
import matplotlib.pyplot as plt

class SigmoidFunction:
    def __init__(self):
        """Initialize the Sigmoid Function class"""
        self.name = "Sigmoid Function"
    
    def sigmoid(self, x):
        """
        Calculate the sigmoid of x
        f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Calculate the derivative of sigmoid
        f'(x) = f(x) * (1 - f(x))
        """
        sx = self.sigmoid(x)
        return sx * (1 - sx)
    
    def plot_sigmoid(self):
        """Create visualization of sigmoid function and its derivative"""
        # Generate input values
        x = np.linspace(-10, 10, 1000)
        
        # Calculate sigmoid and its derivative
        y_sigmoid = self.sigmoid(x)
        y_derivative = self.sigmoid_derivative(x)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot sigmoid function
        plt.plot(x, y_sigmoid, 'b-', label='Sigmoid Function', linewidth=2)
        plt.plot(x, y_derivative, 'r--', label='Sigmoid Derivative', linewidth=2)
        
        # Add horizontal lines at y=0, y=0.5, and y=1
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
        
        # Add vertical line at x=0
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Customize the plot
        plt.grid(True, alpha=0.3)
        plt.title('Sigmoid Function and its Derivative', fontsize=14)
        plt.xlabel('Input (x)', fontsize=12)
        plt.ylabel('Output', fontsize=12)
        plt.legend(fontsize=10)
        
        # Show the plot
        plt.show()
    
    def practical_examples(self):
        """Show practical examples of sigmoid function"""
        # Test values
        test_values = [-5, -2, -1, 0, 1, 2, 5]
        
        print("\nPractical Examples of Sigmoid Function:")
        print("\nInput → Sigmoid Output → Derivative")
        print("-" * 45)
        
        for x in test_values:
            sig = self.sigmoid(x)
            der = self.sigmoid_derivative(x)
            print(f"{x:4.1f} → {sig:14.6f} → {der:11.6f}")
    
    def binary_classification_example(self):
        """Demonstrate sigmoid function in binary classification"""
        # Generate some example scores
        scores = np.array([-3.2, -1.5, 0.2, 1.8, 4.0])
        probabilities = self.sigmoid(scores)
        
        print("\nBinary Classification Example:")
        print("\nRaw Score → Probability → Predicted Class")
        print("-" * 50)
        
        for score, prob in zip(scores, probabilities):
            predicted_class = 1 if prob >= 0.5 else 0
            print(f"{score:8.2f} → {prob:11.6f} → {predicted_class}")

# Create instance and run demonstrations
if __name__ == "__main__":
    sigmoid = SigmoidFunction()
    
    # Show sigmoid visualization
    sigmoid.plot_sigmoid()
    
    # Show practical examples
    sigmoid.practical_examples()
    
    # Show binary classification example
    sigmoid.binary_classification_example()