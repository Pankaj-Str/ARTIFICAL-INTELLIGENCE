import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

class OptimizationDemo:
    def __init__(self):
        # Initialize example parameters
        self.learning_rate = 0.1
        self.beta1 = 0.9  # For Adam and Momentum
        self.beta2 = 0.999  # For Adam
        self.epsilon = 1e-8  # Small value to prevent division by zero

    def generate_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a simple quadratic function with noise for demonstration"""
        X = np.linspace(-5, 5, 100)
        y = X**2 + np.random.normal(0, 1, 100)
        return X.reshape(-1, 1), y.reshape(-1, 1)

    def gradient_descent(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Basic Gradient Descent
        params: Current parameters
        gradients: Computed gradients
        Returns: Updated parameters
        """
        return params - self.learning_rate * gradients

    def momentum(self, params: np.ndarray, gradients: np.ndarray, 
                velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Momentum Optimization
        Helps accelerate gradients in the right direction and dampens oscillations
        """
        # Update velocity
        velocity = self.beta1 * velocity + (1 - self.beta1) * gradients
        # Update parameters
        params = params - self.learning_rate * velocity
        return params, velocity

    def rmsprop(self, params: np.ndarray, gradients: np.ndarray, 
                cache: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RMSprop (Root Mean Square Propagation)
        Adapts learning rates based on the magnitude of recent gradients
        """
        # Update cache
        cache = self.beta2 * cache + (1 - self.beta2) * (gradients ** 2)
        # Update parameters
        params = params - self.learning_rate * gradients / (np.sqrt(cache) + self.epsilon)
        return params, cache

    def adam(self, params: np.ndarray, gradients: np.ndarray, 
            moment: np.ndarray, velocity: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Adam Optimization (Adaptive Moment Estimation)
        Combines benefits of Momentum and RMSprop
        """
        # Update moment and velocity
        moment = self.beta1 * moment + (1 - self.beta1) * gradients
        velocity = self.beta2 * velocity + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        moment_corrected = moment / (1 - self.beta1 ** t)
        velocity_corrected = velocity / (1 - self.beta2 ** t)
        
        # Update parameters
        params = params - self.learning_rate * moment_corrected / (np.sqrt(velocity_corrected) + self.epsilon)
        return params, moment, velocity

    def demonstrate_optimizers(self):
        """Demonstrate different optimizers on a simple problem"""
        # Generate sample data
        X, y = self.generate_sample_data()
        
        # Initialize parameters and states for different optimizers
        params_gd = np.random.randn(1)
        params_momentum = np.random.randn(1)
        params_rmsprop = np.random.randn(1)
        params_adam = np.random.randn(1)
        
        velocity = np.zeros_like(params_momentum)
        cache = np.zeros_like(params_rmsprop)
        moment = np.zeros_like(params_adam)
        velocity_adam = np.zeros_like(params_adam)
        
        # Training history
        history = {
            'GD': [], 'Momentum': [], 'RMSprop': [], 'Adam': []
        }
        
        # Training loop
        for t in range(1, 101):
            # Compute gradients (simplified for demonstration)
            gradients = 2 * X.T.dot(X.dot(params_gd) - y) / len(X)
            
            # Update parameters using different optimizers
            params_gd = self.gradient_descent(params_gd, gradients)
            params_momentum, velocity = self.momentum(params_momentum, gradients, velocity)
            params_rmsprop, cache = self.rmsprop(params_rmsprop, gradients, cache)
            params_adam, moment, velocity_adam = self.adam(params_adam, gradients, moment, velocity_adam, t)
            
            # Record loss
            loss_gd = np.mean((X.dot(params_gd) - y) ** 2)
            loss_momentum = np.mean((X.dot(params_momentum) - y) ** 2)
            loss_rmsprop = np.mean((X.dot(params_rmsprop) - y) ** 2)
            loss_adam = np.mean((X.dot(params_adam) - y) ** 2)
            
            history['GD'].append(loss_gd)
            history['Momentum'].append(loss_momentum)
            history['RMSprop'].append(loss_rmsprop)
            history['Adam'].append(loss_adam)
        
        return history

    def plot_results(self, history: dict):
        """Plot the learning curves for different optimizers"""
        plt.figure(figsize=(10, 6))
        for optimizer, losses in history.items():
            plt.plot(losses, label=optimizer)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Comparison of Optimization Algorithms')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    demo = OptimizationDemo()
    history = demo.demonstrate_optimizers()
    demo.plot_results(history)