import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MSEAnalysis:
    """
    A comprehensive implementation and analysis of Mean Squared Error (MSE)
    with various examples and visualizations.
    """
    
    def __init__(self):
        np.random.seed(42)
        self.scaler = StandardScaler()
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_mse_gradient(self, X: np.ndarray, y_true: np.ndarray, 
                             weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of MSE with respect to weights
        ∂MSE/∂w = (-2/n) * X^T * (y - Xw)
        
        Args:
            X: Input features
            y_true: True values
            weights: Current weight values
            
        Returns:
            np.ndarray: Gradient of MSE
        """
        n = len(y_true)
        y_pred = np.dot(X, weights)
        return (-2/n) * np.dot(X.T, (y_true - y_pred))
    
    def generate_regression_data(self, n_samples: int = 100, 
                               noise: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample regression data with controlled noise
        
        Args:
            n_samples: Number of samples to generate
            noise: Standard deviation of noise
            
        Returns:
            Tuple containing features and target values
        """
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y_true = 0.5 * X**2 + 2 * X + 1
        y = y_true + np.random.normal(0, noise, y_true.shape)
        return X, y
    
    def linear_regression_with_mse(self, X: np.ndarray, y: np.ndarray, 
                                 learning_rate: float = 0.01, 
                                 n_iterations: int = 1000) -> Tuple[List[float], np.ndarray]:
        """
        Implement linear regression using MSE as loss function
        
        Args:
            X: Input features
            y: Target values
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            
        Returns:
            Tuple containing MSE history and final weights
        """
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        
        # Initialize weights
        weights = np.random.randn(n_features)
        mse_history = []
        
        # Gradient descent
        for i in range(n_iterations):
            # Calculate predictions
            y_pred = np.dot(X_b, weights)
            
            # Calculate MSE
            mse = self.calculate_mse(y, y_pred)
            mse_history.append(mse)
            
            # Update weights using gradient descent
            gradient = self.calculate_mse_gradient(X_b, y, weights)
            weights = weights - learning_rate * gradient
            
        return mse_history, weights
    
    def visualize_mse_surface(self, X: np.ndarray, y: np.ndarray):
        """
        Visualize MSE surface for different weight values
        
        Args:
            X: Input features
            y: Target values
        """
        # Create weight space
        w0 = np.linspace(-10, 10, 100)
        w1 = np.linspace(-10, 10, 100)
        W0, W1 = np.meshgrid(w0, w1)
        
        # Calculate MSE for each weight combination
        MSE = np.zeros(W0.shape)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        for i in range(len(w0)):
            for j in range(len(w1)):
                weights = np.array([W0[i,j], W1[i,j]])
                y_pred = np.dot(X_b, weights)
                MSE[i,j] = self.calculate_mse(y, y_pred)
        
        # Plot MSE surface
        plt.figure(figsize=(12, 5))
        
        # 3D surface plot
        plt.subplot(121, projection='3d')
        surf = plt.gca().plot_surface(W0, W1, MSE, cmap='viridis')
        plt.colorbar(surf)
        plt.xlabel('Weight 0 (bias)')
        plt.ylabel('Weight 1')
        plt.title('MSE Surface')
        
        # Contour plot
        plt.subplot(122)
        plt.contour(W0, W1, MSE, levels=50)
        plt.colorbar()
        plt.xlabel('Weight 0 (bias)')
        plt.ylabel('Weight 1')
        plt.title('MSE Contours')
        
        plt.tight_layout()
        plt.show()
    
    def compare_predictions(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        """
        Compare true values with predictions
        
        Args:
            X: Input features
            y: True values
            weights: Trained weights
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = np.dot(X_b, weights)
        
        plt.figure(figsize=(10, 5))
        
        # Data and predictions
        plt.subplot(121)
        plt.scatter(X, y, color='blue', alpha=0.5, label='True values')
        plt.plot(X, y_pred, color='red', label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Data and Predictions')
        plt.legend()
        
        # Residuals
        plt.subplot(122)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_mse_properties(self):
        """
        Demonstrate key properties of MSE
        """
        # Generate sample data
        X, y = self.generate_regression_data(n_samples=100, noise=0.3)
        
        # Train model
        mse_history, weights = self.linear_regression_with_mse(X, y)
        
        # Plot training progress
        plt.figure(figsize=(10, 5))
        plt.plot(mse_history)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('MSE During Training')
        plt.grid(True)
        plt.show()
        
        # Visualize MSE surface
        self.visualize_mse_surface(X, y)
        
        # Compare predictions
        self.compare_predictions(X, y, weights)
        
        # Print final MSE
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = np.dot(X_b, weights)
        final_mse = self.calculate_mse(y, y_pred)
        print(f"\nFinal MSE: {final_mse:.4f}")
        
        return final_mse, weights

# Example usage
if __name__ == "__main__":
    # Create MSE analysis object
    mse_analyzer = MSEAnalysis()
    
    print("Step 1: Generating sample data...")
    X, y = mse_analyzer.generate_regression_data()
    
    print("\nStep 2: Training model with MSE loss...")
    final_mse, weights = mse_analyzer.demonstrate_mse_properties()
    
    print("\nStep 3: Model Summary")
    print(f"Trained weights: {weights}")
    print(f"Final MSE: {final_mse:.4f}")