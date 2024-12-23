import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from typing import Tuple, List

class CostFunctions:
    """
    A comprehensive demonstration of common cost functions used in machine learning
    with visualization capabilities and practical examples.
    """
    
    def __init__(self):
        np.random.seed(42)
    
    def generate_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data for regression and classification tasks
        """
        # Regression data
        X_reg = np.linspace(-5, 5, 100).reshape(-1, 1)
        y_reg = 2 * X_reg + 1 + np.random.normal(0, 1, (100, 1))
        
        # Classification data
        X_clf, y_clf = make_classification(
            n_samples=100, n_features=1, n_classes=2, 
            n_clusters_per_class=1, flip_y=0.1
        )
        
        return (X_reg, y_reg), (X_clf, y_clf)

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (MSE)
        Commonly used in regression tasks
        MSE = (1/n) * Σ(y_true - y_pred)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)
        Less sensitive to outliers compared to MSE
        MAE = (1/n) * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """
        Huber Loss
        Combines best properties of MSE and MAE
        Less sensitive to outliers than MSE, more stable than MAE
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Binary Cross-Entropy Loss
        Common for binary classification tasks
        BCE = -Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def categorical_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Categorical Cross-Entropy Loss
        Used for multi-class classification tasks
        CCE = -Σ(y_true * log(y_pred))
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def hinge_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Hinge Loss
        Common in Support Vector Machines (SVM)
        Used for binary classification with class labels {-1, 1}
        """
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def visualize_regression_losses(self):
        """
        Visualize different regression loss functions
        """
        # Generate predictions ranging from -10 to 10
        y_true = np.zeros(1000)
        y_pred = np.linspace(-10, 10, 1000)
        
        # Calculate losses
        mse_losses = [(y_true - pred) ** 2 for pred in y_pred]
        mae_losses = [np.abs(y_true - pred) for pred in y_pred]
        huber_losses = [self.huber_loss(np.array([0]), np.array([pred])) for pred in y_pred]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_pred, mse_losses, label='MSE')
        plt.plot(y_pred, mae_losses, label='MAE')
        plt.plot(y_pred, huber_losses, label='Huber')
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        plt.title('Comparison of Regression Loss Functions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def demonstrate_cost_functions(self):
        """
        Demonstrate the usage of different cost functions on sample data
        """
        # Generate data
        (X_reg, y_reg), (X_clf, y_clf) = self.generate_sample_data()
        
        # Regression predictions (with intentional error for demonstration)
        y_pred_reg = 2 * X_reg + 1.5 + np.random.normal(0, 0.5, (100, 1))
        
        # Classification predictions (probabilities)
        y_pred_clf_proba = 1 / (1 + np.exp(-X_clf))  # Sigmoid
        
        # Calculate and display regression metrics
        mse = self.mean_squared_error(y_reg, y_pred_reg)
        mae = self.mean_absolute_error(y_reg, y_pred_reg)
        huber = self.huber_loss(y_reg, y_pred_reg)
        
        # Calculate classification metrics
        bce = self.binary_cross_entropy(y_clf.reshape(-1, 1), y_pred_clf_proba.reshape(-1, 1))
        hinge = self.hinge_loss(2 * y_clf - 1, y_pred_clf_proba.reshape(-1))  # Convert to {-1, 1}
        
        results = {
            'Regression Metrics': {
                'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'Huber Loss': huber
            },
            'Classification Metrics': {
                'Binary Cross-Entropy': bce,
                'Hinge Loss': hinge
            }
        }
        
        return results

    def plot_regression_data(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Plot regression data and predictions
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y_true, label='True Values', alpha=0.5)
        plt.scatter(X, y_pred, label='Predictions', alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Regression Data and Predictions')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    cost_functions = CostFunctions()
    
    # Demonstrate cost functions
    results = cost_functions.demonstrate_cost_functions()
    
    # Print results
    print("\nCost Function Results:")
    for category, metrics in results.items():
        print(f"\n{category}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Visualize regression losses
    cost_functions.visualize_regression_losses()
    
    # Generate and plot sample data
    (X_reg, y_reg), _ = cost_functions.generate_sample_data()
    y_pred_reg = 2 * X_reg + 1.5 + np.random.normal(0, 0.5, (100, 1))
    cost_functions.plot_regression_data(X_reg, y_reg, y_pred_reg)