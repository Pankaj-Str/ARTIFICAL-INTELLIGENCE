
1. **Mean Squared Error (MSE)** - Used in regression problems where we predict continuous values (like predicting prices or heights).
2. **Cross-Entropy Loss** - Used in classification problems where we predict categories (like whether an email is spam or not).

### Simple Example: Cost Functions in Python

We will use some example data to understand these concepts.

```python
import numpy as np

# Example 1: Regression Problem with Mean Squared Error (MSE)

# Let's create some data for a simple regression problem
# Imagine we are predicting a score based on study hours
np.random.seed(0)
study_hours = np.random.rand(10, 1) * 10  # 10 random study hours between 0 and 10
true_scores = 5 * study_hours.squeeze() + 20 + np.random.randn(10) * 5  # True scores with some noise

# Suppose our model predicted these scores
predicted_scores = 5 * study_hours.squeeze() + 18  # Predicted scores

# Mean Squared Error (MSE) Function
def mean_squared_error(true, predicted):
    return np.mean((true - predicted) ** 2)

mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error: {mse:.2f}")

# Example 2: Classification Problem with Cross-Entropy Loss

# Creating data for a simple classification problem
# Imagine predicting whether a student passes (1) or fails (0) based on study hours
study_hours_classification = np.random.rand(10, 1) * 10  # 10 random study hours between 0 and 10
true_labels = (study_hours_classification.squeeze() > 5).astype(int)  # Label 1 if study hours > 5, else 0

# Suppose our model predicted these probabilities of passing
predicted_probs = np.clip(study_hours_classification.squeeze() / 10, 0.1, 0.9)  # Predicted probabilities between 0.1 and 0.9

# Cross-Entropy Loss Function
def cross_entropy_loss(true, predicted):
    epsilon = 1e-15  # Small value to avoid log(0)
    predicted = np.clip(predicted, epsilon, 1 - epsilon)  # Ensure predictions are between 0 and 1
    return -np.mean(true * np.log(predicted) + (1 - true) * np.log(1 - predicted))

cross_entropy = cross_entropy_loss(true_labels, predicted_probs)
print(f"Cross-Entropy Loss: {cross_entropy:.2f}")
```

### What is Happening Here?

1. **Mean Squared Error (MSE):**
   - We have some study hours and true scores (with some randomness).
   - Our model tries to predict the scores based on study hours.
   - MSE measures the average squared difference between the true scores and the predicted scores. The smaller the MSE, the better the model.

2. **Cross-Entropy Loss:**
   - We use study hours to predict if a student passes (1) or fails (0).
   - Our model gives a probability of passing based on study hours.
   - Cross-Entropy Loss measures how well these predicted probabilities match the true labels. The smaller the loss, the better the predictions.

