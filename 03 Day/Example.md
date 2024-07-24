### 1. Mean Squared Error (MSE), 
### 2. Mean Absolute Error (MAE), 
### 3. Cross-Entropy Loss, and Hinge Loss. 

### A. Mean Squared Error (MSE)
**Description:** MSE measures the average squared difference between actual and predicted values. This squaring implies that larger errors are penalized more heavily, which can be particularly useful in regression scenarios where outliers are undesirable.

**Python Example:** Predicting house prices using a simple dataset.
```python
import numpy as np

# Actual and predicted prices of houses
actual_prices = np.array([150000, 180000, 240000, 190000, 300000])
predicted_prices = np.array([145000, 175000, 235000, 185000, 295000])

# Calculate Mean Squared Error
mse = np.mean((actual_prices - predicted_prices) ** 2)
print(f"Mean Squared Error: {mse}")
```

### B. Mean Absolute Error (MAE)
**Description:** MAE measures the average magnitude of the absolute differences between actual and predictions. Unlike MSE, it treats all individual differences equally, making it robust against outliers.

**Python Example:** Predicting delivery times where some delays could be significantly longer due to unforeseen circumstances.
```python
import numpy as np

# Actual and predicted delivery times in days
actual_times = np.array([2, 3, 5, 7, 4])
predicted_times = np.array([3, 3, 4, 6, 5])

# Calculate Mean Absolute Error
mae = np.mean(np.abs(actual_times - predicted_times))
print(f"Mean Absolute Error: {mae}")
```

### C. Cross-Entropy Loss or Log Loss
**Description:** Used in classification tasks, this loss measures the performance of a model whose output is a probability value between 0 and 1. It's perfect for scenarios like binary classification, where it helps in assessing how close the prediction probabilities are to actual class labels.

**Python Example:** Using TensorFlow to calculate cross-entropy loss for a binary classification task, such as email spam detection.
```python
import tensorflow as tf

# Actual labels and predicted probabilities for a binary classification (0 = not spam, 1 = spam)
actual_labels = tf.constant([0, 1, 1, 0, 1])
predicted_probabilities = tf.constant([0.05, 0.9, 0.8, 0.1, 0.7])

# Calculate Cross-Entropy Loss
cross_entropy_loss = tf.keras.losses.binary_crossentropy(actual_labels, predicted_probabilities)
print(f"Cross-Entropy Loss: {cross_entropy_loss.numpy()}")
```

### D. Hinge Loss
**Description:** Hinge loss is commonly used with Support Vector Machines (SVMs) for binary classification tasks. It aims to not only classify the instances correctly but also to ensure that the decision boundary is as far from the nearest data points of each class as possible, thereby maximizing the margin.

**Python Example:** Classifying whether images contain a cat or not, aiming for a confident margin.
```python
import numpy as np

# True binary labels (1 = cat, -1 = not cat) and predicted scores from some classifier
true_labels = np.array([1, -1, 1, 1, -1])
predicted_scores = np.array([0.8, -0.3, 0.6, 1.2, -1.0])

# Calculate Hinge Loss
hinge_loss = np.mean(np.maximum(0, 1 - true_labels * predicted_scores))
print(f"Hinge Loss: {hinge_loss}")
```

