

### 1. NumPy version (like before – manual gradient descent)

```python
import numpy as np
import matplotlib.pyplot as plt

# Tiny dataset
X = np.array([50, 80, 120], dtype=float)   # house sizes
y = np.array([25, 40, 60], dtype=float)    # prices in lakhs

# Normalize for stability (very helpful in practice)
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# Parameters
w = 0.0          # weight
b = 0.0          # bias (intercept)
lr = 0.05        # learning rate (bigger now because we normalized)
epochs = 800

losses = []

for epoch in range(epochs):
    # Forward
    y_pred = w * X_norm + b
    
    # Loss (MSE)
    error = y_pred - y
    loss = (error ** 2).mean()
    losses.append(loss)
    
    # Gradients
    dw = (2 / len(X)) * (X_norm * error).sum()
    db = (2 / len(X)) * error.sum()
    
    # Update
    w -= lr * dw
    b -= lr * db
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | loss {loss:8.3f} | w {w:6.3f} | b {b:5.2f}")

# Results
print("\nFinal model:  price ≈", round(w,3), "× (normalized size) +", round(b,2))
print("Real weight on original scale ≈", round(w / X_std, 4))

# Plot loss
plt.plot(losses)
plt.title("Training Loss (NumPy version)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
```

### 2. PyTorch version (most popular way in deep learning today – 2025/2026 style)

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Data → tensors
X = torch.tensor([50., 80., 120.])
y = torch.tensor([25., 40., 60.])

# Normalize (very common in practice)
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

# Model: just one weight + bias
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

optimizer = optim.SGD([w, b], lr=0.1)   # SGD = Stochastic Gradient Descent
losses = []

for epoch in range(1200):
    # Forward
    y_pred = w * X_norm + b
    
    # Loss
    loss = ((y_pred - y) ** 2).mean()
    losses.append(loss.item())
    
    # Backward + optimize
    optimizer.zero_grad()      # ← very important!
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | loss {loss.item():8.4f} | w {w.item():6.3f} | b {b.item():5.2f}")

print("\nFinal PyTorch model:")
print(f"price ≈ {w.item()/X_std.item():.4f} × size + {b.item() - w.item()*X_mean.item()/X_std.item():.2f}")

plt.plot(losses)
plt.title("Training Loss (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
```

### 3. scikit-learn version (easiest – when you just want results fast)

```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data
X = np.array([[50], [80], [120]])   # 2D input expected
y = np.array([25, 40, 60])

# Scale features (almost always needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SGDRegressor = gradient descent under the hood
model = SGDRegressor(
    learning_rate='constant',
    eta0=0.05,           # initial learning rate
    max_iter=1500,
    tol=1e-5,
    random_state=42
)

model.fit(X_scaled, y)

print("scikit-learn results:")
print(f"Weight (slope)  = {model.coef_[0]:.4f}")
print(f"Bias (intercept)= {model.intercept_[0]:.2f}")
print(f"Final model     = {model.coef_[0]:.4f} × scaled_size + {model.intercept_[0]:.2f}")

# Predict on original data
y_pred = model.predict(X_scaled)

print("\nPredictions vs Real:")
for size, real, pred in zip(X.ravel(), y, y_pred):
    print(f"{size:3.0f} m² → real {real:5.1f} | pred {pred:5.1f}")

# Plot (simple)
plt.scatter(X, y, color='blue', label='Real')
plt.plot(X, y_pred, color='red', label='Prediction')
plt.xlabel("Size (m²)")
plt.ylabel("Price (lakhs)")
plt.legend()
plt.grid(True)
plt.show()
```

### Quick comparison table

| Version       | Lines of training code | Best for                  | Learning curve | Real-world usage (2026) |
|---------------|------------------------|---------------------------|----------------|--------------------------|
| NumPy         | ~15–20                 | Understanding GD deeply   | ★★★☆☆          | Teaching, prototypes     |
| PyTorch       | ~10–15                 | Deep learning, research   | ★★☆☆☆          | Most cutting-edge AI     |
| scikit-learn  | ~4–6                   | Fast results, production  | ★☆☆☆☆          | Quick ML, non-DL tasks   |

