# Sigmoid Function in Neural Networks:

 **Sigmoid** – one of the classic activation functions. I'll keep it super simple, like explaining to a beginner over vada pav.

We'll cover: What it is, math, pros/cons, when to use it in 2025–2026, a manual calculation example, code to visualize it, and a full neural network example with the Iris dataset (building on our previous chats).

#### Step 1: What is the Sigmoid Function? (The Basics)
- **Simple Story**: Imagine a neuron that needs to "decide" based on input. Sigmoid takes any number (from -∞ to +∞) and squashes it into a smooth value between 0 and 1.
  - Negative input? → Close to 0 (neuron "off").
  - Zero? → Exactly 0.5 (halfway).
  - Positive input? → Close to 1 (neuron "on").
- It's like a "probability maker" – great for binary decisions (yes/no, 0/1).
- In neural networks: Used as an **activation function** after the weighted sum (z = weights * inputs + bias).
- Without it: Networks stay linear and can't learn complex patterns (as we saw in our activation chat).

Key: Sigmoid was huge in the 1990s–2010s but is less common now (2025–2026) due to issues like vanishing gradients. Still, it's educational and used in output layers for binary classification.

#### Step 2: The Math Behind Sigmoid (Step-by-Step)
The formula is simple:  
**σ(z) = 1 / (1 + e^(-z))**  
Where:
- z = Your raw input (e.g., from linear layer).
- e ≈ 2.718 (Euler's number).
- ^ means "raised to power".

**Manual Calculation Example** (Step-by-Step with a Real Number):
Let's say z = 2.0 (a positive weighted sum).

1. Compute e^(-z) = e^(-2.0) ≈ 2.718^(-2) ≈ 0.1353.
2. Add 1: 1 + 0.1353 = 1.1353.
3. Divide 1 by that: 1 / 1.1353 ≈ 0.8808.
4. Result: σ(2.0) ≈ 0.88 (neuron is "mostly on").

Another one: z = -3.0 (negative).  
1. e^(-(-3)) = e^(3) ≈ 20.0855.  
2. 1 + 20.0855 = 21.0855.  
3. 1 / 21.0855 ≈ 0.0474.  
4. Result: Almost 0 (neuron "off").

For z=0: σ(0) = 1 / (1 + 1) = 0.5.

**Quick Table of Examples** (Real Values):

| Input z       | e^(-z)       | 1 + e^(-z)   | σ(z) ≈       | Interpretation          |
|---------------|--------------|--------------|--------------|-------------------------|
| -5.0          | 148.413     | 149.413     | 0.0067      | Very off (near 0)       |
| -1.0          | 2.718       | 3.718       | 0.269       | Somewhat off            |
| 0.0           | 1.0         | 2.0         | 0.5         | Neutral                 |
| 1.0           | 0.368       | 1.368       | 0.731       | Somewhat on             |
| 5.0           | 0.0067      | 1.0067      | 0.993       | Very on (near 1)        |

Notice: It's S-shaped (sigmoid curve) – smooth and differentiable (key for backprop).

#### Step 3: Pros and Cons of Sigmoid (2025–2026 Perspective)
| Pros                          | Cons                                      |
|-------------------------------|-------------------------------------------|
| Outputs 0–1 (like probabilities). | Vanishing gradients: For large |z|, gradients near 0 → slow training in deep nets. |
| Smooth and easy to differentiate. | Not zero-centered: Outputs always positive → zigzag updates in optimizers. |
| Simple to implement/understand. | Rarely used in hidden layers now (ReLU/GELU better). |
| Good for binary output layers. | Exploding gradients rare, but saturation (stuck at 0/1) common. |

**When to Use in 2025–2026**:
- Output layer for binary classification (with Binary Cross-Entropy loss).
- Avoid in hidden layers – use ReLU, Leaky ReLU, or GELU instead.
- In old papers or logistic regression (which is a 1-layer net with Sigmoid).

#### Step 4: Visualize Sigmoid (Code Example)
Let's plot it to see the S-shape. Here's beginner Python code with NumPy/Matplotlib (run in Jupyter/Colab).

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step-by-step: Generate z values from -10 to 10
z = np.linspace(-10, 10, 100)  # 100 points

# Compute sigmoid
sig = sigmoid(z)

# Plot
plt.plot(z, sig)
plt.title('Sigmoid Function')
plt.xlabel('Input z')
plt.ylabel('Sigmoid(z)')
plt.grid(True)
plt.axhline(0.5, color='r', linestyle='--')  # Neutral line
plt.show()
```

**What You'll See**: An S-curve from 0 to 1, steep around z=0, flat at edges.

#### Step 5: Full Example – Sigmoid in a Neural Network with Iris Dataset
Let's use Iris (150 flowers, 3 classes) like before. We'll build a simple net with Sigmoid in hidden layer, train it, and compare accuracy to ReLU (from our activation chat). This shows Sigmoid in action.

**Setup**: Use PyTorch. Load Iris via sklearn for ease.

**Code** (Full Tutorial – Copy-Paste into Colab):
```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Load and Prepare Data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Step 2: Define Network with Sigmoid
class SigmoidNN(nn.Module):
    def __init__(self):
        super(SigmoidNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Input 4 → Hidden 10
        self.sigmoid = nn.Sigmoid()   # Sigmoid activation!
        self.fc2 = nn.Linear(10, 3)   # Hidden → Output 3 classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)           # Apply Sigmoid here
        x = self.fc2(x)
        return x

# Step 3: Train the Model
model = SigmoidNN()
criterion = nn.CrossEntropyLoss()  # Cost function from last chat
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Step 4: Evaluate
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test).float().mean().item()
print(f"Test Accuracy with Sigmoid: {accuracy:.4f}")
```

**Step-by-Step What Happens**:
1. **Data**: 120 train, 30 test samples.
2. **Forward Pass**: For each input, compute z1 = fc1(x), then sigmoid(z1), then z2 = fc2(that).
3. **Loss**: Cross-Entropy compares predictions to true labels.
4. **Training**: Adam minimizes loss over 200 epochs – watch it drop from ~1.1 to ~0.3.
5. **Results** (Typical from my run): Accuracy ~0.9333. Compare to ReLU's ~0.9667 – Sigmoid works but slower/converges worse in deeper nets.

**Why Sigmoid Here?**: It squashes hidden outputs to 0–1, but notice potential saturation if many z are extreme.

#### Step 6: Practice and Next Steps
- Run the code: Change to `nn.ReLU()` and re-run – see better/faster training?
- Experiment: Add more hidden layers – Sigmoid might struggle (vanishing gradients).
- Math Challenge: Compute sigmoid(3.5) manually (answer: ~0.9707).

