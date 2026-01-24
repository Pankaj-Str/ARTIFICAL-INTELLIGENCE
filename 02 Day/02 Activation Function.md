# Activation Function


**Activation Functions in Neural Networks**

Think of a neuron in your brain (or in a neural network) like a tiny decision-maker.

It gets some numbers from previous neurons → does some math → and then has to **decide** whether it should "fire" (send a strong signal) or stay quiet.

The **activation function** is exactly that **decision rule** — it takes the raw number (after multiplication + bias) and turns it into the final output of the neuron.

Without activation function → the whole network would just be one big linear equation → no matter how many layers → it still behaves like a single straight line → very weak model.

### Most Popular Activation Functions (explained like a 12-year-old would understand)

| Name              | What it does (very simply)                          | Output range     | When people mostly use it today (2024–2025) | Feels like...                     |
|-------------------|------------------------------------------------------|------------------|-----------------------------------------------|------------------------------------|
| Sigmoid           | Squashes everything to 0–1                           | 0 to 1           | Almost never (old school)                     | "How sure are you? (0–100%)"      |
| Tanh              | Squashes everything to -1 to +1                      | –1 to +1         | Rarely now                                    | Centered version of sigmoid       |
| ReLU (most popular) | If number > 0 → keep it, if < 0 → make it 0         | 0 to +∞          | Almost everywhere in hidden layers            | "If it's positive → let it pass!" |
| Leaky ReLU        | Same as ReLU but very small negative slope           | –small to +∞     | When ReLU causes "dead neurons"               | ReLU but lets zombies through     |
| GELU              | Smooth version — kind of like ReLU but curved        | –small to +∞     | Very popular now (Transformers, BERT, GPT…)   | "Fancy modern ReLU"               |
| SiLU / Swish      | x × sigmoid(x) — smooth & sometimes better           | –small to +∞     | Very common in newer models                   | ReLU + a little brain             |
| Softmax           | Turns numbers into probabilities that sum to 1       | 0 to 1 (sum=1)   | **Only last layer** in classification         | "Choose one class – vote percentages" |

### Quick cheat-sheet – What to use in 2025

Layer type               | Most common & safe choice right now
------------------------|--------------------------------------
Hidden layers (CNNs)    | ReLU or GELU
Hidden layers (Transformers) | GELU or SiLU / Swish
Output – Binary classification | Sigmoid
Output – Multi-class classification | Softmax
Output – Regression (any number) | No activation / Linear

### Super simple picture in words

Raw number coming in → Activation function → Output

```
-3.2  → ReLU → 0          (dies / turns off)
-0.1  → ReLU → 0
 0.0  → ReLU → 0
+1.8  → ReLU → 1.8        (lives & passes signal)
+4.7  → ReLU → 4.7        (very excited neuron!)

-3.2  → GELU → ≈ -0.005   (tiny negative allowed)
+4.7  → GELU → ≈ 4.7      (almost same as ReLU)
```

**Quick memory trick**  
ReLU = "Rectified Linear Unit" → if positive → keep, else → kill (set to 0)

Most networks today are basically stacks of  
**Linear (weights + bias) → ReLU or GELU → Linear → ReLU or GELU → ...**

That's the magic combination that lets neural networks learn very complicated patterns.


----


We'll use **Python + NumPy only** (no deep learning libraries like TensorFlow or PyTorch at first — so it's very clear what's happening).

### Goal: 
See with your own eyes how ReLU, Sigmoid and a tiny network changes its behavior with/without activation.

```python
import numpy as np

# ───────────────────────────────────────────────
#   1. The most common activation functions
# ───────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)          # if x > 0 keep it, else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))      # squashes to 0 … 1

def linear(x):                       # no activation = straight line
    return x

# ───────────────────────────────────────────────
#   2. Tiny fake neuron example
# ───────────────────────────────────────────────

# Imagine one neuron that gets 3 inputs
inputs = np.array([1.0, -2.0, 0.5])   # ← our data example

weights = np.array([0.8, 1.2, -0.3])  # ← learned (or random) importance
bias    = 0.1

# Step 1: weighted sum (what the neuron calculates before decision)
z = np.dot(inputs, weights) + bias
print("Raw calculation (z) =", z)               # ← this is BEFORE activation

# Step 2: apply activation (the "decision")
print("\nDifferent activations give different outputs:\n")

print("Linear  (no activation) →", linear(z))
print("ReLU                       →", relu(z))
print("Sigmoid                    →", sigmoid(z))
print("Sigmoid rounded nicely    →", round(sigmoid(z), 4))

# ───────────────────────────────────────────────
#   3. Why activation matters – tiny 2-layer network
# ───────────────────────────────────────────────

print("\n=== Without any activation (bad) ===")

# Layer 1: 3 inputs → 4 hidden neurons
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros(4)

# Layer 2: 4 hidden → 1 output
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

# Forward pass — NO activation
hidden = np.dot(inputs, W1) + b1
output = np.dot(hidden, W2) + b2

print("Output without activation →", output[0])

print("\n=== With ReLU (good for most cases today) ===")

hidden = relu(np.dot(inputs, W1) + b1)          # ← activation here!
output = np.dot(hidden, W2) + b2

print("Output with ReLU           →", output[0])

print("\n=== With Sigmoid (old school style) ===")

hidden = sigmoid(np.dot(inputs, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)       # often used at the end

print("Output with Sigmoid        →", round(output[0], 4))
```

### What to do with this code (beginner practice plan)

1. Copy → paste into any Python environment (Google Colab, Jupyter, VSCode, even online python interpreters)
2. Run it once → see numbers
3. Change these lines and re-run each time:

   ```python
   inputs = np.array([1.0, -2.0, 0.5])   # ← try different numbers
   inputs = np.array([-1, -1, -1])       # ← all negative
   inputs = np.array([3, 4, 5])          # ← all big positive
   ```

4. Try replacing `relu` with `sigmoid` or remove it completely (`hidden = np.dot(...) + b1`)
5. Add `print(hidden)` after each version to see how many neurons "die" (become 0) with ReLU

### Quick Summary Table – What you will notice

| Version               | Output behavior                              | Good for beginners to see? |
|-----------------------|----------------------------------------------|-----------------------------|
| No activation         | Output is almost same no matter layers       | YES – shows why we need it  |
| Only ReLU             | Many values become 0 → network gets "selective" | YES – most common today     |
| Only Sigmoid          | Everything squeezed to 0–1                   | YES – old but educational   |
| ReLU hidden + Sigmoid output | Typical 2020–2025 classification network | Very good combo to remember |

Try running it a few times — change inputs, watch how ReLU "kills" negative values, and how without activation the network can't really learn interesting patterns.


