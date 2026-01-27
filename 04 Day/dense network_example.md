## **Example** of a Dense Neural Network in Python using **Keras** (TensorFlow).

We’ll solve a classic toy problem:

**Predict whether a student will pass or fail**  
based on 3 features:  
- hours studied per day  
- attendance percentage  
- previous test score (out of 100)

### Goal
Binary classification → output 0 = Fail, 1 = Pass

### Step-by-step example

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────
# 1. Create fake (but realistic-looking) data
# ────────────────────────────────────────────────
np.random.seed(42)

# Generate 200 students
hours_studied    = np.random.uniform(1, 12, 200)          # 1 to 12 hours
attendance       = np.random.uniform(40, 100, 200)        # 40% to 100%
prev_test_score  = np.random.uniform(35, 98, 200)         # 35 to 98 marks

# Very simple rule to decide pass/fail (just for creating labels)
# (you would never do this in real life — this is only for demo)
pass_probability = (
    0.25 * hours_studied +
    0.004 * attendance +
    0.008 * prev_test_score
)
pass_probability = np.clip(pass_probability, 0, 1)   # keep between 0 and 1

y = (np.random.random(200) < pass_probability).astype(int)   # 0 or 1

X = np.column_stack((hours_studied, attendance, prev_test_score))

# ────────────────────────────────────────────────
# 2. Split + Normalize (very important for Dense networks)
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ────────────────────────────────────────────────
# 3. Build a small Dense network
# ────────────────────────────────────────────────
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),     # Input: 3 features
    Dense(8,  activation='relu'),                       # Hidden layer
    Dense(1,  activation='sigmoid')                     # Output: probability of PASS
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Show model summary (very useful to see number of parameters)
model.summary()

# ────────────────────────────────────────────────
# 4. Train
# ────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_split=0.2,    # use 20% of training data for validation
    epochs=50,
    batch_size=16,
    verbose=1
)

# ────────────────────────────────────────────────
# 5. Evaluate on test set
# ────────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# ────────────────────────────────────────────────
# 6. Make prediction on one new student (example)
# ────────────────────────────────────────────────
new_student = np.array([[8.5, 92, 78]])   # hours, attendance %, prev score
new_student_scaled = scaler.transform(new_student)

prob = model.predict(new_student_scaled)[0][0]
print(f"Probability of passing = {prob:.4f}")
print("Prediction:", "PASS" if prob > 0.5 else "FAIL")
```

### What you see in this example

| Layer              | Neurons | Activation   | Connections / Parameters (approx) |
|--------------------|---------|--------------|------------------------------------|
| Input              | 3       | —            | —                                  |
| Dense 1            | 16      | ReLU         | 3×16 + 16 = 64                     |
| Dense 2            | 8       | ReLU         | 16×8 + 8 = 136                     |
| Dense 3 (output)   | 1       | Sigmoid      | 8×1 + 1 = 9                        |
| **Total**          |         |              | **~209 parameters**                |

### Quick summary – why this is a Dense network

- Every input feature is connected to **every** neuron in the first hidden layer
- Every neuron in layer 1 is connected to **every** neuron in layer 2
- Every neuron in layer 2 is connected to the **single output** neuron

→ **Fully connected = Dense**
![image](https://github.com/user-attachments/assets/0f074201-7e6c-4b84-a595-6913d9615941)

