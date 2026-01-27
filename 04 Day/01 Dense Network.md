# Dense Network
**Dense Network** (also called **Fully Connected Network** or **Dense Layer**) is the simplest and most classic type of neural network layer.

In very simple words:

**Every neuron is connected to every neuron in the previous layer.**

It's like saying:  
"No one is left out — everyone talks to everyone."

### Visual Comparison – Very Important to understand

Layer →          Input Layer          Dense Layer           Output

```
     x₁ ───────┐
               │
     x₂ ───────┼───→  n₁   n₂   n₃     ← every input connects to every neuron
               │     ╱ ╲  ╱ ╲  ╱ ╲
     x₃ ───────┘    ╱   ╲╱   ╲╱   ╲
                   ╱     ╲     ╲
                  n₁     n₂     n₃
```

In a **Dense layer** → full connections (all arrows exist)

In a **Convolutional layer** → only local connections (few arrows)

In a **Recurrent layer** → connections also go back in time

### Real-life analogy (very simple)

Imagine 4 friends (inputs) want to decide where to eat tonight.

They ask 3 other friends (neurons in dense layer) for their opinion.

**In a Dense network:**
- Each of the 4 friends tells **all 3** opinion friends what they want
- So there are 4 × 3 = **12 conversations**

That's exactly what a Dense layer does — full communication.

### Most common simple example most people see first

Task: Predict house price  
Inputs:  
- size (sq ft)  
- number of bedrooms  
- number of bathrooms  
- age of house

```python
Input (4 numbers)
     ↓
Dense layer (10 neurons)       ← every input connects to all 10 neurons
     ↓
Dense layer (8 neurons)        ← every of 10 connects to all 8
     ↓
Dense layer (1 neuron)         ← final price prediction
```

In Keras / TensorFlow it looks like this:

```python
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),    # 4 inputs → 10 neurons
    Dense(8, activation='relu'),
    Dense(1)                                           # output: price
])
```

### Quick Summary Table

| Property              | Dense Layer                     | Typical use case                     |
|-----------------------|----------------------------------|--------------------------------------|
| Connections           | Every input → every neuron      | Small/medium data, final layers      |
| Parameters (weights)  | Lots (input_size × neurons)     | Can become very big quickly          |
| Also called           | Fully Connected, FC             | —                                    |
| Main disadvantage     | Too many parameters, slow, overfits easily | —                                 |
| When we use it most   | At the end of CNNs / after Flatten | Classification / regression head    |

### One-line memory trick

**Dense = "Don't leave anyone out" layer**  
Every input talks to every neuron.

