# Regularization:

Regularization = techniques that **force** the neural network to be simpler  
→ so it **generalizes better** (works well on new/unseen data)  
→ instead of just memorizing the training data perfectly

Think of it like this:

Without regularization → student who memorizes every answer for the exam questions  
With regularization → student who really understands the concepts (so can solve slightly different questions too)

### Most popular regularization methods (2025 view)

| Method                  | What it actually does                              | Simple analogy                              | When people use it most today          | Code example (Keras)                     |
|-------------------------|----------------------------------------------------|---------------------------------------------|----------------------------------------|------------------------------------------|
| **L2 Regularization**   | Adds penalty for **large weights**                 | "Don't make any single idea too important"  | Almost everywhere (default choice)     | `kernel_regularizer='l2'`                |
| **L1 Regularization**   | Adds penalty for **non-zero weights** → sparsity   | "Try to use as few ideas as possible"       | Feature selection, very deep nets      | `kernel_regularizer='l1'`                |
| **Dropout**             | Randomly "turns off" some neurons during training  | "Learn even if some team members are absent"| Almost every modern network            | `Dropout(0.3)`                           |
| **DropConnect**         | Randomly turns off some **connections** (weights)  | Slightly more random version of dropout     | Less common now                        | —                                        |
| **Batch Normalization** | Normalizes activations + has small regularization side-effect | "Keep the excitement level stable"          | Very common (especially in CNNs)       | `BatchNormalization()`                   |
| **Weight Decay**        | Same as L2 (just different name in some libraries) | —                                           | PyTorch, optimizers                    | `weight_decay=1e-4` in optimizer         |
| **Label Smoothing**     | Makes target labels softer (e.g. 0.9 instead of 1) | "Don't be 100% overconfident"               | Transformers, classification           | Built-in in many loss functions          |
| **Stochastic Depth**    | Randomly skips whole layers during training        | "Sometimes skip school but still pass"      | Very deep ResNets                      | Custom or in libraries                   |

### Most common combination in 2025 (very practical)

```python
model = Sequential([
    Dense(512, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # small L2
    BatchNormalization(),
    Dropout(0.3),                                               # quite strong dropout
    
    Dense(256, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])
```

### Quick "which one to use" cheat sheet

| Situation                                 | Recommended starting combo                     |
|-------------------------------------------|------------------------------------------------|
| Small dataset (<5k–10k samples)           | Dropout 0.2–0.5 + L2 1e-4 to 1e-3             |
| Medium dataset                            | Dropout 0.1–0.3 + L2 5e-5 to 5e-4             |
| Very deep CNN / Vision models             | BatchNorm + Dropout(0.1–0.3) + small L2       |
| Transformers / large language models      | Dropout + LayerNorm + Label Smoothing         |
| You see very high train acc, low val acc  | Increase dropout rate or L2 strength first    |
| Model is too slow to train                | Prefer BatchNorm over heavy dropout           |

### Visual intuition – what regularization does

Imagine the loss surface:

```
Without regularization: very sharp, deep valleys → easy to overfit
With regularization: surface becomes more smooth/wavy → harder to fit noise
```

