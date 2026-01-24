# Optimization Functions in Neural Networks

### First: What are optimizers? (Very simple story)

You trained a neural network → it makes predictions → you calculate how wrong it is → that's the **loss** (error).

Now you want to make the model **less wrong** next time.

→ You need to slightly change the **weights** (those numbers inside the network) so next time the loss becomes smaller.

**The optimizer is the "coach" that decides:**

- In which direction to change the weights  
- How big/small each step should be  
- How fast or carefully to move

Without a good optimizer → training is either super slow, gets stuck, or explodes!

### Most Popular Optimizers – 2025/2026 Style (for beginners)

| Optimizer       | Super simple explanation (like telling a friend)                          | Speed       | Stability   | When people use it most (2025–2026)              | Memory trick / feels like                  |
|-----------------|---------------------------------------------------------------------------|-------------|-------------|--------------------------------------------------|--------------------------------------------|
| **SGD**         | Classic: take a small step in the direction that reduces error            | Slow        | Okay        | Rarely alone now, but good for understanding     | Walking slowly downhill                    |
| **SGD + Momentum** | SGD but with "speed" — if going downhill, keep accelerating!            | Faster      | Better      | Still used in some research & simple models      | Bicycle going downhill with momentum       |
| **Adam**        | Very smart: remembers past directions + adapts step size for each weight  | Fast        | Very good   | Still #1 most used in practice (huge default)    | Personal coach who adjusts for everyone    |
| **AdamW**       | Adam + better weight decay (helps generalization a lot)                   | Fast        | Excellent   | Transformers, LLMs, almost everything now        | Adam but with diet control                 |
| **RMSprop**     | Good at handling changing landscapes (very popular in 2015–2018)          | Medium-Fast | Good        | Less now, but still in some RNNs/old projects    | Step size shrinks when bumpy road          |
| **Adagrad**     | Makes big updates early, tiny updates later (good for sparse data)        | Medium      | Okay        | Rarely now                                       | Learns quickly then becomes very careful   |
| **Lion** / **Sophia** / **AdEMAMix** | Newer 2023–2025 optimizers — sometimes beat AdamW on large models        | Fast        | Very good   | Gaining popularity in big model training         | New kids on the block                      |

### Quick 2025–2026 cheat sheet (what most people actually use)

Task / Model type               | Safest & most common optimizer right now
--------------------------------|-----------------------------------------
Almost everything (beginner)    | **Adam** or **AdamW**
Transformers / LLMs / BERT-like | **AdamW** + learning rate warmup + cosine decay
Computer Vision (ResNet, ViT)   | **AdamW** or **SGD + Momentum** (with scheduler)
Very large models (2025+)       | **AdamW**, **Lion**, sometimes **Sophia**
Quick experiments / toy models  | **Adam**

### Mini code example – See them in action (PyTorch – beginner style)

```python
import torch
import torch.optim as optim

# Suppose you have a tiny model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Different optimizers – just change this line!
optimizer = optim.Adam(model.parameters(), lr=0.001)          # ← most common
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # ← even better for most cases
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# In training loop you do:
# loss.backward()           # calculate gradients
# optimizer.step()          # ← optimizer makes the update!
# optimizer.zero_grad()     # clear old gradients
```

### One-line memory tricks

- **Adam** = "Adaptive Moments" → remembers speed + adapts step size → usually wins for beginners
- **AdamW** = Adam + better regularization → became the real king after 2019–2020
- If training is unstable / loss explodes → lower learning rate or try AdamW
- If training is too slow / stuck → try bigger learning rate + warmup or switch to Lion/Sophia (newer)

